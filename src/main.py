"""
Modular Schema Graph Builder

Extracts schema from PostgreSQL databases (including Supabase) and builds a knowledge graph in Neo4j.
Includes column statistics: distinct count, sample values, null percentage.
Uses async I/O for concurrent stats fetching.
"""

import asyncio
from dataclasses import dataclass
from neo4j import GraphDatabase


@dataclass
class PostgresCredentials:
    host: str
    port: int
    database: str
    user: str
    password: str
    
    @classmethod
    def from_supabase(cls, project_ref: str, password: str, region: str = "us-east-1") -> "PostgresCredentials":
        """Create credentials for a Supabase project using the connection pooler."""
        return cls(
            host=f"aws-1-{region}.pooler.supabase.com",
            port=5432,
            database="postgres",
            user=f"postgres.{project_ref}",
            password=password
        )


@dataclass
class Neo4jCredentials:
    uri: str
    user: str
    password: str


class PostgresConnector:
    """Async connector for PostgreSQL databases (including Supabase)."""
    
    SCHEMA_SQL = """
    SELECT t.table_name, c.column_name, c.data_type, c.is_nullable,
           c.column_default, 
           CASE WHEN pk.column_name IS NOT NULL THEN true ELSE false END as is_pk
    FROM information_schema.tables t
    JOIN information_schema.columns c 
        ON t.table_name = c.table_name AND t.table_schema = c.table_schema
    LEFT JOIN (
        SELECT ku.table_name, ku.column_name 
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage ku ON tc.constraint_name = ku.constraint_name
        WHERE tc.constraint_type = 'PRIMARY KEY' AND tc.table_schema = 'public'
    ) pk ON c.table_name = pk.table_name AND c.column_name = pk.column_name
    WHERE t.table_schema = 'public' AND t.table_type = 'BASE TABLE'
    ORDER BY t.table_name, c.ordinal_position;
    """
    
    FK_SQL = """
    SELECT tc.table_name, kcu.column_name, 
           ccu.table_name AS ref_table, ccu.column_name AS ref_column
    FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage kcu 
        ON tc.constraint_name = kcu.constraint_name
    JOIN information_schema.constraint_column_usage ccu 
        ON ccu.constraint_name = tc.constraint_name
    WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public';
    """
    
    def __init__(self, credentials: PostgresCredentials, max_concurrency: int = 10):
        self.credentials = credentials
        self.max_concurrency = max_concurrency
        self._pool = None
    
    async def connect(self) -> None:
        import asyncpg
        self._pool = await asyncpg.create_pool(
            host=self.credentials.host,
            port=self.credentials.port,
            database=self.credentials.database,
            user=self.credentials.user,
            password=self.credentials.password,
            ssl="require",
            min_size=1,
            max_size=self.max_concurrency
        )
    
    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None
    
    async def fetch_tables(self) -> list[dict]:
        """Fetch tables with row counts."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
            )
            tables = []
            for row in rows:
                name = row["table_name"]
                count = await conn.fetchval(f'SELECT COUNT(*) FROM "{name}"')
                tables.append({"name": name, "row_count": count})
            return tables
    
    async def fetch_columns(self) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(self.SCHEMA_SQL)
            return [
                {"table": r["table_name"], "name": r["column_name"], "type": r["data_type"],
                 "nullable": r["is_nullable"] == "YES", "default": r["column_default"], "is_pk": r["is_pk"]}
                for r in rows
            ]
    
    async def fetch_foreign_keys(self) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(self.FK_SQL)
            return [
                {"table": r["table_name"], "column": r["column_name"],
                 "ref_table": r["ref_table"], "ref_col": r["ref_column"]}
                for r in rows
            ]
    
    async def _fetch_column_stats(self, col: dict, semaphore: asyncio.Semaphore) -> dict:
        """Fetch stats for a single column with concurrency limit."""
        async with semaphore:
            table, column, col_type = col["table"], col["name"], col["type"]
            stats = {"distinct_count": 0, "sample_values": [], "null_count": 0, "total_count": 0}
            
            try:
                async with self._pool.acquire() as conn:
                    # Get total and null counts
                    row = await conn.fetchrow(f'''
                        SELECT COUNT(*) as total, COUNT(*) - COUNT("{column}") as nulls
                        FROM "{table}"
                    ''')
                    stats["total_count"] = row["total"]
                    stats["null_count"] = row["nulls"]
                    
                    # Get distinct count
                    stats["distinct_count"] = await conn.fetchval(
                        f'SELECT COUNT(DISTINCT "{column}") FROM "{table}"'
                    )
                    
                    # Get sample values (skip binary types)
                    skip_types = ('bytea', 'json', 'jsonb', 'xml')
                    if not any(t in col_type.lower() for t in skip_types):
                        rows = await conn.fetch(f'''
                            SELECT "{column}"::text as val, COUNT(*) as freq 
                            FROM "{table}" 
                            WHERE "{column}" IS NOT NULL 
                            GROUP BY "{column}" 
                            ORDER BY freq DESC 
                            LIMIT 5
                        ''')
                        stats["sample_values"] = [r["val"] for r in rows if r["val"] is not None]
            except Exception as e:
                print(f"Warning: Could not fetch stats for {table}.{column}: {e}")
            
            return {**col, **stats}
    
    async def fetch_all_column_stats(self, columns: list[dict]) -> list[dict]:
        """Fetch statistics for all columns concurrently."""
        semaphore = asyncio.Semaphore(self.max_concurrency)
        tasks = [self._fetch_column_stats(col, semaphore) for col in columns]
        return await asyncio.gather(*tasks)


class SchemaGraphBuilder:
    """Builds a Neo4j schema graph from a PostgreSQL database."""
    
    def __init__(self, db_connector: PostgresConnector, neo4j_credentials: Neo4jCredentials):
        self.connector = db_connector
        self.neo4j_creds = neo4j_credentials
        self._driver = None
    
    def _connect_neo4j(self) -> None:
        self._driver = GraphDatabase.driver(
            self.neo4j_creds.uri, 
            auth=(self.neo4j_creds.user, self.neo4j_creds.password)
        )
    
    async def close(self) -> None:
        await self.connector.close()
        if self._driver:
            self._driver.close()
    
    async def build(self, graph_name: str = "schema", clear: bool = True, include_stats: bool = True) -> dict:
        """Extract schema and build Neo4j graph."""
        await self.connector.connect()
        self._connect_neo4j()
        
        try:
            tables = await self.connector.fetch_tables()
            columns = await self.connector.fetch_columns()
            fks = await self.connector.fetch_foreign_keys()
            
            if include_stats:
                print(f"Fetching column statistics for {len(columns)} columns...")
                columns = await self.connector.fetch_all_column_stats(columns)
            
            self._build_graph(graph_name, tables, columns, fks, clear)
            
            return {
                "graph": graph_name,
                "tables": len(tables),
                "columns": len(columns),
                "foreign_keys": len(fks),
                "total_rows": sum(t["row_count"] for t in tables)
            }
        finally:
            await self.close()
    
    def _build_graph(self, name: str, tables: list, columns: list, fks: list, clear: bool):
        if clear:
            self._driver.execute_query(
                "MATCH (n) WHERE n:Database OR n:Table OR n:Column DETACH DELETE n"
            )
        
        self._driver.execute_query("MERGE (:Database {name: $name})", name=name)
        
        self._driver.execute_query("""
            MATCH (d:Database {name: $db})
            UNWIND $tables AS tbl
            MERGE (t:Table {name: tbl.name})
            SET t.row_count = tbl.row_count
            MERGE (d)-[:HAS_TABLE]->(t)
        """, db=name, tables=tables)
        
        self._driver.execute_query("""
            UNWIND $columns AS col
            MATCH (t:Table {name: col.table})
            MERGE (c:Column {table: col.table, name: col.name})
            SET c.type = col.type, 
                c.nullable = col.nullable,
                c.is_pk = col.is_pk, 
                c.default = col.default,
                c.distinct_count = col.distinct_count,
                c.sample_values = col.sample_values,
                c.null_count = col.null_count,
                c.total_count = col.total_count
            MERGE (t)-[:HAS_COLUMN]->(c)
        """, columns=columns)
        
        if fks:
            self._driver.execute_query("""
                UNWIND $fks AS fk
                MATCH (c1:Column {table: fk.table, name: fk.column})
                MATCH (c2:Column {table: fk.ref_table, name: fk.ref_col})
                MERGE (c1)-[:REFERENCES]->(c2)
            """, fks=fks)


def create_builder(credentials: PostgresCredentials, neo4j_credentials: Neo4jCredentials, max_concurrency: int = 10) -> SchemaGraphBuilder:
    """Factory function to create a SchemaGraphBuilder.
    
    Args:
        max_concurrency: Max concurrent database queries (default: 10)
    """
    connector = PostgresConnector(credentials, max_concurrency=max_concurrency)
    return SchemaGraphBuilder(connector, neo4j_credentials)


async def main():
    neo4j_creds = Neo4jCredentials(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )
    
    pg_creds = PostgresCredentials.from_supabase(
        project_ref="gjmssqesjbwcboybclbl",
        password="nIFcTJbRNZ8ITnZE",
        region="ap-southeast-1"
    )
    
    builder = create_builder(pg_creds, neo4j_creds, max_concurrency=10)
    result = await builder.build("my_database", include_stats=True)
    print(f"Built: {result}")


if __name__ == "__main__":
    asyncio.run(main())
"""
Modular Schema Graph Builder

Extracts schema from PostgreSQL databases (including Supabase) and builds a knowledge graph in Neo4j.
Includes column statistics, LLM-generated descriptions, and embeddings.
Uses async I/O for concurrent operations.
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
    def from_supabase(cls, project_ref: str, password: str, region: str) -> "PostgresCredentials":
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


class SchemaEnricher:
    """Generates descriptions and embeddings for schema elements using LangChain."""
    
    TABLE_PROMPT = """Describe this database table in 1-2 sentences.
Table: {name}
Columns: {columns}

Explain what real-world data this table stores and when someone would query it."""

    COLUMN_PROMPT = """Describe this database column in 1-2 sentences.
Table: {table}
Column: {name}
Type: {type}
Sample Values: {sample_values}

Explain what this column represents in plain business terms."""

    def __init__(self, llm_model: str, embed_model: str, max_concurrency: int):
        from langchain.chat_models import init_chat_model
        from langchain.embeddings import init_embeddings
        
        self.llm = init_chat_model(llm_model)
        self.embeddings = init_embeddings(embed_model)
        self.max_concurrency = max_concurrency
    
    async def _generate_description(self, prompt: str, semaphore: asyncio.Semaphore) -> str:
        """Generate a description using the LLM."""
        async with semaphore:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
    
    async def _generate_embedding(self, text: str, semaphore: asyncio.Semaphore) -> list[float]:
        """Generate an embedding for text."""
        async with semaphore:
            return await self.embeddings.aembed_query(text)
    
    async def enrich_tables(self, tables: list[dict], columns: list[dict]) -> list[dict]:
        """Add descriptions and embeddings to tables."""
        sem = asyncio.Semaphore(self.max_concurrency)
        
        # Group columns by table
        cols_by_table = {}
        for c in columns:
            cols_by_table.setdefault(c["table"], []).append(c["name"])
        
        async def enrich_table(t: dict) -> dict:
            col_list = ", ".join(cols_by_table.get(t["name"], []))
            prompt = self.TABLE_PROMPT.format(name=t["name"], columns=col_list)
            desc = await self._generate_description(prompt, sem)
            emb = await self._generate_embedding(desc, sem)
            return {**t, "description": desc, "embedding": emb}
        
        return await asyncio.gather(*[enrich_table(t) for t in tables])
    
    async def enrich_columns(self, columns: list[dict]) -> list[dict]:
        """Add descriptions and embeddings to columns."""
        sem = asyncio.Semaphore(self.max_concurrency)
        
        async def enrich_column(c: dict) -> dict:
            samples = c.get("sample_values", [])[:5]
            prompt = self.COLUMN_PROMPT.format(
                table=c["table"], name=c["name"], type=c["type"],
                sample_values=", ".join(str(s) for s in samples) if samples else "none"
            )
            desc = await self._generate_description(prompt, sem)
            emb = await self._generate_embedding(desc, sem)
            return {**c, "description": desc, "embedding": emb}
        
        return await asyncio.gather(*[enrich_column(c) for c in columns])


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
    
    def __init__(self, credentials: PostgresCredentials, max_concurrency: int):
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
        async with semaphore:
            table, column, col_type = col["table"], col["name"], col["type"]
            stats = {"distinct_count": 0, "sample_values": [], "null_count": 0, "total_count": 0}
            
            try:
                async with self._pool.acquire() as conn:
                    row = await conn.fetchrow(f'''
                        SELECT COUNT(*) as total, COUNT(*) - COUNT("{column}") as nulls
                        FROM "{table}"
                    ''')
                    stats["total_count"] = row["total"]
                    stats["null_count"] = row["nulls"]
                    
                    stats["distinct_count"] = await conn.fetchval(
                        f'SELECT COUNT(DISTINCT "{column}") FROM "{table}"'
                    )
                    
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
        semaphore = asyncio.Semaphore(self.max_concurrency)
        tasks = [self._fetch_column_stats(col, semaphore) for col in columns]
        return await asyncio.gather(*tasks)


class SchemaGraphBuilder:
    """Builds a Neo4j schema graph from a PostgreSQL database."""
    
    def __init__(self, db_connector: PostgresConnector, neo4j_credentials: Neo4jCredentials,
                 enricher: SchemaEnricher):
        self.connector = db_connector
        self.neo4j_creds = neo4j_credentials
        self.enricher = enricher
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
    
    async def build(self, graph_name: str, clear: bool, 
                    include_stats: bool, include_descriptions: bool) -> dict:
        """Extract schema and build Neo4j graph.
        
        Args:
            graph_name: Name for the database node
            clear: Whether to clear existing schema nodes
            include_stats: Fetch column statistics (distinct count, samples, etc.)
            include_descriptions: Generate LLM descriptions and embeddings (requires enricher)
        """
        await self.connector.connect()
        self._connect_neo4j()
        
        try:
            print("Fetching schema...")
            tables = await self.connector.fetch_tables()
            columns = await self.connector.fetch_columns()
            fks = await self.connector.fetch_foreign_keys()
            
            if include_stats:
                print(f"Fetching column statistics for {len(columns)} columns...")
                columns = await self.connector.fetch_all_column_stats(columns)
            
            if include_descriptions:
                print(f"Generating descriptions for {len(tables)} tables...")
                tables = await self.enricher.enrich_tables(tables, columns)
                print(f"Generating descriptions for {len(columns)} columns...")
                columns = await self.enricher.enrich_columns(columns)
            
            print("Building Neo4j graph...")
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
        
        # Create Table nodes
        self._driver.execute_query("""
            MATCH (d:Database {name: $db})
            UNWIND $tables AS tbl
            MERGE (t:Table {name: tbl.name})
            SET t.row_count = tbl.row_count,
                t.description = tbl.description
            MERGE (d)-[:HAS_TABLE]->(t)
            WITH t, tbl
            CALL db.create.setNodeVectorProperty(t, 'embedding', tbl.embedding)
        """, db=name, tables=tables)
        
        # Create Column nodes
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
                c.total_count = col.total_count,
                c.description = col.description
            MERGE (t)-[:HAS_COLUMN]->(c)
            WITH c, col
            CALL db.create.setNodeVectorProperty(c, 'embedding', col.embedding)
        """, columns=columns)
        
        if fks:
            self._driver.execute_query("""
                UNWIND $fks AS fk
                MATCH (c1:Column {table: fk.table, name: fk.column})
                MATCH (c2:Column {table: fk.ref_table, name: fk.ref_col})
                MERGE (c1)-[:REFERENCES]->(c2)
            """, fks=fks)


def create_builder(
    credentials: PostgresCredentials, 
    neo4j_credentials: Neo4jCredentials, 
    max_concurrency: int,
    llm_model: str,
    embed_model: str
) -> SchemaGraphBuilder:
    """Factory function to create a SchemaGraphBuilder."""
    connector = PostgresConnector(credentials, max_concurrency)
    enricher = SchemaEnricher(llm_model, embed_model, max_concurrency)
    return SchemaGraphBuilder(connector, neo4j_credentials, enricher)


async def main():
    neo4j_creds = Neo4jCredentials(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )
    
    pg_creds = PostgresCredentials.from_supabase(
        project_ref="",
        password="",
        region="ap-southeast-1"
    )
    
    builder = create_builder(
        credentials=pg_creds,
        neo4j_credentials=neo4j_creds,
        max_concurrency=5,
        llm_model="openai:gpt-5-mini",
        embed_model="openai:text-embedding-3-small"
    )
    
    result = await builder.build(
        graph_name="my_database",
        clear=True,
        include_stats=True,
        include_descriptions=True
    )
    print(f"Built: {result}")


if __name__ == "__main__":
    asyncio.run(main())
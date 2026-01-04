"""
Modular Schema Graph Builder

Extracts schema from PostgreSQL databases (including Supabase) and builds a knowledge graph in Neo4j.
Includes column statistics: distinct count, sample values, null percentage.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from neo4j import GraphDatabase


@dataclass
class PostgresCredentials:
    host: str
    port: int
    database: str
    user: str
    password: str
    
    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @classmethod
    def from_supabase(cls, project_ref: str, password: str, region: str = "us-east-1") -> "PostgresCredentials":
        """Create credentials for a Supabase project using the connection pooler.
        
        Args:
            project_ref: Your project reference (e.g., 'gjmssqesjbwcboybclbl')
            password: Database password from Supabase dashboard
            region: AWS region (check dashboard, e.g., 'us-east-1', 'ap-southeast-1')
        """
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


class DatabaseConnector(ABC):
    """Abstract base class for database connectors."""
    
    @abstractmethod
    def connect(self) -> None:
        pass
    
    @abstractmethod
    def close(self) -> None:
        pass
    
    @abstractmethod
    def fetch_tables(self) -> list[dict]:
        """Returns list of dicts: name, row_count"""
        pass
    
    @abstractmethod
    def fetch_columns(self) -> list[dict]:
        """Returns list of dicts: table, name, type, nullable, default, is_pk"""
        pass
    
    @abstractmethod
    def fetch_foreign_keys(self) -> list[dict]:
        """Returns list of dicts: table, column, ref_table, ref_col"""
        pass
    
    @abstractmethod
    def fetch_column_stats(self, table: str, column: str, col_type: str) -> dict:
        """Returns dict: distinct_count, sample_values, null_count, total_count"""
        pass


class PostgresConnector(DatabaseConnector):
    """Connector for PostgreSQL databases (including Supabase)."""
    
    DEFAULT_MAX_WORKERS = 5
    
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
    
    def __init__(self, credentials: PostgresCredentials, max_workers: int = None):
        self.credentials = credentials
        self._conn = None
        self.max_workers = max_workers or self.DEFAULT_MAX_WORKERS
        self._pool: Queue = None
    
    def connect(self) -> None:
        import psycopg2
        self._conn = psycopg2.connect(self.credentials.dsn, sslmode="require")
    
    def _new_connection(self):
        """Create a new database connection."""
        import psycopg2
        return psycopg2.connect(self.credentials.dsn, sslmode="require")
    
    def _init_pool(self):
        """Initialize connection pool for parallel operations."""
        self._pool = Queue()
        for _ in range(self.max_workers):
            self._pool.put(self._new_connection())
    
    def _close_pool(self):
        """Close all connections in the pool."""
        if self._pool:
            while not self._pool.empty():
                try:
                    conn = self._pool.get_nowait()
                    conn.close()
                except:
                    pass
            self._pool = None
    
    def close(self) -> None:
        self._close_pool()
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def _execute(self, sql: str, params: tuple = None) -> list[tuple]:
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    
    def fetch_tables(self) -> list[dict]:
        """Fetch tables with row counts."""
        rows = self._execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
        )
        tables = []
        for (name,) in rows:
            count_result = self._execute(f'SELECT COUNT(*) FROM "{name}"')
            tables.append({"name": name, "row_count": count_result[0][0]})
        return tables
    
    def fetch_columns(self) -> list[dict]:
        rows = self._execute(self.SCHEMA_SQL)
        return [
            {"table": r[0], "name": r[1], "type": r[2], 
             "nullable": r[3] == "YES", "default": r[4], "is_pk": r[5]}
            for r in rows
        ]
    
    def fetch_foreign_keys(self) -> list[dict]:
        rows = self._execute(self.FK_SQL)
        return [
            {"table": r[0], "column": r[1], "ref_table": r[2], "ref_col": r[3]}
            for r in rows
        ]
    
    def fetch_column_stats(self, table: str, column: str, col_type: str) -> dict:
        """Fetch column statistics: distinct count, sample values, null stats."""
        return self._fetch_stats_with_conn(self._conn, table, column, col_type)
    
    def _fetch_stats_with_conn(self, conn, table: str, column: str, col_type: str) -> dict:
        """Fetch column statistics using a specific connection."""
        stats = {"distinct_count": 0, "sample_values": [], "null_count": 0, "total_count": 0}
        
        def execute(sql):
            with conn.cursor() as cur:
                cur.execute(sql)
                return cur.fetchall()
        
        try:
            # Get total and null counts
            count_sql = f'''
                SELECT COUNT(*) as total, 
                       COUNT(*) - COUNT("{column}") as nulls
                FROM "{table}"
            '''
            result = execute(count_sql)
            stats["total_count"] = result[0][0]
            stats["null_count"] = result[0][1]
            
            # Get distinct count
            distinct_sql = f'SELECT COUNT(DISTINCT "{column}") FROM "{table}"'
            result = execute(distinct_sql)
            stats["distinct_count"] = result[0][0]
            
            # Get sample values (top 5 most common non-null values)
            # Skip for binary/bytea types
            skip_types = ('bytea', 'json', 'jsonb', 'xml')
            if not any(t in col_type.lower() for t in skip_types):
                sample_sql = f'''
                    SELECT "{column}"::text, COUNT(*) as freq 
                    FROM "{table}" 
                    WHERE "{column}" IS NOT NULL 
                    GROUP BY "{column}" 
                    ORDER BY freq DESC 
                    LIMIT 5
                '''
                result = execute(sample_sql)
                stats["sample_values"] = [r[0] for r in result if r[0] is not None]
        except Exception as e:
            print(f"Warning: Could not fetch stats for {table}.{column}: {e}")
        
        return stats
    
    def fetch_all_column_stats(self, columns: list[dict]) -> list[dict]:
        """Fetch statistics for all columns in parallel using a connection pool."""
        if not columns:
            return columns
        
        self._init_pool()
        
        def fetch_single(col: dict) -> dict:
            conn = self._pool.get()
            try:
                stats = self._fetch_stats_with_conn(conn, col["table"], col["name"], col["type"])
                return {**col, **stats}
            finally:
                self._pool.put(conn)
        
        try:
            enriched = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(fetch_single, col): i for i, col in enumerate(columns)}
                for future in as_completed(futures):
                    enriched.append(future.result())
            return enriched
        finally:
            self._close_pool()


class SchemaGraphBuilder:
    """Builds a Neo4j schema graph from any supported database."""
    
    def __init__(self, db_connector: DatabaseConnector, neo4j_credentials: Neo4jCredentials):
        self.connector = db_connector
        self.neo4j_creds = neo4j_credentials
        self._driver = None
    
    def _connect_neo4j(self) -> None:
        self._driver = GraphDatabase.driver(
            self.neo4j_creds.uri, 
            auth=(self.neo4j_creds.user, self.neo4j_creds.password)
        )
    
    def close(self) -> None:
        self.connector.close()
        if self._driver:
            self._driver.close()
    
    def build(self, graph_name: str = "schema", clear: bool = True, include_stats: bool = True) -> dict:
        """Extract schema and build Neo4j graph.
        
        Args:
            graph_name: Name for the database node
            clear: Whether to clear existing schema nodes
            include_stats: Whether to fetch column statistics (slower but more info)
        """
        self.connector.connect()
        self._connect_neo4j()
        
        try:
            tables = self.connector.fetch_tables()
            columns = self.connector.fetch_columns()
            fks = self.connector.fetch_foreign_keys()
            
            # Enrich columns with statistics
            if include_stats:
                print(f"Fetching column statistics for {len(columns)} columns ({self.connector.max_workers} workers)...")
                columns = self.connector.fetch_all_column_stats(columns)
            
            self._build_graph(graph_name, tables, columns, fks, clear)
            
            return {
                "graph": graph_name,
                "tables": len(tables),
                "columns": len(columns),
                "foreign_keys": len(fks),
                "total_rows": sum(t["row_count"] for t in tables)
            }
        finally:
            self.close()
    
    def _build_graph(self, name: str, tables: list, columns: list, fks: list, clear: bool):
        if clear:
            self._driver.execute_query(
                "MATCH (n) WHERE n:Database OR n:Table OR n:Column DETACH DELETE n"
            )
        
        self._driver.execute_query("MERGE (:Database {name: $name})", name=name)
        
        # Create Table nodes with row_count
        self._driver.execute_query("""
            MATCH (d:Database {name: $db})
            UNWIND $tables AS tbl
            MERGE (t:Table {name: tbl.name})
            SET t.row_count = tbl.row_count
            MERGE (d)-[:HAS_TABLE]->(t)
        """, db=name, tables=tables)
        
        # Create Column nodes with all stats
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


def create_builder(credentials: PostgresCredentials, neo4j_credentials: Neo4jCredentials, max_workers: int = 5) -> SchemaGraphBuilder:
    """Factory function to create a SchemaGraphBuilder.
    
    Args:
        credentials: PostgreSQL credentials
        neo4j_credentials: Neo4j credentials  
        max_workers: Number of parallel connections for fetching column stats (default: 5)
    """
    connector = PostgresConnector(credentials, max_workers=max_workers)
    return SchemaGraphBuilder(connector, neo4j_credentials)


# =============================================================================
# Usage Examples
# =============================================================================

if __name__ == "__main__":
    neo4j_creds = Neo4jCredentials(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )
    
    # Option 1: Direct PostgreSQL
    # pg_creds = PostgresCredentials(
    #     host="localhost",
    #     port=5432,
    #     database="mydb",
    #     user="postgres",
    #     password="password"
    # )
    
    # Option 2: Supabase (uses connection pooler for IPv4 compatibility)
    pg_creds = PostgresCredentials.from_supabase(
        project_ref="",
        password="",
        region="ap-southeast-1"
    )
    
    builder = create_builder(pg_creds, neo4j_creds, max_workers=5)
    
    # Set include_stats=False for faster extraction without column statistics
    result = builder.build("my_database", include_stats=True)
    print(f"Built: {result}")
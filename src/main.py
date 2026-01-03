"""
Modular Schema Graph Builder

Extracts schema from PostgreSQL databases (including Supabase) and builds a knowledge graph in Neo4j.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
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
            region: AWS region (check dashboard, e.g., 'us-east-1', 'us-west-1', 'ap-southeast-1')
        """
        return cls(
            host=f"aws-1-{region}.pooler.supabase.com",
            port=5432,  # Session mode (supports prepared statements)
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
    def fetch_tables(self) -> list[str]:
        pass
    
    @abstractmethod
    def fetch_columns(self) -> list[dict]:
        """Returns list of dicts: table, name, type, nullable, default, is_pk"""
        pass
    
    @abstractmethod
    def fetch_foreign_keys(self) -> list[dict]:
        """Returns list of dicts: table, column, ref_table, ref_col"""
        pass


class PostgresConnector(DatabaseConnector):
    """Connector for PostgreSQL databases (including Supabase)."""
    
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
    
    def __init__(self, credentials: PostgresCredentials):
        self.credentials = credentials
        self._conn = None
    
    def connect(self) -> None:
        import psycopg2
        self._conn = psycopg2.connect(self.credentials.dsn, sslmode="require")
    
    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def _execute(self, sql: str) -> list[tuple]:
        with self._conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()
    
    def fetch_tables(self) -> list[str]:
        rows = self._execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
        )
        return [r[0] for r in rows]
    
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
    
    def build(self, graph_name: str = "schema", clear: bool = True) -> dict:
        """Extract schema and build Neo4j graph."""
        self.connector.connect()
        self._connect_neo4j()
        
        try:
            tables = self.connector.fetch_tables()
            columns = self.connector.fetch_columns()
            fks = self.connector.fetch_foreign_keys()
            
            self._build_graph(graph_name, tables, columns, fks, clear)
            
            return {
                "graph": graph_name,
                "tables": len(tables),
                "columns": len(columns),
                "foreign_keys": len(fks)
            }
        finally:
            self.close()
    
    def _build_graph(self, name: str, tables: list, columns: list, fks: list, clear: bool):
        if clear:
            self._driver.execute_query(
                "MATCH (n) WHERE n:Database OR n:Table OR n:Column DETACH DELETE n"
            )
        
        self._driver.execute_query("MERGE (:Database {name: $name})", name=name)
        
        self._driver.execute_query("""
            MATCH (d:Database {name: $db})
            UNWIND $tables AS tbl
            MERGE (t:Table {name: tbl})
            MERGE (d)-[:HAS_TABLE]->(t)
        """, db=name, tables=tables)
        
        self._driver.execute_query("""
            UNWIND $columns AS col
            MATCH (t:Table {name: col.table})
            MERGE (c:Column {table: col.table, name: col.name})
            SET c.type = col.type, 
                c.nullable = col.nullable,
                c.is_pk = col.is_pk, 
                c.default = col.default
            MERGE (t)-[:HAS_COLUMN]->(c)
        """, columns=columns)
        
        if fks:
            self._driver.execute_query("""
                UNWIND $fks AS fk
                MATCH (c1:Column {table: fk.table, name: fk.column})
                MATCH (c2:Column {table: fk.ref_table, name: fk.ref_col})
                MERGE (c1)-[:REFERENCES]->(c2)
            """, fks=fks)


def create_builder(credentials: PostgresCredentials, neo4j_credentials: Neo4jCredentials) -> SchemaGraphBuilder:
    """Factory function to create a SchemaGraphBuilder."""
    connector = PostgresConnector(credentials)
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
        region="ap-southeast-1"  # Check your dashboard for the correct region
    )
    
    builder = create_builder(pg_creds, neo4j_creds)
    result = builder.build("my_database")
    print(f"Built: {result}")
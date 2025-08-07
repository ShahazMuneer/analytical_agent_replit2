import pandas as pd
import sqlite3
import os
from typing import Optional, Dict, Any
import streamlit as st

try:
    import psycopg2
    from sqlalchemy import create_engine, text
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import pymysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    from pyhive import hive
    HIVE_AVAILABLE = True
except ImportError:
    HIVE_AVAILABLE = False

try:
    import teradatasql
    TERADATA_AVAILABLE = True
except ImportError:
    TERADATA_AVAILABLE = False

# Also support Teradata via ODBC
try:
    import pyodbc
    ODBC_AVAILABLE = True
except ImportError:
    ODBC_AVAILABLE = False

class DatabaseConnector:
    def __init__(self):
        self.connection = None
        self.db_type = None
        self.connection_params = {}
    
    def get_available_databases(self) -> list:
        """Get list of available database types"""
        available = ['SQLite (Local)']
        
        if POSTGRES_AVAILABLE:
            available.append('PostgreSQL')
        if MYSQL_AVAILABLE:
            available.append('MySQL')
        if HIVE_AVAILABLE:
            available.append('Hive (Hadoop)')
        if TERADATA_AVAILABLE or ODBC_AVAILABLE:
            available.append('Teradata')
        if ODBC_AVAILABLE:
            available.append('SQL Server (ODBC)')
            available.append('Oracle (ODBC)')
        
        return available
    
    def connect_sqlite(self, db_path: str = "analytics.db") -> bool:
        """Connect to SQLite database"""
        try:
            self.connection = sqlite3.connect(db_path)
            self.db_type = 'sqlite'
            self.connection_params = {'db_path': db_path}
            return True
        except Exception as e:
            st.error(f"SQLite connection failed: {str(e)}")
            return False
    
    def connect_postgresql(self, host: str, port: str, database: str, 
                          username: str, password: str) -> bool:
        """Connect to PostgreSQL database"""
        if not POSTGRES_AVAILABLE:
            st.error("PostgreSQL driver not available. Install psycopg2-binary.")
            return False
        
        try:
            connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            engine = create_engine(connection_string)
            self.connection = engine
            self.db_type = 'postgresql'
            self.connection_params = {
                'host': host, 'port': port, 'database': database,
                'username': username, 'password': password
            }
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            st.error(f"PostgreSQL connection failed: {str(e)}")
            return False
    
    def connect_mysql(self, host: str, port: str, database: str,
                     username: str, password: str) -> bool:
        """Connect to MySQL database"""
        if not MYSQL_AVAILABLE:
            st.error("MySQL driver not available. Install PyMySQL.")
            return False
        
        try:
            connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
            engine = create_engine(connection_string)
            self.connection = engine
            self.db_type = 'mysql'
            self.connection_params = {
                'host': host, 'port': port, 'database': database,
                'username': username, 'password': password
            }
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            st.error(f"MySQL connection failed: {str(e)}")
            return False
    
    def connect_hive(self, host: str, port: str, database: str,
                    username: str) -> bool:
        """Connect to Hive database"""
        if not HIVE_AVAILABLE:
            st.error("Hive driver not available. Install pyhive.")
            return False
        
        try:
            connection_string = f"hive://{username}@{host}:{port}/{database}"
            engine = create_engine(connection_string)
            self.connection = engine
            self.db_type = 'hive'
            self.connection_params = {
                'host': host, 'port': port, 'database': database,
                'username': username
            }
            return True
        except Exception as e:
            st.error(f"Hive connection failed: {str(e)}")
            return False
    
    def connect_teradata(self, host: str, username: str, password: str) -> bool:
        """Connect to Teradata database"""
        if not TERADATA_AVAILABLE:
            st.error("Teradata driver not available. Install teradatasql.")
            return False
        
        try:
            connection_string = f"teradatasql://{username}:{password}@{host}"
            engine = create_engine(connection_string)
            self.connection = engine
            self.db_type = 'teradata'
            self.connection_params = {
                'host': host, 'username': username, 'password': password
            }
            return True
        except Exception as e:
            st.error(f"Teradata connection failed: {str(e)}")
            return False
    
    def execute_query(self, query: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Execute SQL query and return results"""
        if not self.connection:
            raise Exception("No database connection established")
        
        try:
            if self.db_type == 'sqlite':
                df = pd.read_sql_query(query, self.connection)
            else:
                # For SQLAlchemy engines
                df = pd.read_sql_query(query, self.connection)
            
            if limit and len(df) > limit:
                return df.head(limit)
            
            return df
        except Exception as e:
            raise Exception(f"Query execution failed: {str(e)}")
    
    def get_table_list(self) -> list:
        """Get list of tables in the database"""
        if not self.connection:
            return []
        
        try:
            if self.db_type == 'sqlite':
                cursor = self.connection.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
            elif self.db_type == 'postgresql':
                query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                """
                df = pd.read_sql_query(query, self.connection)
                tables = df['table_name'].tolist()
            elif self.db_type == 'mysql':
                query = "SHOW TABLES"
                df = pd.read_sql_query(query, self.connection)
                tables = df.iloc[:, 0].tolist()
            else:
                tables = []
            
            return tables
        except Exception as e:
            st.error(f"Failed to get table list: {str(e)}")
            return []
    
    def get_table_schema(self, table_name: str) -> pd.DataFrame:
        """Get schema information for a table"""
        if not self.connection:
            return pd.DataFrame()
        
        try:
            if self.db_type == 'sqlite':
                query = f"PRAGMA table_info({table_name})"
                return pd.read_sql_query(query, self.connection)
            elif self.db_type == 'postgresql':
                query = f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_name = '{table_name}'
                """
                return pd.read_sql_query(query, self.connection)
            elif self.db_type == 'mysql':
                query = f"DESCRIBE {table_name}"
                return pd.read_sql_query(query, self.connection)
            else:
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Failed to get table schema: {str(e)}")
            return pd.DataFrame()
    
    def test_connection(self) -> bool:
        """Test if the current connection is working"""
        if not self.connection:
            return False
        
        try:
            if self.db_type == 'sqlite':
                cursor = self.connection.cursor()
                cursor.execute("SELECT 1")
                return True
            else:
                with self.connection.connect() as conn:
                    conn.execute(text("SELECT 1"))
                return True
        except:
            return False
    
    def close_connection(self):
        """Close the database connection"""
        if self.connection:
            if self.db_type == 'sqlite':
                self.connection.close()
            else:
                self.connection.dispose()
            self.connection = None
            self.db_type = None
            self.connection_params = {}
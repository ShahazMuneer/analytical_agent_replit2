import streamlit as st
import pandas as pd
import sqlite3
import psycopg2
import pymysql
import sqlalchemy
from sqlalchemy import create_engine, text
import logging
import json
from typing import Dict, Any, Optional, Tuple
import os

logger = logging.getLogger(__name__)

class UniversalDatabaseConnector:
    """Universal database connector supporting multiple database types"""
    
    def __init__(self):
        self.connection = None
        self.engine = None
        self.db_type = None
        
    def create_connection_string(self, db_config: Dict[str, Any]) -> str:
        """Create connection string based on database type"""
        db_type = db_config['type'].lower()
        
        if db_type == 'sqlite':
            return f"sqlite:///{db_config.get('database', 'analytics.db')}"
        
        elif db_type == 'postgresql':
            return f"postgresql://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        
        elif db_type == 'mysql':
            return f"mysql+pymysql://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        
        elif db_type == 'oracle':
            return f"oracle+cx_oracle://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        
        elif db_type == 'sqlserver':
            return f"mssql+pyodbc://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}?driver=ODBC+Driver+17+for+SQL+Server"
        
        elif db_type == 'teradata':
            return f"teradatasql://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config.get('port', 1025)}/DBC"
        
        elif db_type == 'hive':
            return f"hive://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config.get('port', 10000)}/{db_config['database']}"
        
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    def test_connection(self, db_config: Dict[str, Any]) -> Tuple[bool, str]:
        """Test database connection"""
        try:
            connection_string = self.create_connection_string(db_config)
            engine = create_engine(connection_string)
            
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            return True, "Connection successful"
            
        except Exception as e:
            return False, str(e)
    
    def connect(self, db_config: Dict[str, Any]) -> bool:
        """Connect to database"""
        try:
            connection_string = self.create_connection_string(db_config)
            self.engine = create_engine(connection_string)
            self.db_type = db_config['type']
            
            # Test the connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            # Store connection config in session state
            st.session_state.db_config = db_config
            st.session_state.connected = True
            
            return True
            
        except Exception as e:
            st.error(f"Connection failed: {str(e)}")
            return False
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame"""
        if not self.engine:
            raise Exception("No database connection established")
        
        try:
            df = pd.read_sql(query, self.engine)
            return df
            
        except Exception as e:
            raise Exception(f"Query execution failed: {str(e)}")
    
    def get_table_info(self) -> Dict[str, Any]:
        """Get database schema information"""
        if not self.engine:
            return {}
        
        try:
            # Get table names
            if self.db_type.lower() == 'sqlite':
                tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
            else:
                tables_query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                """
            
            tables_df = pd.read_sql(tables_query, self.engine)
            tables = tables_df.iloc[:, 0].tolist()
            
            # Get column info for each table
            schema_info = {}
            for table in tables[:10]:  # Limit to first 10 tables
                try:
                    if self.db_type.lower() == 'sqlite':
                        cols_query = f"PRAGMA table_info({table})"
                        cols_df = pd.read_sql(cols_query, self.engine)
                        columns = cols_df['name'].tolist() if not cols_df.empty else []
                    else:
                        cols_query = f"""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = '{table}'
                        """
                        cols_df = pd.read_sql(cols_query, self.engine)
                        columns = cols_df['column_name'].tolist() if not cols_df.empty else []
                    
                    schema_info[table] = {
                        'columns': columns,
                        'row_count': self._get_table_row_count(table)
                    }
                except:
                    schema_info[table] = {'columns': [], 'row_count': 0}
            
            return {
                'tables': tables,
                'schema': schema_info,
                'total_tables': len(tables)
            }
            
        except Exception as e:
            logger.error(f"Failed to get table info: {str(e)}")
            return {}
    
    def _get_table_row_count(self, table_name: str) -> int:
        """Get row count for a specific table"""
        try:
            count_query = f"SELECT COUNT(*) as count FROM {table_name}"
            result = pd.read_sql(count_query, self.engine)
            return int(result['count'].iloc[0])
        except:
            return 0
    
    def disconnect(self):
        """Disconnect from database"""
        if self.engine:
            self.engine.dispose()
            self.engine = None
        
        if 'connected' in st.session_state:
            st.session_state.connected = False
        if 'db_config' in st.session_state:
            del st.session_state.db_config

def render_database_connection_ui():
    """Render the database connection interface"""
    st.markdown("""
    <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;'>
        <h1 style='color: white; margin: 0; font-size: 2.5rem;'>🚀 AI Analytics Platform</h1>
        <p style='color: white; margin: 0.5rem 0; font-size: 1.2rem;'>Connect any database and get instant AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize connector
    if 'db_connector' not in st.session_state:
        st.session_state.db_connector = UniversalDatabaseConnector()
    
    # Connection status
    if st.session_state.get('connected', False):
        st.success(f"✅ Connected to {st.session_state.db_config['type']} database")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Disconnect", type="secondary"):
                st.session_state.db_connector.disconnect()
                st.rerun()
        
        return True
    
    # Database selection
    st.markdown("### 🔌 Database Connection")
    
    # Database type cards
    col1, col2, col3 = st.columns(3)
    
    db_options = {
        "SQLite": {"icon": "🗃️", "desc": "Local file database", "demo": True},
        "PostgreSQL": {"icon": "🐘", "desc": "Advanced open source", "demo": False},
        "MySQL": {"icon": "🐬", "desc": "Popular web database", "demo": False},
        "Oracle": {"icon": "🏢", "desc": "Enterprise database", "demo": False},
        "SQL Server": {"icon": "🔷", "desc": "Microsoft platform", "demo": False},
        "Teradata": {"icon": "🏦", "desc": "Enterprise data warehouse", "demo": False},
        "Hive": {"icon": "🐝", "desc": "Big data warehouse", "demo": False}
    }
    
    selected_db = st.selectbox(
        "Choose Database Type:",
        list(db_options.keys()),
        index=0
    )
    
    # Connection form
    with st.form("db_connection_form"):
        st.markdown(f"### {db_options[selected_db]['icon']} {selected_db} Connection")
        
        if selected_db == "SQLite":
            st.info("💡 SQLite demo mode with sample data")
            db_config = {
                'type': 'sqlite',
                'database': 'analytics.db'
            }
            
        elif selected_db == "PostgreSQL":
            # Pre-filled with provided credentials
            st.info("💡 PostgreSQL configured with sample data")
            col1, col2 = st.columns(2)
            
            with col1:
                host = st.text_input("Host", value="localhost")
                username = st.text_input("Username", value="postgres")
                database = st.text_input("Database Name", value="postgres")
                
            with col2:
                port = st.number_input("Port", value=5432)
                password = st.text_input("Password", value="rootpo", type="password")
                
            db_config = {
                'type': 'postgresql',
                'host': host,
                'port': port,
                'username': username,
                'password': password,
                'database': database
            }
            
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                host = st.text_input("Host", value="localhost")
                username = st.text_input("Username")
                database = st.text_input("Database Name")
                
            with col2:
                port_defaults = {"PostgreSQL": 5432, "MySQL": 3306, "Oracle": 1521, "SQL Server": 1433}
                port = st.number_input("Port", value=port_defaults.get(selected_db, 5432))
                password = st.text_input("Password", type="password")
                
            db_config = {
                'type': selected_db.lower().replace(" ", ""),
                'host': host,
                'port': port,
                'username': username,
                'password': password,
                'database': database
            }
        
        submitted = st.form_submit_button("Connect Database", type="primary")
        
        if submitted:
            if selected_db == "SQLite":
                # Initialize SQLite with sample data
                initialize_sample_database()
                
            with st.spinner("Testing connection..."):
                if st.session_state.db_connector.connect(db_config):
                    st.success("✅ Connected successfully!")
                    st.rerun()
    
    return False

def initialize_sample_database():
    """Initialize SQLite database with comprehensive sample data"""
    conn = sqlite3.connect('analytics.db')
    cursor = conn.cursor()
    
    # Create comprehensive sample tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS customers (
        customer_id INTEGER PRIMARY KEY,
        name TEXT,
        email TEXT,
        city TEXT,
        country TEXT,
        signup_date DATE,
        customer_type TEXT,
        credit_score INTEGER
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS products (
        product_id INTEGER PRIMARY KEY,
        product_name TEXT,
        category TEXT,
        price DECIMAL(10,2),
        cost DECIMAL(10,2),
        launch_date DATE
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sales (
        sale_id INTEGER PRIMARY KEY,
        customer_id INTEGER,
        product_id INTEGER,
        sale_date DATE,
        quantity INTEGER,
        total_amount DECIMAL(10,2),
        discount DECIMAL(5,2),
        sales_rep TEXT,
        region TEXT,
        FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
        FOREIGN KEY (product_id) REFERENCES products (product_id)
    )
    ''')
    
    # Check if data exists
    cursor.execute('SELECT COUNT(*) FROM customers')
    if cursor.fetchone()[0] == 0:
        
        # Sample customers
        customers_data = [
            (1, 'Acme Corp', 'info@acme.com', 'New York', 'USA', '2023-01-15', 'Enterprise', 750),
            (2, 'Tech Solutions', 'contact@tech.com', 'London', 'UK', '2023-02-20', 'SMB', 680),
            (3, 'Global Industries', 'hello@global.com', 'Tokyo', 'Japan', '2023-03-10', 'Enterprise', 820),
            (4, 'Startup Inc', 'team@startup.com', 'San Francisco', 'USA', '2023-04-05', 'SMB', 650),
            (5, 'Mega Bank', 'banking@mega.com', 'Frankfurt', 'Germany', '2023-01-30', 'Enterprise', 900),
            (6, 'Finance Plus', 'info@finplus.com', 'Singapore', 'Singapore', '2023-05-15', 'SMB', 720),
            (7, 'Data Corp', 'data@corp.com', 'Toronto', 'Canada', '2023-02-28', 'Enterprise', 780),
            (8, 'Cloud Services', 'cloud@services.com', 'Sydney', 'Australia', '2023-06-10', 'SMB', 690)
        ]
        
        cursor.executemany(
            'INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            customers_data
        )
        
        # Sample products
        products_data = [
            (1, 'Analytics Pro', 'Software', 999.00, 200.00, '2023-01-01'),
            (2, 'Data Insights', 'Software', 1499.00, 300.00, '2023-01-15'),
            (3, 'Business Intelligence Suite', 'Software', 2499.00, 500.00, '2023-02-01'),
            (4, 'Reporting Tools', 'Software', 799.00, 150.00, '2023-02-15'),
            (5, 'Forecasting Engine', 'Software', 1999.00, 400.00, '2023-03-01'),
            (6, 'Dashboard Creator', 'Software', 899.00, 180.00, '2023-03-15')
        ]
        
        cursor.executemany(
            'INSERT INTO products VALUES (?, ?, ?, ?, ?, ?)',
            products_data
        )
        
        # Sample sales with more comprehensive data
        sales_data = []
        import random
        from datetime import datetime, timedelta
        
        base_date = datetime(2024, 1, 1)
        regions = ['North America', 'Europe', 'Asia Pacific', 'South America']
        reps = ['Alice Johnson', 'Bob Smith', 'Carol Davis', 'David Wilson']
        
        for i in range(1, 151):  # 150 sales records
            customer_id = random.randint(1, 8)
            product_id = random.randint(1, 6)
            days_offset = random.randint(0, 200)
            sale_date = (base_date + timedelta(days=days_offset)).strftime('%Y-%m-%d')
            quantity = random.randint(1, 10)
            
            # Get product price (simplified)
            product_prices = {1: 999, 2: 1499, 3: 2499, 4: 799, 5: 1999, 6: 899}
            base_amount = product_prices[product_id] * quantity
            discount = random.uniform(0, 0.2)  # 0-20% discount
            total_amount = base_amount * (1 - discount)
            
            sales_data.append((
                i, customer_id, product_id, sale_date, quantity,
                total_amount, discount * 100, random.choice(reps), random.choice(regions)
            ))
        
        cursor.executemany(
            'INSERT INTO sales VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            sales_data
        )
        
        # Create a comprehensive view
        cursor.execute('''
        CREATE VIEW IF NOT EXISTS sales_analysis_view AS
        SELECT 
            s.sale_id,
            s.sale_date,
            c.name as customer_name,
            c.customer_type,
            c.city,
            c.country,
            c.credit_score,
            p.product_name,
            p.category,
            p.price as product_price,
            s.quantity,
            s.total_amount,
            s.discount,
            s.sales_rep,
            s.region,
            (s.total_amount - (p.cost * s.quantity)) as profit
        FROM sales s
        JOIN customers c ON s.customer_id = c.customer_id
        JOIN products p ON s.product_id = p.product_id
        ''')
    
    conn.commit()
    conn.close()
"""
Database Manager for Enterprise Analytics Hub
Handles multiple database connections and operations
"""

import sqlite3
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import random
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = "analytics.db"):
        """Initialize database manager with SQLite as default"""
        self.db_path = db_path
        self.engine = None
        self.connection_type = "sqlite"
        self.setup_sqlite_database()
        
    def setup_sqlite_database(self):
        """Setup SQLite database with sample data"""
        try:
            # Create SQLAlchemy engine for SQLite
            self.engine = create_engine(f'sqlite:///{self.db_path}', echo=False)
            
            # Check if database exists and has data
            if not os.path.exists(self.db_path) or not self._has_sample_data():
                logger.info("Creating sample database...")
                self._create_sample_data()
                self._create_analytics_view()
                logger.info("Sample database created successfully")
            else:
                logger.info("Using existing database")
                
        except Exception as e:
            logger.error(f"Error setting up SQLite database: {str(e)}")
            raise
    
    def connect_external_database(self, connection_params: Dict[str, str]) -> bool:
        """Connect to external database (PostgreSQL, MySQL, etc.)"""
        try:
            db_type = connection_params.get('type', 'postgresql').lower()
            host = connection_params.get('host', 'localhost')
            port = connection_params.get('port', '5432')
            database = connection_params.get('database', '')
            username = connection_params.get('username', '')
            password = connection_params.get('password', '')
            
            # Build connection string based on database type
            if db_type == 'postgresql':
                conn_str = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            elif db_type == 'mysql':
                conn_str = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
            elif db_type == 'oracle':
                conn_str = f"oracle+cx_oracle://{username}:{password}@{host}:{port}/{database}"
            elif db_type == 'sqlserver':
                conn_str = f"mssql+pyodbc://{username}:{password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
            
            # Create new engine
            self.engine = create_engine(conn_str, echo=False)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self.connection_type = db_type
            logger.info(f"Successfully connected to {db_type} database")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to external database: {str(e)}")
            return False
    
    def _has_sample_data(self) -> bool:
        """Check if sample data exists in the database"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='sales'"))
                return result.fetchone() is not None
        except:
            return False
    
    def _create_sample_data(self):
        """Create comprehensive sample business data"""
        # Sample companies
        companies = [
            {'id': 1, 'name': 'TechCorp Solutions', 'country': 'USA', 'industry': 'Technology'},
            {'id': 2, 'name': 'Global Finance Ltd', 'country': 'UK', 'industry': 'Finance'},
            {'id': 3, 'name': 'Innovation Systems', 'country': 'Germany', 'industry': 'Manufacturing'},
            {'id': 4, 'name': 'Digital Dynamics', 'country': 'Canada', 'industry': 'Software'},
            {'id': 5, 'name': 'Future Industries', 'country': 'Australia', 'industry': 'Technology'}
        ]
        
        # Sample customers
        customers = []
        customer_names = [
            'Alice Johnson', 'Bob Smith', 'Carol Davis', 'David Wilson', 'Eva Brown',
            'Frank Miller', 'Grace Lee', 'Henry Taylor', 'Ivy Chen', 'Jack Anderson',
            'Kate Williams', 'Luke Martinez', 'Mary Garcia', 'Nick Thompson', 'Olivia Rodriguez',
            'Paul Jackson', 'Quinn White', 'Rachel Harris', 'Sam Clark', 'Tina Lewis'
        ]
        
        countries = ['USA', 'UK', 'Germany', 'France', 'Canada', 'Australia', 'Japan', 'Singapore']
        
        for i, name in enumerate(customer_names):
            customers.append({
                'id': i + 1,
                'name': name,
                'country': random.choice(countries),
                'created_at': datetime.now() - timedelta(days=random.randint(30, 365))
            })
        
        # Sample products
        products = [
            {'id': 1, 'name': 'Enterprise Software License', 'sku': 'ESL-001', 'category': 'Software'},
            {'id': 2, 'name': 'Cloud Storage Plan', 'sku': 'CSP-002', 'category': 'Cloud'},
            {'id': 3, 'name': 'Security Suite', 'sku': 'SEC-003', 'category': 'Security'},
            {'id': 4, 'name': 'Analytics Platform', 'sku': 'ANA-004', 'category': 'Analytics'},
            {'id': 5, 'name': 'Database Solution', 'sku': 'DB-005', 'category': 'Database'},
            {'id': 6, 'name': 'Mobile App Framework', 'sku': 'MAF-006', 'category': 'Mobile'},
            {'id': 7, 'name': 'AI Development Kit', 'sku': 'AI-007', 'category': 'AI'},
            {'id': 8, 'name': 'IoT Sensor Package', 'sku': 'IOT-008', 'category': 'IoT'},
            {'id': 9, 'name': 'Blockchain Platform', 'sku': 'BCH-009', 'category': 'Blockchain'},
            {'id': 10, 'name': 'DevOps Tools', 'sku': 'DEV-010', 'category': 'DevOps'}
        ]
        
        # Generate sales data
        sales = []
        sale_details = []
        sale_id = 1
        
        # Create sales over the last 24 months
        start_date = datetime.now() - timedelta(days=730)
        
        for month_offset in range(24):
            month_start = start_date + timedelta(days=month_offset * 30)
            
            # Generate 15-45 sales per month with seasonal variation
            base_sales = 30
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * month_offset / 12)  # Annual seasonality
            monthly_sales = int(base_sales * seasonal_factor) + random.randint(-5, 5)
            
            for _ in range(monthly_sales):
                sale_date = month_start + timedelta(days=random.randint(0, 29))
                customer = random.choice(customers)
                company = random.choice(companies)
                
                # Currency variations
                currencies = ['USD', 'EUR', 'GBP', 'CAD', 'AUD']
                exchange_rates = {'USD': 1.0, 'EUR': 0.85, 'GBP': 0.73, 'CAD': 1.25, 'AUD': 1.35}
                currency = random.choice(currencies)
                exchange_rate = exchange_rates[currency]
                
                sale = {
                    'id': sale_id,
                    'sale_date': sale_date,
                    'customer_id': customer['id'],
                    'company_id': company['id'],
                    'currency': currency,
                    'exchange_rate': exchange_rate,
                    'created_at': sale_date
                }
                
                sales.append(sale)
                
                # Generate 1-5 items per sale
                num_items = random.randint(1, 5)
                sale_total = 0
                
                for _ in range(num_items):
                    product = random.choice(products)
                    quantity = random.randint(1, 10)
                    base_price = random.uniform(100, 5000)
                    
                    # Add price variation based on quantity (bulk discount)
                    if quantity >= 5:
                        base_price *= 0.9  # 10% bulk discount
                    
                    sell_price = base_price
                    total_price = sell_price * quantity
                    vat_rate = 0.2  # 20% VAT
                    vat_amount = total_price * vat_rate
                    net_price = total_price - vat_amount
                    
                    sale_detail = {
                        'sale_id': sale_id,
                        'product_id': product['id'],
                        'quantity': quantity,
                        'sell_price': sell_price,
                        'total_price': total_price,
                        'vat_amount': vat_amount,
                        'net_price': net_price,
                        'created_at': sale_date
                    }
                    
                    sale_details.append(sale_detail)
                    sale_total += total_price
                
                # Update sale total
                sales[sale_id - 1]['total'] = sale_total
                sale_id += 1
        
        # Create DataFrames and save to database
        companies_df = pd.DataFrame(companies)
        customers_df = pd.DataFrame(customers)
        products_df = pd.DataFrame(products)
        sales_df = pd.DataFrame(sales)
        sale_details_df = pd.DataFrame(sale_details)
        
        # Save to database
        with self.engine.connect() as conn:
            companies_df.to_sql('companies', conn, if_exists='replace', index=False)
            customers_df.to_sql('customers', conn, if_exists='replace', index=False)
            products_df.to_sql('products', conn, if_exists='replace', index=False)
            sales_df.to_sql('sales', conn, if_exists='replace', index=False)
            sale_details_df.to_sql('sale_details', conn, if_exists='replace', index=False)
            
            conn.commit()
        
        logger.info(f"Created sample data: {len(sales)} sales, {len(sale_details)} sale items")
    
    def _create_analytics_view(self):
        """Create a comprehensive analytics view"""
        view_sql = """
        CREATE VIEW IF NOT EXISTS sales_product_customer_view AS
        SELECT 
            s.id as sale_id,
            s.sale_date,
            s.total as sale_total,
            s.currency,
            s.exchange_rate,
            s.company_id,
            s.customer_id,
            c.name as customer_name,
            c.country as customer_country,
            sd.product_id,
            p.name as product_name,
            p.sku,
            sd.sell_price,
            sd.quantity as qty,
            sd.total_price,
            sd.vat_amount,
            sd.net_price,
            sd.sell_price as price,
            sd.created_at as sale_detail_created_at
        FROM sales s
        JOIN customers c ON s.customer_id = c.id
        JOIN sale_details sd ON s.id = sd.sale_id
        JOIN products p ON sd.product_id = p.id
        ORDER BY s.sale_date DESC
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(view_sql))
                conn.commit()
            logger.info("Analytics view created successfully")
        except Exception as e:
            logger.error(f"Error creating analytics view: {str(e)}")
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        try:
            with self.engine.connect() as conn:
                result = pd.read_sql(query, conn)
            logger.info(f"Query executed successfully, returned {len(result)} rows")
            return result
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise
    
    def get_table_info(self) -> Dict[str, Any]:
        """Get information about available tables and views"""
        try:
            with self.engine.connect() as conn:
                if self.connection_type == 'sqlite':
                    # Get tables
                    tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
                    tables = pd.read_sql(tables_query, conn)['name'].tolist()
                    
                    # Get views
                    views_query = "SELECT name FROM sqlite_master WHERE type='view'"
                    views = pd.read_sql(views_query, conn)['name'].tolist()
                    
                    # Get view columns
                    view_columns = []
                    if 'sales_product_customer_view' in views:
                        columns_query = "PRAGMA table_info(sales_product_customer_view)"
                        columns = pd.read_sql(columns_query, conn)
                        view_columns = columns['name'].tolist()
                else:
                    # For other databases, adapt queries accordingly
                    tables = []
                    views = []
                    view_columns = []
                
                return {
                    'tables': tables,
                    'views': views,
                    'view_columns': view_columns
                }
        except Exception as e:
            logger.error(f"Error getting table info: {str(e)}")
            return {'tables': [], 'views': [], 'view_columns': []}
    
    def save_dataframe_to_db(self, df: pd.DataFrame, table_name: str) -> bool:
        """Save DataFrame to database as a new table"""
        try:
            with self.engine.connect() as conn:
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                conn.commit()
            logger.info(f"DataFrame saved to table '{table_name}' successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving DataFrame to database: {str(e)}")
            return False
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current database connection status"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            return {
                'connected': True,
                'type': self.connection_type,
                'database': self.db_path if self.connection_type == 'sqlite' else 'External'
            }
        except Exception as e:
            return {
                'connected': False,
                'type': self.connection_type,
                'error': str(e)
            }

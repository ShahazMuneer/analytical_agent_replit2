import sqlite3
import pandas as pd
import os
from datetime import datetime, timedelta
import random
from typing import Optional

class DatabaseManager:
    def __init__(self, db_path: str = "analytics.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with sample data if it doesn't exist"""
        if not os.path.exists(self.db_path):
            self._create_sample_database()
    
    def _create_sample_database(self):
        """Create sample database with realistic data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT NOT NULL,
            sku TEXT UNIQUE NOT NULL,
            price DECIMAL(10,2) NOT NULL
        )
        """)
        
        cursor.execute("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            customer_name TEXT NOT NULL,
            customer_country TEXT NOT NULL,
            company_id INTEGER
        )
        """)
        
        cursor.execute("""
        CREATE TABLE sales (
            sale_id INTEGER PRIMARY KEY,
            sale_date DATE NOT NULL,
            sale_total DECIMAL(10,2) NOT NULL,
            currency TEXT DEFAULT 'USD',
            exchange_rate DECIMAL(10,4) DEFAULT 1.0000,
            customer_id INTEGER,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        )
        """)
        
        cursor.execute("""
        CREATE TABLE sale_details (
            sale_detail_id INTEGER PRIMARY KEY,
            sale_id INTEGER,
            product_id INTEGER,
            qty INTEGER NOT NULL,
            sell_price DECIMAL(10,2) NOT NULL,
            total_price DECIMAL(10,2) NOT NULL,
            vat_amount DECIMAL(10,2) DEFAULT 0,
            net_price DECIMAL(10,2) NOT NULL,
            sale_detail_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (sale_id) REFERENCES sales(sale_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        )
        """)
        
        # Generate sample data
        self._populate_sample_data(cursor)
        
        # Create the main view
        cursor.execute("""
        CREATE VIEW sales_product_customer_view AS
        SELECT 
            s.sale_id,
            s.sale_date,
            s.sale_total,
            s.currency,
            s.exchange_rate,
            c.company_id,
            c.customer_id,
            c.customer_name,
            c.customer_country,
            p.product_id,
            p.product_name,
            p.sku,
            sd.sell_price,
            sd.qty,
            sd.total_price,
            sd.vat_amount,
            sd.net_price,
            p.price,
            sd.sale_detail_created_at
        FROM sales s
        JOIN customers c ON s.customer_id = c.customer_id
        JOIN sale_details sd ON s.sale_id = sd.sale_id
        JOIN products p ON sd.product_id = p.product_id
        """)
        
        conn.commit()
        conn.close()
    
    def _populate_sample_data(self, cursor):
        """Populate database with sample data"""
        
        # Sample products
        products = [
            ("Laptop Pro", "LP001", 1299.99),
            ("Wireless Mouse", "WM002", 29.99),
            ("Mechanical Keyboard", "MK003", 129.99),
            ("4K Monitor", "4K004", 399.99),
            ("USB-C Hub", "UC005", 89.99),
            ("Bluetooth Headphones", "BH006", 199.99),
            ("Webcam HD", "WH007", 79.99),
            ("Smartphone", "SP008", 799.99),
            ("Tablet", "TB009", 449.99),
            ("Smart Watch", "SW010", 299.99)
        ]
        
        cursor.executemany(
            "INSERT INTO products (product_name, sku, price) VALUES (?, ?, ?)",
            products
        )
        
        # Sample customers
        countries = ["USA", "UK", "Germany", "France", "Canada", "Australia", "Japan", "Brazil"]
        customers = []
        for i in range(1, 51):
            customers.append((
                f"Customer {i}",
                random.choice(countries),
                random.randint(1, 10)  # company_id
            ))
        
        cursor.executemany(
            "INSERT INTO customers (customer_name, customer_country, company_id) VALUES (?, ?, ?)",
            customers
        )
        
        # Generate sales data for the last 18 months
        start_date = datetime.now() - timedelta(days=540)  # 18 months ago
        current_date = start_date
        sale_id = 1
        sale_detail_id = 1
        
        while current_date <= datetime.now():
            # Generate 3-8 sales per day
            daily_sales = random.randint(3, 8)
            
            for _ in range(daily_sales):
                customer_id = random.randint(1, 50)
                
                # Generate 1-4 items per sale
                items_count = random.randint(1, 4)
                sale_total = 0
                sale_details = []
                
                for _ in range(items_count):
                    product_id = random.randint(1, 10)
                    qty = random.randint(1, 3)
                    
                    # Get product price
                    cursor.execute("SELECT price FROM products WHERE product_id = ?", (product_id,))
                    base_price = cursor.fetchone()[0]
                    
                    # Add some price variation
                    sell_price = base_price * random.uniform(0.9, 1.1)
                    total_price = sell_price * qty
                    vat_amount = total_price * 0.2  # 20% VAT
                    net_price = total_price - vat_amount
                    
                    sale_total += total_price
                    
                    sale_details.append((
                        sale_detail_id,
                        sale_id,
                        product_id,
                        qty,
                        round(sell_price, 2),
                        round(total_price, 2),
                        round(vat_amount, 2),
                        round(net_price, 2),
                        current_date.strftime('%Y-%m-%d %H:%M:%S')
                    ))
                    sale_detail_id += 1
                
                # Insert sale
                currency = random.choice(["USD", "EUR", "GBP"])
                exchange_rate = 1.0 if currency == "USD" else random.uniform(0.8, 1.2)
                
                cursor.execute(
                    "INSERT INTO sales (sale_id, sale_date, sale_total, currency, exchange_rate, customer_id) VALUES (?, ?, ?, ?, ?, ?)",
                    (sale_id, current_date.strftime('%Y-%m-%d'), round(sale_total, 2), currency, round(exchange_rate, 4), customer_id)
                )
                
                # Insert sale details
                cursor.executemany(
                    "INSERT INTO sale_details (sale_detail_id, sale_id, product_id, qty, sell_price, total_price, vat_amount, net_price, sale_detail_created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    sale_details
                )
                
                sale_id += 1
            
            current_date += timedelta(days=1)
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception as e:
            raise Exception(f"Database query failed: {str(e)}")
    
    def get_table_info(self) -> dict:
        """Get information about available tables and views"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Get views
        cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
        views = [row[0] for row in cursor.fetchall()]
        
        # Get view schema
        cursor.execute("PRAGMA table_info(sales_product_customer_view)")
        view_columns = [row[1] for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            'tables': tables,
            'views': views,
            'view_columns': view_columns
        }

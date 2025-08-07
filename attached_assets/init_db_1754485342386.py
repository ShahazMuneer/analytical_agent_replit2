#!/usr/bin/env python3
"""
Database initialization script
Run this script to set up the sample database with realistic data
"""

from database import DatabaseManager
import os

def main():
    """Initialize the database"""
    print("Initializing database...")
    
    # Remove existing database if it exists
    db_path = "analytics.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        print("Removed existing database")
    
    # Create new database with sample data
    db_manager = DatabaseManager(db_path)
    print("Created new database with sample data")
    
    # Verify the database
    info = db_manager.get_table_info()
    print(f"Tables created: {info['tables']}")
    print(f"Views created: {info['views']}")
    print(f"View columns: {len(info['view_columns'])}")
    
    # Test query
    test_df = db_manager.execute_query("SELECT COUNT(*) as total_records FROM sales_product_customer_view")
    print(f"Total records in view: {test_df.iloc[0]['total_records']}")
    
    print("Database initialization complete!")

if __name__ == "__main__":
    main()

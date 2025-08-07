#!/usr/bin/env python3
"""
Database Initialization Script for Enterprise Analytics Hub
Sets up the sample database with realistic business data
"""

import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from database import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Initialize the Enterprise Analytics Hub database"""
    try:
        print("🚀 Enterprise Analytics Hub - Database Initialization")
        print("=" * 60)
        
        # Configuration
        db_path = os.getenv("DATABASE_PATH", "analytics.db")
        
        # Remove existing database if it exists
        if os.path.exists(db_path):
            print(f"📄 Removing existing database: {db_path}")
            os.remove(db_path)
            logger.info(f"Removed existing database: {db_path}")
        
        # Create new database with sample data
        print("🔧 Creating new database with sample data...")
        db_manager = DatabaseManager(db_path)
        
        # Verify the database setup
        print("✅ Verifying database setup...")
        info = db_manager.get_table_info()
        
        print(f"📊 Database Information:")
        print(f"   • Tables created: {len(info['tables'])}")
        print(f"   • Views created: {len(info['views'])}")
        print(f"   • Available tables: {', '.join(info['tables'])}")
        print(f"   • Available views: {', '.join(info['views'])}")
        
        if info['view_columns']:
            print(f"   • View columns: {len(info['view_columns'])}")
            print(f"   • Sample columns: {', '.join(info['view_columns'][:5])}...")
        
        # Test database connectivity
        print("🔍 Testing database connectivity...")
        test_query = "SELECT COUNT(*) as total_records FROM sales_product_customer_view"
        test_result = db_manager.execute_query(test_query)
        
        if not test_result.empty:
            total_records = test_result.iloc[0]['total_records']
            print(f"✅ Database test successful!")
            print(f"   • Total records in main view: {total_records:,}")
        
        # Generate sample queries for testing
        print("📝 Testing sample queries...")
        sample_queries = [
            ("Monthly Revenue", "SELECT strftime('%Y-%m', sale_date) as month, SUM(total_price) as revenue FROM sales_product_customer_view GROUP BY strftime('%Y-%m', sale_date) ORDER BY month DESC LIMIT 12"),
            ("Top Products", "SELECT product_name, SUM(total_price) as total_sales FROM sales_product_customer_view GROUP BY product_id, product_name ORDER BY total_sales DESC LIMIT 10"),
            ("Customer Countries", "SELECT customer_country, COUNT(DISTINCT customer_id) as customers, SUM(total_price) as revenue FROM sales_product_customer_view GROUP BY customer_country ORDER BY revenue DESC")
        ]
        
        for query_name, query in sample_queries:
            try:
                result = db_manager.execute_query(query)
                print(f"   ✅ {query_name}: {len(result)} rows returned")
            except Exception as e:
                print(f"   ❌ {query_name}: Error - {str(e)}")
                logger.error(f"Sample query failed: {query_name} - {str(e)}")
        
        # Database statistics
        print("📈 Database Statistics:")
        
        # Get table sizes
        for table in info['tables']:
            try:
                count_result = db_manager.execute_query(f"SELECT COUNT(*) as count FROM {table}")
                if not count_result.empty:
                    count = count_result.iloc[0]['count']
                    print(f"   • {table}: {count:,} records")
            except Exception as e:
                print(f"   • {table}: Error getting count - {str(e)}")
        
        # Connection status
        status = db_manager.get_connection_status()
        print(f"🔗 Connection Status:")
        print(f"   • Connected: {status['connected']}")
        print(f"   • Database Type: {status['type']}")
        print(f"   • Database File: {status['database']}")
        
        print("=" * 60)
        print("✅ Database initialization completed successfully!")
        print(f"📁 Database file created at: {os.path.abspath(db_path)}")
        print("🚀 You can now run the Enterprise Analytics Hub application")
        print("\nTo start the application:")
        print("   streamlit run app.py --server.port 5000")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during database initialization: {str(e)}")
        logger.error(f"Database initialization failed: {str(e)}")
        return False

def verify_requirements():
    """Verify that required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'sqlalchemy', 'streamlit'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def cleanup_database(db_path: str = "analytics.db"):
    """Cleanup database file"""
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"✅ Cleaned up database file: {db_path}")
            return True
    except Exception as e:
        print(f"❌ Error cleaning up database: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enterprise Analytics Hub Database Initialization")
    parser.add_argument("--cleanup", action="store_true", help="Clean up existing database")
    parser.add_argument("--verify", action="store_true", help="Verify requirements only")
    parser.add_argument("--db-path", default="analytics.db", help="Database file path")
    
    args = parser.parse_args()
    
    if args.verify:
        print("🔍 Verifying requirements...")
        if verify_requirements():
            print("✅ All requirements satisfied")
            sys.exit(0)
        else:
            sys.exit(1)
    
    if args.cleanup:
        print("🧹 Cleaning up database...")
        if cleanup_database(args.db_path):
            sys.exit(0)
        else:
            sys.exit(1)
    
    # Verify requirements before proceeding
    if not verify_requirements():
        sys.exit(1)
    
    # Set database path from arguments
    if args.db_path != "analytics.db":
        os.environ["DATABASE_PATH"] = args.db_path
    
    # Run main initialization
    success = main()
    sys.exit(0 if success else 1)

"""
Groq Client for Natural Language to SQL Conversion
Enhanced with better error handling and fallback capabilities
"""

import os
import json
from groq import Groq
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class GroqClient:
    def __init__(self):
        """Initialize Groq client with API key from environment"""
        self.api_key = os.getenv("GROQ_API_KEY", "gsk_default_key")
        
        try:
            self.client = Groq(api_key=self.api_key)
            self.model = "llama-3.1-70b-versatile"
            logger.info("Groq client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Groq client: {str(e)}")
            self.client = None
    
    def convert_to_sql(self, natural_language_prompt: str) -> str:
        """Convert natural language prompt to SQL query"""
        
        system_prompt = """You are an expert SQL analyst. Convert natural language queries to optimized SQL queries using the sales_product_customer_view.

Available view: sales_product_customer_view
Columns: sale_id, sale_date, sale_total, currency, exchange_rate, company_id, customer_id, customer_name, customer_country, product_id, product_name, sku, sell_price, qty, total_price, vat_amount, net_price, price, sale_detail_created_at

Guidelines:
1. Always use the sales_product_customer_view (never reference other tables)
2. For time-based queries, use sale_date column
3. For revenue/sales totals, use total_price or sale_total
4. When grouping by time periods, use DATE() functions appropriately
5. Always include ORDER BY for logical result ordering
6. Use LIMIT when appropriate for top/bottom queries
7. Consider currency conversion using exchange_rate when needed
8. For SQLite, use strftime() for date formatting

Return ONLY the SQL query, no explanations or additional text."""

        if not self.client:
            logger.warning("Groq client not available, using fallback")
            return self._generate_fallback_query(natural_language_prompt)

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": natural_language_prompt}
                ],
                model=self.model,
                temperature=0.1,
                max_tokens=1000
            )
            
            sql_query = chat_completion.choices[0].message.content.strip()
            
            # Clean up the response (remove any markdown formatting)
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]
            
            # Validate the query contains expected elements
            if not self._validate_sql_query(sql_query):
                logger.warning("Generated SQL query failed validation, using fallback")
                return self._generate_fallback_query(natural_language_prompt)
            
            logger.info("SQL query generated successfully via Groq")
            return sql_query.strip()
            
        except Exception as e:
            logger.error(f"Error calling Groq API: {str(e)}")
            return self._generate_fallback_query(natural_language_prompt)
    
    def _validate_sql_query(self, query: str) -> bool:
        """Basic validation of generated SQL query"""
        query_lower = query.lower()
        
        # Check for required elements
        if "sales_product_customer_view" not in query_lower:
            return False
        if not query_lower.strip().startswith("select"):
            return False
        
        # Check for potentially dangerous operations
        dangerous_keywords = ["drop", "delete", "insert", "update", "alter", "create"]
        for keyword in dangerous_keywords:
            if keyword in query_lower:
                return False
        
        return True
    
    def _generate_fallback_query(self, prompt: str) -> str:
        """Generate a fallback SQL query when Groq API fails"""
        prompt_lower = prompt.lower()
        
        # Enhanced keyword-based query generation
        if any(word in prompt_lower for word in ["monthly revenue", "revenue by month", "monthly sales"]):
            return """
            SELECT 
                strftime('%Y-%m', sale_date) as month,
                SUM(total_price) as revenue,
                COUNT(DISTINCT sale_id) as number_of_sales,
                AVG(total_price) as average_sale_value
            FROM sales_product_customer_view 
            GROUP BY strftime('%Y-%m', sale_date)
            ORDER BY month DESC
            LIMIT 12
            """
        
        elif any(word in prompt_lower for word in ["top", "best"]) and "product" in prompt_lower:
            return """
            SELECT 
                product_name,
                SUM(total_price) as total_sales,
                SUM(qty) as total_quantity,
                AVG(sell_price) as average_price,
                COUNT(DISTINCT sale_id) as number_of_orders
            FROM sales_product_customer_view 
            GROUP BY product_id, product_name
            ORDER BY total_sales DESC
            LIMIT 10
            """
        
        elif any(word in prompt_lower for word in ["top", "best"]) and "customer" in prompt_lower:
            return """
            SELECT 
                customer_name,
                customer_country,
                SUM(total_price) as total_spent,
                COUNT(DISTINCT sale_id) as number_of_orders,
                AVG(total_price) as average_order_value
            FROM sales_product_customer_view 
            GROUP BY customer_id, customer_name, customer_country
            ORDER BY total_spent DESC
            LIMIT 10
            """
        
        elif "country" in prompt_lower or "geographic" in prompt_lower:
            return """
            SELECT 
                customer_country,
                SUM(total_price) as total_sales,
                COUNT(DISTINCT sale_id) as total_orders,
                COUNT(DISTINCT customer_id) as unique_customers,
                AVG(total_price) as average_order_value
            FROM sales_product_customer_view 
            GROUP BY customer_country
            ORDER BY total_sales DESC
            LIMIT 15
            """
        
        elif any(word in prompt_lower for word in ["daily", "day by day", "daily sales"]):
            return """
            SELECT 
                DATE(sale_date) as sale_day,
                SUM(total_price) as daily_revenue,
                COUNT(DISTINCT sale_id) as number_of_sales
            FROM sales_product_customer_view 
            WHERE DATE(sale_date) >= DATE('now', '-30 days')
            GROUP BY DATE(sale_date)
            ORDER BY sale_day DESC
            LIMIT 30
            """
        
        elif any(word in prompt_lower for word in ["quarterly", "quarter", "q1", "q2", "q3", "q4"]):
            return """
            SELECT 
                CASE 
                    WHEN CAST(strftime('%m', sale_date) AS INTEGER) BETWEEN 1 AND 3 THEN strftime('%Y', sale_date) || '-Q1'
                    WHEN CAST(strftime('%m', sale_date) AS INTEGER) BETWEEN 4 AND 6 THEN strftime('%Y', sale_date) || '-Q2'
                    WHEN CAST(strftime('%m', sale_date) AS INTEGER) BETWEEN 7 AND 9 THEN strftime('%Y', sale_date) || '-Q3'
                    ELSE strftime('%Y', sale_date) || '-Q4'
                END as quarter,
                SUM(total_price) as quarterly_revenue,
                COUNT(DISTINCT sale_id) as number_of_sales
            FROM sales_product_customer_view 
            GROUP BY quarter
            ORDER BY quarter DESC
            LIMIT 8
            """
        
        elif "forecast" in prompt_lower or "prediction" in prompt_lower:
            # Return time series data suitable for forecasting
            return """
            SELECT 
                strftime('%Y-%m', sale_date) as month,
                SUM(total_price) as revenue
            FROM sales_product_customer_view 
            GROUP BY strftime('%Y-%m', sale_date)
            ORDER BY month ASC
            """
        
        elif "trend" in prompt_lower or "growth" in prompt_lower:
            return """
            SELECT 
                strftime('%Y-%m', sale_date) as month,
                SUM(total_price) as revenue,
                COUNT(DISTINCT customer_id) as unique_customers,
                AVG(total_price) as average_order_value
            FROM sales_product_customer_view 
            GROUP BY strftime('%Y-%m', sale_date)
            ORDER BY month ASC
            """
        
        else:
            # Default comprehensive query
            return """
            SELECT 
                sale_date,
                customer_name,
                customer_country,
                product_name,
                qty,
                sell_price,
                total_price,
                currency
            FROM sales_product_customer_view 
            ORDER BY sale_date DESC
            LIMIT 50
            """
    
    def analyze_query_intent(self, prompt: str) -> Dict[str, Any]:
        """Analyze the intent of the natural language query"""
        
        system_prompt = """Analyze the user's natural language query and return a JSON object with the following structure:
{
    "query_type": "trend_analysis|comparison|forecasting|summary|geographic_analysis",
    "time_dimension": "daily|weekly|monthly|quarterly|yearly|none",
    "needs_forecasting": true|false,
    "chart_type": "line|bar|pie|table|scatter|heatmap",
    "primary_metric": "revenue|sales|quantity|customers|profit",
    "grouping": "product|customer|country|time|category|none",
    "complexity": "simple|medium|complex"
}

Return ONLY the JSON object, no additional text."""
        
        if not self.client:
            return self._generate_fallback_intent(prompt)
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.1,
                max_tokens=200
            )
            
            response = chat_completion.choices[0].message.content.strip()
            
            # Clean up JSON response
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            
            return json.loads(response)
            
        except Exception as e:
            logger.error(f"Error analyzing query intent: {str(e)}")
            return self._generate_fallback_intent(prompt)
    
    def _generate_fallback_intent(self, prompt: str) -> Dict[str, Any]:
        """Generate fallback intent analysis when Groq API fails"""
        prompt_lower = prompt.lower()
        
        # Determine query type
        if any(word in prompt_lower for word in ["trend", "over time", "monthly", "daily", "growth"]):
            query_type = "trend_analysis"
            chart_type = "line"
            time_dimension = "monthly"
        elif any(word in prompt_lower for word in ["top", "best", "compare", "vs", "versus"]):
            query_type = "comparison"
            chart_type = "bar"
            time_dimension = "none"
        elif any(word in prompt_lower for word in ["forecast", "predict", "future", "next"]):
            query_type = "forecasting"
            chart_type = "line"
            time_dimension = "monthly"
        elif any(word in prompt_lower for word in ["country", "region", "geographic", "location"]):
            query_type = "geographic_analysis"
            chart_type = "bar"
            time_dimension = "none"
        else:
            query_type = "summary"
            chart_type = "table"
            time_dimension = "none"
        
        # Determine primary metric
        if any(word in prompt_lower for word in ["revenue", "sales", "money", "income"]):
            primary_metric = "revenue"
        elif any(word in prompt_lower for word in ["quantity", "volume", "units"]):
            primary_metric = "quantity"
        elif any(word in prompt_lower for word in ["customer", "client", "buyer"]):
            primary_metric = "customers"
        else:
            primary_metric = "revenue"
        
        # Determine grouping
        if "product" in prompt_lower:
            grouping = "product"
        elif "customer" in prompt_lower:
            grouping = "customer"
        elif "country" in prompt_lower:
            grouping = "country"
        elif any(word in prompt_lower for word in ["time", "month", "day", "year"]):
            grouping = "time"
        else:
            grouping = "none"
        
        return {
            "query_type": query_type,
            "time_dimension": time_dimension,
            "needs_forecasting": "forecast" in prompt_lower or "predict" in prompt_lower,
            "chart_type": chart_type,
            "primary_metric": primary_metric,
            "grouping": grouping,
            "complexity": "medium"
        }

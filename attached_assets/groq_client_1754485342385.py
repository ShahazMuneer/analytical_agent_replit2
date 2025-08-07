import os
import json
from groq import Groq
from typing import Dict, Any

class GroqClient:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "gsk_default_key")
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.1-70b-versatile"
    
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

Return ONLY the SQL query, no explanations or additional text."""

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
            
            return sql_query.strip()
            
        except Exception as e:
            # Fallback to basic query if Groq fails
            return self._generate_fallback_query(natural_language_prompt)
    
    def _generate_fallback_query(self, prompt: str) -> str:
        """Generate a fallback SQL query when Groq API fails"""
        prompt_lower = prompt.lower()
        
        # Simple keyword-based query generation
        if "monthly revenue" in prompt_lower or "revenue by month" in prompt_lower:
            return """
            SELECT 
                strftime('%Y-%m', sale_date) as month,
                SUM(total_price) as revenue
            FROM sales_product_customer_view 
            GROUP BY strftime('%Y-%m', sale_date)
            ORDER BY month DESC
            LIMIT 12
            """
        
        elif "top" in prompt_lower and "product" in prompt_lower:
            return """
            SELECT 
                product_name,
                SUM(total_price) as total_sales,
                SUM(qty) as total_quantity
            FROM sales_product_customer_view 
            GROUP BY product_id, product_name
            ORDER BY total_sales DESC
            LIMIT 10
            """
        
        elif "country" in prompt_lower:
            return """
            SELECT 
                customer_country,
                SUM(total_price) as total_sales,
                COUNT(DISTINCT sale_id) as total_orders
            FROM sales_product_customer_view 
            GROUP BY customer_country
            ORDER BY total_sales DESC
            LIMIT 10
            """
        
        else:
            # Default query
            return """
            SELECT 
                sale_date,
                customer_name,
                product_name,
                qty,
                total_price
            FROM sales_product_customer_view 
            ORDER BY sale_date DESC
            LIMIT 10
            """
    
    def analyze_query_intent(self, prompt: str) -> Dict[str, Any]:
        """Analyze the intent of the natural language query"""
        
        system_prompt = """Analyze the user's natural language query and return a JSON object with the following structure:
{
    "query_type": "trend_analysis|comparison|forecasting|summary",
    "time_dimension": "daily|weekly|monthly|yearly|none",
    "needs_forecasting": true|false,
    "chart_type": "line|bar|pie|table",
    "primary_metric": "revenue|sales|quantity|customers",
    "grouping": "product|customer|country|time|none"
}

Return ONLY the JSON object, no additional text."""
        
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
            return json.loads(response)
            
        except Exception as e:
            # Return default analysis if parsing fails
            return {
                "query_type": "summary",
                "time_dimension": "none",
                "needs_forecasting": False,
                "chart_type": "table",
                "primary_metric": "revenue",
                "grouping": "none"
            }

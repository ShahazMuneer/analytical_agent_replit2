import streamlit as st
import pandas as pd
import openai
import re
import json
from typing import Dict, Any, Optional, List
import logging
from groq import Groq
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AIQueryEngine:
    """AI-powered natural language to SQL query engine"""
    
    def __init__(self):
        self.groq_client = None
        self.query_history = []
        self.initialize_client()
    
    def initialize_client(self):
        """Initialize AI client"""
        try:
            api_key = st.secrets.get("GROQ_API_KEY") or "gsk_default_key"
            if api_key and api_key != "gsk_default_key":
                self.groq_client = Groq(api_key=api_key)
            else:
                st.warning("⚠️ GROQ_API_KEY not configured. Using demo mode.")
        except Exception as e:
            logger.error(f"Failed to initialize AI client: {str(e)}")
    
    def get_database_context(self, db_connector) -> str:
        """Get database schema context for AI"""
        try:
            schema_info = db_connector.get_table_info()
            
            context = "Database Schema Information:\n"
            context += f"Total Tables: {schema_info.get('total_tables', 0)}\n\n"
            
            for table_name, table_info in schema_info.get('schema', {}).items():
                context += f"Table: {table_name}\n"
                context += f"Columns: {', '.join(table_info.get('columns', []))}\n"
                context += f"Row Count: {table_info.get('row_count', 0)}\n\n"
            
            return context
        except:
            return "No schema information available"
    
    def natural_language_to_sql(self, question: str, db_connector) -> Dict[str, Any]:
        """Convert natural language question to SQL"""
        
        # Get database context
        db_context = self.get_database_context(db_connector)
        
        # Demo mode fallback patterns
        if not self.groq_client:
            return self._demo_query_processing(question, db_connector)
        
        try:
            system_prompt = f"""You are a SQL expert. Convert natural language questions to SQL queries.
            
{db_context}

Rules:
1. Generate ONLY valid SQL SELECT statements
2. Use proper table and column names from the schema
3. Include appropriate WHERE, GROUP BY, ORDER BY clauses as needed
4. For time-based queries, assume date columns are in standard formats
5. Return only the SQL query, no explanations
6. Use LIMIT 1000 for safety"""
            
            response = self.groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Convert this question to SQL: {question}"}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Clean the SQL query
            sql_query = self._clean_sql_query(sql_query)
            
            return {
                'sql': sql_query,
                'explanation': f"Generated SQL query for: {question}",
                'confidence': 0.9
            }
            
        except Exception as e:
            logger.error(f"AI query generation failed: {str(e)}")
            return self._demo_query_processing(question, db_connector)
    
    def _clean_sql_query(self, sql: str) -> str:
        """Clean and validate SQL query"""
        # Remove markdown formatting
        sql = re.sub(r'```sql\n', '', sql)
        sql = re.sub(r'```', '', sql)
        
        # Remove extra whitespace
        sql = ' '.join(sql.split())
        
        # Ensure it's a SELECT statement
        if not sql.upper().strip().startswith('SELECT'):
            raise ValueError("Generated query is not a SELECT statement")
        
        # Add LIMIT if not present
        if 'LIMIT' not in sql.upper():
            sql += ' LIMIT 1000'
        
        return sql
    
    def _demo_query_processing(self, question: str, db_connector) -> Dict[str, Any]:
        """Fallback demo query processing"""
        question_lower = question.lower()
        
        # Common query patterns
        if any(word in question_lower for word in ['revenue', 'sales', 'total', 'sum']):
            if 'product' in question_lower:
                return {
                    'sql': 'SELECT product_name, SUM(total_amount) as total_revenue FROM sales_analysis_view GROUP BY product_name ORDER BY total_revenue DESC LIMIT 10',
                    'explanation': 'Revenue by product analysis',
                    'confidence': 0.8
                }
            elif 'region' in question_lower:
                return {
                    'sql': 'SELECT region, SUM(total_amount) as total_revenue FROM sales_analysis_view GROUP BY region ORDER BY total_revenue DESC',
                    'explanation': 'Revenue by region analysis',
                    'confidence': 0.8
                }
            elif 'customer' in question_lower:
                return {
                    'sql': 'SELECT customer_name, customer_type, SUM(total_amount) as total_revenue FROM sales_analysis_view GROUP BY customer_name, customer_type ORDER BY total_revenue DESC LIMIT 10',
                    'explanation': 'Revenue by customer analysis',
                    'confidence': 0.8
                }
            else:
                return {
                    'sql': 'SELECT SUM(total_amount) as total_revenue, COUNT(*) as total_sales FROM sales_analysis_view',
                    'explanation': 'Overall revenue summary',
                    'confidence': 0.8
                }
        
        elif any(word in question_lower for word in ['trend', 'time', 'monthly', 'daily']):
            return {
                'sql': "SELECT DATE(sale_date) as sale_date, SUM(total_amount) as daily_revenue FROM sales_analysis_view GROUP BY DATE(sale_date) ORDER BY sale_date",
                'explanation': 'Revenue trend over time',
                'confidence': 0.8
            }
        
        elif any(word in question_lower for word in ['top', 'best', 'highest']):
            return {
                'sql': 'SELECT product_name, SUM(total_amount) as revenue FROM sales_analysis_view GROUP BY product_name ORDER BY revenue DESC LIMIT 10',
                'explanation': 'Top performing products',
                'confidence': 0.8
            }
        
        elif any(word in question_lower for word in ['customer', 'client']):
            return {
                'sql': 'SELECT customer_name, customer_type, city, country, SUM(total_amount) as total_spent FROM sales_analysis_view GROUP BY customer_name, customer_type, city, country ORDER BY total_spent DESC LIMIT 20',
                'explanation': 'Customer analysis',
                'confidence': 0.8
            }
        
        else:
            # Default query
            return {
                'sql': 'SELECT * FROM sales_analysis_view ORDER BY sale_date DESC LIMIT 50',
                'explanation': 'Recent sales data overview',
                'confidence': 0.6
            }
    
    def execute_query(self, sql: str, db_connector) -> pd.DataFrame:
        """Execute SQL query safely"""
        try:
            # Additional safety checks
            if not sql.upper().strip().startswith('SELECT'):
                raise ValueError("Only SELECT queries are allowed")
            
            # Execute query
            df = db_connector.execute_query(sql)
            
            # Add to history
            self.query_history.append({
                'timestamp': datetime.now(),
                'sql': sql,
                'row_count': len(df)
            })
            
            return df
            
        except Exception as e:
            raise Exception(f"Query execution failed: {str(e)}")
    
    def suggest_visualizations(self, df: pd.DataFrame, question: str) -> List[Dict[str, Any]]:
        """Suggest appropriate visualizations for the data"""
        suggestions = []
        
        if df.empty:
            return suggestions
        
        # Analyze data types
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        question_lower = question.lower()
        
        # Revenue/Sales analysis
        if any(word in question_lower for word in ['revenue', 'sales', 'total', 'amount']):
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                suggestions.append({
                    'type': 'bar',
                    'title': 'Revenue by Category',
                    'x': categorical_cols[0],
                    'y': numeric_cols[0],
                    'description': 'Bar chart showing revenue distribution'
                })
                
                if len(categorical_cols) >= 2:
                    suggestions.append({
                        'type': 'pie',
                        'title': 'Revenue Distribution',
                        'values': numeric_cols[0],
                        'names': categorical_cols[0],
                        'description': 'Pie chart showing revenue breakdown'
                    })
        
        # Trend analysis
        if date_cols and numeric_cols:
            suggestions.append({
                'type': 'line',
                'title': 'Trend Over Time',
                'x': date_cols[0],
                'y': numeric_cols[0],
                'description': 'Time series trend analysis'
            })
        
        # Comparison analysis
        if len(numeric_cols) >= 2:
            suggestions.append({
                'type': 'scatter',
                'title': 'Correlation Analysis',
                'x': numeric_cols[0],
                'y': numeric_cols[1],
                'description': 'Scatter plot for correlation analysis'
            })
        
        # Geographic analysis
        if any(col.lower() in ['country', 'region', 'city', 'location'] for col in categorical_cols):
            geo_col = next((col for col in categorical_cols if col.lower() in ['country', 'region', 'city', 'location']), None)
            if geo_col and numeric_cols:
                suggestions.append({
                    'type': 'map',
                    'title': 'Geographic Distribution',
                    'locations': geo_col,
                    'values': numeric_cols[0],
                    'description': 'Geographic visualization of data'
                })
        
        return suggestions[:4]  # Limit to 4 suggestions
    
    def create_visualization(self, df: pd.DataFrame, viz_config: Dict[str, Any]) -> go.Figure:
        """Create visualization based on configuration"""
        
        if df.empty:
            return go.Figure().add_annotation(text="No data to visualize", x=0.5, y=0.5)
        
        try:
            viz_type = viz_config['type']
            
            if viz_type == 'bar':
                fig = px.bar(df, x=viz_config['x'], y=viz_config['y'], 
                           title=viz_config['title'],
                           color=viz_config['y'],
                           color_continuous_scale='Viridis')
                
            elif viz_type == 'pie':
                fig = px.pie(df, values=viz_config['values'], names=viz_config['names'],
                           title=viz_config['title'])
                
            elif viz_type == 'line':
                fig = px.line(df, x=viz_config['x'], y=viz_config['y'],
                            title=viz_config['title'])
                
            elif viz_type == 'scatter':
                fig = px.scatter(df, x=viz_config['x'], y=viz_config['y'],
                               title=viz_config['title'],
                               trendline="ols")
                
            elif viz_type == 'map':
                # Simple map visualization (would need proper geographic data)
                grouped_data = df.groupby(viz_config['locations'])[viz_config['values']].sum().reset_index()
                fig = px.bar(grouped_data, x=viz_config['locations'], y=viz_config['values'],
                           title=viz_config['title'])
            
            else:
                # Default to bar chart
                if len(df.columns) >= 2:
                    fig = px.bar(df, x=df.columns[0], y=df.columns[1], title="Data Overview")
                else:
                    fig = go.Figure().add_annotation(text="Unable to create visualization", x=0.5, y=0.5)
            
            # Update layout for better appearance
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(size=12),
                title_font_size=16,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {str(e)}")
            return go.Figure().add_annotation(text=f"Visualization error: {str(e)}", x=0.5, y=0.5)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import re
from datetime import datetime, timedelta
import time

from database import DatabaseManager
from groq_client import GroqClient
from forecasting import ForecastingEngine
from visualizations import ChartGenerator
from database_connectors import DatabaseConnector
from advanced_analytics import AdvancedAnalytics
from step_by_step_analytics import StepByStepAnalytics
from data_profiling_tools import DataProfilingTools

# Page configuration
st.set_page_config(
    page_title="Enterprise Analytics Hub",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

if 'current_results' not in st.session_state:
    st.session_state.current_results = None

if 'db_connector' not in st.session_state:
    st.session_state.db_connector = DatabaseConnector()
    # Auto-connect to default SQLite
    st.session_state.db_connector.connect_sqlite()

if 'connected_db' not in st.session_state:
    st.session_state.connected_db = "SQLite (Local)"

if 'show_forecasting' not in st.session_state:
    st.session_state.show_forecasting = False

# Initialize components
@st.cache_resource
def init_components():
    db_manager = DatabaseManager()
    groq_client = GroqClient()
    forecasting_engine = ForecastingEngine()
    chart_generator = ChartGenerator()
    advanced_analytics = AdvancedAnalytics()
    step_analytics = StepByStepAnalytics()
    profiling_tools = DataProfilingTools()
    return db_manager, groq_client, forecasting_engine, chart_generator, advanced_analytics, step_analytics, profiling_tools

def detect_forecasting_keywords(prompt):
    """Detect if the prompt requires forecasting"""
    forecasting_keywords = ['forecast', 'predict', 'future', 'next months', 'next year', 
                           'projection', 'upcoming', 'trend ahead', 'coming months']
    return any(keyword in prompt.lower() for keyword in forecasting_keywords)

def detect_time_series_keywords(prompt):
    """Detect if the prompt requires time series visualization"""
    time_keywords = ['trend', 'over time', 'monthly', 'daily', 'yearly', 'revenue over', 
                    'sales over', 'growth', 'progression', 'timeline']
    return any(keyword in prompt.lower() for keyword in time_keywords)

def create_database_connection_ui():
    """Create database connection interface"""
    with st.sidebar:
        st.header("🗄️ Database Connections")
        
        # Available databases
        available_dbs = st.session_state.db_connector.get_available_databases()
        selected_db = st.selectbox(
            "Select Database Type", 
            available_dbs,
            index=available_dbs.index(st.session_state.connected_db) if st.session_state.connected_db in available_dbs else 0
        )
        
        # Connection forms based on database type
        if selected_db == "SQLite (Local)":
            if st.button("Connect to SQLite", type="primary"):
                if st.session_state.db_connector.connect_sqlite():
                    st.session_state.connected_db = selected_db
                    st.success("Connected to SQLite")
                    st.rerun()
        
        elif selected_db == "PostgreSQL":
            with st.expander("PostgreSQL Connection", expanded=True):
                host = st.text_input("Host", value="localhost")
                port = st.text_input("Port", value="5432")
                database = st.text_input("Database")
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                
                if st.button("Connect to PostgreSQL", type="primary"):
                    if st.session_state.db_connector.connect_postgresql(host, port, database, username, password):
                        st.session_state.connected_db = selected_db
                        st.success("Connected to PostgreSQL")
                        st.rerun()
        
        elif selected_db == "MySQL":
            with st.expander("MySQL Connection", expanded=True):
                host = st.text_input("Host", value="localhost")
                port = st.text_input("Port", value="3306")
                database = st.text_input("Database")
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                
                if st.button("Connect to MySQL", type="primary"):
                    if st.session_state.db_connector.connect_mysql(host, port, database, username, password):
                        st.session_state.connected_db = selected_db
                        st.success("Connected to MySQL")
                        st.rerun()
        
        elif selected_db == "Teradata":
            with st.expander("Teradata Connection", expanded=True):
                st.markdown("**🏦 Enterprise Teradata Integration**")
                connection_method = st.radio(
                    "Connection Method", 
                    ["Direct Connection", "ODBC Connection"],
                    help="Choose your preferred connection method"
                )
                
                if connection_method == "Direct Connection":
                    host = st.text_input("Host/Server", placeholder="teradata-server.company.com")
                    database = st.text_input("Database", placeholder="DW_PROD")
                    username = st.text_input("Username", placeholder="user123")
                    password = st.text_input("Password", type="password")
                    
                    # Advanced options
                    with st.expander("Advanced Settings"):
                        port = st.text_input("Port", value="1025")
                        logmech = st.selectbox("Authentication", ["TD2", "LDAP", "JWT"])
                        charset = st.selectbox("Character Set", ["UTF8", "ASCII"])
                    
                    if st.button("🔗 Connect to Teradata", type="primary"):
                        if st.session_state.db_connector.connect_teradata(host, username, password):
                            st.session_state.connected_db = selected_db
                            st.success("✅ Connected to Teradata Enterprise Database")
                            st.rerun()
                
                else:  # ODBC Connection
                    dsn = st.text_input("DSN Name", placeholder="TeradataDSN")
                    username = st.text_input("Username", placeholder="user123") 
                    password = st.text_input("Password", type="password")
                    
                    if st.button("🔗 Connect via ODBC", type="primary"):
                        st.info("ODBC connection support coming soon...")
        
        elif selected_db == "SQL Server (ODBC)":
            with st.expander("SQL Server Connection", expanded=True):
                server = st.text_input("Server", placeholder="server.company.com")
                database = st.text_input("Database")
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                
                if st.button("Connect to SQL Server", type="primary"):
                    st.info("SQL Server connection support coming soon...")
        
        elif selected_db == "Oracle (ODBC)":
            with st.expander("Oracle Connection", expanded=True):
                host = st.text_input("Host", placeholder="oracle-server.company.com")
                port = st.text_input("Port", value="1521")
                service_name = st.text_input("Service Name", placeholder="ORCL")
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                
                if st.button("Connect to Oracle", type="primary"):
                    st.info("Oracle connection support coming soon...")
        
        elif selected_db == "Hive (Hadoop)":
            with st.expander("Hive Connection", expanded=True):
                host = st.text_input("Host")
                port = st.text_input("Port", value="10000")
                database = st.text_input("Database", value="default")
                username = st.text_input("Username")
                
                if st.button("Connect to Hive", type="primary"):
                    if st.session_state.db_connector.connect_hive(host, port, database, username):
                        st.session_state.connected_db = selected_db
                        st.success("Connected to Hive")
                        st.rerun()
        
        # Show connection status
        st.divider()
        if st.session_state.db_connector.test_connection():
            st.success(f"✅ Connected to {st.session_state.connected_db}")
            
            # Show tables
            tables = st.session_state.db_connector.get_table_list()
            if tables:
                st.write("**Available Tables:**")
                for table in tables[:10]:  # Show first 10 tables
                    st.write(f"• {table}")
        else:
            st.error("❌ No active connection")

def create_kpi_dashboard(data: pd.DataFrame, analytics: AdvancedAnalytics):
    """Create KPI dashboard"""
    kpis = analytics.create_kpi_cards(data)
    
    if kpis:
        st.subheader("📊 Key Performance Indicators")
        
        # Create columns for KPIs
        cols = st.columns(len(kpis))
        for i, (kpi_name, kpi_data) in enumerate(kpis.items()):
            with cols[i % len(cols)]:
                st.metric(
                    label=kpi_name,
                    value=kpi_data['value']
                )

def display_full_results_table(data: pd.DataFrame):
    """Display full results with pagination and download options"""
    st.subheader("📋 Query Results")
    
    # Results summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", len(data))
    with col2:
        st.metric("Columns", len(data.columns))
    with col3:
        csv = data.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Display options
    col1, col2 = st.columns(2)
    with col1:
        rows_to_show = st.selectbox(
            "Rows to display:",
            [10, 25, 50, 100, len(data)],
            index=3  # Default to 100
        )
    with col2:
        show_all = st.checkbox("Show all columns", value=True)
    
    # Display data
    if show_all:
        display_data = data.head(rows_to_show)
    else:
        # Show only first 10 columns if too many
        display_cols = data.columns[:10] if len(data.columns) > 10 else data.columns
        display_data = data[display_cols].head(rows_to_show)
    
    st.dataframe(display_data, use_container_width=True, height=400)
    
    # Show pagination info
    if len(data) > rows_to_show:
        st.info(f"Showing {rows_to_show} of {len(data)} rows. Use download button to get complete data.")

def main():
    # Custom CSS for enterprise styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
    }
    .insight-card {
        background: #f0f9ff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #0ea5e9;
    }
    .step-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        border: none;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize components
    try:
        db_manager, groq_client, forecasting_engine, chart_generator, advanced_analytics, step_analytics, profiling_tools = init_components()
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        st.stop()
    
    # Check if this is the first visit to show welcome screen
    if 'show_welcome' not in st.session_state:
        st.session_state.show_welcome = True
    
    # Show welcome interface first
    if st.session_state.show_welcome:
        step_analytics.create_welcome_interface()
        
        # Feature showcase
        st.markdown("""
        <div class="feature-grid">
            <div class="feature-card">
                <h3>📊 Analyze Your Data Intelligently</h3>
                <p>→ Instantly visualize DataFrames with intelligent analysis<br>
                → Get automatic summaries, comparisons, and anomaly detection<br>
                → Advanced statistical profiling and data quality assessment</p>
            </div>
            <div class="feature-card">
                <h3>🛡️ Ensure Data Quality & Trust</h3>
                <p>→ Run real-time data validation and quality checks<br>
                → Validate schemas, flag missing or inconsistent data<br>
                → Ensure compliance with industry standards</p>
            </div>
            <div class="feature-card">
                <h3>🔄 Track Lineage & Audit Everything</h3>
                <p>→ Trace where data came from and how it transformed<br>
                → Stay compliant, reduce risk, gain 100% traceability<br>
                → Comprehensive audit trails for regulatory compliance</p>
            </div>
            <div class="feature-card">
                <h3>🤖 Ask Anything with AI Insight</h3>
                <p>→ Use natural language prompts to generate reports<br>
                → Detect fraud, forecast trends, segment customers with AI<br>
                → Integrated intelligent analytics powered by advanced ML</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick start section
        st.markdown("### 💬 **Try It Now**")
        st.markdown("Type your query or task below and let the assistant handle the complexity:")
        
        # Quick start examples
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📈 Analyze Revenue Trends", use_container_width=True):
                st.session_state.current_prompt = "Show me monthly revenue for the past 12 months"
                st.session_state.show_welcome = False
                st.rerun()
        with col2:
            if st.button("🔍 Data Quality Check", use_container_width=True):
                st.session_state.enable_step_analysis = True
                st.session_state.show_welcome = False
                st.rerun()
        with col3:
            if st.button("🎯 Business Intelligence", use_container_width=True):
                st.session_state.current_prompt = "Top 5 products by total sales with growth trends"
                st.session_state.show_welcome = False
                st.rerun()
        
        # Manual continue option
        if st.button("🚀 Continue to Analytics Platform", type="primary", use_container_width=True):
            st.session_state.show_welcome = False
            st.rerun()
        
        return  # Exit early if showing welcome
    
    # Main application header (compact version)
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%); padding: 1rem 2rem; border-radius: 10px; margin-bottom: 1rem; color: white;">
        <h2 style="margin: 0;">🏦 Enterprise Analytics Hub</h2>
        <p style="margin: 0; opacity: 0.9;">AI-Powered Business Intelligence for Banking & Finance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Database connection UI
    create_database_connection_ui()
    
    # Single main tab interface as requested
    st.markdown("---")
    

    # Query interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Natural Language Query Interface")
        
        # Business-focused example prompts
        examples = [
            "Show me monthly revenue for the past 12 months",
            "Forecast product-wise sales for next 6 months", 
            "Top 5 products by total sales with growth trends",
            "Sales performance by customer country with comparison",
            "Identify revenue trends and seasonal patterns",
            "Customer segmentation analysis by purchase behavior"
        ]
        
        st.write("**Business Intelligence Examples:**")
        example_cols = st.columns(2)
        for i, example in enumerate(examples):
            with example_cols[i % 2]:
                if st.button(example, key=f"example_{i}", use_container_width=True):
                    st.session_state.current_prompt = example
                    st.rerun()
    
    with col2:
        st.subheader("Analysis Settings")
        forecast_months = st.slider("Forecast Periods", 1, 24, 6)
        show_confidence = st.checkbox("Show Confidence Intervals", True)
        enable_advanced = st.checkbox("Enable Advanced Analytics", True)
    
    # Query input
    prompt = st.text_area(
        "Enter your business question in natural language:",
        value=st.session_state.get('current_prompt', ''),
        height=100,
        placeholder="e.g., Show me the revenue trend for the last 12 months and forecast the next 6 months with customer segmentation analysis"
    )
    
    # Process query button
    if st.button("🚀 Analyze Data", type="primary", use_container_width=True):
            if not prompt.strip():
                st.warning("Please enter a query")
                return
            
            with st.spinner("Processing your business query..."):
                try:
                    # Step 1: Convert natural language to SQL
                    with st.expander("📝 SQL Generation", expanded=False):
                        sql_query = groq_client.convert_to_sql(prompt)
                        st.code(sql_query, language="sql")
                    
                    # Step 2: Execute query
                    if st.session_state.db_connector.connection:
                        results_df = st.session_state.db_connector.execute_query(sql_query)
                    else:
                        results_df = db_manager.execute_query(sql_query)
                    
                    if results_df.empty:
                        st.warning("Query returned no results")
                        return
                    
                    st.success(f"✅ Analysis Complete - {len(results_df)} records processed")
                    
                    # Store results
                    st.session_state.current_results = results_df
                    
                    # Executive Summary
                    if enable_advanced:
                        summary = advanced_analytics.create_executive_summary(results_df)
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📊 Executive Summary")
                            st.metric("Total Records", f"{summary['total_records']:,}")
                            if summary['date_range']:
                                st.write(f"**Date Range:** {summary['date_range']['start']} to {summary['date_range']['end']}")
                                st.write(f"**Analysis Period:** {summary['date_range']['days']} days")
                        
                        with col2:
                            if summary['insights']:
                                st.subheader("💡 Key Insights")
                                for insight in summary['insights']:
                                    st.write(f"• {insight}")
                    
                    # KPI Dashboard
                    create_kpi_dashboard(results_df, advanced_analytics)
                    
                    # Display full results table
                    display_full_results_table(results_df)
                    
                    # Visualization section
                    st.divider()
                    
                    # Detect analysis type
                    needs_forecasting = detect_forecasting_keywords(prompt)
                    needs_time_series = detect_time_series_keywords(prompt) or needs_forecasting
                    
                    if needs_time_series:
                        # Time series analysis
                        chart_data = chart_generator.prepare_time_series_data(results_df)
                        
                        if chart_data is not None and not chart_data.empty:
                            fig = chart_generator.create_time_series_chart(chart_data, "Business Trend Analysis")
                            
                            if needs_forecasting:
                                st.session_state.show_forecasting = True
                                try:
                                    forecast_df = forecasting_engine.generate_forecast(
                                        chart_data, periods=forecast_months
                                    )
                                    
                                    if forecast_df is not None:
                                        fig = chart_generator.add_forecast_to_chart(
                                            fig, forecast_df, show_confidence
                                        )
                                        
                                        col1, col2 = st.columns([2, 1])
                                        with col1:
                                            st.subheader("📈 Trend Analysis with Forecast")
                                            st.plotly_chart(fig, use_container_width=True)
                                        
                                        with col2:
                                            st.subheader("🔮 Forecast Data")
                                            st.dataframe(forecast_df, use_container_width=True)
                                    else:
                                        st.subheader("📈 Trend Analysis")
                                        st.plotly_chart(fig, use_container_width=True)
                                        st.warning("Advanced forecasting not available for this data pattern")
                                
                                except Exception as e:
                                    st.subheader("📈 Trend Analysis")
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.error(f"Forecasting failed: {str(e)}")
                            else:
                                st.subheader("📈 Business Trend Analysis")
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Create alternative visualization
                            fig = chart_generator.create_chart_from_data(results_df, prompt)
                            if fig:
                                st.subheader("📊 Data Analysis")
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Chart visualization not available for this data structure")
                    
                    else:
                        # Non-time series analysis
                        fig = chart_generator.create_chart_from_data(results_df, prompt)
                        if fig:
                            st.subheader("📊 Business Analysis")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Advanced analytics
                    if enable_advanced:
                        st.divider()
                        st.subheader("🎯 Advanced Business Analytics")
                        
                        # Generate multiple charts
                        charts = advanced_analytics.create_multi_chart_dashboard(results_df, prompt)
                        
                        if len(charts) > 1:
                            chart_cols = st.columns(2)
                            for i, chart in enumerate(charts[:4]):  # Show up to 4 charts
                                with chart_cols[i % 2]:
                                    st.plotly_chart(chart, use_container_width=True)
                        
                        # Business insights
                        insights = advanced_analytics.generate_business_insights(results_df, prompt)
                        if insights:
                            st.subheader("💼 Business Insights")
                            for insight in insights:
                                st.markdown(f"• {insight}")
                    
                    # Save to history
                    query_record = {
                        'prompt': prompt,
                        'sql': sql_query,
                        'timestamp': datetime.now().isoformat(),
                        'rows_returned': len(results_df),
                        'database': st.session_state.connected_db
                    }
                    st.session_state.query_history.append(query_record)
                    
                    # Clear the prompt
                    if 'current_prompt' in st.session_state:
                        del st.session_state.current_prompt
                    
                    # Add data profiling tools dropdown 
                    st.markdown("---")
                    with st.expander("🔬 Professional Data Analysis Tools", expanded=False):
                        profiling_tools.create_profiling_interface(results_df, prompt)
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.exception(e)
    
    # Display current results if available
    if st.session_state.current_results is not None:
        st.markdown("---")
        st.subheader("📊 Real-time Business Dashboard")
        
        # Create comprehensive dashboard
        data = st.session_state.current_results
        
        # KPI Overview
        create_kpi_dashboard(data, advanced_analytics)
        
        # Multi-chart dashboard
        charts = advanced_analytics.create_multi_chart_dashboard(data, "comprehensive dashboard")
        
        if charts:
            for i in range(0, len(charts), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(charts):
                        with col:
                            st.plotly_chart(charts[i + j], use_container_width=True, key=f"dashboard_chart_{i+j}")
        
        # Business insights
        insights = advanced_analytics.generate_business_insights(data, "dashboard insights")
        if insights:
            st.subheader("💡 Strategic Insights")
            for insight in insights:
                st.markdown(f"• {insight}")
    
        # Advanced analytics section
        st.markdown("---")
        st.subheader("📈 Advanced Forecasting & Predictions")
        
        if st.session_state.show_forecasting:
            data = st.session_state.current_results
            
            # Forecasting controls
            col1, col2, col3 = st.columns(3)
            with col1:
                forecast_periods = st.slider("Forecast Periods", 1, 36, 12)
            with col2:
                confidence_level = st.slider("Confidence Level", 80, 99, 95)
            with col3:
                seasonality = st.selectbox("Seasonality", ["Auto", "Daily", "Weekly", "Monthly", "Yearly"])
            
            if st.button("🔮 Generate Advanced Forecast", type="primary"):
                with st.spinner("Generating advanced forecasting..."):
                    try:
                        # Prepare time series data
                        chart_data = chart_generator.prepare_time_series_data(data)
                        
                        if chart_data is not None:
                            # Generate forecast
                            forecast_df = forecasting_engine.generate_forecast(
                                chart_data, periods=forecast_periods
                            )
                            
                            if forecast_df is not None:
                                # Create enhanced forecast visualization
                                fig = chart_generator.create_time_series_chart(
                                    chart_data, "Advanced Business Forecast"
                                )
                                fig = chart_generator.add_forecast_to_chart(
                                    fig, forecast_df, True
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Forecast summary
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.subheader("📊 Forecast Summary")
                                    avg_forecast = forecast_df['forecast'].mean()
                                    st.metric("Average Forecast", f"${avg_forecast:,.0f}")
                                    
                                    growth_rate = ((forecast_df['forecast'].iloc[-1] - chart_data['value'].iloc[-1]) / chart_data['value'].iloc[-1]) * 100
                                    st.metric("Projected Growth", f"{growth_rate:.1f}%")
                                
                                with col2:
                                    st.subheader("🔮 Forecast Data")
                                    st.dataframe(forecast_df, use_container_width=True)
                            else:
                                st.error("Unable to generate forecast for this data")
                        else:
                            st.error("No time series data found for forecasting")
                    
                    except Exception as e:
                        st.error(f"Forecasting failed: {str(e)}")
        else:
            st.info("Run a time-series query first to enable advanced forecasting")
        
        # Step-by-step analysis section  
        st.markdown("---")
        st.subheader("🎯 Step-by-Step Data Analysis")
        data = st.session_state.current_results
        
        # Analysis controls
        col1, col2 = st.columns(2)
        with col1:
            analysis_depth = st.selectbox("Analysis Depth", ["Quick", "Comprehensive", "Deep Dive"])
        with col2:
            include_recommendations = st.checkbox("Include Strategic Recommendations", True)
        
        if st.button("🚀 Start Step-by-Step Analysis", type="primary", use_container_width=True):
                with st.spinner("Performing comprehensive step-by-step analysis..."):
                    # Get the last prompt for context
                    last_prompt = st.session_state.query_history[-1]['prompt'] if st.session_state.query_history else "general analysis"
                    
                    # Perform step-by-step analysis
                    analysis_results = step_analytics.analyze_data_step_by_step(data, last_prompt)
                    
                    # Display each step
                    for step in analysis_results['steps']:
                        with st.expander(f"Step {step['step_number']}: {step['title']}", expanded=True):
                            
                            if step['step_number'] == 1:  # Data Overview
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Rows", f"{step['details']['total_rows']:,}")
                                with col2:
                                    st.metric("Total Columns", step['details']['total_columns'])
                                with col3:
                                    st.metric("Memory Usage", f"{step['details']['memory_usage']/1024:.1f} KB")
                                
                                # Column information
                                col_df = pd.DataFrame(step['details']['column_info'])
                                st.dataframe(col_df, use_container_width=True)
                            
                            elif step['step_number'] == 2:  # Data Quality
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    score = step['details']['overall_score']
                                    color = "green" if score >= 80 else "orange" if score >= 60 else "red"
                                    st.metric("Quality Score", f"{score:.1f}%")
                                with col2:
                                    st.metric("Completeness", f"{step['details']['completeness']['score']:.1f}%")
                                with col3:
                                    st.metric("Consistency", f"{step['details']['consistency']['score']:.1f}%")
                                
                                # Create quality gauge
                                quality_fig = step_analytics.create_step_visualization(step)
                                if quality_fig:
                                    st.plotly_chart(quality_fig, use_container_width=True, key="quality_gauge")
                                
                                # Quality issues and strengths
                                if step['details']['issues']:
                                    st.warning("**Data Quality Issues:**")
                                    for issue in step['details']['issues']:
                                        st.write(f"• {issue}")
                                
                                if step['details']['strengths']:
                                    st.success("**Data Quality Strengths:**")
                                    for strength in step['details']['strengths']:
                                        st.write(f"• {strength}")
                            
                            elif step['step_number'] == 3:  # Statistical Analysis
                                if 'summary_statistics' in step['details']:
                                    st.write("**Summary Statistics:**")
                                    stats_df = pd.DataFrame(step['details']['summary_statistics'])
                                    st.dataframe(stats_df, use_container_width=True)
                                
                                # Correlation heatmap
                                corr_fig = step_analytics.create_step_visualization(step)
                                if corr_fig:
                                    st.plotly_chart(corr_fig, use_container_width=True, key="correlation_heatmap")
                                
                                # Strong correlations
                                if 'strong_correlations' in step['details'] and step['details']['strong_correlations']:
                                    st.write("**Strong Correlations Found:**")
                                    for corr in step['details']['strong_correlations']:
                                        st.write(f"• {corr['var1']} ↔ {corr['var2']}: {corr['correlation']:.3f}")
                            
                            elif step['step_number'] == 4:  # Pattern Detection
                                # Growth trends
                                if step['details']['growth_trends']:
                                    trend_fig = step_analytics.create_step_visualization(step)
                                    if trend_fig:
                                        st.plotly_chart(trend_fig, use_container_width=True, key="trend_chart")
                                
                                # Seasonal patterns
                                if step['details']['seasonal_patterns']:
                                    st.write("**Seasonal Patterns Detected:**")
                                    for pattern in step['details']['seasonal_patterns']:
                                        st.write(f"• {pattern['metric']}: Peak in month {pattern['peak_month']}, Low in month {pattern['low_month']}")
                            
                            elif step['step_number'] == 5:  # Business Intelligence
                                if analysis_results['insights']:
                                    st.write("**Key Business Insights:**")
                                    for insight in analysis_results['insights']:
                                        st.info(f"💡 {insight}")
                            
                            elif step['step_number'] == 6:  # Recommendations
                                if include_recommendations and analysis_results['recommendations']:
                                    for rec in analysis_results['recommendations']:
                                        with st.container():
                                            priority_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                                            st.markdown(f"""
                                            **{priority_color.get(rec['priority'], '⚪')} {rec['category']}** ({rec['priority']} Priority)
                                            
                                            *Recommendation:* {rec['recommendation']}
                                            
                                            *Expected Impact:* {rec['impact']}
                                            """)
                    
                    # Summary section
                    st.divider()
                    st.subheader("📋 Analysis Summary")
                    
                    summary_col1, summary_col2 = st.columns(2)
                    with summary_col1:
                        st.metric("Data Quality Score", f"{analysis_results['steps'][1]['details']['overall_score']:.1f}%")
                        st.metric("Key Insights", len(analysis_results['insights']))
                    
                    with summary_col2:
                        st.metric("Recommendations", len(analysis_results['recommendations']))
                        if analysis_results['steps'][1]['details']['issues']:
                            st.metric("Issues Found", len(analysis_results['steps'][1]['details']['issues']))
                        else:
                            st.metric("Issues Found", 0)
        
    else:
        if st.session_state.get('enable_step_analysis'):
            st.info("Step-by-step analysis is enabled. Run a query to start comprehensive data analysis.")
        else:
            st.info("Run a query first to enable step-by-step analysis, or enable it from the welcome screen.")
    
    # Query history sidebar
    with st.sidebar:
        st.divider()
        st.header("📚 Query History")
        
        if st.session_state.query_history:
            for i, query in enumerate(reversed(st.session_state.query_history[-10:])):
                timestamp = datetime.fromisoformat(query['timestamp']).strftime('%H:%M')
                if st.button(
                    f"{timestamp} - {query['prompt'][:30]}...", 
                    key=f"history_{i}",
                    help=f"Database: {query.get('database', 'Unknown')}\nRows: {query['rows_returned']}"
                ):
                    st.session_state.current_prompt = query['prompt']
                    st.rerun()
        else:
            st.write("No queries yet")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.exception(e)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import io
import logging
from typing import Dict, Any, Optional
import base64

# Import custom modules
from database_connector import UniversalDatabaseConnector, render_database_connection_ui, initialize_sample_database
from ai_query_engine import AIQueryEngine
from data_profiling import DataProfilingEngine, check_profiling_tools
from forecasting import ForecastingEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Analytics Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Perplexity-like interface
st.markdown("""
<style>
    /* Clean white background */
    .main > div {
        background-color: white;
        padding: 0;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    /* Query interface styling */
    .query-container {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 15px;
        padding: 2rem;
        margin-bottom: 2rem;
    }
    
    /* Result container */
    .result-container {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* History sidebar */
    .history-item {
        background: white;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .history-item:hover {
        background: #f8f9fa;
        border-color: #007bff;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 25px;
        border: none;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    /* Download section */
    .download-section {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        text-align: center;
    }
    
    /* Tools section */
    .tools-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .tool-card {
        background: white;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s;
    }
    
    .tool-card:hover {
        border-color: #007bff;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    
    if 'connected' not in st.session_state:
        st.session_state.connected = False
    
    if 'db_connector' not in st.session_state:
        st.session_state.db_connector = UniversalDatabaseConnector()
    
    if 'ai_engine' not in st.session_state:
        st.session_state.ai_engine = AIQueryEngine()
    
    if 'profiling_engine' not in st.session_state:
        st.session_state.profiling_engine = DataProfilingEngine()
    
    if 'forecasting_engine' not in st.session_state:
        st.session_state.forecasting_engine = ForecastingEngine()

def render_sidebar_history():
    """Render query history in sidebar"""
    with st.sidebar:
        st.markdown("### 📝 Query History")
        
        if not st.session_state.query_history:
            st.info("No queries yet. Start by asking a question!")
        else:
            for i, query_item in enumerate(reversed(st.session_state.query_history[-10:])):  # Show last 10
                with st.container():
                    st.markdown(f"""
                    <div class="history-item">
                        <strong>{query_item.get('timestamp', 'Unknown').strftime('%H:%M')}</strong><br>
                        <small>{query_item.get('question', 'Unknown query')[:50]}...</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Reload", key=f"reload_{i}"):
                        st.session_state.current_result = query_item
                        st.rerun()

def render_main_query_interface():
    """Render the main query interface"""
    
    # Header
    st.markdown("""
    <div class="header-container">
        <h1>🚀 AI-Powered Analytics Platform</h1>
        <p>Connect any database and get instant AI insights with advanced analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Connection status and query interface
    if not st.session_state.connected:
        connected = render_database_connection_ui()
        if not connected:
            return
    
    # Main query interface
    st.markdown("""
    <div class="query-container">
        <h3>💬 Ask anything about your data</h3>
        <p>Use natural language to query your database and get instant insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample questions
    sample_questions = [
        "What are the top 10 customers by revenue?",
        "Show me monthly sales trends for this year",
        "Which products have the highest profit margins?",
        "What's the revenue breakdown by region?",
        "Show me customers with declining purchase patterns",
        "What are the seasonal trends in our data?"
    ]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_question = st.text_input(
            "Your question:",
            placeholder="e.g., Show me the top performing products by revenue",
            key="main_query"
        )
    
    with col2:
        sample_question = st.selectbox(
            "Or try a sample:",
            [""] + sample_questions,
            key="sample_selector"
        )
    
    # Use sample question if selected
    if sample_question:
        user_question = sample_question
    
    # Execute query button
    if st.button("🔍 Analyze Data", type="primary", disabled=not user_question):
        execute_query(user_question)

def execute_query(question: str):
    """Execute user query and display results"""
    
    with st.spinner("🤖 Processing your question with AI..."):
        try:
            # Generate SQL query
            query_result = st.session_state.ai_engine.natural_language_to_sql(
                question, st.session_state.db_connector
            )
            
            # Display SQL query (collapsible)
            with st.expander("🔍 Generated SQL Query", expanded=False):
                st.code(query_result['sql'], language='sql')
                st.info(f"Confidence: {query_result['confidence']*100:.1f}%")
            
            # Execute query
            df = st.session_state.ai_engine.execute_query(
                query_result['sql'], st.session_state.db_connector
            )
            
            # Store result
            result_item = {
                'timestamp': datetime.now(),
                'question': question,
                'sql': query_result['sql'],
                'data': df,
                'explanation': query_result['explanation']
            }
            
            st.session_state.query_history.append(result_item)
            st.session_state.current_result = result_item
            
            # Display results
            display_query_results(result_item)
            
        except Exception as e:
            st.error(f"Query execution failed: {str(e)}")
            logger.error(f"Query execution error: {str(e)}")

def display_query_results(result_item: Dict[str, Any]):
    """Display query results with visualizations and download options"""
    
    df = result_item['data']
    question = result_item['question']
    
    st.markdown("""
    <div class="result-container">
        <h3>📊 Query Results</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if df.empty:
        st.warning("No data found for your query.")
        return
    
    # Results overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(df):,}</h3>
            <p>Total Rows</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(df.columns)}</h3>
            <p>Columns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        numeric_cols = len(df.select_dtypes(include=['number']).columns)
        st.markdown(f"""
        <div class="metric-card">
            <h3>{numeric_cols}</h3>
            <p>Numeric Columns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.markdown(f"""
        <div class="metric-card">
            <h3>{memory_mb:.1f} MB</h3>
            <p>Memory Usage</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Data preview
    st.markdown("### 📋 Data Preview")
    st.dataframe(df.head(100), use_container_width=True)
    
    # Download section
    render_download_section(df, question)
    
    # Visualizations
    render_visualizations(df, question)
    
    # Forecasting section
    render_forecasting_section(df, question)
    
    # Data profiling and quality
    render_data_profiling_section(df, question)

def render_download_section(df: pd.DataFrame, query_name: str):
    """Render download options for results"""
    
    st.markdown("### 📥 Download Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # CSV download
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV",
            data=csv,
            file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Excel download
    with col2:
        buffer = io.BytesIO()
        try:
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Query Results', index=False)
            excel_data = buffer.getvalue()
            
            st.download_button(
                label="📊 Download Excel",
                data=excel_data,
                file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"Excel export error: {str(e)}")
    
    # JSON download
    with col3:
        json_data = df.to_json(orient='records', indent=2)
        st.download_button(
            label="🔗 Download JSON",
            data=json_data,
            file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    # SQL script download
    with col4:
        if st.session_state.current_result:
            sql_script = f"-- Query: {query_name}\n-- Generated: {datetime.now()}\n\n{st.session_state.current_result.get('sql', '')}"
            st.download_button(
                label="💾 Download SQL",
                data=sql_script,
                file_name=f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql",
                mime="text/plain"
            )

def render_visualizations(df: pd.DataFrame, question: str):
    """Render automatic visualizations"""
    
    st.markdown("### 📈 Smart Visualizations")
    
    # Get visualization suggestions
    viz_suggestions = st.session_state.ai_engine.suggest_visualizations(df, question)
    
    if not viz_suggestions:
        st.info("No suitable visualizations found for this data.")
        return
    
    # Create tabs for different visualizations
    viz_tabs = st.tabs([f"📊 {viz['title']}" for viz in viz_suggestions])
    
    for tab, viz_config in zip(viz_tabs, viz_suggestions):
        with tab:
            try:
                fig = st.session_state.ai_engine.create_visualization(df, viz_config)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(viz_config['description'])
            except Exception as e:
                st.error(f"Visualization error: {str(e)}")

def render_forecasting_section(df: pd.DataFrame, question: str):
    """Render forecasting section with checkbox option"""
    
    st.markdown("### 🔮 Predictive Analytics")
    
    # Check if forecasting is applicable
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not date_cols or not numeric_cols:
        st.info("Forecasting requires date and numeric columns. Not applicable for current data.")
        return
    
    # Forecasting checkbox
    show_forecasting = st.checkbox("🔮 Show Predictive Forecasting", key="forecast_checkbox")
    
    if show_forecasting:
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                date_col = st.selectbox("Select Date Column:", date_cols)
            
            with col2:
                value_col = st.selectbox("Select Value Column:", numeric_cols)
            
            if st.button("Generate Forecast", type="primary"):
                with st.spinner("Generating forecast..."):
                    
                    # Prepare data for forecasting
                    forecast_df = df[[date_col, value_col]].copy()
                    forecast_df[date_col] = pd.to_datetime(forecast_df[date_col])
                    forecast_df = forecast_df.dropna().sort_values(by=date_col)
                    
                    # Generate forecast
                    forecast_result = st.session_state.forecasting_engine.generate_forecast(
                        forecast_df, date_col, value_col
                    )
                    
                    if forecast_result:
                        # Create forecast visualization
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=forecast_df[date_col],
                            y=forecast_df[value_col],
                            mode='lines+markers',
                            name='Historical Data',
                            line=dict(color='blue')
                        ))
                        
                        # Forecast
                        forecast_data = forecast_result.get('forecast', pd.DataFrame())
                        if not forecast_data.empty:
                            fig.add_trace(go.Scatter(
                                x=forecast_data['ds'],
                                y=forecast_data['yhat'],
                                mode='lines',
                                name='Forecast',
                                line=dict(color='red', dash='dash')
                            ))
                            
                            # Confidence intervals
                            fig.add_trace(go.Scatter(
                                x=forecast_data['ds'],
                                y=forecast_data['yhat_upper'],
                                fill=None,
                                mode='lines',
                                line_color='rgba(0,0,0,0)',
                                showlegend=False
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=forecast_data['ds'],
                                y=forecast_data['yhat_lower'],
                                fill='tonexty',
                                mode='lines',
                                line_color='rgba(0,0,0,0)',
                                name='Confidence Interval',
                                fillcolor='rgba(255,0,0,0.2)'
                            ))
                        
                        fig.update_layout(
                            title=f'Forecast: {value_col} over Time',
                            xaxis_title='Date',
                            yaxis_title=value_col,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast metrics
                        if 'metrics' in forecast_result:
                            metrics = forecast_result['metrics']
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
                            with col2:
                                st.metric("MAE", f"{metrics.get('mae', 0):.2f}")
                            with col3:
                                st.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
        
        except Exception as e:
            st.error(f"Forecasting error: {str(e)}")

def render_data_profiling_section(df: pd.DataFrame, query_name: str):
    """Render comprehensive data profiling section"""
    
    st.markdown("### 🔍 Data Profiling & Quality Analysis")
    
    # Data quality summary
    quality_report = st.session_state.profiling_engine.render_data_quality_summary(df)
    
    # Profiling tools
    st.markdown("### 🛠️ Advanced Profiling Tools")
    
    # Tool status
    tools_status = check_profiling_tools()
    
    # Tools grid
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Available Tools:")
        for tool, available in tools_status.items():
            status = "✅" if available else "❌"
            st.write(f"{status} {tool.replace('_', '-').title()}")
    
    with col2:
        if st.button("🚀 Launch All Available Tools"):
            st.session_state.profiling_engine.render_profiling_tools(df, query_name)

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar_history()
    
    # Main content
    try:
        render_main_query_interface()
        
        # Display current result if available
        if st.session_state.current_result:
            display_query_results(st.session_state.current_result)
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Main application error: {str(e)}")

if __name__ == "__main__":
    main()
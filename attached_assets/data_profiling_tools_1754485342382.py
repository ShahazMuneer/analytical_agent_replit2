"""
Data Profiling Tools Integration
Provides one-click access to various data profiling and exploration tools
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
import os
from typing import Optional, Dict, Any
import plotly.express as px
import plotly.graph_objects as go

# Import profiling tools with error handling
try:
    import pygwalker as pyg
    PYGWALKER_AVAILABLE = True
except ImportError:
    PYGWALKER_AVAILABLE = False
    pyg = None

try:
    import dtale
    DTALE_AVAILABLE = True
except ImportError:
    DTALE_AVAILABLE = False
    dtale = None

try:
    import sweetviz as sv
    SWEETVIZ_AVAILABLE = True
except ImportError:
    SWEETVIZ_AVAILABLE = False
    sv = None

# Always available tools
PANDAS_PROFILING_AVAILABLE = True  # Basic pandas profiling
DATA_QUALITY_CHECKER_AVAILABLE = True  # Custom data quality checker


class DataProfilingTools:
    """Unified interface for data profiling and exploration tools"""
    
    def __init__(self):
        self.tools_status = {
            "PyGWalker": PYGWALKER_AVAILABLE,
            "D-Tale": DTALE_AVAILABLE, 
            "SweetViz": SWEETVIZ_AVAILABLE,
            "Pandas Profiling": PANDAS_PROFILING_AVAILABLE,
            "Data Quality Checker": DATA_QUALITY_CHECKER_AVAILABLE
        }
    
    def get_available_tools(self) -> list:
        """Get list of available profiling tools"""
        return [tool for tool, available in self.tools_status.items() if available]
    
    def create_profiling_interface(self, data: pd.DataFrame, query_description: str = ""):
        """Create the main profiling tools interface"""
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 12px; color: white; margin-bottom: 1rem;">
            <h3 style="margin: 0;">🔍 Advanced Data Profiling & Exploration Tools</h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">One-click access to professional data analysis tools</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tool selection
        available_tools = self.get_available_tools()
        
        if not available_tools:
            st.error("No data profiling tools are currently available. Please install the required packages.")
            return
        
        # Create columns for tool buttons
        cols = st.columns(len(available_tools))
        
        for i, tool in enumerate(available_tools):
            with cols[i]:
                self._create_tool_button(tool, data, query_description)
    
    def _create_tool_button(self, tool_name: str, data: pd.DataFrame, query_description: str):
        """Create individual tool button with description"""
        
        tool_info = {
            "PyGWalker": {
                "icon": "📊",
                "description": "Interactive visual analytics with drag-and-drop interface",
                "color": "#FF6B6B"
            },
            "D-Tale": {
                "icon": "🔬", 
                "description": "Comprehensive data exploration and analysis platform",
                "color": "#4ECDC4"
            },
            "SweetViz": {
                "icon": "🍯",
                "description": "Beautiful automated EDA with target analysis",
                "color": "#F7B731"
            },
            "Pandas Profiling": {
                "icon": "🐼",
                "description": "Quick statistical overview and data profiling",
                "color": "#45B7D1"
            },
            "Data Quality Checker": {
                "icon": "✅",
                "description": "Custom data quality validation and testing",
                "color": "#5F27CD"
            }
        }
        
        info = tool_info.get(tool_name, {"icon": "🔧", "description": "Data analysis tool", "color": "#666"})
        
        # Tool card
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; 
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-left: 4px solid {info['color']}; 
                    text-align: center; margin-bottom: 1rem;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{info['icon']}</div>
            <h4 style="margin: 0; color: #333;">{tool_name}</h4>
            <p style="margin: 0.5rem 0; color: #666; font-size: 0.9rem;">{info['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button(f"Launch {tool_name}", key=f"launch_{tool_name}", use_container_width=True):
            self._launch_tool(tool_name, data, query_description)
    
    def _launch_tool(self, tool_name: str, data: pd.DataFrame, query_description: str):
        """Launch the selected profiling tool"""
        
        try:
            if tool_name == "PyGWalker" and PYGWALKER_AVAILABLE:
                self._launch_pygwalker(data, query_description)
            
            elif tool_name == "D-Tale" and DTALE_AVAILABLE:
                self._launch_dtale(data, query_description)
            
            elif tool_name == "SweetViz" and SWEETVIZ_AVAILABLE:
                self._launch_sweetviz(data, query_description)
            
            elif tool_name == "Pandas Profiling":
                self._launch_pandas_profiling(data, query_description)
            
            elif tool_name == "Data Quality Checker":
                self._launch_data_quality_checker(data, query_description)
            
            else:
                st.error(f"{tool_name} is not available or not installed.")
                
        except Exception as e:
            st.error(f"Error launching {tool_name}: {str(e)}")
    
    def _launch_pygwalker(self, data: pd.DataFrame, query_description: str):
        """Launch PyGWalker interactive visualization"""
        st.subheader("📊 PyGWalker - Interactive Visual Analytics")
        st.markdown(f"**Analysis Context:** {query_description}")
        
        if pyg is not None:
            # Create PyGWalker visualization
            pyg_html = pyg.to_html(data)
            st.components.v1.html(pyg_html, height=800, scrolling=True)
        else:
            st.error("PyGWalker is not available. Please install it: pip install pygwalker")
    
    def _launch_dtale(self, data: pd.DataFrame, query_description: str):
        """Launch D-Tale data exploration"""
        st.subheader("🔬 D-Tale - Data Exploration Platform")
        st.markdown(f"**Analysis Context:** {query_description}")
        
        if dtale is not None:
            # Launch D-Tale instance
            d = dtale.show(data)
            st.success(f"D-Tale launched! Access your data exploration at: {d._main_url}")
            
            # Embed D-Tale in iframe
            st.components.v1.iframe(d._main_url, height=800, scrolling=True)
        else:
            st.error("D-Tale is not available. Please install it: pip install dtale")
    
    def _launch_ydata_profiling(self, data: pd.DataFrame, query_description: str):
        """Generate YData Profiling report"""
        st.subheader("📈 YData Profiling - Statistical Data Report")
        st.markdown(f"**Analysis Context:** {query_description}")
        
        with st.spinner("Generating comprehensive data profiling report..."):
            # Create profile report
            profile = ProfileReport(
                data, 
                title=f"Data Profiling Report - {query_description}",
                explorative=True,
                dark_mode=False
            )
            
            # Convert to HTML
            html_report = profile.to_html()
            
            # Display in Streamlit
            st.components.v1.html(html_report, height=800, scrolling=True)
            
            # Download option
            st.download_button(
                label="📥 Download Full Report (HTML)",
                data=html_report,
                file_name=f"data_profile_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html"
            )
    
    def _launch_sweetviz(self, data: pd.DataFrame, query_description: str):
        """Generate SweetViz analysis"""
        st.subheader("🍯 SweetViz - Automated EDA Report")
        st.markdown(f"**Analysis Context:** {query_description}")
        
        if sv is not None:
            with st.spinner("Generating SweetViz analysis report..."):
                # Create SweetViz report
                report = sv.analyze(data)
                
                # Generate HTML report
                report.show_html(filepath='sweetviz_report.html', open_browser=False, layout='vertical')
                
                # Read and display the HTML
                with open('sweetviz_report.html', 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                st.components.v1.html(html_content, height=800, scrolling=True)
                
                # Download option
                st.download_button(
                    label="📥 Download SweetViz Report (HTML)",
                    data=html_content,
                    file_name=f"sweetviz_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )
                
                # Clean up
                if os.path.exists('sweetviz_report.html'):
                    os.remove('sweetviz_report.html')
        else:
            st.error("SweetViz is not available. Please install it: pip install sweetviz")
    
    def _launch_pandas_profiling(self, data: pd.DataFrame, query_description: str):
        """Generate basic pandas profiling report"""
        st.subheader("🐼 Pandas Profiling - Statistical Overview")
        st.markdown(f"**Analysis Context:** {query_description}")
        
        with st.spinner("Generating pandas profiling report..."):
            # Basic statistical overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Rows", f"{len(data):,}")
            with col2:
                st.metric("Columns", len(data.columns))
            with col3:
                missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
                st.metric("Missing Data", f"{missing_pct:.1f}%")
            with col4:
                memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
                st.metric("Memory Usage", f"{memory_mb:.1f} MB")
            
            # Data types and missing values
            st.subheader("Column Analysis")
            col_info = []
            for col in data.columns:
                col_info.append({
                    'Column': col,
                    'Type': str(data[col].dtype),
                    'Non-Null Count': data[col].count(),
                    'Null Count': data[col].isnull().sum(),
                    'Unique Values': data[col].nunique(),
                    'Memory Usage': data[col].memory_usage(deep=True)
                })
            
            col_df = pd.DataFrame(col_info)
            st.dataframe(col_df, use_container_width=True)
            
            # Statistical summary for numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.subheader("Statistical Summary")
                st.dataframe(data[numeric_cols].describe(), use_container_width=True)
            
            # Sample data
            st.subheader("Sample Data")
            st.dataframe(data.head(10), use_container_width=True)
    
    def _launch_data_quality_checker(self, data: pd.DataFrame, query_description: str):
        """Launch custom data quality validation"""
        st.subheader("✅ Great Expectations - Data Quality Validation")
        st.markdown(f"**Analysis Context:** {query_description}")
        
        with st.spinner("Setting up data quality validation..."):
            try:
                # Create Great Expectations context
                context = gx.get_context()
                
                # Create data source
                datasource = context.sources.add_pandas("pandas_datasource")
                data_asset = datasource.add_dataframe_asset(name="data_asset", dataframe=data)
                
                # Create batch request
                batch_request = data_asset.build_batch_request()
                
                # Create expectation suite
                suite_name = f"data_quality_suite_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
                suite = context.add_expectation_suite(expectation_suite_name=suite_name)
                
                # Add basic expectations
                validator = context.get_validator(
                    batch_request=batch_request,
                    expectation_suite=suite
                )
                
                # Basic data quality checks
                st.subheader("🔍 Data Quality Checks")
                
                results = []
                
                # Check for null values
                for column in data.columns:
                    if data[column].isnull().any():
                        expectation = validator.expect_column_values_to_not_be_null(column)
                        results.append({
                            'Check': f'No null values in {column}',
                            'Status': '✅ Pass' if expectation.success else '❌ Fail',
                            'Details': f"{data[column].isnull().sum()} null values found"
                        })
                
                # Check for duplicates
                duplicate_count = data.duplicated().sum()
                results.append({
                    'Check': 'No duplicate rows',
                    'Status': '✅ Pass' if duplicate_count == 0 else '❌ Fail',
                    'Details': f"{duplicate_count} duplicate rows found"
                })
                
                # Display results
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    passed = sum(1 for r in results if 'Pass' in r['Status'])
                    st.metric("Checks Passed", passed)
                with col2:
                    failed = sum(1 for r in results if 'Fail' in r['Status'])
                    st.metric("Checks Failed", failed)
                with col3:
                    score = (passed / len(results)) * 100 if results else 0
                    st.metric("Quality Score", f"{score:.1f}%")
                
            except Exception as e:
                st.error(f"Great Expectations setup failed: {str(e)}")
                st.info("Great Expectations requires additional configuration for full functionality.")
    
    def create_quick_stats_overview(self, data: pd.DataFrame):
        """Create a quick statistical overview"""
        st.markdown("### 📊 Quick Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rows", f"{len(data):,}")
        with col2:
            st.metric("Columns", len(data.columns))
        with col3:
            missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        with col4:
            memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("Memory Usage", f"{memory_mb:.1f} MB")
        
        # Data types summary
        st.markdown("**Data Types:**")
        type_counts = data.dtypes.value_counts()
        type_df = pd.DataFrame({
            'Data Type': type_counts.index.astype(str),
            'Count': type_counts.values
        })
        st.dataframe(type_df, use_container_width=True)
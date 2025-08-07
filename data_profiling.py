import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sweetviz as sv
import io
import base64
from typing import Dict, Any, List, Tuple
import logging

# Import profiling libraries with fallbacks
try:
    import dtale
    DTALE_AVAILABLE = True
except ImportError:
    DTALE_AVAILABLE = False

try:
    import pygwalker as pyg
    PYGWALKER_AVAILABLE = True
except ImportError:
    PYGWALKER_AVAILABLE = False

try:
    from ydata_profiling import ProfileReport
    YDATA_PROFILING_AVAILABLE = True
except ImportError:
    try:
        from pandas_profiling import ProfileReport
        YDATA_PROFILING_AVAILABLE = True
    except ImportError:
        YDATA_PROFILING_AVAILABLE = False

logger = logging.getLogger(__name__)

class DataProfilingEngine:
    """Comprehensive data profiling and quality assessment engine"""
    
    def __init__(self):
        self.profile_cache = {}
    
    def generate_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data quality assessment"""
        
        quality_report = {
            'overview': self._get_overview_metrics(df),
            'completeness': self._assess_completeness(df),
            'consistency': self._assess_consistency(df),
            'validity': self._assess_validity(df),
            'uniqueness': self._assess_uniqueness(df),
            'recommendations': self._generate_recommendations(df)
        }
        
        return quality_report
    
    def _get_overview_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic overview metrics"""
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': len(df.select_dtypes(include=['datetime']).columns),
            'duplicate_rows': df.duplicated().sum(),
            'total_missing_values': df.isnull().sum().sum()
        }
    
    def _assess_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data completeness"""
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data / len(df)) * 100
        
        completeness_score = (1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))) * 100
        
        return {
            'completeness_score': round(completeness_score, 2),
            'missing_by_column': missing_data.to_dict(),
            'missing_percentage_by_column': missing_percentage.to_dict(),
            'columns_with_missing': missing_data[missing_data > 0].index.tolist(),
            'high_missing_columns': missing_percentage[missing_percentage > 50].index.tolist()
        }
    
    def _assess_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data consistency"""
        consistency_issues = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for inconsistent formatting
                unique_values = df[col].dropna().unique()
                if len(unique_values) > 1:
                    # Check for case inconsistencies
                    lower_values = set(str(v).lower() for v in unique_values)
                    if len(lower_values) < len(unique_values):
                        consistency_issues.append({
                            'column': col,
                            'issue': 'case_inconsistency',
                            'details': f'Found {len(unique_values)} values that could be {len(lower_values)} unique values'
                        })
                    
                    # Check for whitespace issues
                    trimmed_values = set(str(v).strip() for v in unique_values)
                    if len(trimmed_values) < len(unique_values):
                        consistency_issues.append({
                            'column': col,
                            'issue': 'whitespace_inconsistency',
                            'details': 'Found values with leading/trailing whitespace'
                        })
        
        return {
            'consistency_score': max(0, 100 - len(consistency_issues) * 10),
            'issues': consistency_issues
        }
    
    def _assess_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data validity"""
        validity_issues = []
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                # Check for outliers using IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                if len(outliers) > 0:
                    validity_issues.append({
                        'column': col,
                        'issue': 'outliers',
                        'count': len(outliers),
                        'percentage': round((len(outliers) / len(df)) * 100, 2)
                    })
            
            elif df[col].dtype == 'object':
                # Check for potential data type mismatches
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    # Try to identify if column should be numeric
                    try:
                        pd.to_numeric(non_null_values.head(100))
                        validity_issues.append({
                            'column': col,
                            'issue': 'potential_numeric_column',
                            'details': 'Column appears to contain numeric data but is stored as text'
                        })
                    except:
                        pass
                    
                    # Try to identify if column should be datetime
                    try:
                        pd.to_datetime(non_null_values.head(100))
                        validity_issues.append({
                            'column': col,
                            'issue': 'potential_datetime_column',
                            'details': 'Column appears to contain date/time data but is stored as text'
                        })
                    except:
                        pass
        
        return {
            'validity_score': max(0, 100 - len(validity_issues) * 5),
            'issues': validity_issues
        }
    
    def _assess_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data uniqueness"""
        uniqueness_metrics = {}
        
        for col in df.columns:
            unique_count = df[col].nunique()
            total_count = len(df[col].dropna())
            uniqueness_ratio = unique_count / total_count if total_count > 0 else 0
            
            uniqueness_metrics[col] = {
                'unique_count': unique_count,
                'uniqueness_ratio': round(uniqueness_ratio, 4),
                'is_unique': uniqueness_ratio == 1.0,
                'has_duplicates': uniqueness_ratio < 1.0
            }
        
        return uniqueness_metrics
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Generate data quality improvement recommendations"""
        recommendations = []
        
        # Check missing data
        missing_data = df.isnull().sum()
        high_missing = missing_data[missing_data > len(df) * 0.5]
        if len(high_missing) > 0:
            recommendations.append({
                'priority': 'high',
                'category': 'completeness',
                'issue': f'Columns with >50% missing data: {", ".join(high_missing.index)}',
                'recommendation': 'Consider removing these columns or investigating why data is missing'
            })
        
        # Check duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            recommendations.append({
                'priority': 'medium',
                'category': 'uniqueness',
                'issue': f'{duplicate_count} duplicate rows found',
                'recommendation': 'Remove duplicate rows or investigate if they represent valid data'
            })
        
        # Check data types
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_numeric(df[col].dropna().head(100))
                    recommendations.append({
                        'priority': 'low',
                        'category': 'validity',
                        'issue': f'Column "{col}" appears numeric but stored as text',
                        'recommendation': f'Convert column "{col}" to numeric data type for better performance'
                    })
                except:
                    pass
        
        return recommendations
    
    def create_data_quality_dashboard(self, df: pd.DataFrame, quality_report: Dict[str, Any]) -> go.Figure:
        """Create interactive data quality dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Data Quality Scores',
                'Missing Data by Column',
                'Data Type Distribution',
                'Uniqueness Ratios'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "bar"}]]
        )
        
        # Quality scores
        scores = [
            quality_report['completeness']['completeness_score'],
            quality_report['consistency']['consistency_score'],
            quality_report['validity']['validity_score'],
            100 - (len(quality_report['recommendations']) * 10)  # Overall score
        ]
        score_labels = ['Completeness', 'Consistency', 'Validity', 'Overall']
        
        fig.add_trace(
            go.Bar(x=score_labels, y=scores, name='Quality Scores', 
                   marker_color=['green' if s >= 80 else 'orange' if s >= 60 else 'red' for s in scores]),
            row=1, col=1
        )
        
        # Missing data
        missing_data = quality_report['completeness']['missing_by_column']
        if missing_data:
            cols_with_missing = {k: v for k, v in missing_data.items() if v > 0}
            if cols_with_missing:
                fig.add_trace(
                    go.Bar(x=list(cols_with_missing.keys()), y=list(cols_with_missing.values()), 
                           name='Missing Values', marker_color='red'),
                    row=1, col=2
                )
        
        # Data type distribution
        overview = quality_report['overview']
        type_counts = [
            overview['numeric_columns'],
            overview['categorical_columns'],
            overview['datetime_columns']
        ]
        type_labels = ['Numeric', 'Categorical', 'DateTime']
        
        fig.add_trace(
            go.Pie(labels=type_labels, values=type_counts, name="Data Types"),
            row=2, col=1
        )
        
        # Uniqueness ratios (top 10 columns)
        uniqueness_data = quality_report['uniqueness']
        unique_ratios = {col: data['uniqueness_ratio'] for col, data in uniqueness_data.items()}
        sorted_ratios = sorted(unique_ratios.items(), key=lambda x: x[1], reverse=True)[:10]
        
        if sorted_ratios:
            fig.add_trace(
                go.Bar(x=[item[0] for item in sorted_ratios], 
                       y=[item[1] for item in sorted_ratios],
                       name='Uniqueness Ratio', marker_color='blue'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=False, title_text="Data Quality Dashboard")
        
        return fig
    
    def render_profiling_tools(self, df: pd.DataFrame, query_name: str = "data"):
        """Render available profiling tools"""
        
        st.markdown("### 🔍 Advanced Data Profiling Tools")
        
        if df.empty:
            st.warning("No data available for profiling")
            return
        
        # Limit data size for performance
        if len(df) > 10000:
            st.info(f"Large dataset detected ({len(df)} rows). Using sample of 10,000 rows for profiling.")
            df_sample = df.sample(n=10000, random_state=42)
        else:
            df_sample = df
        
        # Tool selection
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔬 Launch D-Tale", disabled=not DTALE_AVAILABLE):
                if DTALE_AVAILABLE:
                    self._launch_dtale(df_sample, query_name)
                else:
                    st.error("D-Tale not available")
        
        with col2:
            if st.button("📊 Launch PyGWalker", disabled=not PYGWALKER_AVAILABLE):
                if PYGWALKER_AVAILABLE:
                    self._launch_pygwalker(df_sample, query_name)
                else:
                    st.error("PyGWalker not available")
        
        with col3:
            if st.button("📋 Generate Profile Report", disabled=not YDATA_PROFILING_AVAILABLE):
                if YDATA_PROFILING_AVAILABLE:
                    self._generate_profile_report(df_sample, query_name)
                else:
                    st.error("ydata-profiling not available")
        
        # SweetViz profiling (always available)
        if st.button("🍭 Generate SweetViz Report"):
            self._generate_sweetviz_report(df_sample, query_name)
    
    def _launch_dtale(self, df: pd.DataFrame, query_name: str):
        """Launch D-Tale interface"""
        try:
            import dtale
            
            # Create D-Tale instance
            d = dtale.show(df, host='0.0.0.0', port=40000, open_browser=False)
            
            st.success("D-Tale launched successfully!")
            st.info(f"Access D-Tale at: {d._url}")
            st.markdown(f"[Open D-Tale Interface]({d._url})")
            
            # Store instance reference
            st.session_state[f'dtale_{query_name}'] = d
            
        except Exception as e:
            st.error(f"Failed to launch D-Tale: {str(e)}")
    
    def _launch_pygwalker(self, df: pd.DataFrame, query_name: str):
        """Launch PyGWalker interface"""
        try:
            import pygwalker as pyg
            
            st.markdown("### PyGWalker Interactive Analysis")
            
            # Create PyGWalker component
            pyg_html = pyg.walk(df, return_html=True)
            
            # Display in Streamlit
            st.components.v1.html(pyg_html, height=1000, scrolling=True)
            
        except Exception as e:
            st.error(f"Failed to launch PyGWalker: {str(e)}")
    
    def _generate_profile_report(self, df: pd.DataFrame, query_name: str):
        """Generate ydata-profiling report"""
        try:
            with st.spinner("Generating comprehensive data profile..."):
                profile = ProfileReport(
                    df, 
                    title=f"Data Profile Report - {query_name}",
                    explorative=True,
                    minimal=False
                )
                
                # Generate HTML
                html_report = profile.to_html()
                
                # Display in Streamlit
                st.markdown("### 📋 Comprehensive Data Profile Report")
                st.components.v1.html(html_report, height=800, scrolling=True)
                
                # Provide download link
                st.download_button(
                    label="📥 Download Full Report (HTML)",
                    data=html_report,
                    file_name=f"profile_report_{query_name}.html",
                    mime="text/html"
                )
                
        except Exception as e:
            st.error(f"Failed to generate profile report: {str(e)}")
    
    def _generate_sweetviz_report(self, df: pd.DataFrame, query_name: str):
        """Generate SweetViz report"""
        try:
            with st.spinner("Generating SweetViz analysis..."):
                # Create SweetViz report
                report = sv.analyze(df)
                
                # Save report
                report_path = f"sweetviz_report_{query_name}.html"
                report.show_html(report_path, open_browser=False)
                
                # Read and display
                with open(report_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                st.markdown("### 🍭 SweetViz Analysis Report")
                st.components.v1.html(html_content, height=800, scrolling=True)
                
                # Provide download
                st.download_button(
                    label="📥 Download SweetViz Report (HTML)",
                    data=html_content,
                    file_name=f"sweetviz_report_{query_name}.html",
                    mime="text/html"
                )
                
        except Exception as e:
            st.error(f"Failed to generate SweetViz report: {str(e)}")
    
    def render_data_quality_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Render data quality summary with recommendations"""
        
        st.markdown("### 📊 Data Quality Assessment")
        
        # Generate quality report
        quality_report = self.generate_data_quality_report(df)
        
        # Display overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        overview = quality_report['overview']
        with col1:
            st.metric("Total Rows", f"{overview['total_rows']:,}")
        
        with col2:
            st.metric("Total Columns", overview['total_columns'])
        
        with col3:
            completeness = quality_report['completeness']['completeness_score']
            st.metric("Data Completeness", f"{completeness}%")
        
        with col4:
            missing_vals = overview['total_missing_values']
            st.metric("Missing Values", f"{missing_vals:,}")
        
        # Quality dashboard
        fig = self.create_data_quality_dashboard(df, quality_report)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        recommendations = quality_report['recommendations']
        if recommendations:
            st.markdown("### 💡 Data Quality Recommendations")
            
            for rec in recommendations:
                priority_color = {
                    'high': '🔴',
                    'medium': '🟡',
                    'low': '🟢'
                }.get(rec['priority'], '⚪')
                
                with st.expander(f"{priority_color} {rec['category'].title()}: {rec['issue']}"):
                    st.write(rec['recommendation'])
        
        return quality_report

# Initialize profiling tools availability status
def check_profiling_tools():
    """Check which profiling tools are available"""
    tools_status = {
        'dtale': DTALE_AVAILABLE,
        'pygwalker': PYGWALKER_AVAILABLE,
        'ydata_profiling': YDATA_PROFILING_AVAILABLE,
        'sweetviz': True  # Always available as it's in requirements
    }
    
    return tools_status
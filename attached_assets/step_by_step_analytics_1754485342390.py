import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json

class StepByStepAnalytics:
    def __init__(self):
        self.steps = []
        self.current_step = 0
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
    
    def create_welcome_interface(self):
        """Create the welcome interface with superpowers"""
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 3rem 2rem;
            border-radius: 15px;
            text-align: center;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        ">
            <h1 style="font-size: 2.5rem; margin-bottom: 1rem;">🚀 Unlock Full Data Superpowers! 💡</h1>
            <p style="font-size: 1.2rem; margin-bottom: 0;">
                Welcome to your <strong>AI-Driven Intelligence Hub</strong> – built for 
                <strong>Banking & Finance Professionals</strong> who demand clarity, trust, and insight from data.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature grid
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📊 **Analyze Your Data Intelligently**
            → Instantly visualize DataFrames with intelligent analysis  
            → Get automatic summaries, comparisons, and anomaly detection – no coding needed  
            → Advanced statistical profiling and data quality assessment
            
            ### 🛡️ **Ensure Data Quality & Trust**
            → Run real-time data validation and quality checks  
            → Validate schemas, flag missing or inconsistent data  
            → Ensure compliance with industry standards
            """)
        
        with col2:
            st.markdown("""
            ### 🔄 **Track Lineage & Audit Everything**
            → Trace where data came from and how it transformed  
            → Stay compliant, reduce risk, and gain 100% traceability  
            → Comprehensive audit trails for regulatory compliance
            
            ### 🤖 **Ask Anything with AI Insight**
            → Use natural language prompts to generate reports  
            → Detect fraud, forecast trends, or segment customers with AI  
            → Integrated intelligent analytics powered by advanced ML
            """)
    
    def analyze_data_step_by_step(self, data: pd.DataFrame, prompt: str) -> Dict[str, Any]:
        """Perform comprehensive step-by-step data analysis"""
        analysis_results = {
            'steps': [],
            'insights': [],
            'recommendations': [],
            'data_quality': {},
            'visualizations': []
        }
        
        # Step 1: Data Overview
        overview = self._step_1_data_overview(data)
        analysis_results['steps'].append(overview)
        
        # Step 2: Data Quality Assessment
        quality = self._step_2_data_quality(data)
        analysis_results['steps'].append(quality)
        analysis_results['data_quality'] = quality['details']
        
        # Step 3: Statistical Analysis
        stats = self._step_3_statistical_analysis(data)
        analysis_results['steps'].append(stats)
        
        # Step 4: Pattern Detection
        patterns = self._step_4_pattern_detection(data, prompt)
        analysis_results['steps'].append(patterns)
        
        # Step 5: Business Intelligence
        business_intel = self._step_5_business_intelligence(data, prompt)
        analysis_results['steps'].append(business_intel)
        analysis_results['insights'] = business_intel['insights']
        
        # Step 6: Recommendations
        recommendations = self._step_6_recommendations(data, prompt)
        analysis_results['steps'].append(recommendations)
        analysis_results['recommendations'] = recommendations['recommendations']
        
        return analysis_results
    
    def _step_1_data_overview(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Step 1: Comprehensive data overview"""
        overview = {
            'step_number': 1,
            'title': '📋 Data Overview & Structure',
            'status': 'completed',
            'details': {
                'total_rows': len(data),
                'total_columns': len(data.columns),
                'data_types': data.dtypes.value_counts().to_dict(),
                'memory_usage': data.memory_usage(deep=True).sum(),
                'column_info': []
            },
            'visualizations': []
        }
        
        # Analyze each column
        for col in data.columns:
            col_info = {
                'name': col,
                'type': str(data[col].dtype),
                'non_null_count': data[col].count(),
                'null_count': data[col].isnull().sum(),
                'null_percentage': (data[col].isnull().sum() / len(data)) * 100,
                'unique_values': data[col].nunique() if data[col].dtype != 'object' or data[col].nunique() < 50 else '50+'
            }
            
            if data[col].dtype in ['int64', 'float64']:
                col_info.update({
                    'mean': data[col].mean(),
                    'median': data[col].median(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max()
                })
            
            overview['details']['column_info'].append(col_info)
        
        return overview
    
    def _step_2_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Step 2: Data quality assessment"""
        quality = {
            'step_number': 2,
            'title': '🔍 Data Quality Assessment',
            'status': 'completed',
            'details': {
                'overall_score': 0,
                'issues': [],
                'strengths': [],
                'completeness': {},
                'consistency': {},
                'validity': {}
            }
        }
        
        total_cells = len(data) * len(data.columns)
        null_cells = data.isnull().sum().sum()
        completeness_score = ((total_cells - null_cells) / total_cells) * 100
        
        quality['details']['completeness'] = {
            'score': completeness_score,
            'missing_values': null_cells,
            'complete_rows': len(data.dropna()),
            'completion_rate': completeness_score
        }
        
        # Check for duplicates
        duplicate_rows = data.duplicated().sum()
        consistency_score = ((len(data) - duplicate_rows) / len(data)) * 100
        
        quality['details']['consistency'] = {
            'score': consistency_score,
            'duplicate_rows': duplicate_rows,
            'unique_rows': len(data) - duplicate_rows
        }
        
        # Data type validity
        validity_issues = []
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if data[col].isin([np.inf, -np.inf]).any():
                validity_issues.append(f"Infinite values in {col}")
            if (data[col] < 0).any() and 'amount' in col.lower():
                validity_issues.append(f"Negative values in amount field: {col}")
        
        validity_score = max(0, 100 - len(validity_issues) * 10)
        quality['details']['validity'] = {
            'score': validity_score,
            'issues': validity_issues
        }
        
        # Overall score
        quality['details']['overall_score'] = (completeness_score + consistency_score + validity_score) / 3
        
        # Generate insights
        if completeness_score > 95:
            quality['details']['strengths'].append("Excellent data completeness")
        elif completeness_score > 85:
            quality['details']['strengths'].append("Good data completeness")
        else:
            quality['details']['issues'].append("Data has significant missing values")
        
        if duplicate_rows > 0:
            quality['details']['issues'].append(f"Found {duplicate_rows} duplicate rows")
        
        return quality
    
    def _step_3_statistical_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Step 3: Statistical analysis"""
        stats = {
            'step_number': 3,
            'title': '📊 Statistical Analysis',
            'status': 'completed',
            'details': {
                'summary_statistics': {},
                'correlations': {},
                'distributions': {},
                'outliers': {}
            }
        }
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Summary statistics
            desc_stats = data[numeric_cols].describe()
            stats['details']['summary_statistics'] = desc_stats.to_dict()
            
            # Correlations
            if len(numeric_cols) > 1:
                corr_matrix = data[numeric_cols].corr()
                stats['details']['correlations'] = corr_matrix.to_dict()
                
                # Find strong correlations
                strong_corrs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            strong_corrs.append({
                                'var1': corr_matrix.columns[i],
                                'var2': corr_matrix.columns[j],
                                'correlation': corr_val
                            })
                stats['details']['strong_correlations'] = strong_corrs
            
            # Outlier detection using IQR method
            outliers = {}
            for col in numeric_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_count = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                outliers[col] = {
                    'count': outlier_count,
                    'percentage': (outlier_count / len(data)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
            
            stats['details']['outliers'] = outliers
        
        return stats
    
    def _step_4_pattern_detection(self, data: pd.DataFrame, prompt: str) -> Dict[str, Any]:
        """Step 4: Pattern detection"""
        patterns = {
            'step_number': 4,
            'title': '🔍 Pattern Detection & Trends',
            'status': 'completed',
            'details': {
                'temporal_patterns': [],
                'seasonal_patterns': [],
                'growth_trends': [],
                'anomalies': []
            }
        }
        
        # Look for date columns
        date_cols = []
        for col in data.columns:
            try:
                pd.to_datetime(data[col].head())
                date_cols.append(col)
            except:
                pass
        
        if date_cols:
            date_col = date_cols[0]
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                value_col = numeric_cols[0]
                
                # Temporal analysis
                temp_data = data.copy()
                temp_data[date_col] = pd.to_datetime(temp_data[date_col])
                temp_data = temp_data.sort_values(date_col)
                
                # Monthly aggregation
                temp_data['month'] = temp_data[date_col].dt.to_period('M')
                # Only sum numeric columns, skip datetime columns
                if pd.api.types.is_numeric_dtype(temp_data[value_col]):
                    monthly_data = temp_data.groupby('month')[value_col].sum()
                else:
                    monthly_data = temp_data.groupby('month')[value_col].count()
                
                # Growth trend
                if len(monthly_data) > 1:
                    growth_rates = monthly_data.pct_change().dropna()
                    avg_growth = growth_rates.mean() * 100
                    
                    patterns['details']['growth_trends'].append({
                        'metric': value_col,
                        'average_monthly_growth': avg_growth,
                        'trend_direction': 'increasing' if avg_growth > 0 else 'decreasing',
                        'volatility': growth_rates.std() * 100
                    })
                
                # Seasonality detection (basic)
                if len(monthly_data) >= 12:
                    temp_data['month_num'] = temp_data[date_col].dt.month
                    seasonal_pattern = temp_data.groupby('month_num')[value_col].mean()
                    seasonal_strength = seasonal_pattern.std() / seasonal_pattern.mean()
                    
                    if seasonal_strength > 0.1:
                        patterns['details']['seasonal_patterns'].append({
                            'metric': value_col,
                            'strength': seasonal_strength,
                            'peak_month': seasonal_pattern.idxmax(),
                            'low_month': seasonal_pattern.idxmin()
                        })
        
        return patterns
    
    def _step_5_business_intelligence(self, data: pd.DataFrame, prompt: str) -> Dict[str, Any]:
        """Step 5: Business intelligence insights"""
        business_intel = {
            'step_number': 5,
            'title': '💼 Business Intelligence & Insights',
            'status': 'completed',
            'insights': [],
            'kpis': {},
            'segments': {}
        }
        
        # Generate business insights based on data
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Revenue insights
        revenue_cols = [col for col in numeric_cols if any(word in col.lower() for word in ['revenue', 'total', 'amount', 'price'])]
        if revenue_cols:
            revenue_col = revenue_cols[0]
            total_revenue = data[revenue_col].sum()
            avg_transaction = data[revenue_col].mean()
            
            business_intel['insights'].append(f"Total {revenue_col.replace('_', ' ')}: ${total_revenue:,.0f}")
            business_intel['insights'].append(f"Average transaction value: ${avg_transaction:,.0f}")
            
            # Performance distribution
            high_value_threshold = data[revenue_col].quantile(0.8)
            high_value_transactions = (data[revenue_col] >= high_value_threshold).sum()
            business_intel['insights'].append(f"High-value transactions (top 20%): {high_value_transactions} transactions")
        
        # Customer insights
        if 'customer_name' in data.columns or 'customer_id' in data.columns:
            customer_col = 'customer_name' if 'customer_name' in data.columns else 'customer_id'
            unique_customers = data[customer_col].nunique()
            transactions_per_customer = len(data) / unique_customers
            
            business_intel['insights'].append(f"Total unique customers: {unique_customers:,}")
            business_intel['insights'].append(f"Average transactions per customer: {transactions_per_customer:.1f}")
        
        # Geographic insights
        if 'customer_country' in data.columns:
            top_country = data.groupby('customer_country').size().idxmax()
            country_revenue = data.groupby('customer_country')[revenue_cols[0]].sum().sort_values(ascending=False)
            business_intel['insights'].append(f"Top market by volume: {top_country}")
            business_intel['insights'].append(f"Top market by revenue: {country_revenue.index[0]}")
        
        # Product insights
        if 'product_name' in data.columns:
            top_product = data.groupby('product_name').size().idxmax()
            business_intel['insights'].append(f"Most popular product: {top_product}")
        
        return business_intel
    
    def _step_6_recommendations(self, data: pd.DataFrame, prompt: str) -> Dict[str, Any]:
        """Step 6: Strategic recommendations"""
        recommendations = {
            'step_number': 6,
            'title': '🎯 Strategic Recommendations',
            'status': 'completed',
            'recommendations': [],
            'action_items': [],
            'risk_alerts': []
        }
        
        # Data quality recommendations
        null_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        if null_percentage > 10:
            recommendations['recommendations'].append({
                'category': 'Data Quality',
                'recommendation': 'Implement data validation processes to reduce missing values',
                'priority': 'High',
                'impact': 'Improve analysis accuracy and decision-making reliability'
            })
        
        # Performance recommendations
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Look for declining trends
            date_cols = []
            for col in data.columns:
                try:
                    pd.to_datetime(data[col].head())
                    date_cols.append(col)
                except:
                    pass
            
            if date_cols:
                # Time series analysis for recommendations
                recommendations['recommendations'].append({
                    'category': 'Performance Monitoring',
                    'recommendation': 'Implement real-time dashboards for trend monitoring',
                    'priority': 'Medium',
                    'impact': 'Enable proactive decision-making and early issue detection'
                })
        
        # Business intelligence recommendations
        if 'customer_country' in data.columns:
            recommendations['recommendations'].append({
                'category': 'Market Expansion',
                'recommendation': 'Analyze underperforming geographic markets for growth opportunities',
                'priority': 'Medium',
                'impact': 'Potential revenue growth through market development'
            })
        
        # Automation recommendations
        recommendations['recommendations'].append({
            'category': 'Process Automation',
            'recommendation': 'Automate recurring analytics workflows to reduce manual effort',
            'priority': 'Low',
            'impact': 'Increase efficiency and reduce human error in reporting'
        })
        
        return recommendations
    
    def create_step_visualization(self, step_data: Dict[str, Any]) -> Optional[go.Figure]:
        """Create visualization for a specific step"""
        step_num = step_data['step_number']
        
        if step_num == 2:  # Data Quality
            return self._create_quality_gauge(step_data)
        elif step_num == 3:  # Statistical Analysis
            return self._create_correlation_heatmap(step_data)
        elif step_num == 4:  # Pattern Detection
            return self._create_trend_chart(step_data)
        
        return None
    
    def _create_quality_gauge(self, step_data: Dict[str, Any]) -> go.Figure:
        """Create data quality gauge chart"""
        score = step_data['details'].get('overall_score', 0)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Data Quality Score"},
            delta = {'reference': 80},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig
    
    def _create_correlation_heatmap(self, step_data: Dict[str, Any]) -> Optional[go.Figure]:
        """Create correlation heatmap"""
        correlations = step_data['details'].get('correlations', {})
        
        if not correlations:
            return None
        
        # Convert to matrix format
        variables = list(correlations.keys())
        matrix = []
        for var1 in variables:
            row = []
            for var2 in variables:
                row.append(correlations[var1].get(var2, 0))
            matrix.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=variables,
            y=variables,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title="Correlation Matrix",
            height=400
        )
        
        return fig
    
    def _create_trend_chart(self, step_data: Dict[str, Any]) -> Optional[go.Figure]:
        """Create trend visualization"""
        growth_trends = step_data['details'].get('growth_trends', [])
        
        if not growth_trends:
            return None
        
        metrics = [trend['metric'] for trend in growth_trends]
        growth_rates = [trend['average_monthly_growth'] for trend in growth_trends]
        
        fig = go.Figure(data=[go.Bar(
            x=metrics,
            y=growth_rates,
            marker_color=['green' if rate > 0 else 'red' for rate in growth_rates]
        )])
        
        fig.update_layout(
            title="Growth Trends by Metric",
            xaxis_title="Metrics",
            yaxis_title="Average Monthly Growth (%)",
            height=300
        )
        
        return fig
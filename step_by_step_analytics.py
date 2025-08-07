"""
Step-by-Step Analytics Engine for Enterprise Analytics Hub
Provides comprehensive 6-step data analysis workflow
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

class StepByStepAnalytics:
    def __init__(self):
        """Initialize step-by-step analytics engine"""
        self.steps = []
        self.current_step = 0
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        logger.info("Step-by-step analytics engine initialized")
    
    def create_welcome_interface(self):
        """Create the welcome interface with enterprise features showcase"""
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
            <h1 style="font-size: 2.5rem; margin-bottom: 1rem;">🚀 Enterprise Analytics Hub</h1>
            <p style="font-size: 1.2rem; margin-bottom: 0;">
                Welcome to your <strong>AI-Driven Intelligence Platform</strong> – built for 
                <strong>Banking & Finance Professionals</strong> who demand precision, compliance, and actionable insights.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature showcase grid
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📊 **Intelligent Data Analysis**
            → Automated 6-step comprehensive analysis workflow  
            → Statistical profiling with advanced metrics and outlier detection  
            → Real-time data quality assessment and validation  
            → Pattern recognition with seasonal and trend analysis
            
            ### 🔮 **Advanced Forecasting & Predictions**
            → Prophet-powered time series forecasting with confidence intervals  
            → Anomaly detection and trend analysis  
            → Scenario modeling and what-if analysis  
            → Executive-ready forecast summaries
            """)
        
        with col2:
            st.markdown("""
            ### 🛡️ **Enterprise Security & Compliance**
            → Bank-grade security headers and data protection  
            → Audit trail and data lineage tracking  
            → Multi-database connectivity with secure credential management  
            → Production-ready Docker containerization
            
            ### 🤖 **AI-Powered Natural Language Interface**
            → Convert plain English to optimized SQL queries  
            → Intelligent chart selection based on data patterns  
            → Executive dashboard with KPI monitoring  
            → Strategic recommendations powered by advanced ML
            """)
    
    def analyze_data_step_by_step(self, data: pd.DataFrame, prompt: str) -> Dict[str, Any]:
        """Perform comprehensive 6-step data analysis workflow"""
        logger.info(f"Starting comprehensive analysis for {len(data)} rows of data")
        
        analysis_results = {
            'steps': [],
            'insights': [],
            'recommendations': [],
            'data_quality': {},
            'visualizations': [],
            'executive_summary': {}
        }
        
        try:
            # Step 1: Data Overview & Structure Assessment
            overview = self._step_1_data_overview(data)
            analysis_results['steps'].append(overview)
            
            # Step 2: Data Quality Assessment
            quality = self._step_2_data_quality(data)
            analysis_results['steps'].append(quality)
            analysis_results['data_quality'] = quality['details']
            
            # Step 3: Statistical Analysis
            stats = self._step_3_statistical_analysis(data)
            analysis_results['steps'].append(stats)
            
            # Step 4: Pattern Detection & Trend Analysis
            patterns = self._step_4_pattern_detection(data, prompt)
            analysis_results['steps'].append(patterns)
            
            # Step 5: Business Intelligence & Insights
            business_intel = self._step_5_business_intelligence(data, prompt)
            analysis_results['steps'].append(business_intel)
            analysis_results['insights'] = business_intel['insights']
            
            # Step 6: Strategic Recommendations
            recommendations = self._step_6_recommendations(data, prompt, analysis_results)
            analysis_results['steps'].append(recommendations)
            analysis_results['recommendations'] = recommendations['recommendations']
            
            # Generate executive summary
            analysis_results['executive_summary'] = self._generate_executive_summary(analysis_results)
            
            logger.info("Comprehensive analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in step-by-step analysis: {str(e)}")
            raise
    
    def _step_1_data_overview(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Step 1: Comprehensive data overview and structure assessment"""
        logger.info("Executing Step 1: Data Overview & Structure Assessment")
        
        overview = {
            'step_number': 1,
            'title': '📋 Data Overview & Structure Assessment',
            'status': 'completed',
            'details': {
                'total_rows': len(data),
                'total_columns': len(data.columns),
                'data_types': data.dtypes.value_counts().to_dict(),
                'memory_usage_mb': round(data.memory_usage(deep=True).sum() / 1024**2, 2),
                'column_info': [],
                'data_shape': f"{len(data)} rows × {len(data.columns)} columns",
                'duplicate_rows': data.duplicated().sum(),
                'empty_cells': data.isnull().sum().sum()
            },
            'visualizations': []
        }
        
        # Analyze each column in detail
        for col in data.columns:
            col_info = {
                'name': col,
                'type': str(data[col].dtype),
                'non_null_count': data[col].count(),
                'null_count': data[col].isnull().sum(),
                'null_percentage': round((data[col].isnull().sum() / len(data)) * 100, 2),
                'unique_values': data[col].nunique(),
                'sample_values': data[col].dropna().head(3).tolist() if not data[col].empty else []
            }
            
            # Add statistics for numeric columns
            if data[col].dtype in ['int64', 'float64']:
                try:
                    col_info.update({
                        'mean': round(data[col].mean(), 2),
                        'median': round(data[col].median(), 2),
                        'std': round(data[col].std(), 2),
                        'min': round(data[col].min(), 2),
                        'max': round(data[col].max(), 2),
                        'range': round(data[col].max() - data[col].min(), 2)
                    })
                except:
                    pass
            
            # Add analysis for categorical columns
            elif data[col].dtype == 'object':
                try:
                    mode_value = data[col].mode().iloc[0] if not data[col].mode().empty else None
                    col_info.update({
                        'most_frequent': mode_value,
                        'frequency_of_mode': data[col].value_counts().iloc[0] if not data[col].empty else 0,
                        'avg_string_length': round(data[col].astype(str).str.len().mean(), 1) if not data[col].empty else 0
                    })
                except:
                    pass
            
            overview['details']['column_info'].append(col_info)
        
        return overview
    
    def _step_2_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Step 2: Comprehensive data quality assessment"""
        logger.info("Executing Step 2: Data Quality Assessment")
        
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
                'validity': {},
                'accuracy': {}
            }
        }
        
        # Completeness Assessment
        total_cells = len(data) * len(data.columns)
        null_cells = data.isnull().sum().sum()
        completeness_score = ((total_cells - null_cells) / total_cells) * 100 if total_cells > 0 else 0
        
        quality['details']['completeness'] = {
            'score': round(completeness_score, 2),
            'missing_values': null_cells,
            'complete_rows': len(data.dropna()),
            'completion_rate': round(completeness_score, 2),
            'columns_with_missing': (data.isnull().sum() > 0).sum()
        }
        
        # Consistency Assessment
        duplicate_rows = data.duplicated().sum()
        consistency_score = ((len(data) - duplicate_rows) / len(data)) * 100 if len(data) > 0 else 0
        
        quality['details']['consistency'] = {
            'score': round(consistency_score, 2),
            'duplicate_rows': duplicate_rows,
            'unique_rows': len(data) - duplicate_rows,
            'duplicate_percentage': round((duplicate_rows / len(data)) * 100, 2) if len(data) > 0 else 0
        }
        
        # Validity Assessment
        validity_issues = []
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Check for infinite values
            if data[col].isin([np.inf, -np.inf]).any():
                validity_issues.append(f"Infinite values detected in {col}")
            
            # Check for negative values in amount/price fields
            if any(keyword in col.lower() for keyword in ['amount', 'price', 'cost', 'revenue', 'sales']) and (data[col] < 0).any():
                validity_issues.append(f"Negative values in financial field: {col}")
            
            # Check for outliers using IQR method
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
                if outliers > len(data) * 0.1:  # More than 10% outliers
                    validity_issues.append(f"High number of outliers in {col}: {outliers} ({round(outliers/len(data)*100, 1)}%)")
        
        validity_score = max(0, 100 - len(validity_issues) * 10)
        quality['details']['validity'] = {
            'score': round(validity_score, 2),
            'issues': validity_issues,
            'numeric_columns_analyzed': len(numeric_cols)
        }
        
        # Accuracy Assessment (basic checks)
        accuracy_issues = []
        for col in data.columns:
            if data[col].dtype == 'object':
                # Check for inconsistent formatting
                unique_vals = data[col].dropna().unique()
                if len(unique_vals) > 1:
                    # Check for potential case inconsistencies
                    lower_vals = [str(v).lower() for v in unique_vals]
                    if len(set(lower_vals)) < len(unique_vals):
                        accuracy_issues.append(f"Potential case inconsistencies in {col}")
        
        accuracy_score = max(0, 100 - len(accuracy_issues) * 15)
        quality['details']['accuracy'] = {
            'score': round(accuracy_score, 2),
            'issues': accuracy_issues
        }
        
        # Calculate overall score
        quality['details']['overall_score'] = round(
            (completeness_score + consistency_score + validity_score + accuracy_score) / 4, 2
        )
        
        # Generate insights based on scores
        if completeness_score > 95:
            quality['details']['strengths'].append("Excellent data completeness (>95%)")
        elif completeness_score > 85:
            quality['details']['strengths'].append("Good data completeness (>85%)")
        else:
            quality['details']['issues'].append(f"Low data completeness ({completeness_score:.1f}%)")
        
        if consistency_score > 98:
            quality['details']['strengths'].append("No duplicate records detected")
        elif duplicate_rows > 0:
            quality['details']['issues'].append(f"Found {duplicate_rows} duplicate rows")
        
        if validity_score > 90:
            quality['details']['strengths'].append("High data validity")
        
        if accuracy_score > 90:
            quality['details']['strengths'].append("Good data formatting consistency")
        
        return quality
    
    def _step_3_statistical_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Step 3: Advanced statistical analysis"""
        logger.info("Executing Step 3: Statistical Analysis")
        
        stats = {
            'step_number': 3,
            'title': '📊 Statistical Analysis',
            'status': 'completed',
            'details': {
                'summary_statistics': {},
                'correlations': {},
                'distributions': {},
                'outliers': {},
                'skewness_analysis': {},
                'variance_analysis': {}
            }
        }
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Summary statistics
            desc_stats = data[numeric_cols].describe()
            stats['details']['summary_statistics'] = desc_stats.round(2).to_dict()
            
            # Correlation analysis
            if len(numeric_cols) > 1:
                corr_matrix = data[numeric_cols].corr()
                stats['details']['correlations'] = corr_matrix.round(3).to_dict()
                
                # Find strong correlations
                strong_corrs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            strong_corrs.append({
                                'var1': corr_matrix.columns[i],
                                'var2': corr_matrix.columns[j],
                                'correlation': round(corr_val, 3),
                                'strength': 'Very Strong' if abs(corr_val) > 0.9 else 'Strong'
                            })
                stats['details']['strong_correlations'] = strong_corrs
            
            # Outlier detection using IQR method
            outliers = {}
            for col in numeric_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
                    outlier_count = outlier_mask.sum()
                    
                    outliers[col] = {
                        'count': outlier_count,
                        'percentage': round((outlier_count / len(data)) * 100, 2),
                        'lower_bound': round(lower_bound, 2),
                        'upper_bound': round(upper_bound, 2),
                        'severity': 'High' if outlier_count > len(data) * 0.1 else 'Medium' if outlier_count > len(data) * 0.05 else 'Low'
                    }
            
            stats['details']['outliers'] = outliers
            
            # Skewness analysis
            skewness = {}
            for col in numeric_cols:
                skew_val = data[col].skew()
                skewness[col] = {
                    'skewness': round(skew_val, 3),
                    'interpretation': self._interpret_skewness(skew_val)
                }
            stats['details']['skewness_analysis'] = skewness
            
            # Variance analysis
            variance_info = {}
            for col in numeric_cols:
                var_val = data[col].var()
                std_val = data[col].std()
                mean_val = data[col].mean()
                cv = (std_val / mean_val) * 100 if mean_val != 0 else 0
                
                variance_info[col] = {
                    'variance': round(var_val, 2),
                    'standard_deviation': round(std_val, 2),
                    'coefficient_of_variation': round(cv, 2),
                    'variability': 'High' if cv > 30 else 'Medium' if cv > 15 else 'Low'
                }
            stats['details']['variance_analysis'] = variance_info
        
        return stats
    
    def _step_4_pattern_detection(self, data: pd.DataFrame, prompt: str) -> Dict[str, Any]:
        """Step 4: Advanced pattern detection and trend analysis"""
        logger.info("Executing Step 4: Pattern Detection & Trend Analysis")
        
        patterns = {
            'step_number': 4,
            'title': '🔍 Pattern Detection & Trend Analysis',
            'status': 'completed',
            'details': {
                'temporal_patterns': [],
                'seasonal_patterns': [],
                'growth_trends': [],
                'anomalies': [],
                'cyclical_behavior': [],
                'data_trends': {}
            }
        }
        
        # Look for date columns
        date_cols = self._identify_date_columns(data)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if date_cols and len(numeric_cols) > 0:
            date_col = date_cols[0]
            value_col = numeric_cols[0]
            
            try:
                # Temporal analysis
                temp_data = data.copy()
                temp_data[date_col] = pd.to_datetime(temp_data[date_col])
                temp_data = temp_data.sort_values(date_col)
                
                # Monthly aggregation for trend analysis
                temp_data['month_year'] = temp_data[date_col].dt.to_period('M')
                
                if pd.api.types.is_numeric_dtype(temp_data[value_col]):
                    monthly_data = temp_data.groupby('month_year')[value_col].agg(['sum', 'mean', 'count']).reset_index()
                    monthly_data['month_year'] = monthly_data['month_year'].astype(str)
                    
                    # Growth trend analysis
                    if len(monthly_data) > 1:
                        monthly_data['growth_rate'] = monthly_data['sum'].pct_change() * 100
                        avg_growth = monthly_data['growth_rate'].mean()
                        
                        patterns['details']['growth_trends'] = {
                            'average_monthly_growth': round(avg_growth, 2),
                            'trend_direction': 'Growing' if avg_growth > 0 else 'Declining' if avg_growth < 0 else 'Stable',
                            'volatility': round(monthly_data['growth_rate'].std(), 2),
                            'periods_analyzed': len(monthly_data)
                        }
                        
                        # Seasonal pattern detection
                        if len(monthly_data) >= 12:
                            seasonal_analysis = self._detect_seasonality(temp_data, date_col, value_col)
                            patterns['details']['seasonal_patterns'] = seasonal_analysis
                
                # Anomaly detection in time series
                anomalies = self._detect_temporal_anomalies(temp_data, date_col, value_col)
                patterns['details']['anomalies'] = anomalies
                
            except Exception as e:
                logger.warning(f"Error in temporal analysis: {str(e)}")
        
        # Pattern detection in categorical data
        categorical_patterns = self._analyze_categorical_patterns(data)
        patterns['details']['categorical_patterns'] = categorical_patterns
        
        # Numerical patterns
        numerical_patterns = self._analyze_numerical_patterns(data)
        patterns['details']['numerical_patterns'] = numerical_patterns
        
        return patterns
    
    def _step_5_business_intelligence(self, data: pd.DataFrame, prompt: str) -> Dict[str, Any]:
        """Step 5: Business intelligence and actionable insights"""
        logger.info("Executing Step 5: Business Intelligence & Insights")
        
        business_intel = {
            'step_number': 5,
            'title': '💡 Business Intelligence & Insights',
            'status': 'completed',
            'insights': [],
            'key_findings': [],
            'performance_metrics': {},
            'risk_indicators': [],
            'opportunities': []
        }
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        # Revenue/Sales Analysis (if applicable)
        revenue_cols = [col for col in numeric_cols if any(keyword in col.lower() for keyword in ['revenue', 'sales', 'amount', 'total', 'price'])]
        if revenue_cols:
            revenue_col = revenue_cols[0]
            total_revenue = data[revenue_col].sum()
            avg_revenue = data[revenue_col].mean()
            
            business_intel['performance_metrics']['total_revenue'] = round(total_revenue, 2)
            business_intel['performance_metrics']['average_transaction'] = round(avg_revenue, 2)
            
            business_intel['insights'].append(f"Total revenue/sales volume: ${total_revenue:,.2f}")
            business_intel['insights'].append(f"Average transaction value: ${avg_revenue:,.2f}")
            
            # Revenue distribution analysis
            if len(data) > 10:
                top_20_percent = data.nlargest(int(len(data) * 0.2), revenue_col)[revenue_col].sum()
                pareto_ratio = (top_20_percent / total_revenue) * 100
                
                if pareto_ratio > 80:
                    business_intel['insights'].append(f"Pareto principle applies: Top 20% of transactions generate {pareto_ratio:.1f}% of revenue")
                    business_intel['opportunities'].append("Focus on high-value transaction patterns for growth")
        
        # Customer Analysis (if applicable)
        customer_cols = [col for col in categorical_cols if any(keyword in col.lower() for keyword in ['customer', 'client', 'name'])]
        if customer_cols and revenue_cols:
            customer_col = customer_cols[0]
            revenue_col = revenue_cols[0]
            
            customer_analysis = data.groupby(customer_col)[revenue_col].agg(['sum', 'count', 'mean']).reset_index()
            top_customers = customer_analysis.nlargest(5, 'sum')
            
            business_intel['key_findings'].append({
                'category': 'Customer Analysis',
                'finding': f"Top customer generates ${top_customers.iloc[0]['sum']:,.2f} in revenue",
                'impact': 'High'
            })
            
            # Customer concentration risk
            top_5_revenue = top_customers['sum'].sum()
            concentration_risk = (top_5_revenue / total_revenue) * 100
            
            if concentration_risk > 50:
                business_intel['risk_indicators'].append(f"Customer concentration risk: Top 5 customers represent {concentration_risk:.1f}% of revenue")
        
        # Geographic Analysis (if applicable)
        geo_cols = [col for col in categorical_cols if any(keyword in col.lower() for keyword in ['country', 'region', 'location', 'city'])]
        if geo_cols and revenue_cols:
            geo_col = geo_cols[0]
            geo_analysis = data.groupby(geo_col)[revenue_cols[0]].agg(['sum', 'count']).reset_index()
            top_regions = geo_analysis.nlargest(3, 'sum')
            
            business_intel['insights'].append(f"Top performing region: {top_regions.iloc[0][geo_col]} with ${top_regions.iloc[0]['sum']:,.2f}")
            
            if len(geo_analysis) > 5:
                business_intel['opportunities'].append("Consider expansion in underperforming geographic markets")
        
        # Product Analysis (if applicable)
        product_cols = [col for col in categorical_cols if any(keyword in col.lower() for keyword in ['product', 'item', 'service'])]
        if product_cols and revenue_cols:
            product_col = product_cols[0]
            product_analysis = data.groupby(product_col)[revenue_cols[0]].agg(['sum', 'count', 'mean']).reset_index()
            product_analysis['avg_order_value'] = product_analysis['sum'] / product_analysis['count']
            
            top_products = product_analysis.nlargest(3, 'sum')
            business_intel['insights'].append(f"Best performing product: {top_products.iloc[0][product_col]}")
            
            # Product diversification analysis
            product_count = len(product_analysis)
            if product_count > 10:
                business_intel['opportunities'].append(f"Portfolio includes {product_count} products - consider focusing on top performers")
        
        # Data quality business impact
        if hasattr(self, '_last_quality_score'):
            quality_score = getattr(self, '_last_quality_score', 0)
            if quality_score < 80:
                business_intel['risk_indicators'].append(f"Data quality score of {quality_score:.1f}% may impact decision accuracy")
        
        # Generate executive insights
        if len(business_intel['insights']) == 0:
            business_intel['insights'].append("Comprehensive dataset suitable for further analysis")
            business_intel['opportunities'].append("Consider time-series analysis for trend identification")
        
        return business_intel
    
    def _step_6_recommendations(self, data: pd.DataFrame, prompt: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Step 6: Strategic recommendations based on analysis"""
        logger.info("Executing Step 6: Strategic Recommendations")
        
        recommendations = {
            'step_number': 6,
            'title': '🎯 Strategic Recommendations',
            'status': 'completed',
            'recommendations': [],
            'action_items': [],
            'priority_areas': [],
            'implementation_roadmap': []
        }
        
        # Data quality recommendations
        data_quality = analysis_results.get('data_quality', {})
        overall_score = data_quality.get('overall_score', 100)
        
        if overall_score < 80:
            recommendations['recommendations'].append("Implement data quality improvement initiatives")
            recommendations['action_items'].append("Establish data validation rules and monitoring processes")
            recommendations['priority_areas'].append("Data Quality Management")
        
        # Missing data recommendations
        completeness = data_quality.get('completeness', {})
        if completeness.get('score', 100) < 90:
            recommendations['recommendations'].append("Address missing data through improved collection processes")
            recommendations['action_items'].append("Investigate root causes of missing data and implement fixes")
        
        # Duplicate data recommendations
        consistency = data_quality.get('consistency', {})
        if consistency.get('duplicate_rows', 0) > 0:
            recommendations['recommendations'].append("Implement deduplication processes to ensure data consistency")
            recommendations['action_items'].append("Create automated duplicate detection and removal workflows")
        
        # Statistical insights recommendations
        if len(analysis_results['steps']) >= 3:
            stats_step = analysis_results['steps'][2]
            strong_corrs = stats_step['details'].get('strong_correlations', [])
            
            if strong_corrs:
                recommendations['recommendations'].append("Leverage identified correlations for predictive modeling")
                recommendations['action_items'].append("Develop correlation-based forecasting models")
                recommendations['priority_areas'].append("Predictive Analytics")
        
        # Business intelligence recommendations
        if len(analysis_results['steps']) >= 5:
            bi_step = analysis_results['steps'][4]
            risk_indicators = bi_step.get('risk_indicators', [])
            opportunities = bi_step.get('opportunities', [])
            
            for risk in risk_indicators:
                recommendations['recommendations'].append(f"Mitigate identified risk: {risk}")
                recommendations['priority_areas'].append("Risk Management")
            
            for opp in opportunities:
                recommendations['recommendations'].append(f"Pursue opportunity: {opp}")
                recommendations['priority_areas'].append("Growth Opportunities")
        
        # Time series recommendations
        date_cols = self._identify_date_columns(data)
        if date_cols:
            recommendations['recommendations'].append("Implement time series forecasting for trend prediction")
            recommendations['action_items'].append("Set up automated forecasting dashboards")
            recommendations['priority_areas'].append("Forecasting & Planning")
        
        # Technology recommendations
        if len(data) > 100000:
            recommendations['recommendations'].append("Consider implementing big data technologies for large dataset processing")
            recommendations['action_items'].append("Evaluate cloud-based analytics platforms")
            recommendations['priority_areas'].append("Technology Infrastructure")
        
        # Implementation roadmap
        recommendations['implementation_roadmap'] = [
            {"phase": "Phase 1 (0-30 days)", "focus": "Data Quality Improvements", "actions": ["Implement validation rules", "Set up monitoring"]},
            {"phase": "Phase 2 (30-60 days)", "focus": "Analytics Enhancement", "actions": ["Deploy forecasting models", "Create automated dashboards"]},
            {"phase": "Phase 3 (60-90 days)", "focus": "Strategic Implementation", "actions": ["Execute growth initiatives", "Implement risk mitigation"]}
        ]
        
        # Default recommendations if none generated
        if not recommendations['recommendations']:
            recommendations['recommendations'] = [
                "Continue monitoring data quality and patterns",
                "Establish regular analysis reporting cycles",
                "Consider expanding data collection for deeper insights"
            ]
        
        return recommendations
    
    def _generate_executive_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of the analysis"""
        summary = {
            'data_overview': {},
            'key_insights': [],
            'critical_findings': [],
            'recommended_actions': [],
            'business_impact': 'Medium'
        }
        
        # Extract key metrics from analysis
        if analysis_results['steps']:
            overview_step = analysis_results['steps'][0]
            data_details = overview_step['details']
            
            summary['data_overview'] = {
                'total_records': data_details['total_rows'],
                'data_quality_score': analysis_results.get('data_quality', {}).get('overall_score', 'N/A'),
                'analysis_depth': f"{len(analysis_results['steps'])} comprehensive steps completed"
            }
        
        # Consolidate insights
        summary['key_insights'] = analysis_results['insights'][:5]  # Top 5 insights
        summary['recommended_actions'] = analysis_results['recommendations'][:3]  # Top 3 recommendations
        
        # Determine business impact
        quality_score = analysis_results.get('data_quality', {}).get('overall_score', 100)
        if quality_score > 90 and len(analysis_results['insights']) > 3:
            summary['business_impact'] = 'High'
        elif quality_score < 70 or len(analysis_results.get('data_quality', {}).get('issues', [])) > 3:
            summary['business_impact'] = 'Critical'
        
        return summary
    
    # Helper methods
    def _identify_date_columns(self, data: pd.DataFrame) -> List[str]:
        """Identify potential date columns in the dataframe"""
        date_cols = []
        for col in data.columns:
            col_str = str(col).lower()
            if any(word in col_str for word in ['date', 'time', 'created', 'updated', 'month', 'year']):
                try:
                    pd.to_datetime(data[col].dropna().head(10))
                    date_cols.append(col)
                except:
                    pass
        return date_cols
    
    def _interpret_skewness(self, skew_val: float) -> str:
        """Interpret skewness value"""
        if abs(skew_val) < 0.5:
            return "Approximately symmetric"
        elif abs(skew_val) < 1:
            return "Moderately skewed"
        else:
            return "Highly skewed"
    
    def _detect_seasonality(self, data: pd.DataFrame, date_col: str, value_col: str) -> Dict[str, Any]:
        """Detect seasonal patterns in time series data"""
        try:
            data['month'] = data[date_col].dt.month
            monthly_avg = data.groupby('month')[value_col].mean()
            
            overall_mean = monthly_avg.mean()
            seasonal_strength = (monthly_avg.max() - monthly_avg.min()) / overall_mean
            
            return {
                'seasonal_strength': round(seasonal_strength, 3),
                'has_seasonality': seasonal_strength > 0.1,
                'peak_month': monthly_avg.idxmax(),
                'low_month': monthly_avg.idxmin(),
                'monthly_averages': monthly_avg.round(2).to_dict()
            }
        except:
            return {'has_seasonality': False, 'error': 'Insufficient data for seasonal analysis'}
    
    def _detect_temporal_anomalies(self, data: pd.DataFrame, date_col: str, value_col: str) -> List[Dict[str, Any]]:
        """Detect anomalies in time series data"""
        try:
            # Simple anomaly detection using rolling statistics
            data = data.sort_values(date_col)
            rolling_mean = data[value_col].rolling(window=7, center=True).mean()
            rolling_std = data[value_col].rolling(window=7, center=True).std()
            
            threshold = 2  # 2 standard deviations
            anomalies = []
            
            for idx, row in data.iterrows():
                if pd.notna(rolling_mean.loc[idx]) and pd.notna(rolling_std.loc[idx]):
                    if abs(row[value_col] - rolling_mean.loc[idx]) > threshold * rolling_std.loc[idx]:
                        anomalies.append({
                            'date': row[date_col].strftime('%Y-%m-%d'),
                            'value': row[value_col],
                            'expected': round(rolling_mean.loc[idx], 2),
                            'severity': 'High' if abs(row[value_col] - rolling_mean.loc[idx]) > 3 * rolling_std.loc[idx] else 'Medium'
                        })
            
            return anomalies[:10]  # Return top 10 anomalies
        except:
            return []
    
    def _analyze_categorical_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in categorical data"""
        categorical_cols = data.select_dtypes(include=['object']).columns
        patterns = {}
        
        for col in categorical_cols:
            value_counts = data[col].value_counts()
            patterns[col] = {
                'unique_values': len(value_counts),
                'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_common_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'distribution_type': 'Uniform' if value_counts.std() < value_counts.mean() * 0.1 else 'Skewed'
            }
        
        return patterns
    
    def _analyze_numerical_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in numerical data"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        patterns = {}
        
        for col in numeric_cols:
            patterns[col] = {
                'distribution_shape': self._classify_distribution(data[col]),
                'has_zeros': (data[col] == 0).sum(),
                'negative_values': (data[col] < 0).sum(),
                'range_analysis': {
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'range': data[col].max() - data[col].min()
                }
            }
        
        return patterns
    
    def _classify_distribution(self, series: pd.Series) -> str:
        """Classify the distribution shape of a numerical series"""
        try:
            skewness = series.skew()
            kurtosis = series.kurtosis()
            
            if abs(skewness) < 0.5 and abs(kurtosis) < 3:
                return "Normal-like"
            elif skewness > 1:
                return "Right-skewed"
            elif skewness < -1:
                return "Left-skewed"
            elif kurtosis > 3:
                return "Heavy-tailed"
            else:
                return "Non-normal"
        except:
            return "Unknown"

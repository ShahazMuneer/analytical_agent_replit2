import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import streamlit as st
from datetime import datetime, timedelta
import json

class AdvancedAnalytics:
    def __init__(self):
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
    
    def suggest_analysis_types(self, data: pd.DataFrame, prompt: str) -> List[Dict]:
        """Suggest appropriate analysis types based on data and prompt"""
        suggestions = []
        
        # Analyze data structure
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        date_cols = []
        categorical_cols = []
        
        for col in data.columns:
            try:
                pd.to_datetime(data[col].head())
                date_cols.append(col)
            except:
                if data[col].dtype == 'object' and data[col].nunique() < len(data) * 0.5:
                    categorical_cols.append(col)
        
        prompt_lower = prompt.lower()
        
        # Time series analysis
        if date_cols and numeric_cols:
            suggestions.append({
                'type': 'Time Series',
                'description': 'Trend analysis over time with forecasting',
                'icon': '📈',
                'suitable': any(word in prompt_lower for word in ['trend', 'time', 'monthly', 'daily', 'forecast'])
            })
        
        # Comparison analysis
        if categorical_cols and numeric_cols:
            suggestions.append({
                'type': 'Comparison',
                'description': 'Compare performance across categories',
                'icon': '📊',
                'suitable': any(word in prompt_lower for word in ['compare', 'top', 'best', 'vs', 'by'])
            })
        
        # Distribution analysis
        if numeric_cols:
            suggestions.append({
                'type': 'Distribution',
                'description': 'Statistical distribution and outlier analysis',
                'icon': '📊',
                'suitable': any(word in prompt_lower for word in ['distribution', 'outlier', 'stats', 'summary'])
            })
        
        # Correlation analysis
        if len(numeric_cols) >= 2:
            suggestions.append({
                'type': 'Correlation',
                'description': 'Relationship analysis between metrics',
                'icon': '🔗',
                'suitable': any(word in prompt_lower for word in ['correlation', 'relationship', 'impact'])
            })
        
        return suggestions
    
    def create_executive_summary(self, data: pd.DataFrame) -> Dict:
        """Create executive summary with key metrics"""
        summary = {
            'total_records': len(data),
            'date_range': None,
            'key_metrics': {},
            'insights': []
        }
        
        # Find date range
        date_cols = []
        for col in data.columns:
            try:
                pd.to_datetime(data[col].head())
                date_cols.append(col)
            except:
                pass
        
        if date_cols:
            date_col = date_cols[0]
            dates = pd.to_datetime(data[date_col])
            summary['date_range'] = {
                'start': dates.min().strftime('%Y-%m-%d'),
                'end': dates.max().strftime('%Y-%m-%d'),
                'days': (dates.max() - dates.min()).days
            }
        
        # Calculate key metrics
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if 'revenue' in col.lower() or 'total' in col.lower() or 'amount' in col.lower():
                summary['key_metrics'][col] = {
                    'total': float(data[col].sum()),
                    'average': float(data[col].mean()),
                    'max': float(data[col].max()),
                    'min': float(data[col].min())
                }
        
        # Generate insights
        if summary['date_range'] and summary['key_metrics']:
            for metric, values in summary['key_metrics'].items():
                if values['total'] > 1000000:
                    summary['insights'].append(f"Total {metric.replace('_', ' ').title()}: ${values['total']:,.0f}M")
                else:
                    summary['insights'].append(f"Total {metric.replace('_', ' ').title()}: ${values['total']:,.0f}")
        
        return summary
    
    def create_multi_chart_dashboard(self, data: pd.DataFrame, prompt: str) -> List[go.Figure]:
        """Create multiple charts for comprehensive analysis"""
        charts = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        date_cols = []
        categorical_cols = []
        
        for col in data.columns:
            try:
                pd.to_datetime(data[col].head())
                date_cols.append(col)
            except:
                if data[col].dtype == 'object' and data[col].nunique() < 20:
                    categorical_cols.append(col)
        
        # Chart 1: Time series if available
        if date_cols and numeric_cols:
            fig = self._create_time_series_chart(data, date_cols[0], numeric_cols[0])
            if fig:
                charts.append(fig)
        
        # Chart 2: Top categories
        if categorical_cols and numeric_cols:
            fig = self._create_top_categories_chart(data, categorical_cols[0], numeric_cols[0])
            if fig:
                charts.append(fig)
        
        # Chart 3: Distribution analysis
        if numeric_cols:
            fig = self._create_distribution_chart(data, numeric_cols[0])
            if fig:
                charts.append(fig)
        
        # Chart 4: Correlation heatmap if multiple numeric columns
        if len(numeric_cols) >= 2:
            fig = self._create_correlation_heatmap(data, numeric_cols[:5])
            if fig:
                charts.append(fig)
        
        return charts
    
    def _create_time_series_chart(self, data: pd.DataFrame, date_col: str, value_col: str) -> Optional[go.Figure]:
        """Create time series chart"""
        try:
            chart_data = data[[date_col, value_col]].copy()
            chart_data[date_col] = pd.to_datetime(chart_data[date_col])
            chart_data = chart_data.sort_values(date_col)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=chart_data[date_col],
                y=chart_data[value_col],
                mode='lines+markers',
                name='Trend',
                line=dict(color=self.color_palette[0], width=3),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title=f"{value_col.replace('_', ' ').title()} Over Time",
                xaxis_title="Date",
                yaxis_title=value_col.replace('_', ' ').title(),
                template="plotly_white",
                height=400
            )
            
            return fig
        except Exception:
            return None
    
    def _create_top_categories_chart(self, data: pd.DataFrame, cat_col: str, value_col: str) -> Optional[go.Figure]:
        """Create top categories bar chart"""
        try:
            agg_data = data.groupby(cat_col)[value_col].sum().sort_values(ascending=False).head(10)
            
            fig = go.Figure(data=[go.Bar(
                x=agg_data.values,
                y=agg_data.index,
                orientation='h',
                marker_color=self.color_palette[1]
            )])
            
            fig.update_layout(
                title=f"Top {cat_col.replace('_', ' ').title()} by {value_col.replace('_', ' ').title()}",
                xaxis_title=value_col.replace('_', ' ').title(),
                yaxis_title=cat_col.replace('_', ' ').title(),
                template="plotly_white",
                height=400
            )
            
            return fig
        except Exception:
            return None
    
    def _create_distribution_chart(self, data: pd.DataFrame, value_col: str) -> Optional[go.Figure]:
        """Create distribution histogram"""
        try:
            fig = go.Figure(data=[go.Histogram(
                x=data[value_col],
                nbinsx=30,
                marker_color=self.color_palette[2],
                opacity=0.7
            )])
            
            fig.update_layout(
                title=f"{value_col.replace('_', ' ').title()} Distribution",
                xaxis_title=value_col.replace('_', ' ').title(),
                yaxis_title="Frequency",
                template="plotly_white",
                height=400
            )
            
            return fig
        except Exception:
            return None
    
    def _create_correlation_heatmap(self, data: pd.DataFrame, numeric_cols: List[str]) -> Optional[go.Figure]:
        """Create correlation heatmap"""
        try:
            corr_matrix = data[numeric_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title="Correlation Matrix",
                template="plotly_white",
                height=400
            )
            
            return fig
        except Exception:
            return None
    
    def generate_business_insights(self, data: pd.DataFrame, prompt: str) -> List[str]:
        """Generate business insights from data"""
        insights = []
        
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            # Revenue insights
            revenue_cols = [col for col in numeric_cols if 'revenue' in col.lower() or 'total' in col.lower()]
            if revenue_cols:
                revenue_col = revenue_cols[0]
                total_revenue = data[revenue_col].sum()
                avg_revenue = data[revenue_col].mean()
                
                insights.append(f"Total revenue: ${total_revenue:,.0f}")
                insights.append(f"Average transaction: ${avg_revenue:,.0f}")
                
                # Growth trends
                if len(data) > 1:
                    recent_avg = data[revenue_col].tail(len(data)//4).mean()
                    earlier_avg = data[revenue_col].head(len(data)//4).mean()
                    if recent_avg > earlier_avg:
                        growth = ((recent_avg - earlier_avg) / earlier_avg) * 100
                        insights.append(f"Recent performance up {growth:.1f}% vs earlier periods")
            
            # Product insights
            if 'product_name' in data.columns and revenue_cols:
                top_product = data.groupby('product_name')[revenue_cols[0]].sum().idxmax()
                insights.append(f"Top performing product: {top_product}")
            
            # Customer insights
            if 'customer_country' in data.columns and revenue_cols:
                top_country = data.groupby('customer_country')[revenue_cols[0]].sum().idxmax()
                insights.append(f"Highest revenue market: {top_country}")
        
        except Exception:
            insights.append("Analysis complete - detailed metrics available in charts")
        
        return insights[:5]  # Limit to top 5 insights
    
    def create_kpi_cards(self, data: pd.DataFrame) -> Dict:
        """Create KPI cards for dashboard"""
        kpis = {}
        
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            # Revenue KPIs
            revenue_cols = [col for col in numeric_cols if any(word in col.lower() for word in ['revenue', 'total', 'amount', 'price'])]
            if revenue_cols:
                col = revenue_cols[0]
                kpis['Total Revenue'] = {
                    'value': f"${data[col].sum():,.0f}",
                    'icon': '💰',
                    'color': 'green'
                }
                kpis['Avg Transaction'] = {
                    'value': f"${data[col].mean():,.0f}",
                    'icon': '📊',
                    'color': 'blue'
                }
            
            # Volume KPIs
            qty_cols = [col for col in numeric_cols if 'qty' in col.lower() or 'quantity' in col.lower()]
            if qty_cols:
                col = qty_cols[0]
                kpis['Total Quantity'] = {
                    'value': f"{data[col].sum():,.0f}",
                    'icon': '📦',
                    'color': 'orange'
                }
            
            # Record count
            kpis['Total Records'] = {
                'value': f"{len(data):,}",
                'icon': '📋',
                'color': 'purple'
            }
            
            # Unique customers if available
            if 'customer_name' in data.columns:
                kpis['Unique Customers'] = {
                    'value': f"{data['customer_name'].nunique():,}",
                    'icon': '👥',
                    'color': 'teal'
                }
        
        except Exception:
            kpis['Records Analyzed'] = {
                'value': f"{len(data):,}",
                'icon': '📊',
                'color': 'blue'
            }
        
        return kpis
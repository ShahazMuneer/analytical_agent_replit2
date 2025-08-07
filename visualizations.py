"""
Enhanced Visualization Generator for Enterprise Analytics Hub
Creates intelligent, interactive charts with professional styling
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
from typing import Optional, Union, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ChartGenerator:
    def __init__(self):
        """Initialize chart generator with enterprise color palette"""
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
        # Enterprise theme colors
        self.enterprise_colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#ff7f0e',
            'danger': '#d62728',
            'info': '#17becf'
        }
        
        # Common layout settings for enterprise charts
        self.base_layout = {
            'template': 'plotly_white',
            'font': {'family': 'Arial, sans-serif', 'size': 12},
            'title': {'font': {'size': 16, 'family': 'Arial, sans-serif'}},
            'showlegend': True,
            'legend': {'orientation': 'h', 'yanchor': 'bottom', 'y': -0.2},
            'margin': {'l': 50, 'r': 50, 't': 80, 'b': 100}
        }
    
    def prepare_time_series_data(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Enhanced time series data preparation with better detection"""
        
        if data.empty:
            logger.warning("Empty dataframe provided for time series preparation")
            return None
        
        # Look for date and numeric columns with improved detection
        date_columns = []
        numeric_columns = []
        
        for col in data.columns:
            col_str = str(col).lower()
            
            # Enhanced date column detection
            date_keywords = ['date', 'time', 'month', 'year', 'day', 'period', 'timestamp', 'created', 'updated']
            if any(word in col_str for word in date_keywords):
                try:
                    # Test date parsing on non-null sample
                    test_data = data[col].dropna().head(10)
                    if len(test_data) > 0:
                        pd.to_datetime(test_data, errors='raise')
                        date_columns.append(col)
                        logger.info(f"Detected date column: {col}")
                except:
                    # Try parsing as period strings (e.g., "2024-01")
                    try:
                        test_val = str(data[col].dropna().iloc[0])
                        if len(test_val) >= 7 and '-' in test_val:
                            pd.to_datetime(test_val + '-01', errors='raise')
                            date_columns.append(col)
                            logger.info(f"Detected period column: {col}")
                    except:
                        pass
            
            # Enhanced numeric column detection
            try:
                test_data = data[col].dropna().head(10)
                if len(test_data) > 0:
                    numeric_values = pd.to_numeric(test_data, errors='raise')
                    # Exclude ID columns and other non-metric columns
                    if not col_str.endswith('_id') and 'id' != col_str and col not in date_columns:
                        numeric_columns.append(col)
                        logger.info(f"Detected numeric column: {col}")
            except:
                pass
        
        # If no explicit date columns found, check first column
        if not date_columns and len(data.columns) >= 2:
            first_col = data.columns[0]
            try:
                test_data = data[first_col].dropna().head(10)
                if len(test_data) > 0:
                    pd.to_datetime(test_data, errors='raise')
                    date_columns.append(first_col)
                    logger.info(f"Using first column as date: {first_col}")
            except:
                # Try period format
                try:
                    test_val = str(data[first_col].iloc[0])
                    if len(test_val) >= 7 and '-' in test_val:
                        pd.to_datetime(test_val + '-01', errors='raise')
                        date_columns.append(first_col)
                        logger.info(f"Using first column as period: {first_col}")
                except:
                    pass
        
        if not date_columns or not numeric_columns:
            logger.warning(f"Insufficient time series data: {len(date_columns)} date columns, {len(numeric_columns)} numeric columns")
            return None
        
        # Use first available columns
        date_col = date_columns[0]
        value_col = numeric_columns[0]
        
        logger.info(f"Using columns - Date: {date_col}, Value: {value_col}")
        
        # Prepare time series data
        ts_data = data[[date_col, value_col]].copy()
        
        # Handle different date formats
        try:
            ts_data[date_col] = pd.to_datetime(ts_data[date_col])
        except:
            try:
                # Handle period strings by adding day
                ts_data[date_col] = pd.to_datetime(ts_data[date_col].astype(str) + '-01')
            except:
                logger.error(f"Failed to parse date column: {date_col}")
                return None
        
        # Convert value column to numeric
        ts_data[value_col] = pd.to_numeric(ts_data[value_col], errors='coerce')
        
        # Remove rows with NaN values
        ts_data = ts_data.dropna()
        
        if len(ts_data) == 0:
            logger.warning("No valid time series data after cleaning")
            return None
        
        # Sort by date
        ts_data = ts_data.sort_values(date_col)
        
        # Standardize column names
        ts_data.columns = ['date', 'value']
        
        logger.info(f"Prepared time series data: {len(ts_data)} rows")
        return ts_data
    
    def create_time_series_chart(self, data: pd.DataFrame, title: str = "Time Series Analysis") -> go.Figure:
        """Create an enhanced time series line chart"""
        
        fig = go.Figure()
        
        # Add main time series line
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['value'],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color=self.enterprise_colors['primary'], width=3),
            marker=dict(size=6, symbol='circle'),
            hovertemplate='<b>Date:</b> %{x}<br><b>Value:</b> %{y:,.2f}<extra></extra>'
        ))
        
        # Add trend line if enough data points
        if len(data) > 3:
            try:
                # Calculate simple trend
                x_numeric = np.arange(len(data))
                z = np.polyfit(x_numeric, data['value'], 1)
                trend_line = np.poly1d(z)(x_numeric)
                
                fig.add_trace(go.Scatter(
                    x=data['date'],
                    y=trend_line,
                    mode='lines',
                    name='Trend',
                    line=dict(color=self.enterprise_colors['secondary'], width=2, dash='dash'),
                    hovertemplate='<b>Trend:</b> %{y:,.2f}<extra></extra>'
                ))
            except Exception as e:
                logger.warning(f"Could not add trend line: {str(e)}")
        
        # Update layout with enterprise styling
        fig.update_layout(
            **self.base_layout,
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Date",
            yaxis_title="Value",
            height=500,
            hovermode='x unified'
        )
        
        # Format axes
        fig.update_xaxis(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxis(showgrid=True, gridwidth=1, gridcolor='lightgray', tickformat=',.0f')
        
        return fig
    
    def add_forecast_to_chart(self, fig: go.Figure, forecast_data: pd.DataFrame, 
                            show_confidence: bool = True) -> go.Figure:
        """Add forecast data to existing time series chart with enhanced styling"""
        
        # Convert date strings to datetime for plotting
        forecast_dates = pd.to_datetime(forecast_data['date'])
        
        # Add forecast line
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_data['forecast'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color=self.enterprise_colors['info'], width=3, dash='dash'),
            marker=dict(size=6, symbol='diamond'),
            hovertemplate='<b>Date:</b> %{x}<br><b>Forecast:</b> %{y:,.2f}<extra></extra>'
        ))
        
        # Add confidence intervals if available
        if show_confidence and 'lower_bound' in forecast_data.columns and 'upper_bound' in forecast_data.columns:
            # Add confidence band
            fig.add_trace(go.Scatter(
                x=list(forecast_dates) + list(forecast_dates)[::-1],
                y=list(forecast_data['upper_bound']) + list(forecast_data['lower_bound'])[::-1],
                fill='toself',
                fillcolor='rgba(23, 190, 207, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                hoverinfo='skip',
                showlegend=True
            ))
        
        # Update title and layout
        fig.update_layout(
            title="Historical Data with Forecast",
            showlegend=True
        )
        
        return fig
    
    def create_chart_from_data(self, data: pd.DataFrame, prompt: str = "") -> Optional[go.Figure]:
        """Create intelligent chart based on data structure and user prompt"""
        
        if data.empty:
            logger.warning("Empty dataframe provided for chart creation")
            return None
        
        prompt_lower = prompt.lower()
        logger.info(f"Creating chart for prompt: {prompt[:50]}...")
        
        # Check if this is time series data
        ts_data = self.prepare_time_series_data(data)
        if ts_data is not None and len(ts_data) > 2:
            return self.create_time_series_chart(ts_data, "Time Series Analysis")
        
        # Analyze data structure for best chart type
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        # Two columns - likely category/value pair
        if len(data.columns) == 2:
            col1, col2 = data.columns
            
            # Try to identify x and y axes
            try:
                if col1 in categorical_cols and col2 in numeric_cols:
                    x_col, y_col = col1, col2
                elif col2 in categorical_cols and col1 in numeric_cols:
                    x_col, y_col = col2, col1
                else:
                    x_col, y_col = col1, col2
                
                values = pd.to_numeric(data[y_col], errors='coerce').dropna()
                categories = data[x_col].astype(str)
                
                # Choose chart type based on context
                if any(keyword in prompt_lower for keyword in ['top', 'rank', 'best', 'worst']) or len(data) <= 15:
                    return self._create_bar_chart(categories, values, x_col, y_col, "Ranking Analysis")
                elif any(keyword in prompt_lower for keyword in ['trend', 'over time', 'growth']):
                    return self._create_line_chart(categories, values, x_col, y_col, "Trend Analysis")
                elif len(data) <= 10 and any(keyword in prompt_lower for keyword in ['share', 'distribution', 'proportion']):
                    return self._create_pie_chart(categories, values, "Distribution Analysis")
                else:
                    return self._create_bar_chart(categories, values, x_col, y_col, "Comparison Analysis")
                    
            except Exception as e:
                logger.error(f"Error creating two-column chart: {str(e)}")
        
        # Multiple numeric columns
        elif len(numeric_cols) >= 2:
            return self._create_multi_series_chart(data, numeric_cols, "Multi-Series Analysis")
        
        # Single numeric column with categories
        elif len(numeric_cols) == 1 and len(categorical_cols) >= 1:
            return self._create_categorical_analysis(data, categorical_cols[0], numeric_cols[0])
        
        # Fallback: summary statistics
        elif len(numeric_cols) > 0:
            return self.create_summary_stats_chart(data)
        
        logger.warning("Could not determine appropriate chart type for data")
        return None
    
    def _create_bar_chart(self, x_data, y_data, x_title: str, y_title: str, title: str) -> go.Figure:
        """Create professional bar chart"""
        fig = go.Figure(data=[go.Bar(
            x=x_data,
            y=y_data,
            marker_color=self.enterprise_colors['primary'],
            text=[f'{val:,.0f}' for val in y_data],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>%{y:,.2f}<extra></extra>'
        )])
        
        fig.update_layout(
            **self.base_layout,
            title={'text': title, 'x': 0.5, 'xanchor': 'center'},
            xaxis_title=x_title.title().replace('_', ' '),
            yaxis_title=y_title.title().replace('_', ' '),
            height=500
        )
        
        fig.update_xaxis(tickangle=45 if len(x_data) > 5 else 0)
        fig.update_yaxis(tickformat=',.0f')
        
        return fig
    
    def _create_line_chart(self, x_data, y_data, x_title: str, y_title: str, title: str) -> go.Figure:
        """Create professional line chart"""
        fig = go.Figure(data=[go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines+markers',
            line=dict(color=self.enterprise_colors['primary'], width=3),
            marker=dict(size=8),
            hovertemplate='<b>%{x}</b><br>%{y:,.2f}<extra></extra>'
        )])
        
        fig.update_layout(
            **self.base_layout,
            title={'text': title, 'x': 0.5, 'xanchor': 'center'},
            xaxis_title=x_title.title().replace('_', ' '),
            yaxis_title=y_title.title().replace('_', ' '),
            height=500
        )
        
        return fig
    
    def _create_pie_chart(self, labels, values, title: str) -> go.Figure:
        """Create professional pie chart"""
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,  # Donut chart
            marker_colors=self.color_palette,
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Value: %{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            **self.base_layout,
            title={'text': title, 'x': 0.5, 'xanchor': 'center'},
            height=500
        )
        
        return fig
    
    def _create_multi_series_chart(self, data: pd.DataFrame, numeric_cols: List[str], title: str) -> go.Figure:
        """Create multi-series chart"""
        fig = go.Figure()
        
        for i, col in enumerate(numeric_cols[:5]):  # Limit to 5 series for readability
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[col],
                mode='lines+markers',
                name=col.title().replace('_', ' '),
                line=dict(color=self.color_palette[i % len(self.color_palette)], width=2),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            **self.base_layout,
            title={'text': title, 'x': 0.5, 'xanchor': 'center'},
            xaxis_title="Index",
            yaxis_title="Values",
            height=500,
            showlegend=True
        )
        
        return fig
    
    def _create_categorical_analysis(self, data: pd.DataFrame, cat_col: str, num_col: str) -> go.Figure:
        """Create categorical analysis chart"""
        # Group by category and sum/average the numeric values
        grouped = data.groupby(cat_col)[num_col].agg(['sum', 'mean', 'count']).reset_index()
        
        # Choose sum or mean based on data characteristics
        value_col = 'sum' if grouped['sum'].sum() > grouped['mean'].sum() * 2 else 'mean'
        
        fig = go.Figure(data=[go.Bar(
            x=grouped[cat_col],
            y=grouped[value_col],
            marker_color=self.enterprise_colors['primary'],
            text=[f'{val:,.0f}' for val in grouped[value_col]],
            textposition='auto',
            hovertemplate=f'<b>%{{x}}</b><br>{value_col.title()}: %{{y:,.2f}}<extra></extra>'
        )])
        
        fig.update_layout(
            **self.base_layout,
            title={'text': f'{num_col.title()} by {cat_col.title()}', 'x': 0.5, 'xanchor': 'center'},
            xaxis_title=cat_col.title().replace('_', ' '),
            yaxis_title=f'{value_col.title()} {num_col.title().replace("_", " ")}',
            height=500
        )
        
        fig.update_xaxis(tickangle=45 if len(grouped) > 5 else 0)
        
        return fig
    
    def create_summary_stats_chart(self, data: pd.DataFrame) -> Optional[go.Figure]:
        """Create enhanced summary statistics visualization"""
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns found for summary statistics")
            return None
        
        # Calculate summary statistics
        stats = data[numeric_cols].describe()
        
        # Create subplot for different metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Mean Values', 'Standard Deviation', 'Min/Max Range', 'Data Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Mean values
        fig.add_trace(
            go.Bar(
                x=[col.replace('_', ' ').title() for col in numeric_cols],
                y=stats.loc['mean'].values,
                name='Mean',
                marker_color=self.enterprise_colors['primary']
            ),
            row=1, col=1
        )
        
        # Standard deviation
        fig.add_trace(
            go.Bar(
                x=[col.replace('_', ' ').title() for col in numeric_cols],
                y=stats.loc['std'].values,
                name='Std Dev',
                marker_color=self.enterprise_colors['secondary']
            ),
            row=1, col=2
        )
        
        # Min/Max range
        fig.add_trace(
            go.Scatter(
                x=[col.replace('_', ' ').title() for col in numeric_cols],
                y=stats.loc['min'].values,
                mode='markers',
                name='Min',
                marker=dict(color=self.enterprise_colors['success'], size=10)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[col.replace('_', ' ').title() for col in numeric_cols],
                y=stats.loc['max'].values,
                mode='markers',
                name='Max',
                marker=dict(color=self.enterprise_colors['danger'], size=10)
            ),
            row=2, col=1
        )
        
        # Data distribution (using first numeric column)
        if len(numeric_cols) > 0:
            first_col = numeric_cols[0]
            fig.add_trace(
                go.Histogram(
                    x=data[first_col].dropna(),
                    name=f'{first_col} Distribution',
                    marker_color=self.enterprise_colors['info'],
                    opacity=0.7
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title={'text': 'Comprehensive Data Summary', 'x': 0.5, 'xanchor': 'center'},
            **self.base_layout
        )
        
        return fig
    
    def create_correlation_heatmap(self, data: pd.DataFrame) -> Optional[go.Figure]:
        """Create correlation heatmap for numeric data"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return None
        
        correlation_matrix = numeric_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=[col.replace('_', ' ').title() for col in correlation_matrix.columns],
            y=[col.replace('_', ' ').title() for col in correlation_matrix.columns],
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={'size': 10},
            hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            **self.base_layout,
            title={'text': 'Correlation Matrix', 'x': 0.5, 'xanchor': 'center'},
            height=600,
            width=600
        )
        
        return fig

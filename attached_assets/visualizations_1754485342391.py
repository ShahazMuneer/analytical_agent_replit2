import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
from typing import Optional, Union

class ChartGenerator:
    def __init__(self):
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
    
    def prepare_time_series_data(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare data for time series visualization"""
        
        if data.empty:
            return None
        
        # Look for date and numeric columns
        date_columns = []
        numeric_columns = []
        
        for col in data.columns:
            col_str = str(col).lower()
            # Check for common date column patterns
            if any(word in col_str for word in ['date', 'time', 'month', 'year', 'day', 'period']):
                try:
                    # Try to parse as datetime or date string
                    test_data = data[col].dropna().head(10)
                    if len(test_data) > 0:
                        pd.to_datetime(test_data, errors='raise')
                        date_columns.append(col)
                except:
                    pass
            
            # Check if column contains numeric data
            try:
                test_data = data[col].dropna().head(10)
                if len(test_data) > 0:
                    pd.to_numeric(test_data, errors='raise')
                    numeric_columns.append(col)
            except:
                pass
        
        # If no explicit date columns found, check if first column could be a date/period
        if not date_columns and len(data.columns) >= 2:
            first_col = data.columns[0]
            try:
                test_data = data[first_col].dropna().head(10)
                if len(test_data) > 0:
                    # Try parsing as date or month strings
                    pd.to_datetime(test_data, errors='raise')
                    date_columns.append(first_col)
            except:
                # Check if it looks like month strings (e.g., "2024-01")
                try:
                    test_val = str(data[first_col].iloc[0])
                    if len(test_val) >= 7 and '-' in test_val:
                        pd.to_datetime(test_val + '-01', errors='raise')  # Add day for month strings
                        date_columns.append(first_col)
                except:
                    pass
        
        if not date_columns or not numeric_columns:
            return None
        
        # Use first date column and first numeric column
        date_col = date_columns[0]
        value_col = numeric_columns[0]
        
        # Prepare time series data
        ts_data = data[[date_col, value_col]].copy()
        
        # Handle different date formats
        try:
            # Try direct conversion first
            ts_data[date_col] = pd.to_datetime(ts_data[date_col])
        except:
            try:
                # Try adding day to month strings
                ts_data[date_col] = pd.to_datetime(ts_data[date_col].astype(str) + '-01')
            except:
                return None
        
        ts_data[value_col] = pd.to_numeric(ts_data[value_col], errors='coerce')
        
        # Remove rows with NaN values
        ts_data = ts_data.dropna()
        
        if len(ts_data) == 0:
            return None
        
        # Sort by date
        ts_data = ts_data.sort_values(date_col)
        
        ts_data.columns = ['date', 'value']
        return ts_data
    
    def create_time_series_chart(self, data: pd.DataFrame, title: str = "Time Series Analysis") -> go.Figure:
        """Create a time series line chart"""
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['value'],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color=self.color_palette[0], width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_white",
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def add_forecast_to_chart(self, fig: go.Figure, forecast_data: pd.DataFrame, 
                            show_confidence: bool = True) -> go.Figure:
        """Add forecast data to existing time series chart"""
        
        # Convert date strings to datetime for plotting
        forecast_dates = pd.to_datetime(forecast_data['date'])
        
        # Add forecast line
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_data['forecast'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color=self.color_palette[1], width=2, dash='dash'),
            marker=dict(size=6)
        ))
        
        # Add confidence intervals if requested
        if show_confidence and 'lower_bound' in forecast_data.columns and 'upper_bound' in forecast_data.columns:
            # Upper bound
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_data['upper_bound'],
                mode='lines',
                name='Upper Bound',
                line=dict(color='rgba(255,127,14,0)', width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Lower bound (fill between)
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_data['lower_bound'],
                mode='lines',
                name='Confidence Interval',
                line=dict(color='rgba(255,127,14,0)', width=0),
                fill='tonexty',
                fillcolor='rgba(255,127,14,0.2)',
                showlegend=True,
                hoverinfo='skip'
            ))
        
        # Update layout
        fig.update_layout(
            title="Historical Data with Forecast",
            showlegend=True
        )
        
        return fig
    
    def create_chart_from_data(self, data: pd.DataFrame, prompt: str = "") -> Optional[go.Figure]:
        """Create appropriate chart based on data structure and prompt"""
        
        if data.empty:
            return None
        
        prompt_lower = prompt.lower()
        
        # Determine chart type based on data and prompt
        if len(data.columns) == 2:
            # Two columns - likely category/value pair
            col1, col2 = data.columns
            
            # Check if first column looks like categories and second like values
            try:
                values = pd.to_numeric(data[col2], errors='raise')
                categories = data[col1].astype(str)
                
                # Choose chart type based on prompt and data
                if "top" in prompt_lower or "rank" in prompt_lower or len(data) <= 10:
                    # Bar chart for rankings/comparisons
                    fig = go.Figure(data=[go.Bar(
                        x=categories,
                        y=values,
                        marker_color=self.color_palette[0]
                    )])
                    
                    fig.update_layout(
                        title="Comparison Analysis",
                        xaxis_title=col1.title().replace('_', ' '),
                        yaxis_title=col2.title().replace('_', ' '),
                        template="plotly_white",
                        height=500
                    )
                    
                    return fig
                
                elif len(data) > 20:
                    # Line chart for many data points
                    fig = go.Figure(data=[go.Scatter(
                        x=categories,
                        y=values,
                        mode='lines+markers',
                        line=dict(color=self.color_palette[0])
                    )])
                    
                    fig.update_layout(
                        title="Trend Analysis",
                        xaxis_title=col1.title().replace('_', ' '),
                        yaxis_title=col2.title().replace('_', ' '),
                        template="plotly_white",
                        height=500
                    )
                    
                    return fig
                    
            except:
                pass
        
        # Multi-column data - create a summary chart
        numeric_cols = []
        for col in data.columns:
            try:
                pd.to_numeric(data[col], errors='raise')
                numeric_cols.append(col)
            except:
                pass
        
        if len(numeric_cols) >= 2:
            # Create a correlation or comparison chart
            fig = go.Figure()
            
            for i, col in enumerate(numeric_cols[:5]):  # Limit to 5 series
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[col],
                    mode='lines',
                    name=col.title().replace('_', ' '),
                    line=dict(color=self.color_palette[i % len(self.color_palette)])
                ))
            
            fig.update_layout(
                title="Multi-Series Analysis",
                xaxis_title="Index",
                yaxis_title="Values",
                template="plotly_white",
                height=500,
                showlegend=True
            )
            
            return fig
        
        return None
    
    def create_summary_stats_chart(self, data: pd.DataFrame) -> Optional[go.Figure]:
        """Create a summary statistics visualization"""
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return None
        
        # Calculate summary statistics
        stats = data[numeric_cols].describe()
        
        # Create a bar chart of means
        fig = go.Figure(data=[go.Bar(
            x=[col.title().replace('_', ' ') for col in numeric_cols],
            y=stats.loc['mean'].values,
            marker_color=self.color_palette[0]
        )])
        
        fig.update_layout(
            title="Average Values by Metric",
            xaxis_title="Metrics",
            yaxis_title="Average Value",
            template="plotly_white",
            height=400
        )
        
        return fig

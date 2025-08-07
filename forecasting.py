import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional, Tuple
import warnings

# Import Prophet with fallback
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# Import scikit-learn for fallback models
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class ForecastingEngine:
    """Advanced forecasting engine with multiple models"""
    
    def __init__(self):
        self.models = {}
        self.forecasts = {}
    
    def generate_forecast(self, df: pd.DataFrame, date_col: str, value_col: str, 
                         periods: int = 30, model_type: str = 'auto') -> Dict[str, Any]:
        """Generate forecast using best available method"""
        
        try:
            # Validate and prepare data
            forecast_data = self._prepare_data(df, date_col, value_col)
            
            if len(forecast_data) < 10:
                raise ValueError("Insufficient data points for forecasting (minimum 10 required)")
            
            # Choose forecasting method
            if model_type == 'auto':
                if PROPHET_AVAILABLE and len(forecast_data) >= 20:
                    return self._prophet_forecast(forecast_data, periods)
                elif SKLEARN_AVAILABLE:
                    return self._linear_forecast(forecast_data, periods)
                else:
                    return self._simple_forecast(forecast_data, periods)
            
            elif model_type == 'prophet' and PROPHET_AVAILABLE:
                return self._prophet_forecast(forecast_data, periods)
            
            elif model_type == 'linear' and SKLEARN_AVAILABLE:
                return self._linear_forecast(forecast_data, periods)
            
            else:
                return self._simple_forecast(forecast_data, periods)
                
        except Exception as e:
            logger.error(f"Forecasting error: {str(e)}")
            return {
                'error': str(e),
                'forecast': pd.DataFrame(),
                'metrics': {}
            }
    
    def _prepare_data(self, df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
        """Prepare data for forecasting"""
        
        # Create a copy and clean data
        data = df[[date_col, value_col]].copy()
        
        # Convert date column
        data[date_col] = pd.to_datetime(data[date_col])
        
        # Remove nulls and sort
        data = data.dropna().sort_values(date_col)
        
        # Remove duplicates (keep last)
        data = data.drop_duplicates(subset=[date_col], keep='last')
        
        # Rename columns for consistency
        data.columns = ['ds', 'y']
        
        return data
    
    def _prophet_forecast(self, data: pd.DataFrame, periods: int) -> Dict[str, Any]:
        """Generate forecast using Facebook Prophet"""
        
        try:
            # Initialize Prophet model
            model = Prophet(
                daily_seasonality=True if len(data) > 60 else False,
                weekly_seasonality=True if len(data) > 14 else False,
                yearly_seasonality=True if len(data) > 365 else False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10,
                uncertainty_samples=100
            )
            
            # Add custom seasonalities if enough data
            if len(data) > 60:
                model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            
            # Fit model
            model.fit(data)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=periods)
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Calculate metrics on historical data
            metrics = self._calculate_metrics(data, forecast[:-periods])
            
            # Prepare result
            result = {
                'forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
                'historical': data,
                'metrics': metrics,
                'model_type': 'prophet',
                'components': model.predict(future)[['ds', 'trend', 'seasonal', 'seasonalities']] if len(data) > 30 else None
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prophet forecasting failed: {str(e)}")
            return self._linear_forecast(data, periods)
    
    def _linear_forecast(self, data: pd.DataFrame, periods: int) -> Dict[str, Any]:
        """Generate forecast using linear regression"""
        
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import PolynomialFeatures
            
            # Prepare features (convert dates to numeric)
            data['ds_numeric'] = (data['ds'] - data['ds'].min()).dt.days
            
            X = data[['ds_numeric']].values
            y = data['y'].values
            
            # Try polynomial features for better fit
            poly_features = PolynomialFeatures(degree=2)
            X_poly = poly_features.fit_transform(X)
            
            # Fit model
            model = LinearRegression()
            model.fit(X_poly, y)
            
            # Generate future dates
            last_date = data['ds'].max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
            
            # Predict future values
            future_numeric = np.array([(d - data['ds'].min()).days for d in future_dates]).reshape(-1, 1)
            future_poly = poly_features.transform(future_numeric)
            future_predictions = model.predict(future_poly)
            
            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                'ds': list(data['ds']) + list(future_dates),
                'yhat': list(y) + list(future_predictions)
            })
            
            # Add simple confidence intervals (±10%)
            forecast_df['yhat_lower'] = forecast_df['yhat'] * 0.9
            forecast_df['yhat_upper'] = forecast_df['yhat'] * 1.1
            
            # Calculate metrics
            y_pred = model.predict(X_poly)
            metrics = {
                'mae': mean_absolute_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'mape': np.mean(np.abs((y - y_pred) / y)) * 100
            }
            
            result = {
                'forecast': forecast_df,
                'historical': data,
                'metrics': metrics,
                'model_type': 'linear_regression'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Linear forecasting failed: {str(e)}")
            return self._simple_forecast(data, periods)
    
    def _simple_forecast(self, data: pd.DataFrame, periods: int) -> Dict[str, Any]:
        """Generate simple forecast using moving average and trend"""
        
        try:
            # Calculate moving average and trend
            window = min(7, len(data) // 3)
            data['ma'] = data['y'].rolling(window=window, center=True).mean()
            
            # Calculate trend
            data['trend'] = np.nan
            if len(data) > 2:
                for i in range(window, len(data) - window):
                    x = np.arange(i - window, i + window + 1)
                    y_subset = data['y'].iloc[i - window:i + window + 1]
                    if not y_subset.isnull().all():
                        slope, intercept = np.polyfit(x, y_subset.fillna(method='ffill'), 1)
                        data.loc[data.index[i], 'trend'] = slope
            
            # Forward fill trend
            data['trend'] = data['trend'].fillna(method='ffill').fillna(method='bfill')
            
            # Generate future predictions
            last_value = data['y'].iloc[-1]
            avg_trend = data['trend'].mean()
            
            future_dates = pd.date_range(start=data['ds'].max() + timedelta(days=1), periods=periods, freq='D')
            future_values = []
            
            current_value = last_value
            for i in range(periods):
                current_value += avg_trend
                future_values.append(current_value)
            
            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                'ds': list(data['ds']) + list(future_dates),
                'yhat': list(data['y']) + future_values
            })
            
            # Add confidence intervals based on historical variance
            std_dev = data['y'].std()
            forecast_df['yhat_lower'] = forecast_df['yhat'] - 1.96 * std_dev
            forecast_df['yhat_upper'] = forecast_df['yhat'] + 1.96 * std_dev
            
            # Calculate simple metrics
            metrics = {
                'mae': std_dev * 0.8,  # Approximation
                'rmse': std_dev,
                'mape': (std_dev / data['y'].mean()) * 100
            }
            
            result = {
                'forecast': forecast_df,
                'historical': data[['ds', 'y']],
                'metrics': metrics,
                'model_type': 'simple_trend'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Simple forecasting failed: {str(e)}")
            return {
                'error': str(e),
                'forecast': pd.DataFrame(),
                'metrics': {}
            }
    
    def _calculate_metrics(self, actual_data: pd.DataFrame, forecast_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate forecast accuracy metrics"""
        
        try:
            # Merge actual and forecast data
            merged = actual_data.merge(forecast_data[['ds', 'yhat']], on='ds', how='inner')
            
            if len(merged) == 0:
                return {}
            
            actual = merged['y'].values
            predicted = merged['yhat'].values
            
            # Calculate metrics
            mae = np.mean(np.abs(actual - predicted))
            mse = np.mean((actual - predicted) ** 2)
            rmse = np.sqrt(mse)
            
            # Avoid division by zero for MAPE
            non_zero_mask = actual != 0
            if np.any(non_zero_mask):
                mape = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100
            else:
                mape = 0
            
            return {
                'mae': round(mae, 4),
                'mse': round(mse, 4),
                'rmse': round(rmse, 4),
                'mape': round(mape, 4)
            }
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {str(e)}")
            return {}
    
    def create_forecast_visualization(self, forecast_result: Dict[str, Any], title: str = "Forecast") -> go.Figure:
        """Create interactive forecast visualization"""
        
        fig = go.Figure()
        
        if 'error' in forecast_result:
            fig.add_annotation(
                text=f"Forecast error: {forecast_result['error']}",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig
        
        try:
            historical = forecast_result.get('historical', pd.DataFrame())
            forecast = forecast_result.get('forecast', pd.DataFrame())
            
            if not historical.empty and not forecast.empty:
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=historical['ds'],
                    y=historical['y'],
                    mode='lines+markers',
                    name='Historical Data',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4)
                ))
                
                # Forecast line
                forecast_future = forecast[forecast['ds'] > historical['ds'].max()]
                if not forecast_future.empty:
                    fig.add_trace(go.Scatter(
                        x=forecast_future['ds'],
                        y=forecast_future['yhat'],
                        mode='lines',
                        name='Forecast',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    
                    # Confidence intervals
                    if 'yhat_lower' in forecast_future.columns and 'yhat_upper' in forecast_future.columns:
                        fig.add_trace(go.Scatter(
                            x=forecast_future['ds'],
                            y=forecast_future['yhat_upper'],
                            fill=None,
                            mode='lines',
                            line_color='rgba(0,0,0,0)',
                            showlegend=False
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_future['ds'],
                            y=forecast_future['yhat_lower'],
                            fill='tonexty',
                            mode='lines',
                            line_color='rgba(0,0,0,0)',
                            name='Confidence Interval',
                            fillcolor='rgba(255,0,0,0.2)'
                        ))
            
            # Update layout
            fig.update_layout(
                title=f'{title} - {forecast_result.get("model_type", "Unknown").title()} Model',
                xaxis_title='Date',
                yaxis_title='Value',
                hovermode='x unified',
                showlegend=True,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            # Add vertical line at forecast start
            if not historical.empty:
                fig.add_vline(
                    x=historical['ds'].max(),
                    line_dash="dot",
                    line_color="gray",
                    annotation_text="Forecast Start"
                )
            
        except Exception as e:
            logger.error(f"Forecast visualization error: {str(e)}")
            fig.add_annotation(
                text=f"Visualization error: {str(e)}",
                x=0.5, y=0.5,
                showarrow=False
            )
        
        return fig
    
    def get_available_models(self) -> Dict[str, bool]:
        """Get available forecasting models"""
        return {
            'prophet': PROPHET_AVAILABLE,
            'linear': SKLEARN_AVAILABLE,
            'simple': True  # Always available
        }
    
    def recommend_model(self, data_length: int, seasonality_detected: bool = False) -> str:
        """Recommend best model based on data characteristics"""
        
        if data_length < 10:
            return None  # Insufficient data
        
        elif data_length >= 60 and PROPHET_AVAILABLE:
            return 'prophet'  # Best for larger datasets with seasonality
        
        elif data_length >= 20 and SKLEARN_AVAILABLE:
            return 'linear'  # Good for medium datasets
        
        else:
            return 'simple'  # Fallback for small datasets
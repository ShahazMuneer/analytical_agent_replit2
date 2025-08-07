import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

class ForecastingEngine:
    def __init__(self):
        self.prophet_available = PROPHET_AVAILABLE
    
    def generate_forecast(self, data: pd.DataFrame, periods: int = 6) -> Optional[pd.DataFrame]:
        """Generate forecast using Prophet or simple trend analysis"""
        
        if not self.prophet_available:
            return self._simple_trend_forecast(data, periods)
        
        try:
            return self._prophet_forecast(data, periods)
        except Exception as e:
            print(f"Prophet forecasting failed: {e}")
            return self._simple_trend_forecast(data, periods)
    
    def _prophet_forecast(self, data: pd.DataFrame, periods: int) -> Optional[pd.DataFrame]:
        """Generate forecast using Facebook Prophet"""
        
        # Prepare data for Prophet (needs 'ds' and 'y' columns)
        if len(data.columns) < 2:
            return None
        
        # Assume first column is date, second is value
        prophet_data = pd.DataFrame({
            'ds': pd.to_datetime(data.iloc[:, 0]),
            'y': pd.to_numeric(data.iloc[:, 1], errors='coerce')
        })
        
        # Remove any rows with NaN values
        prophet_data = prophet_data.dropna()
        
        if len(prophet_data) < 10:  # Need minimum data points
            return self._simple_trend_forecast(data, periods)
        
        # Initialize and fit Prophet model
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            interval_width=0.8
        )
        
        model.fit(prophet_data)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq='M')
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Extract forecast data (only future periods)
        forecast_data = forecast.tail(periods)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        
        forecast_data.columns = ['date', 'forecast', 'lower_bound', 'upper_bound']
        forecast_data['date'] = forecast_data['date'].dt.strftime('%Y-%m')
        
        return forecast_data
    
    def _simple_trend_forecast(self, data: pd.DataFrame, periods: int) -> Optional[pd.DataFrame]:
        """Simple linear trend-based forecasting as fallback"""
        
        if len(data) < 3:
            return None
        
        try:
            # Assume first column is date/period, second is value
            dates = pd.to_datetime(data.iloc[:, 0])
            values = pd.to_numeric(data.iloc[:, 1], errors='coerce')
            
            # Remove NaN values
            valid_indices = ~values.isna()
            dates = dates[valid_indices]
            values = values[valid_indices]
            
            if len(values) < 3:
                return None
            
            # Calculate simple linear trend
            x = np.arange(len(values))
            coeffs = np.polyfit(x, values, 1)
            slope, intercept = coeffs
            
            # Generate future dates
            last_date = dates.iloc[-1]
            future_dates = []
            for i in range(1, periods + 1):
                future_date = last_date + timedelta(days=30 * i)  # Approximate monthly
                future_dates.append(future_date)
            
            # Calculate forecasted values
            future_x = np.arange(len(values), len(values) + periods)
            forecasted_values = slope * future_x + intercept
            
            # Calculate confidence intervals (simple approach)
            residuals = values - (slope * x + intercept)
            std_error = np.std(residuals)
            
            forecast_df = pd.DataFrame({
                'date': [d.strftime('%Y-%m') for d in future_dates],
                'forecast': forecasted_values,
                'lower_bound': forecasted_values - 1.96 * std_error,
                'upper_bound': forecasted_values + 1.96 * std_error
            })
            
            return forecast_df
            
        except Exception as e:
            print(f"Simple forecasting failed: {e}")
            return None
    
    def validate_forecast_data(self, data: pd.DataFrame) -> bool:
        """Validate if data is suitable for forecasting"""
        
        if data.empty or len(data) < 3:
            return False
        
        # Check if we have time-based data
        try:
            pd.to_datetime(data.iloc[:, 0])
            pd.to_numeric(data.iloc[:, 1], errors='coerce')
            return True
        except:
            return False

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DemandForecaster:
    def __init__(self):
        self.product_models = {}
        self.category_models = {}
        self.is_trained = False
        
    def prepare_product_demand_data(self, df):
        """Prepare product-level demand data"""
        if not all(col in df.columns for col in ['product_name', 'last_purchase_date', 'quantity']):
            raise ValueError("Required columns missing: product_name, last_purchase_date, quantity")
        
        # Group by product and date
        product_demand = df.groupby(['product_name', df['last_purchase_date'].dt.date]).agg({
            'quantity': 'sum',
            'order_id': 'count',
            'total_revenue': 'sum' if 'total_revenue' in df.columns else 'first'
        }).reset_index()
        
        product_demand.columns = ['product_name', 'date', 'quantity_demanded', 'orders_count', 'revenue']
        product_demand['date'] = pd.to_datetime(product_demand['date'])
        
        return product_demand
    
    def create_demand_features(self, demand_data, product_name):
        """Create features for demand forecasting"""
        product_data = demand_data[demand_data['product_name'] == product_name].copy()
        product_data = product_data.sort_values('date')
        
        # Fill missing dates
        date_range = pd.date_range(start=product_data['date'].min(), 
                                 end=product_data['date'].max(), freq='D')
        product_data = product_data.set_index('date').reindex(date_range, fill_value=0).reset_index()
        product_data.columns = ['date', 'product_name', 'quantity_demanded', 'orders_count', 'revenue']
        product_data['product_name'] = product_name
        
        # Time features
        product_data['day_of_week'] = product_data['date'].dt.dayofweek
        product_data['month'] = product_data['date'].dt.month
        product_data['day_of_month'] = product_data['date'].dt.day
        product_data['is_weekend'] = (product_data['day_of_week'] >= 5).astype(int)
        
        # Lag features
        for lag in [1, 7, 14, 30]:
            product_data[f'demand_lag_{lag}'] = product_data['quantity_demanded'].shift(lag)
        
        # Rolling averages
        for window in [7, 14, 30]:
            product_data[f'demand_rolling_{window}'] = product_data['quantity_demanded'].rolling(window=window).mean()
        
        # Trend features
        product_data['demand_trend_7d'] = product_data['quantity_demanded'].rolling(7).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 7 else 0
        )
        
        return product_data.dropna()
    
    def forecast_product_demand(self, demand_data, product_name, days_ahead=30):
        """Forecast demand for a specific product using simple methods"""
        product_data = self.create_demand_features(demand_data, product_name)
        
        if len(product_data) < 14:  # Need at least 2 weeks of data
            return None
        
        # Simple forecasting methods
        forecasts = {}
        
        # Method 1: Moving Average
        recent_avg = product_data.tail(14)['quantity_demanded'].mean()
        forecasts['moving_average'] = [recent_avg] * days_ahead
        
        # Method 2: Exponential Smoothing (simple)
        alpha = 0.3
        last_value = product_data['quantity_demanded'].iloc[-1]
        exp_forecast = []
        for _ in range(days_ahead):
            last_value = alpha * last_value + (1 - alpha) * recent_avg
            exp_forecast.append(last_value)
        forecasts['exponential_smoothing'] = exp_forecast
        
        # Method 3: Seasonal Average (by day of week)
        seasonal_avg = product_data.groupby('day_of_week')['quantity_demanded'].mean()
        last_date = product_data['date'].max()
        seasonal_forecast = []
        for i in range(days_ahead):
            future_date = last_date + timedelta(days=i+1)
            day_of_week = future_date.dayofweek
            seasonal_forecast.append(seasonal_avg.get(day_of_week, recent_avg))
        forecasts['seasonal'] = seasonal_forecast
        
        # Method 4: Trend-adjusted forecast
        recent_trend = product_data.tail(14)['demand_trend_7d'].mean()
        trend_forecast = []
        base_value = recent_avg
        for i in range(days_ahead):
            trend_forecast.append(max(0, base_value + recent_trend * i))
        forecasts['trend_adjusted'] = trend_forecast
        
        # Ensemble forecast (average of methods)
        ensemble_forecast = []
        for i in range(days_ahead):
            avg_forecast = np.mean([
                forecasts['moving_average'][i],
                forecasts['exponential_smoothing'][i],
                forecasts['seasonal'][i],
                forecasts['trend_adjusted'][i]
            ])
            ensemble_forecast.append(max(0, avg_forecast))
        
        # Create forecast dataframe
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                   periods=days_ahead, freq='D')
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'product_name': product_name,
            'predicted_demand': ensemble_forecast,
            'moving_average': forecasts['moving_average'],
            'exponential_smoothing': forecasts['exponential_smoothing'],
            'seasonal': forecasts['seasonal'],
            'trend_adjusted': forecasts['trend_adjusted']
        })
        
        return forecast_df
    
    def forecast_category_demand(self, df, days_ahead=30):
        """Forecast demand by product category"""
        if 'category' not in df.columns:
            return None
        
        category_forecasts = {}
        
        for category in df['category'].unique():
            category_data = df[df['category'] == category]
            
            if len(category_data) < 30:  # Skip categories with insufficient data
                continue
            
            # Aggregate category demand
            category_demand = category_data.groupby(category_data['last_purchase_date'].dt.date).agg({
                'quantity': 'sum',
                'total_revenue': 'sum' if 'total_revenue' in category_data.columns else 'first'
            }).reset_index()
            
            category_demand.columns = ['date', 'quantity_demanded', 'revenue']
            category_demand['date'] = pd.to_datetime(category_demand['date'])
            category_demand = category_demand.sort_values('date')
            
            # Simple forecast using recent patterns
            recent_avg = category_demand.tail(14)['quantity_demanded'].mean()
            seasonal_pattern = category_demand.groupby(category_demand['date'].dt.dayofweek)['quantity_demanded'].mean()
            
            future_dates = pd.date_range(start=category_demand['date'].max() + timedelta(days=1), 
                                       periods=days_ahead, freq='D')
            
            forecasted_demand = []
            for future_date in future_dates:
                day_of_week = future_date.dayofweek
                seasonal_factor = seasonal_pattern.get(day_of_week, 1) / seasonal_pattern.mean()
                forecasted_demand.append(max(0, recent_avg * seasonal_factor))
            
            category_forecasts[category] = pd.DataFrame({
                'date': future_dates,
                'category': category,
                'predicted_demand': forecasted_demand
            })
        
        return category_forecasts
    
    def identify_top_products(self, df, metric='revenue', top_n=10):
        """Identify top products by specified metric"""
        if metric == 'revenue' and 'total_revenue' in df.columns:
            top_products = df.groupby('product_name')['total_revenue'].sum().nlargest(top_n)
        elif metric == 'quantity':
            top_products = df.groupby('product_name')['quantity'].sum().nlargest(top_n)
        elif metric == 'frequency':
            top_products = df.groupby('product_name')['order_id'].count().nlargest(top_n)
        else:
            raise ValueError("Metric must be 'revenue', 'quantity', or 'frequency'")
        
        return top_products
    
    def analyze_demand_patterns(self, df):
        """Analyze demand patterns and seasonality"""
        analysis = {}
        
        # Overall demand trends
        if 'last_purchase_date' in df.columns:
            df['month'] = df['last_purchase_date'].dt.month
            df['day_of_week'] = df['last_purchase_date'].dt.day_name()
            df['hour'] = df['last_purchase_date'].dt.hour
            
            # Seasonal patterns
            analysis['monthly_demand'] = df.groupby('month')['quantity'].sum().to_dict()
            analysis['daily_demand'] = df.groupby('day_of_week')['quantity'].sum().to_dict()
            
            # Peak periods
            analysis['peak_month'] = df.groupby('month')['quantity'].sum().idxmax()
            analysis['peak_day'] = df.groupby('day_of_week')['quantity'].sum().idxmax()
        
        # Product-level analysis
        if 'product_name' in df.columns:
            product_stats = df.groupby('product_name').agg({
                'quantity': ['sum', 'mean', 'std'],
                'total_revenue': 'sum' if 'total_revenue' in df.columns else 'first',
                'order_id': 'count'
            }).round(2)
            
            # Flatten column names
            product_stats.columns = ['_'.join(col).strip() for col in product_stats.columns]
            analysis['product_statistics'] = product_stats.to_dict('index')
        
        # Category-level analysis
        if 'category' in df.columns:
            category_stats = df.groupby('category').agg({
                'quantity': ['sum', 'mean'],
                'total_revenue': 'sum' if 'total_revenue' in df.columns else 'first',
                'product_name': 'nunique'
            }).round(2)
            
            category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns]
            analysis['category_statistics'] = category_stats.to_dict('index')
        
        return analysis
    
    def calculate_inventory_recommendations(self, demand_forecasts, safety_stock_days=7):
        """Calculate inventory recommendations based on demand forecasts"""
        recommendations = {}
        
        if isinstance(demand_forecasts, dict):
            # Handle category forecasts
            for category, forecast_df in demand_forecasts.items():
                avg_daily_demand = forecast_df['predicted_demand'].mean()
                safety_stock = avg_daily_demand * safety_stock_days
                reorder_point = avg_daily_demand * 14 + safety_stock  # 2 weeks + safety stock
                
                recommendations[category] = {
                    'avg_daily_demand': round(avg_daily_demand, 2),
                    'safety_stock': round(safety_stock, 2),
                    'reorder_point': round(reorder_point, 2),
                    'monthly_demand': round(avg_daily_demand * 30, 2)
                }
        else:
            # Handle single product forecast
            if demand_forecasts is not None:
                avg_daily_demand = demand_forecasts['predicted_demand'].mean()
                safety_stock = avg_daily_demand * safety_stock_days
                reorder_point = avg_daily_demand * 14 + safety_stock
                
                product_name = demand_forecasts['product_name'].iloc[0]
                recommendations[product_name] = {
                    'avg_daily_demand': round(avg_daily_demand, 2),
                    'safety_stock': round(safety_stock, 2),
                    'reorder_point': round(reorder_point, 2),
                    'monthly_demand': round(avg_daily_demand * 30, 2)
                }
        
        return recommendations
    
    def plot_demand_analysis(self, df, demand_forecasts=None):
        """Plot demand analysis and forecasts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Monthly demand pattern
        if 'last_purchase_date' in df.columns:
            monthly_demand = df.groupby(df['last_purchase_date'].dt.month)['quantity'].sum()
            axes[0, 0].bar(monthly_demand.index, monthly_demand.values, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('Monthly Demand Pattern')
            axes[0, 0].set_xlabel('Month')
            axes[0, 0].set_ylabel('Total Quantity')
        
        # Daily demand pattern
        if 'last_purchase_date' in df.columns:
            daily_demand = df.groupby(df['last_purchase_date'].dt.day_name())['quantity'].sum()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_demand = daily_demand.reindex(day_order)
            axes[0, 1].bar(daily_demand.index, daily_demand.values, color='lightcoral', alpha=0.7)
            axes[0, 1].set_title('Daily Demand Pattern')
            axes[0, 1].set_xlabel('Day of Week')
            axes[0, 1].set_ylabel('Total Quantity')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Top products by quantity
        if 'product_name' in df.columns:
            top_products = df.groupby('product_name')['quantity'].sum().nlargest(10)
            axes[1, 0].barh(range(len(top_products)), top_products.values, color='lightgreen', alpha=0.7)
            axes[1, 0].set_yticks(range(len(top_products)))
            axes[1, 0].set_yticklabels(top_products.index)
            axes[1, 0].set_title('Top 10 Products by Demand')
            axes[1, 0].set_xlabel('Total Quantity')
        
        # Category demand distribution
        if 'category' in df.columns:
            category_demand = df.groupby('category')['quantity'].sum()
            axes[1, 1].pie(category_demand.values, labels=category_demand.index, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Demand Distribution by Category')
        
        plt.tight_layout()
        return fig

# Example usage
if __name__ == "__main__":
    print("Demand Forecasting Model Ready!")
    print("Use this class to forecast product and category demand for inventory management.")

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
        self._last_cols = None

    # -------------------- Helpers for adaptive schemas --------------------
    def _first_present_col(self, df, candidates):
        lower_map = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        return None

    def _to_numeric(self, s, clip_min=None, clip_max=None):
        s = pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)
        if clip_min is not None or clip_max is not None:
            s = np.clip(s, a_min=clip_min if clip_min is not None else -np.inf,
                        a_max=clip_max if clip_max is not None else np.inf)
        return s

    def _ensure_datetime(self, s):
        return pd.to_datetime(s, errors='coerce')

    def _detect_cols(self, df):
        cols = {}
        cols['product'] = self._first_present_col(df, ['product_name', 'product', 'product_id', 'sku', 'item', 'itemname', 'productname'])
        cols['date'] = self._first_present_col(df, ['last_purchase_date', 'order_date', 'invoice_date', 'date', 'purchase_date', 'transaction_date', 'orderdate'])
        cols['quantity'] = self._first_present_col(df, ['quantity', 'qty', 'units', 'quantity_demanded', 'order_quantity'])
        cols['revenue'] = self._first_present_col(df, ['total_revenue', 'revenue', 'amount', 'sales', 'sales_amount', 'gross_sales', 'net_sales', 'line_total'])
        cols['order_id'] = self._first_present_col(df, ['order_id', 'orderid', 'invoice_no', 'invoice', 'transaction_id', 'orderid'])
        cols['category'] = self._first_present_col(df, ['category', 'product_category', 'segment'])
        self._last_cols = cols
        return cols
        
    def prepare_product_demand_data(self, df):
        """Prepare product-level demand data with adaptive column detection"""
        cols = self._detect_cols(df)
        prod_col, date_col, qty_col = cols['product'], cols['date'], cols['quantity']
        if not all([prod_col, date_col, qty_col]):
            raise ValueError("Required columns missing. Need product, date, and quantity. Consider renaming or mapping your columns.")

        work = df.copy()
        work[date_col] = self._ensure_datetime(work[date_col])
        work = work.dropna(subset=[date_col])
        if work.empty:
            raise ValueError("No valid dates after parsing; cannot prepare demand data")

        # Sanitize numerics
        work[qty_col] = self._to_numeric(work[qty_col], clip_min=0, clip_max=1e6)
        # Revenue handling
        rev_col = cols['revenue']
        if rev_col is None:
            # Try to compute from unit price if available
            price_col = self._first_present_col(work, ['unit_price', 'price'])
            if price_col is not None:
                work['__revenue_tmp__'] = self._to_numeric(work[price_col], 0, 1e6) * work[qty_col]
                rev_col = '__revenue_tmp__'
        if rev_col is None:
            work['__revenue_fallback__'] = 0.0
            rev_col = '__revenue_fallback__'

        # Order id handling for counting frequency
        oid_col = cols['order_id']
        if oid_col is None:
            # Use a pseudo order count by treating each row as one order
            work['__order_count__'] = 1
            oid_col = '__order_count__'

        grouped = work.groupby([prod_col, work[date_col].dt.date]).agg({
            qty_col: 'sum',
            oid_col: 'count' if oid_col != '__order_count__' else 'sum',
            rev_col: 'sum'
        }).reset_index()

        grouped.columns = ['product_name', 'date', 'quantity_demanded', 'orders_count', 'revenue']
        grouped['date'] = pd.to_datetime(grouped['date'])
        return grouped
    
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
    
    def forecast_product_demand(self, demand_data, product_name, days_ahead=30, confidence: int | float | None = None, seasonal: bool = True):
        """Forecast demand for a specific product.

        Parameters:
        - demand_data: prepared product demand data (from prepare_product_demand_data)
        - product_name: name of product to forecast
        - days_ahead: forecast horizon in days
        - confidence: optional confidence level (e.g., 80..99) for yhat_lower/yhat_upper
        - seasonal: include seasonal and trend components in the ensemble if True
        """
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
        components = ['moving_average', 'exponential_smoothing']
        if seasonal:
            components += ['seasonal', 'trend_adjusted']
        for i in range(days_ahead):
            values = [forecasts[c][i] for c in components]
            avg_forecast = np.mean(values)
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

        # Optional confidence interval based on in-sample residual stddev of chosen ensemble
        if confidence is not None:
            try:
                # Build naive in-sample ensemble fit
                hist_values = product_data['quantity_demanded'].values
                # Align last len(hist_values) days with available features; use last 14 days for sigma
                window = min(30, len(product_data))
                # Recreate component fits for last window days where possible
                # For simplicity, use residuals to moving avg baseline as proxy
                baseline = product_data.tail(window)['quantity_demanded'].rolling(7).mean().bfill()
                residuals = product_data.tail(window)['quantity_demanded'] - baseline
                sigma = float(np.nanstd(residuals.values, ddof=1)) if len(residuals) > 1 else 0.0
                if sigma > 0:
                    z_map = {80: 1.282, 85: 1.440, 90: 1.645, 95: 1.960, 98: 2.326, 99: 2.576}
                    c = int(round(float(confidence)))
                    z = z_map.get(c, z_map[min(z_map.keys(), key=lambda k: abs(k - c))])
                    lower = np.clip(np.array(ensemble_forecast) - z * sigma, a_min=0.0, a_max=None)
                    upper = np.clip(np.array(ensemble_forecast) + z * sigma, a_min=0.0, a_max=None)
                    forecast_df['yhat_lower'] = lower
                    forecast_df['yhat_upper'] = upper
            except Exception:
                pass
        
        return forecast_df
    
    def forecast_category_demand(self, df, days_ahead=30):
        """Forecast demand by product category with adaptive columns"""
        cols = self._detect_cols(df)
        cat_col, date_col, qty_col = cols['category'], cols['date'], cols['quantity']
        if cat_col is None or date_col is None or qty_col is None:
            return None

        work = df.copy()
        work[date_col] = self._ensure_datetime(work[date_col])
        work = work.dropna(subset=[date_col])
        if work.empty:
            return None
        work[qty_col] = self._to_numeric(work[qty_col], 0, 1e6)
        rev_col = cols['revenue']
        if rev_col is None:
            work['__rev__'] = 0.0
            rev_col = '__rev__'

        category_forecasts = {}
        for category, category_data in work.groupby(cat_col):
            if len(category_data) < 30:
                continue
            category_demand = category_data.groupby(category_data[date_col].dt.date).agg({
                qty_col: 'sum',
                rev_col: 'sum'
            }).reset_index()
            category_demand.columns = ['date', 'quantity_demanded', 'revenue']
            category_demand['date'] = pd.to_datetime(category_demand['date'])
            category_demand = category_demand.sort_values('date')

            if len(category_demand) == 0:
                continue
            recent_avg = category_demand.tail(14)['quantity_demanded'].mean()
            if np.isnan(recent_avg) or recent_avg <= 0:
                continue
            seasonal_pattern = category_demand.groupby(category_demand['date'].dt.dayofweek)['quantity_demanded'].mean()

            future_dates = pd.date_range(start=category_demand['date'].max() + timedelta(days=1),
                                         periods=days_ahead, freq='D')
            forecasted_demand = []
            for future_date in future_dates:
                dow = future_date.dayofweek
                seasonal_factor = seasonal_pattern.get(dow, seasonal_pattern.mean()) / max(seasonal_pattern.mean(), 1e-6)
                forecasted_demand.append(max(0, recent_avg * seasonal_factor))
            category_forecasts[category] = pd.DataFrame({
                'date': future_dates,
                'category': category,
                'predicted_demand': forecasted_demand
            })
        return category_forecasts
    
    def identify_top_products(self, df, metric='revenue', top_n=10):
        """Identify top products by specified metric with adaptive columns"""
        cols = self._detect_cols(df)
        prod_col = cols['product'] or 'product_name'
        if metric == 'revenue':
            rev_col = cols['revenue']
            if rev_col is None:
                # Attempt compute from unit price * quantity
                price_col = self._first_present_col(df, ['unit_price', 'price'])
                qty_col = cols['quantity']
                if price_col and qty_col:
                    tmp = self._to_numeric(df[price_col], 0, 1e6) * self._to_numeric(df[qty_col], 0, 1e6)
                    top_products = pd.Series(tmp).groupby(df[prod_col]).sum().nlargest(top_n)
                else:
                    raise ValueError("No revenue column found and cannot compute from price*quantity")
            else:
                top_products = self._to_numeric(df[rev_col], 0, 1e9).groupby(df[prod_col]).sum().nlargest(top_n)
        elif metric == 'quantity':
            qty_col = cols['quantity']
            if qty_col is None:
                raise ValueError("No quantity column found")
            top_products = self._to_numeric(df[qty_col], 0, 1e6).groupby(df[prod_col]).sum().nlargest(top_n)
        elif metric == 'frequency':
            oid_col = cols['order_id']
            if oid_col:
                top_products = df.groupby(prod_col)[oid_col].count().nlargest(top_n)
            else:
                # Fallback: count rows per product
                top_products = df.groupby(prod_col).size().nlargest(top_n)
        else:
            raise ValueError("Metric must be 'revenue', 'quantity', or 'frequency'")
        return top_products
    
    def analyze_demand_patterns(self, df):
        """Analyze demand patterns and seasonality (adaptive columns)"""
        analysis = {}
        cols = self._detect_cols(df)
        date_col, qty_col, prod_col, cat_col, rev_col, oid_col = (
            cols['date'], cols['quantity'], cols['product'], cols['category'], cols['revenue'], cols['order_id']
        )

        if date_col and qty_col:
            work = df.copy()
            work[date_col] = self._ensure_datetime(work[date_col])
            work = work.dropna(subset=[date_col])
            work[qty_col] = self._to_numeric(work[qty_col], 0, 1e6)
            work['month'] = work[date_col].dt.month
            work['day_of_week'] = work[date_col].dt.day_name()
            # Seasonal patterns
            analysis['monthly_demand'] = work.groupby('month')[qty_col].sum().to_dict()
            analysis['daily_demand'] = work.groupby('day_of_week')[qty_col].sum().to_dict()
            if len(analysis['monthly_demand']):
                analysis['peak_month'] = max(analysis['monthly_demand'], key=analysis['monthly_demand'].get)
            if len(analysis['daily_demand']):
                analysis['peak_day'] = max(analysis['daily_demand'], key=analysis['daily_demand'].get)

        # Product-level analysis
        if prod_col and qty_col:
            agg_dict = {qty_col: ['sum', 'mean', 'std']}
            if rev_col:
                agg_dict[rev_col] = 'sum'
            if oid_col:
                agg_dict[oid_col] = 'count'
            product_stats = df.groupby(prod_col).agg(agg_dict).round(2)
            product_stats.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in product_stats.columns]
            analysis['product_statistics'] = product_stats.to_dict('index')

        # Category-level analysis
        if cat_col and qty_col and prod_col:
            agg_dict = {qty_col: ['sum', 'mean']}
            if rev_col:
                agg_dict[rev_col] = 'sum'
            agg_dict[prod_col] = 'nunique'
            category_stats = df.groupby(cat_col).agg(agg_dict).round(2)
            category_stats.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in category_stats.columns]
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
        """Plot demand analysis and forecasts (adaptive columns)"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        cols = self._detect_cols(df)
        date_col, qty_col, prod_col, cat_col = cols['date'], cols['quantity'], cols['product'], cols['category']

        # Monthly demand pattern
        if date_col and qty_col:
            work = df.copy()
            work[date_col] = self._ensure_datetime(work[date_col])
            work = work.dropna(subset=[date_col])
            work[qty_col] = self._to_numeric(work[qty_col], 0, 1e6)
            monthly_demand = work.groupby(work[date_col].dt.month)[qty_col].sum()
            axes[0, 0].bar(monthly_demand.index, monthly_demand.values, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('Monthly Demand Pattern')
            axes[0, 0].set_xlabel('Month')
            axes[0, 0].set_ylabel('Total Quantity')
        
        # Daily demand pattern
        if date_col and qty_col:
            daily_demand = work.groupby(work[date_col].dt.day_name())[qty_col].sum()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_demand = daily_demand.reindex(day_order)
            axes[0, 1].bar(daily_demand.index, daily_demand.values, color='lightcoral', alpha=0.7)
            axes[0, 1].set_title('Daily Demand Pattern')
            axes[0, 1].set_xlabel('Day of Week')
            axes[0, 1].set_ylabel('Total Quantity')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Top products by quantity
        if prod_col and qty_col:
            top_products = self._to_numeric(df[qty_col], 0, 1e6).groupby(df[prod_col]).sum().nlargest(10)
            axes[1, 0].barh(range(len(top_products)), top_products.values, color='lightgreen', alpha=0.7)
            axes[1, 0].set_yticks(range(len(top_products)))
            axes[1, 0].set_yticklabels(top_products.index)
            axes[1, 0].set_title('Top 10 Products by Demand')
            axes[1, 0].set_xlabel('Total Quantity')
        
        # Category demand distribution
        if cat_col and qty_col:
            category_demand = self._to_numeric(df[qty_col], 0, 1e6).groupby(df[cat_col]).sum()
            axes[1, 1].pie(category_demand.values, labels=category_demand.index, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Demand Distribution by Category')
        
        plt.tight_layout()
        return fig

# Example usage
if __name__ == "__main__":
    print("Demand Forecasting Model Ready!")
    print("Use this class to forecast product and category demand for inventory management.")

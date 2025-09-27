import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class _ModelResult:
	mae: float
	mse: float
	r2: float


class AdvancedSalesForecaster:
	"""Simple sales forecasting helper used by app.py.

	Provides:
	  - prepare_sales_data(df): daily aggregation with revenue, quantity_sold, orders_count
	  - train_forecasting_models(sales_data): evaluates a few naive models on revenue
	  - forecast_future_sales(days, target_metric): produce future predictions
	  - analyze_sales_trends(sales_data): seasonal patterns for UI
	"""

	def __init__(self):
		self.best_model_name: str | None = None
		self._last_date = None
		self._seasonal_by_dow = None
		self._recent_avg_window = 14
		self._linreg = None  # LinearRegression on time index
		self._train_df: pd.DataFrame | None = None

	def prepare_sales_data(self, df: pd.DataFrame) -> pd.DataFrame | None:
		if df is None or df.empty:
			return None

		data = df.copy()
		# Parse date
		if 'last_purchase_date' not in data.columns:
			return None
		data['last_purchase_date'] = pd.to_datetime(data['last_purchase_date'], errors='coerce')
		data = data.dropna(subset=['last_purchase_date'])

		# Compute revenue if not present
		if 'total_revenue' in data.columns:
			revenue = pd.to_numeric(data['total_revenue'], errors='coerce').fillna(0)
		else:
			unit_price = pd.to_numeric(data.get('unit_price', 0), errors='coerce').fillna(0)
			qty = pd.to_numeric(data.get('quantity', 0), errors='coerce').fillna(0)
			revenue = unit_price * qty

		data['_revenue'] = revenue

		# Aggregate per day
		date_key = data['last_purchase_date'].dt.date
		grouped = data.groupby(date_key).agg(
			revenue=('_revenue', 'sum'),
			quantity_sold=('quantity', 'sum') if 'quantity' in data.columns else ('_revenue', 'size'),
			orders_count=('order_id', 'nunique') if 'order_id' in data.columns else ('_revenue', 'size'),
		).reset_index(names='date')

		grouped['date'] = pd.to_datetime(grouped['date'])
		grouped = grouped.sort_values('date')

		# Fill missing dates to create a continuous series
		if not grouped.empty:
			full_range = pd.date_range(grouped['date'].min(), grouped['date'].max(), freq='D')
			grouped = grouped.set_index('date').reindex(full_range).fillna(0.0).rename_axis('date').reset_index()

		return grouped

	def _evaluate_backtest(self, y: np.ndarray) -> dict[str, _ModelResult]:
		# simple backtest on last 30 days (if available)
		n = len(y)
		test_size = min(30, max(1, n // 5))
		if n <= test_size + 7:
			# Too short for reliable metrics; fallback to naive values
			return {
				'moving_average': _ModelResult(mae=float('nan'), mse=float('nan'), r2=float('nan')),
				'seasonal_dow': _ModelResult(mae=float('nan'), mse=float('nan'), r2=float('nan')),
				'linear_trend': _ModelResult(mae=float('nan'), mse=float('nan'), r2=float('nan')),
			}

		train = y[:-test_size]
		test = y[-test_size:]

		# 1) Moving average
		w = min(self._recent_avg_window, len(train))
		ma_val = train[-w:].mean() if w > 0 else train.mean()
		ma_pred = np.full_like(test, fill_value=ma_val, dtype=float)

		# 2) Seasonal by day-of-week (approximate using cycle of 7)
		# Build seasonal factors from train using modulo-7 buckets
		idx = np.arange(len(train))
		buckets = {k: train[idx % 7 == k].mean() if (idx % 7 == k).any() else train.mean() for k in range(7)}
		start_mod = len(train) % 7
		seasonal_pred = np.array([buckets[(start_mod + i) % 7] for i in range(len(test))], dtype=float)

		# 3) Linear trend
		X = np.arange(len(train)).reshape(-1, 1)
		lr = LinearRegression()
		lr.fit(X, train)
		X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
		lr_pred = lr.predict(X_test)

		def _metrics(y_true, y_hat):
			return _ModelResult(
				mae=mean_absolute_error(y_true, y_hat),
				mse=mean_squared_error(y_true, y_hat),
				r2=r2_score(y_true, y_hat),
			)

		return {
			'moving_average': _metrics(test, ma_pred),
			'seasonal_dow': _metrics(test, seasonal_pred),
			'linear_trend': _metrics(test, lr_pred),
		}

	def train_forecasting_models(self, sales_data: pd.DataFrame) -> dict:
		"""Train/evaluate a few simple models on revenue and select the best."""
		if sales_data is None or sales_data.empty:
			return {}

		self._train_df = sales_data.copy()
		y = sales_data['revenue'].astype(float).values
		self._last_date = sales_data['date'].max()

		# Store seasonal by DOW on full data for later forecasting
		# Map each date to dow bucket 0..6 using actual weekday
		dow = sales_data['date'].dt.weekday
		self._seasonal_by_dow = sales_data.groupby(dow)['revenue'].mean().to_dict()

		# Fit linear trend on full data for future forecasting
		X_full = np.arange(len(y)).reshape(-1, 1)
		self._linreg = LinearRegression().fit(X_full, y)

		results = self._evaluate_backtest(y)

		# choose best by highest r2 (fallback to moving average)
		self.best_model_name = max(results.keys(), key=lambda k: (results[k].r2 if not np.isnan(results[k].r2) else -np.inf))
		if results[self.best_model_name].r2 == -np.inf or np.isnan(results[self.best_model_name].r2):
			self.best_model_name = 'moving_average'

		# Return as primitive dict for app display
		return {k: {'mae': v.mae, 'mse': v.mse, 'r2': v.r2} for k, v in results.items()}

	def forecast_future_sales(self, days_ahead: int, target_metric: str = 'revenue') -> pd.DataFrame:
		if self._train_df is None or self._train_df.empty:
			raise ValueError("Model not trained. Call train_forecasting_models() first.")

		df = self._train_df
		if target_metric not in {'revenue', 'quantity_sold', 'orders_count'}:
			target_metric = 'revenue'

		# Build series for selected metric
		series = df[target_metric].astype(float).values
		last_date = self._last_date
		future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead, freq='D')

		# Generate predictions using the chosen best model (recomputed on selected metric)
		preds = None
		if self.best_model_name == 'moving_average':
			w = min(self._recent_avg_window, len(series))
			avg_val = series[-w:].mean() if w > 0 else series.mean()
			preds = np.full(shape=days_ahead, fill_value=max(0.0, avg_val), dtype=float)
		elif self.best_model_name == 'seasonal_dow':
			# Compute seasonal avg by actual weekday for this metric
			dow = df['date'].dt.weekday
			seasonal = df.groupby(dow)[target_metric].mean().to_dict()
			start_dow = (last_date.weekday() + 1) % 7
			preds = np.array([float(seasonal.get((start_dow + i) % 7, series[-14:].mean())) for i in range(days_ahead)], dtype=float)
		elif self.best_model_name == 'linear_trend':
			X_full = np.arange(len(series)).reshape(-1, 1)
			lr = LinearRegression().fit(X_full, series)
			X_fut = np.arange(len(series), len(series) + days_ahead).reshape(-1, 1)
			preds = lr.predict(X_fut)
			preds = np.clip(preds, a_min=0.0, a_max=None)
		else:
			# Fallback
			avg_val = series[-14:].mean() if len(series) >= 14 else series.mean()
			preds = np.full(shape=days_ahead, fill_value=max(0.0, avg_val), dtype=float)

		return pd.DataFrame({
			'date': future_dates,
			f'predicted_{target_metric}': preds
		})

	def analyze_sales_trends(self, sales_data: pd.DataFrame) -> dict:
		if sales_data is None or sales_data.empty:
			return {}
		df = sales_data.copy()
		df['month'] = df['date'].dt.month
		df['day_of_week'] = df['date'].dt.day_name()

		return {
			'seasonal_patterns': {
				'by_month': df.groupby('month')['revenue'].mean().to_dict(),
				'by_day_of_week': df.groupby('day_of_week')['revenue'].mean().to_dict(),
			}
		}


# Example usage
if __name__ == "__main__":
	print("AdvancedSalesForecaster ready.")

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
		self._detected_cols: dict | None = None

	# -------------------- Helpers for adaptive schemas --------------------
	def _first_present_col(self, df: pd.DataFrame, candidates: list[str]) -> str | None:
		lower_map = {c.lower(): c for c in df.columns}
		for cand in candidates:
			if cand.lower() in lower_map:
				return lower_map[cand.lower()]
		return None

	def _to_numeric(self, s: pd.Series, clip_min=None, clip_max=None) -> pd.Series:
		s = pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)
		if clip_min is not None or clip_max is not None:
			s = np.clip(s, a_min=clip_min if clip_min is not None else -np.inf,
						a_max=clip_max if clip_max is not None else np.inf)
		return s

	def _ensure_datetime(self, s: pd.Series) -> pd.Series:
		return pd.to_datetime(s, errors='coerce')

	def prepare_sales_data(self, df: pd.DataFrame) -> pd.DataFrame | None:
		if df is None or df.empty:
			return None

		data = df.copy()
		# Detect date/quantity/revenue/order columns
		date_col = self._first_present_col(data, ['last_purchase_date', 'order_date', 'invoice_date', 'date', 'purchase_date', 'transaction_date'])
		if date_col is None:
			return None
		data[date_col] = self._ensure_datetime(data[date_col])
		data = data.dropna(subset=[date_col])

		# Compute revenue if not present
		revenue_col = self._first_present_col(data, ['total_revenue', 'revenue', 'amount', 'sales'])
		if revenue_col is not None:
			revenue = self._to_numeric(data[revenue_col], 0, 1e12)
		else:
			unit_price_col = self._first_present_col(data, ['unit_price', 'price'])
			qty_col = self._first_present_col(data, ['quantity', 'qty', 'units'])
			unit_price = self._to_numeric(data[unit_price_col], 0, 1e6) if unit_price_col else 0
			qty = self._to_numeric(data[qty_col], 0, 1e6) if qty_col else 0
			revenue = unit_price * qty

		data['_revenue'] = revenue

		# Aggregate per day
		date_key = data[date_col].dt.date
		# Quantity and order fields (fallbacks)
		qty_col = self._first_present_col(data, ['quantity', 'qty', 'units'])
		order_id_col = self._first_present_col(data, ['order_id', 'invoice_no', 'transaction_id', 'orderid'])
		grouped = data.groupby(date_key).agg(
			revenue=('_revenue', 'sum'),
			quantity_sold=(qty_col, 'sum') if qty_col else ('_revenue', 'size'),
			orders_count=(order_id_col, 'nunique') if order_id_col else ('_revenue', 'size'),
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

	def forecast_future_sales(self, days_ahead: int, target_metric: str = 'revenue', confidence: int | float | None = None) -> pd.DataFrame:
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
		residual_sigma = None
		if self.best_model_name == 'moving_average':
			w = min(self._recent_avg_window, len(series))
			avg_val = series[-w:].mean() if w > 0 else series.mean()
			preds = np.full(shape=days_ahead, fill_value=max(0.0, avg_val), dtype=float)
			# residuals relative to recent average
			if w > 1:
				residuals = series[-w:] - avg_val
				residual_sigma = float(np.nanstd(residuals, ddof=1)) if len(residuals) > 1 else 0.0
		elif self.best_model_name == 'seasonal_dow':
			# Compute seasonal avg by actual weekday for this metric
			dow = df['date'].dt.weekday
			seasonal = df.groupby(dow)[target_metric].mean().to_dict()
			start_dow = (last_date.weekday() + 1) % 7
			preds = np.array([float(seasonal.get((start_dow + i) % 7, series[-14:].mean())) for i in range(days_ahead)], dtype=float)
			# residuals against seasonal means for training period
			train_pred = np.array([float(seasonal.get(int(d), series.mean())) for d in dow], dtype=float)
			residuals = series - train_pred
			residual_sigma = float(np.nanstd(residuals, ddof=1)) if len(residuals) > 1 else 0.0
		elif self.best_model_name == 'linear_trend':
			X_full = np.arange(len(series)).reshape(-1, 1)
			lr = LinearRegression().fit(X_full, series)
			X_fut = np.arange(len(series), len(series) + days_ahead).reshape(-1, 1)
			preds = lr.predict(X_fut)
			preds = np.clip(preds, a_min=0.0, a_max=None)
			# in-sample residuals
			train_fit = lr.predict(X_full)
			residuals = series - train_fit
			residual_sigma = float(np.nanstd(residuals, ddof=1)) if len(residuals) > 1 else 0.0
		else:
			# Fallback
			avg_val = series[-14:].mean() if len(series) >= 14 else series.mean()
			preds = np.full(shape=days_ahead, fill_value=max(0.0, avg_val), dtype=float)
			residual_sigma = float(np.nanstd(series - avg_val, ddof=1)) if len(series) > 1 else 0.0

		out = pd.DataFrame({
			'date': future_dates,
			f'predicted_{target_metric}': preds
		})

		# Optional confidence interval
		if confidence is not None and residual_sigma is not None and residual_sigma > 0:
			# Map common confidence levels to z-scores; default to nearest key if not exact
			z_map = {80: 1.282, 85: 1.440, 90: 1.645, 95: 1.960, 98: 2.326, 99: 2.576}
			try:
				c = float(confidence)
				nc = int(round(c))
				z = z_map.get(nc)
				if z is None:
					# clamp to available keys
					closest = min(z_map.keys(), key=lambda k: abs(k - nc))
					z = z_map[closest]
			except Exception:
				z = 1.645  # default ~90%
			lower = preds - z * residual_sigma
			upper = preds + z * residual_sigma
			# clip lower bound at 0 for non-negative metrics
			out['yhat_lower'] = np.clip(lower, a_min=0.0, a_max=None)
			out['yhat_upper'] = np.clip(upper, a_min=0.0, a_max=None)

		return out

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

	# -------------------- Quarterly forecasting --------------------
	def prepare_quarterly_data(self, df: pd.DataFrame) -> pd.DataFrame | None:
		"""Aggregate input data into quarterly revenue time series with columns [ds, y].

		Accepts raw transactional dataframe or daily-aggregated output of prepare_sales_data.
		"""
		if df is None or df.empty:
			return None

		# If already looks like daily aggregated output
		if {'date', 'revenue'}.issubset(df.columns):
			daily = df[['date', 'revenue']].copy()
			if not np.issubdtype(daily['date'].dtype, np.datetime64):
				daily['date'] = pd.to_datetime(daily['date'], errors='coerce')
			daily = daily.dropna(subset=['date'])
		else:
			# Build from raw using existing helper
			daily = self.prepare_sales_data(df)
			if daily is None or daily.empty:
				return None

		q = (
			daily.set_index('date')['revenue']
			.resample('Q').sum()
			.reset_index()
			.rename(columns={'date': 'ds', 'revenue': 'y'})
		)
		q = q.sort_values('ds').reset_index(drop=True)
		return q

	def forecast_quarterly_sales(self, df: pd.DataFrame, periods: int = 4, inventory_cap: float = 0.0) -> dict:
		"""Forecast next `periods` quarters of revenue.

		Returns dict with keys:
		- historical: DataFrame [ds, y]
		- forecast: DataFrame [ds, yhat, yhat_capped]
		"""
		q = self.prepare_quarterly_data(df)
		if q is None or len(q) < 4:
			raise ValueError('Need at least 4 quarters of data to forecast.')

		last_known = pd.to_datetime(q['ds'].max())
		forecast_df = None

		# Try Prophet if available
		try:
			from prophet import Prophet  # type: ignore
			m = Prophet(interval_width=0.90, yearly_seasonality=True)
			m.add_seasonality(name='quarterly', period=91.25, fourier_order=6)
			m.fit(q[['ds', 'y']])
			future = m.make_future_dataframe(periods=periods, freq='Q')
			pred_full = m.predict(future)[['ds', 'yhat']]
			forecast_df = pred_full[pred_full['ds'] > last_known].reset_index(drop=True)
		except Exception:
			# Fallback: simple linear trend on quarterly index
			y = q['y'].astype(float).values
			X = np.arange(len(y)).reshape(-1, 1)
			lr = LinearRegression().fit(X, y)
			X_fut = np.arange(len(y), len(y) + periods).reshape(-1, 1)
			yhat = lr.predict(X_fut).clip(min=0.0)
			future_ds = pd.date_range(start=last_known + pd.offsets.QuarterEnd(), periods=periods, freq='Q')
			forecast_df = pd.DataFrame({'ds': future_ds, 'yhat': yhat})

		if inventory_cap and inventory_cap > 0:
			forecast_df['yhat_capped'] = forecast_df['yhat'].clip(upper=inventory_cap)
		else:
			forecast_df['yhat_capped'] = forecast_df['yhat']

		return {'historical': q, 'forecast': forecast_df}


# Example usage
if __name__ == "__main__":
	print("AdvancedSalesForecaster ready.")

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

class MetricsCalculator:
    """Class for calculating and displaying various business and model metrics"""
    
    def __init__(self):
        pass
    
    def calculate_churn_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive churn prediction metrics"""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
        
        # AUC score if probabilities are provided
        if y_pred_proba is not None:
            metrics['auc_score'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    def calculate_sales_metrics(self, y_true, y_pred):
        """Calculate sales forecasting metrics"""
        metrics = {}
        
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2_score'] = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        eps = 1e-8
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.abs((y_true - y_pred) / np.where(np.abs(y_true) < eps, eps, y_true))
            mape = mape[~np.isnan(mape)]
            metrics['mape'] = float(np.mean(mape) * 100) if mape.size > 0 else np.nan
        
        # Mean Directional Accuracy
        if len(y_true) >= 2 and len(y_pred) >= 2:
            direction_true = np.diff(y_true) > 0
            direction_pred = np.diff(y_pred) > 0
            metrics['directional_accuracy'] = float(np.mean(direction_true == direction_pred) * 100)
        else:
            metrics['directional_accuracy'] = np.nan
        
        return metrics
    
    def calculate_business_metrics(self, df):
        """Calculate key business metrics"""
        metrics = {}
        
        # Customer metrics
        if 'customer_id' in df.columns:
            metrics['total_customers'] = df['customer_id'].nunique()
            
        # Revenue metrics
        if 'total_revenue' in df.columns:
            metrics['total_revenue'] = df['total_revenue'].sum()
            metrics['avg_revenue_per_customer'] = df.groupby('customer_id')['total_revenue'].sum().mean() if 'customer_id' in df.columns else df['total_revenue'].mean()
            metrics['revenue_std'] = df['total_revenue'].std()
        
        # Churn metrics
        if 'is_churned' in df.columns:
            metrics['churn_rate'] = df['is_churned'].mean()
            metrics['churned_customers'] = df['is_churned'].sum()
            metrics['active_customers'] = len(df) - metrics['churned_customers']
            
            # Revenue at risk
            if 'total_revenue' in df.columns:
                metrics['revenue_at_risk'] = df[df['is_churned'] == 1]['total_revenue'].sum()
                metrics['revenue_at_risk_percentage'] = (metrics['revenue_at_risk'] / metrics['total_revenue']) * 100
        
        # Purchase behavior metrics
        if 'purchase_frequency' in df.columns:
            metrics['avg_purchase_frequency'] = df['purchase_frequency'].mean()
            metrics['purchase_frequency_std'] = df['purchase_frequency'].std()
        
        if 'days_since_last_purchase' in df.columns:
            metrics['avg_days_since_purchase'] = df['days_since_last_purchase'].mean()
            metrics['customers_inactive_30_days'] = (df['days_since_last_purchase'] > 30).sum()
            metrics['customers_inactive_90_days'] = (df['days_since_last_purchase'] > 90).sum()
        
        # Customer lifetime metrics
        if 'customer_lifetime_days' in df.columns:
            metrics['avg_customer_lifetime'] = df['customer_lifetime_days'].mean()
            metrics['customer_lifetime_std'] = df['customer_lifetime_days'].std()
        
        # Product metrics
        if 'category' in df.columns:
            metrics['total_categories'] = df['category'].nunique()
            metrics['top_category'] = df['category'].value_counts().index[0]
            metrics['top_category_percentage'] = (df['category'].value_counts().iloc[0] / len(df)) * 100
        
        if 'product_name' in df.columns:
            metrics['total_products'] = df['product_name'].nunique()
        
        # Geographic metrics
        if 'country' in df.columns:
            metrics['total_countries'] = df['country'].nunique()
            metrics['top_country'] = df['country'].value_counts().index[0]
            metrics['top_country_percentage'] = (df['country'].value_counts().iloc[0] / len(df)) * 100
        
        return metrics
    
    def calculate_segment_metrics(self, segmented_df, segment_profiles):
        """Calculate metrics for customer segments"""
        segment_metrics = {}
        
        for segment_id in segmented_df['cluster'].unique():
            segment_data = segmented_df[segmented_df['cluster'] == segment_id]
            segment_name = segment_profiles.get(f'Segment_{segment_id}', {}).get('profile_name', f'Segment {segment_id}')
            
            metrics = {}
            metrics['size'] = len(segment_data)
            metrics['percentage'] = (len(segment_data) / len(segmented_df)) * 100
            
            # Revenue metrics
            if 'total_revenue' in segment_data.columns:
                metrics['total_revenue'] = segment_data['total_revenue'].sum()
                metrics['avg_revenue'] = segment_data['total_revenue'].mean()
                metrics['revenue_percentage'] = (metrics['total_revenue'] / segmented_df['total_revenue'].sum()) * 100
            
            # Churn metrics
            if 'is_churned' in segment_data.columns:
                metrics['churn_rate'] = segment_data['is_churned'].mean()
                metrics['churned_customers'] = segment_data['is_churned'].sum()
            
            # Behavioral metrics
            if 'purchase_frequency' in segment_data.columns:
                metrics['avg_purchase_frequency'] = segment_data['purchase_frequency'].mean()
            
            if 'days_since_last_purchase' in segment_data.columns:
                metrics['avg_recency'] = segment_data['days_since_last_purchase'].mean()
            
            if 'customer_lifetime_days' in segment_data.columns:
                metrics['avg_lifetime_days'] = segment_data['customer_lifetime_days'].mean()
                metrics['lifetime_std'] = segment_data['customer_lifetime_days'].std()

            # Spending/ratings if present
            if 'total_spent' in segment_data.columns:
                metrics['avg_total_spent'] = segment_data['total_spent'].mean()
            if 'Ratings' in segment_data.columns:
                metrics['avg_rating'] = segment_data['Ratings'].mean()

            # Top categories/products/countries
            if 'category' in segment_data.columns and not segment_data['category'].empty:
                vc = segment_data['category'].value_counts(normalize=True)
                top_cat = vc.index[0]
                metrics['top_category'] = top_cat
                metrics['top_category_share'] = float(vc.iloc[0] * 100)
            if 'product_name' in segment_data.columns and not segment_data['product_name'].empty:
                vp = segment_data['product_name'].value_counts(normalize=True)
                metrics['top_product'] = vp.index[0]
                metrics['top_product_share'] = float(vp.iloc[0] * 100)
            if 'country' in segment_data.columns and not segment_data['country'].empty:
                vco = segment_data['country'].value_counts(normalize=True)
                metrics['top_country'] = vco.index[0]
                metrics['top_country_share'] = float(vco.iloc[0] * 100)

            segment_metrics[segment_name] = metrics

        return segment_metrics

    # ---------- Optional helpers for Streamlit rendering ----------
    def display_metrics(self, metrics: dict, title: str | None = None, columns: int = 3):
        """Render a dictionary of metrics as Streamlit metric widgets.

        Args:
            metrics: dict of name -> value
            title: optional section title
            columns: number of columns to layout
        """
        if title:
            st.subheader(title)
        if not metrics:
            st.info("No metrics to display.")
            return

        keys = list(metrics.keys())
        cols = st.columns(columns)
        for i, key in enumerate(keys):
            val = metrics[key]
            col = cols[i % columns]
            # Format floats nicely
            if isinstance(val, float):
                if 'rate' in key or 'percentage' in key or 'share' in key:
                    text = f"{val:.2f}%"
                elif key in {"accuracy", "precision", "recall", "f1_score"}:
                    text = f"{val:.3f}"
                else:
                    text = f"{val:,.3f}"
            else:
                text = f"{val}"
            with col:
                st.metric(key.replace('_', ' ').title(), text)

    def display_segment_metrics(self, segment_metrics: dict):
        """Render segment-level metrics in an expandable table format."""
        if not segment_metrics:
            st.info("No segment metrics available.")
            return
        rows = []
        for seg, m in segment_metrics.items():
            row = {"segment": seg}
            row.update(m)
            rows.append(row)
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

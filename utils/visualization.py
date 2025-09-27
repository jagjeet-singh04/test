import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

class InteractiveCharts:
    """Class for creating interactive charts and visualizations"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set1
        self.theme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd'
        }
    
    def create_churn_dashboard(self, df, churn_predictions=None):
        """Create comprehensive churn analysis dashboard"""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Churn Rate by Age Group',
                'Churn Rate by Country', 
                'Churn Rate by Category',
                'Revenue at Risk',
                'Customer Lifetime vs Churn',
                'Churn Probability Distribution'
            ],
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "scatter"}, {"type": "histogram"}]]
        )
        
        # Age group analysis
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], 
                                   labels=['18-25', '26-35', '36-45', '46-55', '55+'])
            age_churn = df.groupby('age_group')['is_churned'].agg(['count', 'sum', 'mean']).reset_index()
            
            fig.add_trace(
                go.Bar(x=age_churn['age_group'], y=age_churn['mean'], 
                      name='Churn Rate', marker_color=self.theme['primary']),
                row=1, col=1
            )
        
        # Country analysis
        if 'country' in df.columns:
            country_churn = df.groupby('country')['is_churned'].mean().nlargest(10).reset_index()
            
            fig.add_trace(
                go.Bar(x=country_churn['country'], y=country_churn['is_churned'],
                      name='Churn Rate', marker_color=self.theme['secondary']),
                row=1, col=2
            )
        
        # Category analysis
        if 'category' in df.columns:
            category_churn = df.groupby('category')['is_churned'].mean().reset_index()
            
            fig.add_trace(
                go.Bar(x=category_churn['category'], y=category_churn['is_churned'],
                      name='Churn Rate', marker_color=self.theme['success']),
                row=1, col=3
            )
        
        # Revenue at risk
        if 'total_revenue' in df.columns:
            churned_revenue = df[df['is_churned'] == 1]['total_revenue'].sum()
            active_revenue = df[df['is_churned'] == 0]['total_revenue'].sum()
            
            fig.add_trace(
                go.Pie(labels=['At Risk', 'Safe'], values=[churned_revenue, active_revenue],
                      marker_colors=[self.theme['warning'], self.theme['success']]),
                row=2, col=1
            )
        
        # Customer lifetime vs churn
        if 'customer_lifetime_days' in df.columns and 'total_revenue' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['customer_lifetime_days'], 
                    y=df['total_revenue'],
                    mode='markers',
                    marker=dict(
                        color=df['is_churned'],
                        colorscale=[[0, self.theme['success']], [1, self.theme['warning']]],
                        size=8,
                        opacity=0.6
                    ),
                    name='Customers'
                ),
                row=2, col=2
            )
        
        # Churn probability distribution
        if churn_predictions is not None:
            fig.add_trace(
                go.Histogram(x=churn_predictions, nbinsx=30, 
                           marker_color=self.theme['info'], opacity=0.7),
                row=2, col=3
            )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Customer Churn Analysis Dashboard",
            title_x=0.5
        )
        
        return fig
    
    def create_sales_trend_chart(self, sales_data, forecast_data=None):
        """Create interactive sales trend chart with forecast"""
        fig = go.Figure()
        
        # Historical sales
        fig.add_trace(go.Scatter(
            x=sales_data['date'],
            y=sales_data['revenue'],
            mode='lines+markers',
            name='Historical Sales',
            line=dict(color=self.theme['primary'], width=2),
            marker=dict(size=4),
            hovertemplate='<b>Date:</b> %{x}<br><b>Revenue:</b> $%{y:,.2f}<extra></extra>'
        ))
        
        # Add forecast if provided
        if forecast_data is not None:
            fig.add_trace(go.Scatter(
                x=forecast_data['date'],
                y=forecast_data['predicted_revenue'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color=self.theme['warning'], width=2, dash='dash'),
                marker=dict(size=4),
                hovertemplate='<b>Date:</b> %{x}<br><b>Forecast:</b> $%{y:,.2f}<extra></extra>'
            ))
            
            # Add confidence interval if available
            if 'confidence_lower' in forecast_data.columns and 'confidence_upper' in forecast_data.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_data['date'].tolist() + forecast_data['date'].tolist()[::-1],
                    y=forecast_data['confidence_upper'].tolist() + forecast_data['confidence_lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255, 127, 14, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval',
                    showlegend=True
                ))
        
        # Add moving average
        if len(sales_data) > 7:
            sales_data['ma_7'] = sales_data['revenue'].rolling(window=7).mean()
            fig.add_trace(go.Scatter(
                x=sales_data['date'],
                y=sales_data['ma_7'],
                mode='lines',
                name='7-Day Moving Average',
                line=dict(color=self.theme['success'], width=1, dash='dot'),
                opacity=0.8
            ))
        
        fig.update_layout(
            title='Sales Trend Analysis',
            xaxis_title='Date',
            yaxis_title='Revenue ($)',
            hovermode='x unified',
            height=500,
            showlegend=True,
            legend=dict(x=0.02, y=0.98)
        )
        
        return fig
    
    def create_customer_segmentation_chart(self, segmented_df, segment_profiles):
        """Create interactive customer segmentation visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Customer Segments (PCA)',
                'Segment Size Distribution',
                'Revenue by Segment',
                'Churn Rate by Segment'
            ],
            specs=[[{"type": "scatter"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # PCA scatter plot
        for cluster in segmented_df['cluster'].unique():
            cluster_data = segmented_df[segmented_df['cluster'] == cluster]
            profile_name = segment_profiles.get(f'Segment_{cluster}', {}).get('profile_name', f'Segment {cluster}')
            
            fig.add_trace(
                go.Scatter(
                    x=cluster_data['pca_1'],
                    y=cluster_data['pca_2'],
                    mode='markers',
                    name=profile_name,
                    marker=dict(
                        size=8,
                        color=self.color_palette[cluster % len(self.color_palette)],
                        opacity=0.7
                    ),
                    hovertemplate=f'<b>{profile_name}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Segment size pie chart
        segment_sizes = segmented_df['cluster'].value_counts()
        segment_labels = [segment_profiles.get(f'Segment_{i}', {}).get('profile_name', f'Segment {i}') 
                         for i in segment_sizes.index]
        
        fig.add_trace(
            go.Pie(
                labels=segment_labels,
                values=segment_sizes.values,
                marker_colors=self.color_palette[:len(segment_sizes)]
            ),
            row=1, col=2
        )
        
        # Revenue by segment
        if 'total_revenue' in segmented_df.columns:
            revenue_by_segment = segmented_df.groupby('cluster')['total_revenue'].sum()
            segment_labels_rev = [segment_profiles.get(f'Segment_{i}', {}).get('profile_name', f'Segment {i}') 
                                 for i in revenue_by_segment.index]
            
            fig.add_trace(
                go.Bar(
                    x=segment_labels_rev,
                    y=revenue_by_segment.values,
                    marker_color=self.color_palette[:len(revenue_by_segment)],
                    name='Revenue'
                ),
                row=2, col=1
            )
        
        # Churn rate by segment
        if 'is_churned' in segmented_df.columns:
            churn_by_segment = segmented_df.groupby('cluster')['is_churned'].mean()
            segment_labels_churn = [segment_profiles.get(f'Segment_{i}', {}).get('profile_name', f'Segment {i}') 
                                   for i in churn_by_segment.index]
            
            fig.add_trace(
                go.Bar(
                    x=segment_labels_churn,
                    y=churn_by_segment.values,
                    marker_color=self.color_palette[:len(churn_by_segment)],
                    name='Churn Rate'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Customer Segmentation Analysis",
            title_x=0.5
        )
        
        return fig
    
    def create_demand_forecast_chart(self, historical_data, forecast_data, product_name):
        """Create demand forecasting chart for a specific product"""
        fig = go.Figure()
        
        # Historical demand
        fig.add_trace(go.Scatter(
            x=historical_data['date'],
            y=historical_data['quantity_demanded'],
            mode='lines+markers',
            name='Historical Demand',
            line=dict(color=self.theme['primary'], width=2),
            marker=dict(size=4),
            hovertemplate='<b>Date:</b> %{x}<br><b>Demand:</b> %{y}<extra></extra>'
        ))
        
        # Forecasted demand
        fig.add_trace(go.Scatter(
            x=forecast_data['date'],
            y=forecast_data['predicted_demand'],
            mode='lines+markers',
            name='Forecasted Demand',
            line=dict(color=self.theme['warning'], width=2, dash='dash'),
            marker=dict(size=4),
            hovertemplate='<b>Date:</b> %{x}<br><b>Forecast:</b> %{y}<extra></extra>'
        ))
        
        # Add different forecasting methods if available
        methods = ['moving_average', 'exponential_smoothing', 'seasonal', 'trend_adjusted']
        colors = [self.theme['success'], self.theme['info'], '#e377c2', '#8c564b']
        
        for i, method in enumerate(methods):
            if method in forecast_data.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_data['date'],
                    y=forecast_data[method],
                    mode='lines',
                    name=method.replace('_', ' ').title(),
                    line=dict(color=colors[i], width=1, dash='dot'),
                    opacity=0.6,
                    visible='legendonly'  # Hidden by default
                ))
        
        fig.update_layout(
            title=f'Demand Forecast for {product_name}',
            xaxis_title='Date',
            yaxis_title='Quantity Demanded',
            hovermode='x unified',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_cohort_analysis_chart(self, df):
        """Create cohort analysis chart for customer retention"""
        if not all(col in df.columns for col in ['customer_id', 'signup_date', 'last_purchase_date']):
            return None
        
        # Prepare cohort data
        df['signup_month'] = df['signup_date'].dt.to_period('M')
        df['purchase_month'] = df['last_purchase_date'].dt.to_period('M')
        
        # Calculate period number
        df['period_number'] = (df['purchase_month'] - df['signup_month']).apply(attrgetter('n'))
        
        # Create cohort table
        cohort_data = df.groupby(['signup_month', 'period_number'])['customer_id'].nunique().reset_index()
        cohort_sizes = df.groupby('signup_month')['customer_id'].nunique()
        
        cohort_table = cohort_data.pivot(index='signup_month', columns='period_number', values='customer_id')
        
        # Calculate retention rates
        cohort_table = cohort_table.divide(cohort_sizes, axis=0)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cohort_table.values,
            x=cohort_table.columns,
            y=[str(idx) for idx in cohort_table.index],
            colorscale='RdYlBu_r',
            hovertemplate='<b>Cohort:</b> %{y}<br><b>Period:</b> %{x}<br><b>Retention:</b> %{z:.1%}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Customer Cohort Analysis',
            xaxis_title='Period Number',
            yaxis_title='Signup Month',
            height=500
        )
        
        return fig
    
    def create_feature_importance_chart(self, feature_importance_df):
        """Create interactive feature importance chart"""
        fig = go.Figure(go.Bar(
            x=feature_importance_df['importance'],
            y=feature_importance_df['feature'],
            orientation='h',
            marker=dict(
                color=feature_importance_df['importance'],
                colorscale='Viridis',
                showscale=True
            ),
            hovertemplate='<b>Feature:</b> %{y}<br><b>Importance:</b> %{x:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Feature Importance for Churn Prediction',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=max(400, len(feature_importance_df) * 25),
            yaxis=dict(autorange='reversed')
        )
        
        return fig
    
    def create_roc_curve_chart(self, y_true, y_pred_proba):
        """Create ROC curve chart"""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color=self.theme['primary'], width=2)
        ))
        
        # Random classifier line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_seasonal_decomposition_chart(self, sales_data):
        """Create seasonal decomposition chart"""
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Prepare data
        ts_data = sales_data.set_index('date')['revenue'].resample('D').sum().fillna(0)
        
        if len(ts_data) < 30:  # Need sufficient data
            return None
        
        # Perform decomposition
        decomposition = seasonal_decompose(ts_data, model='additive', period=7)
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'],
            vertical_spacing=0.08
        )
        
        # Original
        fig.add_trace(go.Scatter(
            x=ts_data.index, y=ts_data.values,
            mode='lines', name='Original',
            line=dict(color=self.theme['primary'])
        ), row=1, col=1)
        
        # Trend
        fig.add_trace(go.Scatter(
            x=decomposition.trend.index, y=decomposition.trend.values,
            mode='lines', name='Trend',
            line=dict(color=self.theme['success'])
        ), row=2, col=1)
        
        # Seasonal
        fig.add_trace(go.Scatter(
            x=decomposition.seasonal.index, y=decomposition.seasonal.values,
            mode='lines', name='Seasonal',
            line=dict(color=self.theme['warning'])
        ), row=3, col=1)
        
        # Residual
        fig.add_trace(go.Scatter(
            x=decomposition.resid.index, y=decomposition.resid.values,
            mode='lines', name='Residual',
            line=dict(color=self.theme['info'])
        ), row=4, col=1)
        
        fig.update_layout(
            height=800,
            title_text="Sales Seasonal Decomposition",
            showlegend=False
        )
        
        return fig

def create_interactive_filter_widget(df, column_name, widget_type='multiselect'):
    """Create interactive filter widgets"""
    if column_name not in df.columns:
        return None
    
    unique_values = df[column_name].unique()
    
    if widget_type == 'multiselect':
        return st.multiselect(
            f"Filter by {column_name.title()}",
            options=unique_values,
            default=unique_values[:min(5, len(unique_values))]
        )
    elif widget_type == 'selectbox':
        return st.selectbox(
            f"Select {column_name.title()}",
            options=['All'] + list(unique_values)
        )
    elif widget_type == 'slider' and df[column_name].dtype in ['int64', 'float64']:
        min_val, max_val = df[column_name].min(), df[column_name].max()
        return st.slider(
            f"Filter by {column_name.title()}",
            min_value=float(min_val),
            max_value=float(max_val),
            value=(float(min_val), float(max_val))
        )
    
    return None

def apply_filters(df, filters):
    """Apply multiple filters to dataframe"""
    filtered_df = df.copy()
    
    for column, values in filters.items():
        if column in df.columns and values:
            if isinstance(values, (list, tuple)) and len(values) == 2 and all(isinstance(v, (int, float)) for v in values):
                # Range filter
                filtered_df = filtered_df[(filtered_df[column] >= values[0]) & (filtered_df[column] <= values[1])]
            elif isinstance(values, list):
                # Multiple selection filter
                filtered_df = filtered_df[filtered_df[column].isin(values)]
            elif values != 'All':
                # Single selection filter
                filtered_df = filtered_df[filtered_df[column] == values]
    
    return filtered_df

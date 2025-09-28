import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import io
import sys
import os

# Add the current directory to Python path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.data_processor import DataProcessor
from models.churn_analysis import AdvancedChurnAnalyzer
from models.customer_segmentation import CustomerSegmentation
from models.sales_forecasting import AdvancedSalesForecaster
from models.demand_forecasting import DemandForecaster
 

# Page configuration
st.set_page_config(
    page_title="FORESIGHT",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'churn_model' not in st.session_state:
    st.session_state.churn_model = None
if 'sales_model' not in st.session_state:
    st.session_state.sales_model = None

def load_data(uploaded_file):
    """Load and process uploaded data"""
    try:
        processor = DataProcessor()
        
        # Save uploaded file temporarily with its original extension
        original_name = getattr(uploaded_file, 'name', 'temp_data')
        _, ext = os.path.splitext(original_name)
        if not ext:
            ext = '.csv'
        temp_path = f"temp_data{ext}"

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load data using the appropriate reader based on extension
        df = processor.load_data(temp_path)
        
        if df is not None:
            # Clean and preprocess
            df_clean = processor.clean_and_preprocess(df)
            
            # Remove temporary file
            try:
                os.remove(temp_path)
            except Exception:
                pass
            
            return df_clean, processor
        else:
            return None, None
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def display_data_overview(df):
    """Display data overview and statistics"""
    st.markdown('<div class="sub-header">📋 Data Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Records", len(df))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Unique Customers", df['customer_id'].nunique() if 'customer_id' in df.columns else 'N/A')
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        total_revenue = df['total_revenue'].sum() if 'total_revenue' in df.columns else 0
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        churn_rate = df['is_churned'].mean() * 100 if 'is_churned' in df.columns else 0
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Data quality check
    st.markdown("### Data Quality Summary")
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        st.warning(f"Found {missing_data.sum()} missing values across {(missing_data > 0).sum()} columns")
        with st.expander("View Missing Data Details"):
            st.dataframe(missing_data[missing_data > 0])
    else:
        st.success("No missing values found in the dataset!")
    
    # Sample data preview
    st.markdown("### Data Preview")
    st.dataframe(df.head(10))

def churn_analysis_page(df):
    """Churn analysis and prediction page"""
    st.markdown('<div class="sub-header">🎯 Customer Churn Analysis</div>', unsafe_allow_html=True)
    
    # Initialize churn analyzer
    churn_analyzer = AdvancedChurnAnalyzer()
    
    try:
        # Prepare features
        X = churn_analyzer.prepare_churn_features(df)
        y = df['is_churned'] if 'is_churned' in df.columns else None
        
        if y is None:
            st.error("Churn labels not found. Please ensure your data has churn information.")
            return
        
        # Train models
        with st.spinner("Training churn prediction models..."):
            model_results = churn_analyzer.train_and_evaluate_models(X, y)
        
        # Display model performance
        st.markdown("### Model Performance Comparison")
        
        performance_df = pd.DataFrame({
            'Model': list(model_results.keys()),
            'Accuracy': [results['accuracy'] for results in model_results.values()],
            'AUC Score': [results['auc_score'] for results in model_results.values()],
            'CV AUC': [results['cv_mean'] for results in model_results.values()]
        })
        
        st.dataframe(performance_df.round(3))
        
        # Best model info
        st.markdown(f'<div class="success-box">✅ Best Model: <strong>{churn_analyzer.best_model_name}</strong> with AUC Score: <strong>{model_results[churn_analyzer.best_model_name]["auc_score"]:.3f}</strong></div>', unsafe_allow_html=True)
        
        # Feature importance
        st.markdown("### Feature Importance")
        feature_importance = churn_analyzer.get_feature_importance()
        
        if not feature_importance.empty:
            fig_importance = px.bar(
                feature_importance.head(10), 
                x='importance', 
                y='feature',
                orientation='h',
                title="Top 10 Most Important Features for Churn Prediction"
            )
            fig_importance.update_layout(height=400)
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Top at-risk customers
        st.markdown("### High-Risk Customers")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            top_n = st.slider("Number of customers to show", 5, 50, 10)
            high_only_flag = st.checkbox("High risk only", value=True, help="Show only customers classified as High risk")
        
        top_risk_customers = churn_analyzer.get_top_risk_customers(df, X, top_n=top_n, high_only=high_only_flag)
        
        st.dataframe(top_risk_customers)
        
        # Download high-risk customers
        csv_buffer = io.StringIO()
        top_risk_customers.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download High-Risk Customers",
            data=csv_buffer.getvalue(),
            file_name=f"high_risk_customers_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Churn insights
        st.markdown("### Business Insights")
        insights = churn_analyzer.generate_churn_insights(df)
        
        if 'churn_by_category' in insights:
            st.markdown("#### Churn Rate by Product Category")
            churn_category = insights['churn_by_category'].reset_index()
            fig_category = px.bar(
                churn_category, 
                x='category', 
                y='mean',
                title="Churn Rate by Product Category"
            )
            st.plotly_chart(fig_category, use_container_width=True)
        
        if 'revenue_at_risk' in insights:
            revenue_info = insights['revenue_at_risk']
            st.markdown(f'<div class="warning-box">⚠️ Revenue at Risk: <strong>${revenue_info["churned_revenue"]:,.2f}</strong> ({revenue_info["percentage_at_risk"]:.1f}% of total revenue)</div>', unsafe_allow_html=True)
        
        # Store model in session state
        st.session_state.churn_model = churn_analyzer
        
    except Exception as e:
        st.error(f"Error in churn analysis: {str(e)}")

def sales_forecasting_page(df):
    """Sales forecasting page"""
    st.markdown('<div class="sub-header">📈 Sales Forecasting</div>', unsafe_allow_html=True)
    
    # Initialize sales forecaster
    sales_forecaster = AdvancedSalesForecaster()
    
    try:
        # Prepare sales data
        sales_data = sales_forecaster.prepare_sales_data(df)
        
        if sales_data is None or len(sales_data) < 30:
            st.error("Insufficient sales data for forecasting. Need at least 30 days of data.")
            return
        
        # Train forecasting models
        with st.spinner("Training sales forecasting models..."):
            model_results = sales_forecaster.train_forecasting_models(sales_data)
        
        # Display model performance
        st.markdown("### Model Performance")
        
        performance_df = pd.DataFrame({
            'Model': list(model_results.keys()),
            'MAE': [results['mae'] for results in model_results.values()],
            'RMSE': [np.sqrt(results['mse']) for results in model_results.values()],
            'R² Score': [results['r2'] for results in model_results.values()]
        })
        
        st.dataframe(performance_df.round(3))
        
        st.markdown(f'<div class="success-box">✅ Best Model: <strong>{sales_forecaster.best_model_name}</strong> with R² Score: <strong>{model_results[sales_forecaster.best_model_name]["r2"]:.3f}</strong></div>', unsafe_allow_html=True)
        
        # Forecasting parameters
        st.markdown("### Generate Sales Forecast")
        
        col1, col2 = st.columns(2)
        with col1:
            forecast_days = st.slider("Forecast period (days)", 30, 365, 90)
        with col2:
            target_metric = st.selectbox("Forecast metric", ["revenue", "quantity_sold", "orders_count"])
        
        # Generate forecast
        with st.spinner("Generating forecast..."):
            forecast_data = sales_forecaster.forecast_future_sales(forecast_days, target_metric)
        
        # Display forecast
        st.markdown("### Sales Forecast Results")
        
        # Create forecast visualization
        fig = go.Figure()
        
        # Historical data (last 60 days)
        recent_data = sales_data.tail(60)
        fig.add_trace(go.Scatter(
            x=recent_data['date'],
            y=recent_data[target_metric],
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Forecast data
        fig.add_trace(go.Scatter(
            x=forecast_data['date'],
            y=forecast_data[f'predicted_{target_metric}'],
            mode='lines',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f'{target_metric.title()} Forecast',
            xaxis_title='Date',
            yaxis_title=target_metric.title(),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast summary
        total_forecast = forecast_data[f'predicted_{target_metric}'].sum()
        avg_daily = forecast_data[f'predicted_{target_metric}'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"Total Forecast ({forecast_days} days)", f"{total_forecast:,.2f}")
        with col2:
            st.metric("Average Daily", f"{avg_daily:.2f}")
        with col3:
            growth_rate = (avg_daily - recent_data[target_metric].tail(30).mean()) / recent_data[target_metric].tail(30).mean() * 100
            st.metric("Growth Rate", f"{growth_rate:+.1f}%")
        
        # Download forecast
        csv_buffer = io.StringIO()
        forecast_data.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Forecast Data",
            data=csv_buffer.getvalue(),
            file_name=f"sales_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Sales trends analysis
        st.markdown("### Sales Trends Analysis")
        trends = sales_forecaster.analyze_sales_trends(sales_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Seasonal Patterns")
            if 'seasonal_patterns' in trends:
                monthly_pattern = trends['seasonal_patterns']['by_month']
                months = list(monthly_pattern.keys())
                values = list(monthly_pattern.values())
                
                fig_monthly = px.bar(
                    x=months, 
                    y=values,
                    title="Average Revenue by Month"
                )
                st.plotly_chart(fig_monthly, use_container_width=True)
        
        with col2:
            st.markdown("#### Weekly Patterns")
            if 'seasonal_patterns' in trends:
                daily_pattern = trends['seasonal_patterns']['by_day_of_week']
                days = list(daily_pattern.keys())
                values = list(daily_pattern.values())
                
                fig_daily = px.bar(
                    x=days, 
                    y=values,
                    title="Average Revenue by Day of Week"
                )
                st.plotly_chart(fig_daily, use_container_width=True)
        
        # Store model in session state
        st.session_state.sales_model = sales_forecaster
        
    except Exception as e:
        st.error(f"Error in sales forecasting: {str(e)}")

    # --- Quarterly forecast (simple) ---
    st.markdown("---")
    st.markdown("### 📆 Quarterly Forecast (simple)")
    try:
        colq1, colq2 = st.columns(2)
        with colq1:
            q_periods = st.slider("Quarters to forecast", 1, 8, 4)
        with colq2:
            q_cap = st.number_input("Quarterly capacity cap (0 = none)", min_value=0.0, value=0.0, step=1000.0)

        res = sales_forecaster.forecast_quarterly_sales(df, periods=int(q_periods), inventory_cap=float(q_cap))
        q_hist = res['historical']
        q_fc = res['forecast'].copy()
        q_fc['Quarter'] = q_fc['ds'].dt.to_period('Q').astype(str)

        # Chart
        figq = go.Figure()
        figq.add_trace(go.Scatter(x=q_hist['ds'], y=q_hist['y'], mode='lines+markers', name='Historical'))
        figq.add_trace(go.Scatter(x=q_fc['ds'], y=q_fc['yhat_capped'], mode='lines+markers', name='Forecast', line=dict(dash='dash')))
        figq.update_layout(height=420, template='plotly_white', title='Quarterly Revenue Forecast')
        st.plotly_chart(figq, use_container_width=True)

        # Table
        st.dataframe(q_fc[['Quarter', 'ds', 'yhat', 'yhat_capped']].rename(columns={'yhat': 'Predicted', 'yhat_capped': 'Predicted (capped)'}))
    except Exception as e:
        st.info(f"Quarterly forecast not available: {e}")

def customer_segmentation_page(df):
    """Customer segmentation page"""
    st.markdown('<div class="sub-header">👥 Customer Segmentation</div>', unsafe_allow_html=True)
    
    # Initialize segmentation model
    segmentation = CustomerSegmentation()
    
    try:
        # Segment customers
        with st.spinner("Performing customer segmentation..."):
            segmented_df = segmentation.segment_customers(df)
        
        # Analyze segments
        segment_analysis = segmentation.analyze_segments(segmented_df)
        segment_profiles = segmentation.create_segment_profiles(segment_analysis)
        
        # Display segment overview
        st.markdown("### Customer Segments Overview")
        
        segment_summary = []
        for segment, profile in segment_profiles.items():
            segment_summary.append({
                'Segment': segment,
                'Profile': profile['profile_name'],
                'Size': profile['size'],
                'Percentage': f"{profile['percentage']:.1f}%",
                'Avg Revenue': f"${profile['key_metrics']['avg_revenue']:.2f}",
                'Churn Rate': f"{profile['key_metrics']['churn_rate']:.1%}"
            })
        
        st.dataframe(pd.DataFrame(segment_summary))
        
        # Segment visualization
        st.markdown("### Segment Visualization")
        
        # PCA plot
        fig_pca = px.scatter(
            segmented_df, 
            x='pca_1', 
            y='pca_2', 
            color='cluster',
            title="Customer Segments (PCA Visualization)",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig_pca, use_container_width=True)
        
        # Segment characteristics
        st.markdown("### Segment Characteristics & Strategies")
        
        for segment, profile in segment_profiles.items():
            with st.expander(f"{profile['profile_name']} ({profile['size']} customers, {profile['percentage']:.1f}%)"):
                st.markdown(f"**Description:** {profile['description']}")
                st.markdown(f"**Recommended Strategy:** {profile['strategy']}")
                
                # Key metrics
                metrics = profile['key_metrics']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Revenue", f"${metrics['avg_revenue']:.2f}")
                with col2:
                    st.metric("Avg Frequency", f"{metrics['avg_frequency']:.1f}")
                with col3:
                    st.metric("Avg Recency", f"{metrics['avg_recency']:.0f} days")
                with col4:
                    st.metric("Churn Rate", f"{metrics['churn_rate']:.1%}")
        
        # Revenue by segment
        st.markdown("### Revenue Analysis by Segment")
        
        if 'total_revenue' in segmented_df.columns:
            segment_revenue = segmented_df.groupby('cluster')['total_revenue'].agg(['sum', 'mean', 'count']).reset_index()
            segment_revenue['cluster'] = segment_revenue['cluster'].apply(lambda x: f"Segment {x}")
            
            fig_revenue = px.bar(
                segment_revenue, 
                x='cluster', 
                y='sum',
                title="Total Revenue by Segment"
            )
            st.plotly_chart(fig_revenue, use_container_width=True)
        
        # Download segmented data
        csv_buffer = io.StringIO()
        segmented_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Segmented Customer Data",
            data=csv_buffer.getvalue(),
            file_name=f"customer_segments_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error in customer segmentation: {str(e)}")

def demand_forecasting_page(df):
    """Demand forecasting and inventory management page"""
    st.markdown('<div class="sub-header">📦 Demand Forecasting & Inventory Management</div>', unsafe_allow_html=True)
    
    # Initialize demand forecaster
    demand_forecaster = DemandForecaster()
    
    try:
        # Prepare demand data
        demand_data = demand_forecaster.prepare_product_demand_data(df)
        
        # Get top products
        st.markdown("### Top Products Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            metric_choice = st.selectbox("Rank by", ["revenue", "quantity", "frequency"])
        with col2:
            top_n_products = st.slider("Number of products", 5, 20, 10)
        
        top_products = demand_forecaster.identify_top_products(df, metric_choice, top_n_products)
        
        # Display top products
        fig_top = px.bar(
            x=top_products.values,
            y=top_products.index,
            orientation='h',
            title=f"Top {top_n_products} Products by {metric_choice.title()}"
        )
        st.plotly_chart(fig_top, use_container_width=True)
        
        # Product demand forecasting
        st.markdown("### Product Demand Forecasting")
        
        available_products = df['product_name'].unique()[:20]  # Limit for performance
        selected_product = st.selectbox("Select product for detailed forecast", available_products)
        
        forecast_days = st.slider("Forecast period (days)", 30, 90, 30)
        
        if st.button("Generate Product Forecast"):
            with st.spinner(f"Forecasting demand for {selected_product}..."):
                product_forecast = demand_forecaster.forecast_product_demand(
                    demand_data, selected_product, forecast_days
                )
                
                if product_forecast is not None:
                    # Display forecast chart
                    fig_forecast = go.Figure()
                    
                    # Historical data for the product
                    product_history = demand_data[demand_data['product_name'] == selected_product].tail(60)
                    if not product_history.empty:
                        fig_forecast.add_trace(go.Scatter(
                            x=product_history['date'],
                            y=product_history['quantity_demanded'],
                            mode='lines',
                            name='Historical Demand',
                            line=dict(color='blue')
                        ))
                    
                    # Forecast
                    fig_forecast.add_trace(go.Scatter(
                        x=product_forecast['date'],
                        y=product_forecast['predicted_demand'],
                        mode='lines',
                        name='Forecasted Demand',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig_forecast.update_layout(
                        title=f'Demand Forecast for {selected_product}',
                        xaxis_title='Date',
                        yaxis_title='Quantity Demanded',
                        height=400
                    )
                    
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Forecast summary
                    total_demand = product_forecast['predicted_demand'].sum()
                    avg_daily_demand = product_forecast['predicted_demand'].mean()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Forecasted Demand", f"{total_demand:.0f}")
                    with col2:
                        st.metric("Average Daily Demand", f"{avg_daily_demand:.1f}")
                    with col3:
                        st.metric("Peak Daily Demand", f"{product_forecast['predicted_demand'].max():.0f}")
                    
                    # Inventory recommendations
                    recommendations = demand_forecaster.calculate_inventory_recommendations(product_forecast)
                    
                    if selected_product in recommendations:
                        rec = recommendations[selected_product]
                        st.markdown("#### Inventory Recommendations")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Safety Stock", f"{rec['safety_stock']:.0f} units")
                        with col2:
                            st.metric("Reorder Point", f"{rec['reorder_point']:.0f} units")
                        with col3:
                            st.metric("Monthly Demand", f"{rec['monthly_demand']:.0f} units")
                        with col4:
                            st.metric("Daily Demand", f"{rec['avg_daily_demand']:.1f} units")
                else:
                    st.warning(f"Insufficient data for {selected_product} to generate forecast.")
        
        # Category demand analysis
        st.markdown("### Category Demand Analysis")
        
        category_forecasts = demand_forecaster.forecast_category_demand(df, 30)
        
        if category_forecasts:
            # Display category forecasts
            category_summary = []
            for category, forecast_df in category_forecasts.items():
                total_demand = forecast_df['predicted_demand'].sum()
                avg_daily = forecast_df['predicted_demand'].mean()
                category_summary.append({
                    'Category': category,
                    'Total 30-Day Demand': f"{total_demand:.0f}",
                    'Average Daily Demand': f"{avg_daily:.1f}"
                })
            
            st.dataframe(pd.DataFrame(category_summary))
            
            # Category demand visualization
            categories = list(category_forecasts.keys())
            total_demands = [category_forecasts[cat]['predicted_demand'].sum() for cat in categories]
            
            fig_category = px.pie(
                values=total_demands,
                names=categories,
                title="30-Day Demand Forecast by Category"
            )
            st.plotly_chart(fig_category, use_container_width=True)
        
        # Demand patterns analysis
        st.markdown("### Demand Patterns Analysis")
        
        patterns = demand_forecaster.analyze_demand_patterns(df)
        
        if 'monthly_demand' in patterns:
            col1, col2 = st.columns(2)
            
            with col1:
                monthly_data = patterns['monthly_demand']
                fig_monthly = px.bar(
                    x=list(monthly_data.keys()),
                    y=list(monthly_data.values()),
                    title="Monthly Demand Pattern"
                )
                st.plotly_chart(fig_monthly, use_container_width=True)
            
            with col2:
                daily_data = patterns['daily_demand']
                fig_daily = px.bar(
                    x=list(daily_data.keys()),
                    y=list(daily_data.values()),
                    title="Daily Demand Pattern"
                )
                st.plotly_chart(fig_daily, use_container_width=True)
        
        # Key insights
        if 'peak_month' in patterns and 'peak_day' in patterns:
            st.markdown(f'<div class="insight-box">📊 <strong>Key Insights:</strong><br>• Peak demand month: <strong>{patterns["peak_month"]}</strong><br>• Peak demand day: <strong>{patterns["peak_day"]}</strong></div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error in demand forecasting: {str(e)}")

def main():
    """Main application"""
    st.markdown('<div class="main-header">📊 FORESIGHT</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # File upload
    st.sidebar.markdown("### 📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your data file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file with customer and sales data"
    )
    
    if uploaded_file is not None:
        if not st.session_state.data_loaded:
            with st.spinner("Loading and processing data..."):
                df, processor = load_data(uploaded_file)
                
                if df is not None:
                    st.session_state.processed_data = df
                    st.session_state.data_loaded = True
                    st.sidebar.success("✅ Data loaded successfully!")
                else:
                    st.sidebar.error("❌ Failed to load data")
    
    if st.session_state.data_loaded and st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        
        # Navigation menu
        page = st.sidebar.selectbox(
            "Select Analysis",
            [
                "📋 Data Overview",
                "🎯 Churn Analysis",
                "📈 Sales Forecasting", 
                "👥 Customer Segmentation",
                "📦 Demand Forecasting"
            ]
        )
        
        # Display selected page
        if page == "📋 Data Overview":
            display_data_overview(df)
        elif page == "🎯 Churn Analysis":
            churn_analysis_page(df)
        elif page == "📈 Sales Forecasting":
            sales_forecasting_page(df)
        elif page == "👥 Customer Segmentation":
            customer_segmentation_page(df)
        elif page == "📦 Demand Forecasting":
            demand_forecasting_page(df)
    
    else:
        # Welcome page
        st.markdown("""
        ## Welcome to the Customer Churn & Sales Forecasting Dashboard! 🚀
        
        This comprehensive analytics platform helps businesses:
        
        ### 🎯 **Churn Prediction**
        - Identify customers at risk of churning
        - Understand key factors driving churn
        - Get actionable insights for retention strategies
        
        ### 📈 **Sales Forecasting**
        - Predict future sales trends
        - Analyze seasonal patterns
        - Plan inventory and resources effectively
        
        ### 👥 **Customer Segmentation**
        - Segment customers based on behavior
        - Develop targeted marketing strategies
        - Optimize customer lifetime value
        
        ### 📦 **Demand Forecasting**
        - Forecast product demand
        - Optimize inventory management
        - Reduce stockouts and overstock
        
        ---
        
        ### 🚀 **Getting Started**
        1. **Upload your data** using the file uploader in the sidebar
        2. **Supported formats**: CSV, Excel (.xlsx, .xls)
        3. **Required columns**: customer_id, order_id, purchase dates, revenue, quantity
        4. **Navigate** through different analysis sections using the sidebar menu
        
        ### 📊 **Sample Data Format**
        Your data should include columns like:
        - `customer_id`, `order_id`, `age`, `gender`, `country`
        - `signup_date`, `last_purchase_date`, `subscription_status`
        - `unit_price`, `quantity`, `product_name`, `category`
        - `purchase_frequency`, `cancellations_count`, `Ratings`
        
        ---
        
        **Ready to get started?** Upload your data file using the sidebar! 📁
        """)
        
        # Sample data info
        with st.expander("📋 View Sample Data Structure"):
            sample_data = {
                'order_id': ['ORD5000', 'ORD5001', 'ORD5002'],
                'customer_id': ['CUST1000', 'CUST1001', 'CUST1002'],
                'age': [39, 61, 26],
                'gender': ['Female', 'Female', 'Female'],
                'country': ['Canada', 'USA', 'Pakistan'],
                'signup_date': ['01-07-2021', '10/19/2020', '06-10-2023'],
                'last_purchase_date': ['2/21/2023', '12-08-2021', '09-04-2023'],
                'subscription_status': ['active', 'active', 'cancelled'],
                'unit_price': [78.21, 64.02, 604.14],
                'quantity': [5, 8, 2],
                'product_name': ['Football', 'Refrigerator', 'Hoodie'],
                'category': ['Sports', 'Home', 'Clothing'],
                'Ratings': [4.2, 4.0, 3.9]
            }
            st.dataframe(pd.DataFrame(sample_data))

if __name__ == "__main__":
    main()

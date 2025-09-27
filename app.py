# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Import your churn predictor
from churn_model import ChurnPredictor

@st.cache_data(show_spinner=False)
def _read_csv_cached(file):
    return pd.read_csv(file)

class ChurnAnalysisDashboard:
    def __init__(self):
        self.churn_predictor = None
        self.data = None
        self.model_path = 'churn_model.pkl'
        
    def load_data(self, file_path=None, df=None):
        """Load customer data"""
        if file_path:
            self.data = pd.read_csv(file_path)
        elif df is not None:
            self.data = df
        else:
            st.error("Please provide data to load")
            return False
        return True
    
    def setup_sidebar(self):
        """Setup the sidebar controls"""
        st.sidebar.title("Churn Prediction Dashboard")
        st.sidebar.header("Controls")
        
        # File upload
        uploaded_file = st.sidebar.file_uploader(
            "Upload Customer Data (CSV)", 
            type=['csv'],
            help="Upload your customer data CSV file"
        )
        
        churn_days = st.sidebar.slider(
            "Churn Threshold (days without purchase)", min_value=30, max_value=365, value=90, step=15,
            help="Number of days since last purchase after which a customer is considered churned"
        )

        # Analysis period
        analysis_period = st.sidebar.selectbox(
            "Analysis Period",
            ["Last Quarter", "Last 6 Months", "Last Year", "All Time"],
            index=0
        )
        
        # Churn probability threshold
        churn_threshold = st.sidebar.slider(
            "High Churn Risk Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.7,
            step=0.1,
            help="Probability above which customers are considered high risk"
        )

        decision_threshold = st.sidebar.slider(
            "Decision Threshold for Churn (affects Precision/Recall)",
            min_value=0.1, max_value=0.9, value=0.5, step=0.05,
            help="Classify a customer as churn if predicted probability ≥ this threshold"
        )
        
        return uploaded_file, analysis_period, churn_threshold, churn_days, decision_threshold
    
    def display_overview_metrics(self, predictions):
        """Display overview metrics"""
        st.subheader("📊 Overview Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_customers = len(predictions)
        churned_customers = predictions['is_churned'].sum()
        high_risk = len(predictions[predictions['churn_probability'] > 0.7])
        avg_churn_prob = predictions['churn_probability'].mean()
        
        with col1:
            st.metric("Total Customers", f"{total_customers:,}")
        with col2:
            st.metric("Churned Customers", f"{churned_customers:,}", 
                     delta=f"{(churned_customers/total_customers*100):.1f}%")
        with col3:
            st.metric("High Risk Customers", f"{high_risk:,}")
        with col4:
            st.metric("Avg Churn Probability", f"{avg_churn_prob:.2%}")

    def display_threshold_metrics(self, predictions, decision_threshold: float):
        """Show precision/recall/f1 and confusion matrix at a chosen threshold."""
        st.subheader("🎯 Threshold Tuning: Precision/Recall")
        if 'is_churned' not in predictions.columns:
            st.info("Ground-truth churn labels not available in the data to compute metrics.")
            return

        y_true = predictions['is_churned'].astype(int).values
        y_pred = (predictions['churn_probability'].values >= decision_threshold).astype(int)

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        c1, c2, c3 = st.columns(3)
        c1.metric("Precision (Churn=1)", f"{prec:.2f}")
        c2.metric("Recall (Churn=1)", f"{rec:.2f}")
        c3.metric("F1 (Churn=1)", f"{f1:.2f}")

        fig_cm = px.imshow(cm, text_auto=True, aspect='auto', color_continuous_scale='Blues',
                           labels=dict(x="Predicted", y="Actual", color="Count"))
        fig_cm.update_xaxes(ticktext=['Non-churn (0)', 'Churn (1)'], tickvals=[0,1])
        fig_cm.update_yaxes(ticktext=['Non-churn (0)', 'Churn (1)'], tickvals=[0,1])
        fig_cm.update_layout(title=f"Confusion Matrix @ Threshold = {decision_threshold:.2f}")
        st.plotly_chart(fig_cm, use_container_width=True)

    def display_feature_importances(self):
        """Show top model feature importances."""
        try:
            importances = pd.Series(self.churn_predictor.model.feature_importances_, index=self.churn_predictor.feature_columns)
            top = importances.sort_values(ascending=False).head(10).reset_index()
            top.columns = ['feature', 'importance']
            st.subheader("🏷️ Top Feature Importances")
            fig = px.bar(top, x='importance', y='feature', orientation='h', title='Top 10 Features')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Feature importances unavailable: {e}")
    
    def display_churn_trends(self, predictions):
        """Display churn rate trends over time"""
        st.subheader("📈 Churn Rate Trends")
        
        # Create monthly churn trends
        pred = predictions.copy()
        pred['last_purchase_date'] = pd.to_datetime(pred['last_purchase_date'], errors='coerce')
        pred['month'] = pred['last_purchase_date'].dt.to_period('M')
        monthly = pred.groupby('month').agg({
            'is_churned': 'mean',
            'churn_probability': 'mean',
            'customer_id': 'count'
        }).reset_index()
        monthly['month'] = monthly['month'].astype(str)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly['month'], y=monthly['is_churned'], name='Actual Churn Rate', mode='lines+markers'))
        fig.add_trace(go.Scatter(x=monthly['month'], y=monthly['churn_probability'], name='Avg Predicted Probability', mode='lines+markers'))
        fig.update_layout(title='Monthly Actual vs Predicted Churn', xaxis_title='Month', yaxis_title='Rate / Probability')
        st.plotly_chart(fig, use_container_width=True)
    
    def display_top_churn_risks(self, predictions, top_n=10):
        """Display top customers with highest churn risk"""
        st.subheader("🔴 Top Customers with Highest Churn Risk")
        
        top_risks = predictions.nlargest(top_n, 'churn_probability')[
            ['customer_id', 'age', 'gender', 'country', 'churn_probability', 
             'churn_risk', 'days_since_last_purchase', 'Ratings']
        ]
        
        # Format the display
        display_df = top_risks.copy()
        display_df['churn_probability'] = display_df['churn_probability'].apply(
            lambda x: f"{x:.1%}"
        )
        display_df['days_since_last_purchase'] = display_df['days_since_last_purchase'].astype(int)
        
        try:
            styled = display_df.style.apply(
                lambda row: ['background-color: #FFBABA' if row.churn_risk == "High" else '' for _ in row], axis=1
            )
            st.dataframe(styled, use_container_width=True)
        except Exception:
            st.dataframe(display_df, use_container_width=True)
    
    def display_customer_segmentation(self, predictions):
        """Display customer segmentation based on churn likelihood"""
        st.subheader("👥 Customer Segmentation by Churn Risk")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk distribution pie chart
            risk_counts = predictions['churn_risk'].value_counts()
            fig_pie = px.pie(values=risk_counts.values, names=risk_counts.index,
                           title='Customer Distribution by Churn Risk')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Risk by demographic factors
            risk_by_country = predictions.groupby('country')['churn_probability'].mean().sort_values(ascending=False)
            fig_bar = px.bar(x=risk_by_country.values, y=risk_by_country.index,
                           orientation='h', title='Average Churn Probability by Country',
                           labels={'x': 'Churn Probability', 'y': 'Country'})
            st.plotly_chart(fig_bar, use_container_width=True)

        # KMeans clustering on probability and spend
        seg_df = predictions[['churn_probability', 'total_spent']].copy()
        seg_df = seg_df.fillna(0)
        try:
            kmeans = KMeans(n_clusters=3, random_state=42).fit(seg_df)
            predictions['segment'] = kmeans.labels_
            fig = px.scatter(predictions, x='total_spent', y='churn_probability',
                             color='segment', hover_data=['customer_id', 'country'],
                             title='Customer Segments (KMeans: Prob vs Spend)')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Segmentation skipped: {e}")

        # RFM-style segmentation (Recency, Frequency, Monetary)
        st.markdown("### RFM Segmentation (Preview)")
        rfm_cols = [c for c in ['days_since_last_purchase','purchase_frequency','total_spent'] if c in predictions.columns]
        if len(rfm_cols) == 3:
            try:
                rfm_df = predictions[rfm_cols].fillna(0)
                kmeans_rfm = KMeans(n_clusters=3, random_state=42).fit(rfm_df)
                predictions['rfm_segment'] = kmeans_rfm.labels_
                fig2 = px.scatter_3d(
                    predictions,
                    x='days_since_last_purchase', y='purchase_frequency', z='total_spent',
                    color='rfm_segment', hover_data=['customer_id','country'],
                    title='RFM Segments'
                )
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.info(f"RFM segmentation skipped: {e}")
    
    def display_high_churn_periods(self, predictions):
        """Identify and display high churn periods"""
        st.subheader("📅 High Churn Periods Analysis")
        
        # Analyze churn by time periods
        predictions['last_purchase_date'] = pd.to_datetime(predictions['last_purchase_date'], errors='coerce')
        predictions['purchase_month'] = predictions['last_purchase_date'].dt.to_period('M')
        monthly_analysis = predictions.groupby('purchase_month').agg({
            'is_churned': ['count', 'mean'],
            'churn_probability': 'mean'
        }).reset_index()
        
        monthly_analysis.columns = ['Month', 'Total_Customers', 'Churn_Rate', 'Avg_Churn_Probability']
        monthly_analysis['Month'] = monthly_analysis['Month'].astype(str)
        
        # Identify high churn periods (top 25%)
        high_churn_threshold = monthly_analysis['Churn_Rate'].quantile(0.75)
        high_churn_periods = monthly_analysis[monthly_analysis['Churn_Rate'] > high_churn_threshold]
        
        st.write("**High Churn Periods (Top 25%):**")
        high_sorted = high_churn_periods.sort_values('Churn_Rate', ascending=False)
        st.dataframe(high_sorted, use_container_width=True)

        # Visualize high churn periods
        try:
            fig_h = px.bar(high_sorted, x='Month', y='Churn_Rate',
                           title='High Churn Periods (Top 25%)', text='Churn_Rate')
            fig_h.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            # Add overall average as reference line
            overall = monthly_analysis['Churn_Rate'].mean()
            fig_h.add_hline(y=overall, line_dash='dash', line_color='gray', annotation_text='Avg Churn Rate')
            st.plotly_chart(fig_h, use_container_width=True)
        except Exception as e:
            st.info(f"High churn periods chart skipped: {e}")
    
    def run_dashboard(self):
        """Main method to run the dashboard"""
        st.set_page_config(
            page_title="Customer Churn Prediction Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🎯 Customer Churn Prediction & Analysis")
        st.markdown("---")
        
        # Setup sidebar
        uploaded_file, analysis_period, churn_threshold, churn_days, decision_threshold = self.setup_sidebar()
        
        if uploaded_file is not None:
            try:
                # Load data
                self.data = _read_csv_cached(uploaded_file)
                st.sidebar.success(f"Data loaded successfully! {len(self.data)} records found.")
                
                # Initialize or load model
                if self.churn_predictor is None:
                    self.churn_predictor = ChurnPredictor(churn_threshold_days=churn_days)

                # Option to reuse saved model
                use_saved = st.sidebar.checkbox("Use saved model if available", value=True)
                retrain = st.sidebar.checkbox("Retrain model now", value=False)

                if use_saved and not retrain:
                    try:
                        self.churn_predictor.load_model(self.model_path)
                        st.sidebar.info("Loaded saved model.")
                    except Exception:
                        st.sidebar.warning("No saved model found. Training a new model...")
                        retrain = True

                if retrain or self.churn_predictor.model is None:
                    with st.spinner("Training churn prediction model..."):
                        self.churn_predictor.train_model(self.data)
                        # Save immediately
                        try:
                            self.churn_predictor.save_model(self.model_path)
                            st.sidebar.success("Model trained and saved.")
                        except Exception as e:
                            st.sidebar.warning(f"Model save failed: {e}")

                # Predict
                predictions = self.churn_predictor.predict_churn_probability(self.data)

                # Recompute churn_risk using user-set threshold (simple rule: High/Medium/Low)
                probs = predictions['churn_probability']
                medium_cut = max(0.3, churn_threshold / 2)
                predictions['churn_risk'] = pd.cut(
                    probs,
                    bins=[0, medium_cut, churn_threshold, 1.0],
                    labels=['Low', 'Medium', 'High'], include_lowest=True
                )

                # Add predicted class for evaluation
                predictions['pred_is_churn'] = (probs >= decision_threshold).astype(int)

                # Apply analysis period filter
                now = pd.Timestamp.now()
                predictions['last_purchase_date'] = pd.to_datetime(predictions['last_purchase_date'], errors='coerce')
                if analysis_period == "Last Quarter":
                    cutoff = now - pd.DateOffset(months=3)
                elif analysis_period == "Last 6 Months":
                    cutoff = now - pd.DateOffset(months=6)
                elif analysis_period == "Last Year":
                    cutoff = now - pd.DateOffset(years=1)
                else:
                    cutoff = None
                if cutoff is not None:
                    predictions = predictions[predictions['last_purchase_date'] >= cutoff]
                
                # Display all sections
                self.display_overview_metrics(predictions)
                st.markdown("---")
                
                tabs = st.tabs(["Churn Analysis", "Sales Forecast (Preview)"])

                with tabs[0]:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        self.display_churn_trends(predictions)
                    with col2:
                        self.display_customer_segmentation(predictions)

                    st.markdown("---")
                    self.display_top_churn_risks(predictions)
                    st.markdown("---")
                    self.display_high_churn_periods(predictions)

                    st.markdown("---")
                    self.display_threshold_metrics(predictions, decision_threshold)
                    st.markdown("---")
                    self.display_feature_importances()

                with tabs[1]:
                    st.info("Sales forecasting module will aggregate monthly sales and fit a time-series model (Prophet/ARIMA). Coming soon.")
                    # Placeholder aggregation
                    try:
                        tmp = predictions.copy()
                        tmp['purchase_date'] = pd.to_datetime(tmp['last_purchase_date'], errors='coerce')
                        monthly_sales = tmp.groupby(pd.Grouper(key='purchase_date', freq='M'))['total_spent'].sum().reset_index()
                        fig_sf = px.line(monthly_sales, x='purchase_date', y='total_spent', title='Historical Monthly Sales')
                        st.plotly_chart(fig_sf, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Sales aggregation failed: {e}")
                
                # Download results
                st.sidebar.markdown("---")
                st.sidebar.subheader("Export Results")
                csv = predictions.to_csv(index=False)
                st.sidebar.download_button(
                    label="Download Churn Predictions CSV",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
        else:
            st.info("👈 Please upload a CSV file to get started with churn analysis")
            
            # Display sample data structure
            st.subheader("Expected Data Format")
            sample_data = {
                'customer_id': ['CUST1000', 'CUST1001', 'CUST1002'],
                'age': [39, 61, 26],
                'gender': ['Female', 'Female', 'Female'],
                'country': ['Canada', 'USA', 'Pakistan'],
                'signup_date': ['01-07-2021', '10/19/2020', '06-10-2023'],
                'last_purchase_date': ['2/21/2023', '12-08-2021', '09-04-2023'],
                'cancellations_count': [0, 0, 3],
                'subscription_status': ['active', 'active', 'cancelled'],
                'unit_price': [78.21, 64.02, 604.14],
                'quantity': [5, 8, 2],
                'purchase_frequency': [37, 35, 44],
                'Ratings': [4.2, 4.0, 3.9]
            }
            st.json(sample_data)

# Run the dashboard
if __name__ == "__main__":
    dashboard = ChurnAnalysisDashboard()
    dashboard.run_dashboard()
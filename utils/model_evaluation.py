import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ModelEvaluationDashboard:
    """Comprehensive model evaluation and performance monitoring"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set1
        
    def create_classification_evaluation(self, y_true, y_pred, y_pred_proba=None, model_name="Model"):
        """Create comprehensive classification model evaluation"""
        st.markdown(f"### 🎯 {model_name} Performance Evaluation")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.3f}")
        with col2:
            st.metric("Precision", f"{precision:.3f}")
        with col3:
            st.metric("Recall", f"{recall:.3f}")
        with col4:
            st.metric("F1-Score", f"{f1:.3f}")
        
        # AUC Score if probabilities available
        if y_pred_proba is not None:
            auc_score = roc_auc_score(y_true, y_pred_proba)
            st.metric("AUC Score", f"{auc_score:.3f}")
        
        # Confusion Matrix
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Not Churned', 'Churned'],
                y=['Not Churned', 'Churned'],
                color_continuous_scale='Blues',
                text_auto=True
            )
            fig_cm.update_layout(height=400)
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            if y_pred_proba is not None:
                st.markdown("#### ROC Curve")
                from sklearn.metrics import roc_curve
                
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'ROC Curve (AUC = {auc_score:.3f})',
                    line=dict(color='blue', width=2)
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random Classifier',
                    line=dict(color='red', width=1, dash='dash')
                ))
                
                fig_roc.update_layout(
                    title='ROC Curve',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    height=400
                )
                st.plotly_chart(fig_roc, use_container_width=True)
        
        # Classification Report
        st.markdown("#### Detailed Classification Report")
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3))
        
        # Prediction Distribution
        if y_pred_proba is not None:
            st.markdown("#### Prediction Probability Distribution")
            
            fig_dist = go.Figure()
            
            # Churned customers
            churned_probs = y_pred_proba[y_true == 1]
            fig_dist.add_trace(go.Histogram(
                x=churned_probs,
                name='Churned Customers',
                opacity=0.7,
                nbinsx=30,
                marker_color='red'
            ))
            
            # Non-churned customers
            non_churned_probs = y_pred_proba[y_true == 0]
            fig_dist.add_trace(go.Histogram(
                x=non_churned_probs,
                name='Active Customers',
                opacity=0.7,
                nbinsx=30,
                marker_color='blue'
            ))
            
            fig_dist.update_layout(
                title='Churn Probability Distribution by Actual Status',
                xaxis_title='Predicted Churn Probability',
                yaxis_title='Count',
                barmode='overlay',
                height=400
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score if y_pred_proba is not None else None
        }
    
    def create_regression_evaluation(self, y_true, y_pred, model_name="Model"):
        """Create comprehensive regression model evaluation"""
        st.markdown(f"### 📈 {model_name} Performance Evaluation")
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("MAE", f"{mae:.2f}")
        with col2:
            st.metric("RMSE", f"{rmse:.2f}")
        with col3:
            st.metric("R² Score", f"{r2:.3f}")
        with col4:
            st.metric("MAPE", f"{mape:.1f}%")
        with col5:
            # Explained variance
            explained_var = 1 - (np.var(y_true - y_pred) / np.var(y_true))
            st.metric("Explained Var", f"{explained_var:.3f}")
        
        # Prediction vs Actual scatter plot
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Predictions vs Actual Values")
            
            fig_scatter = go.Figure()
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            
            fig_scatter.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            
            # Actual predictions
            fig_scatter.add_trace(go.Scatter(
                x=y_true,
                y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(
                    color='blue',
                    opacity=0.6,
                    size=6
                ),
                hovertemplate='Actual: %{x}<br>Predicted: %{y}<extra></extra>'
            ))
            
            fig_scatter.update_layout(
                title='Predictions vs Actual',
                xaxis_title='Actual Values',
                yaxis_title='Predicted Values',
                height=400
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            st.markdown("#### Residuals Distribution")
            
            residuals = y_true - y_pred
            
            fig_residuals = go.Figure()
            fig_residuals.add_trace(go.Histogram(
                x=residuals,
                nbinsx=30,
                marker_color='lightblue',
                opacity=0.7
            ))
            
            fig_residuals.update_layout(
                title='Residuals Distribution',
                xaxis_title='Residuals (Actual - Predicted)',
                yaxis_title='Frequency',
                height=400
            )
            st.plotly_chart(fig_residuals, use_container_width=True)
        
        # Residuals vs Predicted
        st.markdown("#### Residuals vs Predicted Values")
        
        fig_residuals_scatter = go.Figure()
        fig_residuals_scatter.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            marker=dict(
                color='green',
                opacity=0.6,
                size=6
            ),
            hovertemplate='Predicted: %{x}<br>Residual: %{y}<extra></extra>'
        ))
        
        # Add horizontal line at y=0
        fig_residuals_scatter.add_hline(y=0, line_dash="dash", line_color="red")
        
        fig_residuals_scatter.update_layout(
            title='Residuals vs Predicted Values',
            xaxis_title='Predicted Values',
            yaxis_title='Residuals',
            height=400
        )
        st.plotly_chart(fig_residuals_scatter, use_container_width=True)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2,
            'mape': mape
        }
    
    def create_model_comparison_dashboard(self, model_results, model_type='classification'):
        """Create model comparison dashboard"""
        st.markdown("### 🏆 Model Comparison Dashboard")
        
        if model_type == 'classification':
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC Score']
        else:
            metrics = ['mae', 'rmse', 'r2_score', 'mape']
            metric_names = ['MAE', 'RMSE', 'R² Score', 'MAPE']
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, results in model_results.items():
            row = {'Model': model_name}
            for metric in metrics:
                if metric in results and results[metric] is not None:
                    row[metric] = results[metric]
                else:
                    row[metric] = np.nan
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display comparison table
        st.markdown("#### Performance Comparison Table")
        display_df = comparison_df.copy()
        for metric in metrics:
            if metric in display_df.columns:
                display_df[metric] = display_df[metric].round(3)
        st.dataframe(display_df)
        
        # Best model identification
        if model_type == 'classification':
            best_metric = 'auc_score' if 'auc_score' in comparison_df.columns else 'accuracy'
            best_model_idx = comparison_df[best_metric].idxmax()
        else:
            best_metric = 'r2_score'
            best_model_idx = comparison_df[best_metric].idxmax()
        
        best_model = comparison_df.loc[best_model_idx, 'Model']
        best_score = comparison_df.loc[best_model_idx, best_metric]
        
        st.success(f"🏆 Best Model: **{best_model}** with {best_metric.upper()}: **{best_score:.3f}**")
        
        # Radar chart for model comparison
        if len(model_results) > 1:
            st.markdown("#### Model Performance Radar Chart")
            
            fig_radar = go.Figure()
            
            for model_name, results in model_results.items():
                values = []
                for metric in metrics:
                    if metric in results and results[metric] is not None:
                        # Normalize metrics for radar chart
                        if model_type == 'classification':
                            values.append(results[metric])
                        else:
                            # For regression, invert MAE and RMSE (lower is better)
                            if metric in ['mae', 'rmse', 'mape']:
                                max_val = comparison_df[metric].max()
                                values.append(1 - (results[metric] / max_val))
                            else:
                                values.append(results[metric])
                    else:
                        values.append(0)
                
                # Close the radar chart
                values.append(values[0])
                metric_names_closed = metric_names + [metric_names[0]]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metric_names_closed,
                    fill='toself',
                    name=model_name,
                    opacity=0.6
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                height=500
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Bar chart comparison
        st.markdown("#### Metric Comparison Bar Chart")
        
        # Select metric for bar chart
        selected_metric = st.selectbox(
            "Select metric for comparison",
            options=metrics,
            format_func=lambda x: x.upper().replace('_', ' ')
        )
        
        if selected_metric in comparison_df.columns:
            fig_bar = px.bar(
                comparison_df,
                x='Model',
                y=selected_metric,
                title=f'{selected_metric.upper()} Comparison',
                color=selected_metric,
                color_continuous_scale='viridis'
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    def create_feature_importance_analysis(self, feature_importance_dict):
        """Create feature importance analysis dashboard"""
        st.markdown("### 🔍 Feature Importance Analysis")
        
        if not feature_importance_dict:
            st.warning("No feature importance data available.")
            return
        
        # Combine feature importance from multiple models
        all_features = set()
        for model_name, importance_df in feature_importance_dict.items():
            all_features.update(importance_df['feature'].tolist())
        
        # Create comparison dataframe
        importance_comparison = pd.DataFrame({'feature': list(all_features)})
        
        for model_name, importance_df in feature_importance_dict.items():
            importance_dict = dict(zip(importance_df['feature'], importance_df['importance']))
            importance_comparison[model_name] = importance_comparison['feature'].map(importance_dict).fillna(0)
        
        # Display top features
        st.markdown("#### Top Important Features")
        
        # Calculate average importance across models
        model_columns = [col for col in importance_comparison.columns if col != 'feature']
        importance_comparison['avg_importance'] = importance_comparison[model_columns].mean(axis=1)
        
        top_features = importance_comparison.nlargest(15, 'avg_importance')
        
        # Create horizontal bar chart
        fig_importance = px.bar(
            top_features,
            x='avg_importance',
            y='feature',
            orientation='h',
            title='Top 15 Most Important Features (Average Across Models)',
            color='avg_importance',
            color_continuous_scale='viridis'
        )
        fig_importance.update_layout(
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Feature importance heatmap
        if len(feature_importance_dict) > 1:
            st.markdown("#### Feature Importance Heatmap Across Models")
            
            # Prepare data for heatmap
            heatmap_data = top_features[model_columns + ['feature']].set_index('feature')
            
            fig_heatmap = px.imshow(
                heatmap_data.T,
                labels=dict(x="Features", y="Models", color="Importance"),
                color_continuous_scale='viridis',
                aspect='auto'
            )
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Feature importance table
        st.markdown("#### Detailed Feature Importance Table")
        display_importance = top_features.round(4)
        st.dataframe(display_importance)
        
        # Download feature importance
        csv_data = importance_comparison.to_csv(index=False)
        st.download_button(
            label="📥 Download Feature Importance Data",
            data=csv_data,
            file_name=f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    def create_prediction_analysis(self, df, predictions, prediction_probabilities=None):
        """Create prediction analysis dashboard"""
        st.markdown("### 🎯 Prediction Analysis")
        
        # Add predictions to dataframe
        analysis_df = df.copy()
        analysis_df['predicted_churn'] = predictions
        
        if prediction_probabilities is not None:
            analysis_df['churn_probability'] = prediction_probabilities
        
        # Prediction distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Prediction Distribution")
            pred_counts = pd.Series(predictions).value_counts()
            
            fig_pred_dist = px.pie(
                values=pred_counts.values,
                names=['Active', 'Churned'],
                title='Predicted Customer Status Distribution'
            )
            st.plotly_chart(fig_pred_dist, use_container_width=True)
        
        with col2:
            if prediction_probabilities is not None:
                st.markdown("#### Churn Probability Distribution")
                
                fig_prob_dist = go.Figure()
                fig_prob_dist.add_trace(go.Histogram(
                    x=prediction_probabilities,
                    nbinsx=30,
                    marker_color='lightblue',
                    opacity=0.7
                ))
                
                fig_prob_dist.update_layout(
                    title='Churn Probability Distribution',
                    xaxis_title='Churn Probability',
                    yaxis_title='Count',
                    height=400
                )
                st.plotly_chart(fig_prob_dist, use_container_width=True)
        
        # High-risk customers analysis
        if prediction_probabilities is not None:
            st.markdown("#### High-Risk Customer Analysis")
            
            # Define risk categories
            analysis_df['risk_category'] = pd.cut(
                analysis_df['churn_probability'],
                bins=[0, 0.3, 0.7, 1.0],
                labels=['Low Risk', 'Medium Risk', 'High Risk']
            )
            
            risk_summary = analysis_df['risk_category'].value_counts()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Low Risk", f"{risk_summary.get('Low Risk', 0):,}")
            with col2:
                st.metric("Medium Risk", f"{risk_summary.get('Medium Risk', 0):,}")
            with col3:
                st.metric("High Risk", f"{risk_summary.get('High Risk', 0):,}")
            
            # Risk by segments
            if 'category' in analysis_df.columns:
                st.markdown("#### Risk Distribution by Product Category")
                
                risk_by_category = pd.crosstab(
                    analysis_df['category'],
                    analysis_df['risk_category'],
                    normalize='index'
                ) * 100
                
                fig_risk_category = px.bar(
                    risk_by_category,
                    title='Risk Distribution by Product Category (%)',
                    barmode='stack',
                    color_discrete_map={
                        'Low Risk': 'green',
                        'Medium Risk': 'orange',
                        'High Risk': 'red'
                    }
                )
                st.plotly_chart(fig_risk_category, use_container_width=True)
        
        # Prediction accuracy by segments (if actual labels available)
        if 'is_churned' in analysis_df.columns:
            st.markdown("#### Model Performance by Segments")
            
            segments = ['country', 'category', 'gender']
            available_segments = [seg for seg in segments if seg in analysis_df.columns]
            
            if available_segments:
                selected_segment = st.selectbox(
                    "Select segment for analysis",
                    options=available_segments
                )
                
                segment_performance = []
                for segment_value in analysis_df[selected_segment].unique():
                    segment_data = analysis_df[analysis_df[selected_segment] == segment_value]
                    
                    if len(segment_data) > 0:
                        accuracy = accuracy_score(
                            segment_data['is_churned'],
                            segment_data['predicted_churn']
                        )
                        
                        segment_performance.append({
                            'Segment': segment_value,
                            'Count': len(segment_data),
                            'Accuracy': accuracy,
                            'Actual Churn Rate': segment_data['is_churned'].mean(),
                            'Predicted Churn Rate': segment_data['predicted_churn'].mean()
                        })
                
                performance_df = pd.DataFrame(segment_performance)
                
                if not performance_df.empty:
                    st.dataframe(performance_df.round(3))
                    
                    # Visualize segment performance
                    fig_segment = px.scatter(
                        performance_df,
                        x='Actual Churn Rate',
                        y='Predicted Churn Rate',
                        size='Count',
                        hover_data=['Segment', 'Accuracy'],
                        title=f'Actual vs Predicted Churn Rate by {selected_segment.title()}'
                    )
                    
                    # Add diagonal line for perfect prediction
                    fig_segment.add_trace(go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(dash='dash', color='red')
                    ))
                    
                    st.plotly_chart(fig_segment, use_container_width=True)

def create_model_monitoring_dashboard():
    """Create model monitoring and drift detection dashboard"""
    st.markdown("### 📊 Model Monitoring Dashboard")
    
    st.info("🚧 Model monitoring features will track performance over time and detect data drift.")
    
    # Placeholder for future monitoring features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Performance Tracking")
        st.write("- Model accuracy over time")
        st.write("- Prediction distribution changes")
        st.write("- Feature importance stability")
    
    with col2:
        st.markdown("#### Data Drift Detection")
        st.write("- Input feature distribution changes")
        st.write("- Target variable drift")
        st.write("- Concept drift detection")
    
    st.markdown("#### Alerts and Notifications")
    st.write("- Performance degradation alerts")
    st.write("- Data quality issues")
    st.write("- Model retraining recommendations")

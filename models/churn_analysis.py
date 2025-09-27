import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedChurnAnalyzer:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(probability=True, random_state=42)
        }
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        
    def prepare_churn_features(self, df):
        """Enhanced feature engineering for churn prediction"""
        features_df = df.copy()
        
        # Basic demographic and behavioral features
        numeric_features = ['age', 'unit_price', 'quantity', 'purchase_frequency', 
                          'cancellations_count', 'Ratings']
        
        # Time-based features
        if 'days_since_signup' in features_df.columns:
            numeric_features.append('days_since_signup')
        if 'days_since_last_purchase' in features_df.columns:
            numeric_features.append('days_since_last_purchase')
        if 'customer_lifetime_days' in features_df.columns:
            numeric_features.append('customer_lifetime_days')
        if 'total_revenue' in features_df.columns:
            numeric_features.append('total_revenue')
            
        # Create additional engineered features
        if 'total_revenue' in features_df.columns and 'customer_lifetime_days' in features_df.columns:
            features_df['revenue_per_day'] = features_df['total_revenue'] / (features_df['customer_lifetime_days'] + 1)
            numeric_features.append('revenue_per_day')
            
        if 'purchase_frequency' in features_df.columns and 'customer_lifetime_days' in features_df.columns:
            features_df['purchase_intensity'] = features_df['purchase_frequency'] / (features_df['customer_lifetime_days'] + 1)
            numeric_features.append('purchase_intensity')
            
        # Risk score based on cancellations and days since last purchase
        if 'cancellations_count' in features_df.columns and 'days_since_last_purchase' in features_df.columns:
            features_df['risk_score'] = (features_df['cancellations_count'] * 0.3 + 
                                       features_df['days_since_last_purchase'] * 0.01)
            numeric_features.append('risk_score')
        
        # Categorical features encoding
        categorical_mappings = {}
        
        if 'gender' in features_df.columns:
            features_df['gender_encoded'] = features_df['gender'].map({'Male': 1, 'Female': 0})
            numeric_features.append('gender_encoded')
            
        if 'subscription_status' in features_df.columns:
            status_mapping = {'active': 0, 'paused': 1, 'cancelled': 2}
            features_df['status_encoded'] = features_df['subscription_status'].map(status_mapping)
            numeric_features.append('status_encoded')
            categorical_mappings['subscription_status'] = status_mapping
            
        # Country risk encoding (based on churn rates)
        if 'country' in features_df.columns and 'is_churned' in features_df.columns:
            country_churn_rates = features_df.groupby('country')['is_churned'].mean().to_dict()
            features_df['country_risk'] = features_df['country'].map(country_churn_rates)
            numeric_features.append('country_risk')
            categorical_mappings['country'] = country_churn_rates
            
        # Category preference encoding
        if 'category' in features_df.columns and 'is_churned' in features_df.columns:
            category_churn_rates = features_df.groupby('category')['is_churned'].mean().to_dict()
            features_df['category_risk'] = features_df['category'].map(category_churn_rates)
            numeric_features.append('category_risk')
            categorical_mappings['category'] = category_churn_rates
        
        # Select final features
        available_features = [col for col in numeric_features if col in features_df.columns]
        X = features_df[available_features].copy()
        # Replace inf/-inf with NaN then fill
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        # Clip extremely large values to reasonable caps to avoid float64 overflow
        caps = {
            'total_revenue': 1e9,
            'revenue_per_day': 1e7,
            'purchase_intensity': 1e4,
            'days_since_signup': 36500,  # ~100 years
            'days_since_last_purchase': 36500,
            'customer_lifetime_days': 36500,
            'unit_price': 1e6,
            'quantity': 1e6,
            'purchase_frequency': 1e5,
            'cancellations_count': 1e4,
            'Ratings': 10,
            'risk_score': 1e6,
            'country_risk': 1.0,
            'category_risk': 1.0,
        }
        for col, cap in caps.items():
            if col in X.columns:
                X[col] = np.clip(X[col].astype(float), a_min=0, a_max=cap)
        
        self.feature_names = available_features
        self.categorical_mappings = categorical_mappings
        
        return X
    
    def train_and_evaluate_models(self, X, y):
        """Train multiple models and select the best one"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features for models that need it
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        model_results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Use scaled data for SVM and Logistic Regression
            if name in ['SVM', 'Logistic Regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = (y_pred == y_test).mean()
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation score
            if name in ['SVM', 'Logistic Regression']:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            
            model_results[name] = {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'model': model
            }
            
            print(f"{name} - Accuracy: {accuracy:.3f}, AUC: {auc_score:.3f}, CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
        # Select best model based on AUC score
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['auc_score'])
        self.best_model = model_results[best_model_name]['model']
        self.best_model_name = best_model_name
        self.is_trained = True
        
        print(f"\nBest model: {best_model_name}")
        
        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test
        self.model_results = model_results
        
        return model_results
    
    def predict_churn_risk(self, X):
        """Predict churn risk for customers"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if self.best_model_name in ['SVM', 'Logistic Regression']:
            X_scaled = self.scaler.transform(X)
            probabilities = self.best_model.predict_proba(X_scaled)[:, 1]
        else:
            probabilities = self.best_model.predict_proba(X)[:, 1]
        
        return probabilities
    
    def get_top_risk_customers(self, df, X, top_n=10):
        """Identify top N customers at risk of churning"""
        probabilities = self.predict_churn_risk(X)
        
        risk_df = df.copy()
        risk_df['churn_probability'] = probabilities
        risk_df['risk_level'] = pd.cut(probabilities, 
                                     bins=[0, 0.3, 0.7, 1.0], 
                                     labels=['Low', 'Medium', 'High'])
        
        # Sort by churn probability and get top N
        top_risk = risk_df.nlargest(top_n, 'churn_probability')
        
        return top_risk[['customer_id', 'age', 'gender', 'country', 'subscription_status', 
                        'total_revenue', 'days_since_last_purchase', 'churn_probability', 'risk_level']]
    
    def get_feature_importance(self):
        """Get feature importance from the best model"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            # For models without feature_importances_ (like SVM)
            return pd.DataFrame({'feature': self.feature_names, 'importance': [0] * len(self.feature_names)})
    
    def generate_churn_insights(self, df):
        """Generate business insights about churn patterns"""
        insights = {}
        
        # Churn rate by different segments
        if 'gender' in df.columns:
            insights['churn_by_gender'] = df.groupby('gender')['is_churned'].agg(['count', 'sum', 'mean']).round(3)
        
        if 'country' in df.columns:
            insights['churn_by_country'] = df.groupby('country')['is_churned'].agg(['count', 'sum', 'mean']).round(3)
        
        if 'category' in df.columns:
            insights['churn_by_category'] = df.groupby('category')['is_churned'].agg(['count', 'sum', 'mean']).round(3)
        
        # Age group analysis
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], 
                                   labels=['18-25', '26-35', '36-45', '46-55', '55+'])
            insights['churn_by_age_group'] = df.groupby('age_group')['is_churned'].agg(['count', 'sum', 'mean']).round(3)
        
        # Revenue impact analysis
        if 'total_revenue' in df.columns:
            churned_revenue = df[df['is_churned'] == 1]['total_revenue'].sum()
            total_revenue = df['total_revenue'].sum()
            insights['revenue_at_risk'] = {
                'churned_revenue': churned_revenue,
                'total_revenue': total_revenue,
                'percentage_at_risk': (churned_revenue / total_revenue * 100) if total_revenue > 0 else 0
            }
        
        return insights
    
    def plot_model_comparison(self):
        """Plot comparison of different models"""
        if not hasattr(self, 'model_results'):
            raise ValueError("Models must be trained first")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Model performance comparison
        models = list(self.model_results.keys())
        accuracies = [self.model_results[model]['accuracy'] for model in models]
        auc_scores = [self.model_results[model]['auc_score'] for model in models]
        
        axes[0, 0].bar(models, accuracies, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].bar(models, auc_scores, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Model AUC Score Comparison')
        axes[0, 1].set_ylabel('AUC Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # ROC Curve for best model
        best_results = self.model_results[self.best_model_name]
        fpr, tpr, _ = roc_curve(self.y_test, best_results['probabilities'])
        
        axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {best_results["auc_score"]:.2f})')
        axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1, 0].set_xlim([0.0, 1.0])
        axes[1, 0].set_ylim([0.0, 1.05])
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title(f'ROC Curve - {self.best_model_name}')
        axes[1, 0].legend(loc="lower right")
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, best_results['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title(f'Confusion Matrix - {self.best_model_name}')
        axes[1, 1].set_ylabel('Actual')
        axes[1, 1].set_xlabel('Predicted')
        
        plt.tight_layout()
        return fig

# Example usage and testing
if __name__ == "__main__":
    # This would be used with real data
    print("Churn Analysis Model Ready!")
    print("Use this class with your processed data to predict customer churn.")

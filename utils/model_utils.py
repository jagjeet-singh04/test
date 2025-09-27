import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class ChurnPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def train(self, X, y):
        """Train the churn prediction model"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Churn Prediction Model Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def predict_churn_probability(self, X):
        """Predict churn probability for customers"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        probabilities = self.model.predict_proba(X)[:, 1]  # Probability of churn (class 1)
        return probabilities
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

class SalesForecaster:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def prepare_time_series_features(self, sales_data):
        """Prepare time series features for sales forecasting"""
        df = sales_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Create time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Create lag features
        for lag in [1, 7, 30]:
            df[f'revenue_lag_{lag}'] = df['revenue'].shift(lag)
            df[f'quantity_lag_{lag}'] = df['quantity_sold'].shift(lag)
        
        # Create rolling averages
        for window in [7, 30]:
            df[f'revenue_rolling_{window}'] = df['revenue'].rolling(window=window).mean()
            df[f'quantity_rolling_{window}'] = df['quantity_sold'].rolling(window=window).mean()
        
        # Drop rows with NaN values created by lag and rolling features
        df = df.dropna()
        
        return df
    
    def train(self, sales_data, target_column='revenue'):
        """Train the sales forecasting model"""
        df = self.prepare_time_series_features(sales_data)
        
        feature_columns = [col for col in df.columns if col not in ['date', target_column]]
        X = df[feature_columns]
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.feature_columns = feature_columns
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Sales Forecasting Model Performance:")
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"R²: {r2:.3f}")
        
        return {
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'predictions': y_pred,
            'actual': y_test
        }
    
    def forecast_future_sales(self, sales_data, days_ahead=90):
        """Forecast future sales"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making forecasts")
        
        df = self.prepare_time_series_features(sales_data)
        
        # Get the last date in the data
        last_date = df['date'].max()
        
        # Create future dates
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead, freq='D')
        
        forecasts = []
        
        for future_date in future_dates:
            # Create features for the future date
            future_features = {
                'year': future_date.year,
                'month': future_date.month,
                'day': future_date.day,
                'day_of_week': future_date.dayofweek,
                'day_of_year': future_date.dayofyear,
                'week_of_year': future_date.isocalendar().week
            }
            
            # Use recent values for lag features (simplified approach)
            recent_data = df.tail(30)
            for lag in [1, 7, 30]:
                if len(recent_data) >= lag:
                    future_features[f'revenue_lag_{lag}'] = recent_data['revenue'].iloc[-lag]
                    future_features[f'quantity_lag_{lag}'] = recent_data['quantity_sold'].iloc[-lag]
                else:
                    future_features[f'revenue_lag_{lag}'] = recent_data['revenue'].mean()
                    future_features[f'quantity_lag_{lag}'] = recent_data['quantity_sold'].mean()
            
            # Use recent rolling averages
            for window in [7, 30]:
                future_features[f'revenue_rolling_{window}'] = recent_data['revenue'].tail(window).mean()
                future_features[f'quantity_rolling_{window}'] = recent_data['quantity_sold'].tail(window).mean()
            
            # Create feature vector
            feature_vector = []
            for col in self.feature_columns:
                if col in future_features:
                    feature_vector.append(future_features[col])
                else:
                    feature_vector.append(0)  # Default value for missing features
            
            # Make prediction
            prediction = self.model.predict([feature_vector])[0]
            forecasts.append({
                'date': future_date,
                'predicted_revenue': max(0, prediction)  # Ensure non-negative predictions
            })
        
        return pd.DataFrame(forecasts)

def save_models(churn_model, sales_model, filepath_prefix='models/'):
    """Save trained models"""
    import os
    os.makedirs(filepath_prefix, exist_ok=True)
    
    if churn_model.is_trained:
        joblib.dump(churn_model, f'{filepath_prefix}churn_model.pkl')
    
    if sales_model.is_trained:
        joblib.dump(sales_model, f'{filepath_prefix}sales_model.pkl')

def load_models(filepath_prefix='models/'):
    """Load trained models"""
    churn_model = joblib.load(f'{filepath_prefix}churn_model.pkl')
    sales_model = joblib.load(f'{filepath_prefix}sales_model.pkl')
    return churn_model, sales_model

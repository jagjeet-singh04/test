# # churn_model.py
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# import joblib
# from datetime import datetime
# import warnings
# warnings.filterwarnings('ignore')

# class ChurnPredictor:
#     def __init__(self, churn_threshold_days: int = 90):
#         """Initialize predictor.

#         Args:
#             churn_threshold_days: Number of days without a purchase to consider a user churned.
#         """
#         self.model = None
#         self.scaler = StandardScaler()
#         self.label_encoders = {}
#         self.feature_columns = []
#         self.churn_threshold_days = churn_threshold_days
        
#     def preprocess_data(self, df):
#         """Preprocess the customer data for churn prediction"""
#         # Create a copy to avoid modifying original data
#         data = df.copy()
        
#         # Convert dates to datetime
#         date_columns = ['signup_date', 'last_purchase_date']
#         for col in date_columns:
#             if col in data.columns:
#                 data[col] = pd.to_datetime(data[col], errors='coerce')
        
#         # Use today's date as reference point instead of max date from data
#         # This prevents issues with future dates in the dataset
#         current_date = pd.Timestamp.now()
        
#         # Calculate days since last purchase (recency)
#         # Handle NaT (Not a Time) values that might result from date parsing errors
#         if 'last_purchase_date' in data.columns:
#             data['days_since_last_purchase'] = (current_date - data['last_purchase_date']).dt.days
#             data['days_since_last_purchase'] = data['days_since_last_purchase'].fillna(0)  # Fill NaN with 0
#         else:
#             data['days_since_last_purchase'] = 0
        
#         # Calculate customer tenure
#         if 'signup_date' in data.columns:
#             data['tenure_days'] = (current_date - data['signup_date']).dt.days
#             data['tenure_days'] = data['tenure_days'].fillna(0)  # Fill NaN with 0
#         else:
#             data['tenure_days'] = 0
        
#         # Create target variable - churn definition (configurable threshold)
#         # Assuming churn if subscription_status is 'cancelled' or no purchase in last N days
#         if 'subscription_status' in data.columns:
#             data['is_churned'] = (
#                 (data['subscription_status'] == 'cancelled') | 
#                 (data['days_since_last_purchase'] > self.churn_threshold_days)
#             ).astype(int)
#         else:
#             data['is_churned'] = (data['days_since_last_purchase'] > self.churn_threshold_days).astype(int)
        
#         # Feature engineering (robust to missing columns)
#         # total_spent
#         if 'total_spent' in data.columns:
#             # keep existing
#             data['total_spent'] = pd.to_numeric(data['total_spent'], errors='coerce').fillna(0)
#         elif 'unit_price' in data.columns and 'quantity' in data.columns:
#             data['total_spent'] = (
#                 pd.to_numeric(data['unit_price'], errors='coerce').fillna(0)
#                 * pd.to_numeric(data['quantity'], errors='coerce').fillna(0)
#             )
#         else:
#             data['total_spent'] = 0.0

#         # avg_purchase_value
#         if 'purchase_frequency' in data.columns:
#             data['avg_purchase_value'] = data['total_spent'] / data['purchase_frequency'].replace(0, 1)
#         else:
#             data['avg_purchase_value'] = data['total_spent']

#         # Additional engineered features
#         # 1) Purchase recency bins (categorical)
#         recency_bins = [0, 30, 60, 90, 180, np.inf]
#         recency_labels = ['0-30', '31-60', '61-90', '91-180', '180+']
#         # Coerce negatives to 0 to avoid issues
#         data['days_since_last_purchase'] = data['days_since_last_purchase'].clip(lower=0)
#         data['purchase_recency_bins'] = pd.cut(
#             data['days_since_last_purchase'], bins=recency_bins, labels=recency_labels, right=True, include_lowest=True
#         ).astype(str)

#         # 2) Spending score (normalized total_spent)
#         ts_mean = data['total_spent'].mean(skipna=True)
#         ts_std = data['total_spent'].std(skipna=True)
#         if pd.isna(ts_std) or ts_std == 0:
#             data['spending_score'] = 0.0
#         else:
#             data['spending_score'] = (data['total_spent'] - ts_mean) / ts_std
        
#         # Fill any remaining NaN values that might have been created during feature engineering
#         numeric_columns = ['total_spent', 'avg_purchase_value', 'days_since_last_purchase', 'tenure_days', 'spending_score']
#         for col in numeric_columns:
#             if col in data.columns:
#                 data[col] = data[col].fillna(0)
        
#         return data
    
#     def prepare_features(self, data, for_prediction=False):
#         """Prepare features for model training"""
#         # Select features for modeling
#         base_features = [
#             'age', 'tenure_days', 'days_since_last_purchase', 'cancellations_count',
#             'purchase_frequency', 'total_spent', 'avg_purchase_value', 'Ratings', 'spending_score'
#         ]
#         # Use only features that actually exist in the provided data for training
#         safe_base_features = [c for c in base_features if c in data.columns]
        
#         # Encode categorical variables
#         categorical_columns = ['gender', 'country', 'category', 'purchase_recency_bins']
        
#         feature_columns = list(safe_base_features)
#         for col in categorical_columns:
#             if col in data.columns:
#                 if for_prediction and col in self.label_encoders:
#                     # Use existing encoders for prediction
#                     # Handle unseen categories by mapping them to a special 'Unknown' class
#                     le = self.label_encoders[col]
#                     known = set(map(str, le.classes_))
#                     # Ensure 'Unknown' exists in classes
#                     if 'Unknown' not in le.classes_:
#                         le.classes_ = np.append(le.classes_, 'Unknown')
#                         known = set(map(str, le.classes_))
#                     # Replace unseen with 'Unknown' then transform
#                     data[col] = data[col].astype(str).apply(lambda x: x if x in known else 'Unknown')
#                     data[col + '_encoded'] = le.transform(data[col].astype(str))
#                     feature_columns.append(col + '_encoded')
#                 elif for_prediction and col not in self.label_encoders:
#                     # During prediction, if the encoder was not seen during training, skip this column entirely
#                     continue
#                 else:
#                     # Create new encoders for training
#                     self.label_encoders[col] = LabelEncoder()
#                     # During training, also add 'Unknown' to classes to stabilize mapping
#                     series = data[col].astype(str).fillna('Unknown')
#                     if 'Unknown' not in series.unique():
#                         series = pd.concat([series, pd.Series(['Unknown'])], ignore_index=True)
#                     data[col + '_encoded'] = self.label_encoders[col].fit_transform(series)[:len(data)]
#                     feature_columns.append(col + '_encoded')
        
#         # Update or align feature columns depending on mode
#         if not for_prediction or not self.feature_columns:
#             self.feature_columns = feature_columns
        
#         # Handle missing values - fill with median/mode instead of dropping
#         target_feature_set = self.feature_columns if for_prediction else feature_columns
#         for col in target_feature_set:
#             if col in data.columns:
#                 if data[col].dtype in ['int64', 'float64']:
#                     data[col] = data[col].fillna(data[col].median() if not data[col].empty else 0)
#                 else:
#                     data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 0)
        
#         # Only drop rows for training if we have the target variable
#         if not for_prediction and 'is_churned' in data.columns:
#             data = data.dropna(subset=['is_churned'])
#             return data[feature_columns], data['is_churned']
#         else:
#             # Ensure all expected columns exist; fill missing with zeros, drop extras, and order columns
#             for col in self.feature_columns:
#                 if col not in data.columns:
#                     data[col] = 0
#             X = data[self.feature_columns]
#             return X, None
    
#     def train_model(self, df):
#         """Train the churn prediction model"""
#         # Preprocess data
#         processed_data = self.preprocess_data(df)
#         X, y = self.prepare_features(processed_data, for_prediction=False)
        
#         # Split data
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42, stratify=y
#         )
        
#         # Scale features
#         X_train_scaled = self.scaler.fit_transform(X_train)
#         X_test_scaled = self.scaler.transform(X_test)
        
#         # Choose model with imbalance handling
#         model_trained = False
#         # Try XGBoost
#         try:
#             from xgboost import XGBClassifier
#             pos = int(y_train.sum()) if hasattr(y_train, 'sum') else int(np.sum(y_train))
#             neg = len(y_train) - pos
#             scale_pos_weight = (neg / pos) if pos > 0 else 1.0
#             self.model = XGBClassifier(
#                 n_estimators=300,
#                 max_depth=6,
#                 learning_rate=0.05,
#                 subsample=0.8,
#                 colsample_bytree=0.8,
#                 reg_lambda=1.0,
#                 random_state=42,
#                 eval_metric='logloss',
#                 scale_pos_weight=scale_pos_weight,
#                 n_jobs=-1,
#             )
#             self.model.fit(X_train_scaled, y_train)
#             model_trained = True
#         except Exception:
#             pass

#         # Try LightGBM if XGBoost not used
#         if not model_trained:
#             try:
#                 from lightgbm import LGBMClassifier
#                 self.model = LGBMClassifier(
#                     n_estimators=400,
#                     max_depth=-1,
#                     learning_rate=0.05,
#                     subsample=0.8,
#                     colsample_bytree=0.8,
#                     class_weight='balanced',
#                     random_state=42,
#                 )
#                 self.model.fit(X_train_scaled, y_train)
#                 model_trained = True
#             except Exception:
#                 pass

#         # Fallback to RandomForest with class imbalance handling
#         if not model_trained:
#             self.model = RandomForestClassifier(
#                 n_estimators=200,
#                 max_depth=12,
#                 min_samples_split=5,
#                 min_samples_leaf=2,
#                 class_weight='balanced',
#                 random_state=42,
#                 n_jobs=-1,
#             )
#             self.model.fit(X_train_scaled, y_train)
        
#         # Evaluate model with richer metrics
#         train_score = self.model.score(X_train_scaled, y_train)
#         test_score = self.model.score(X_test_scaled, y_test)
#         y_pred = self.model.predict(X_test_scaled)
#         y_proba = self.model.predict_proba(X_test_scaled)[:, 1]

#         print(f"Training Accuracy: {train_score:.3f}")
#         print(f"Test Accuracy: {test_score:.3f}")
#         print("\nClassification Report:\n", classification_report(y_test, y_pred))
#         print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
#         try:
#             auc = roc_auc_score(y_test, y_proba)
#             print(f"AUC Score: {auc:.3f}")
#         except Exception:
#             pass
#         try:
#             importances = pd.Series(self.model.feature_importances_, index=self.feature_columns)
#             top_features = importances.sort_values(ascending=False).head(10)
#             print("Top features contributing to churn:\n", top_features)
#         except Exception:
#             pass
        
#         return processed_data, X_test_scaled, y_test
    
#     def predict_churn_probability(self, df):
#         """Predict churn probability for all customers"""
#         # Create a copy to avoid modifying original data
#         processed_data = self.preprocess_data(df)
#         X, _ = self.prepare_features(processed_data, for_prediction=True)
        
#         if self.model is None:
#             raise ValueError("Model not trained yet. Call train_model first.")
        
#         X_scaled = self.scaler.transform(X)
#         churn_probs = self.model.predict_proba(X_scaled)[:, 1]
        
#         # Create results DataFrame with same length as processed_data
#         results = processed_data.copy()
#         results['churn_probability'] = churn_probs
#         # Default risk binning; UI may override with custom threshold
#         results['churn_risk'] = pd.cut(
#             churn_probs,
#             bins=[0, 0.3, 0.7, 1],
#             labels=['Low', 'Medium', 'High'], include_lowest=True, right=True
#         )
        
#         return results
    
#     def get_top_churn_risks(self, df, top_n=10):
#         """Get top N customers with highest churn probability"""
#         predictions = self.predict_churn_probability(df)
#         # Select available columns
#         available_cols = ['customer_id', 'age', 'gender', 'country', 'churn_probability', 'churn_risk',
#                          'days_since_last_purchase', 'cancellations_count', 'Ratings']
#         cols_to_use = [col for col in available_cols if col in predictions.columns]
#         top_risks = predictions.nlargest(top_n, 'churn_probability')[cols_to_use]
#         return top_risks
    
#     def save_model(self, filepath):
#         """Save the trained model and preprocessing objects"""
#         model_data = {
#             'model': self.model,
#             'scaler': self.scaler,
#             'label_encoders': self.label_encoders,
#             'feature_columns': self.feature_columns,
#             'churn_threshold_days': self.churn_threshold_days
#         }
#         joblib.dump(model_data, filepath)
    
#     def load_model(self, filepath):
#         """Load a trained model and preprocessing objects"""
#         model_data = joblib.load(filepath)
#         self.model = model_data['model']
#         self.scaler = model_data['scaler']
#         self.label_encoders = model_data['label_encoders']
#         self.feature_columns = model_data['feature_columns']
#         self.churn_threshold_days = model_data.get('churn_threshold_days', 90)

# # Example usage and testing
# if __name__ == "__main__":
#     # Create sample data for testing
#     sample_data = pd.DataFrame({
#         'customer_id': range(1000),
#         'age': np.random.randint(18, 70, 1000),
#         'gender': np.random.choice(['Male', 'Female'], 1000),
#         'country': np.random.choice(['US', 'UK', 'Canada'], 1000),
#         'signup_date': pd.date_range('2020-01-01', periods=1000, freq='D'),
#         'last_purchase_date': pd.date_range('2023-01-01', periods=1000, freq='D'),
#         'subscription_status': np.random.choice(['active', 'cancelled'], 1000, p=[0.8, 0.2]),
#         'total_spent': np.random.exponential(100, 1000),
#         'purchase_frequency': np.random.poisson(5, 1000),
#         'cancellations_count': np.random.randint(0, 3, 1000),
#         'Ratings': np.random.uniform(1, 5, 1000)
#     })
    
#     # Initialize and train model
#     churn_predictor = ChurnPredictor()
#     processed_data, X_test, y_test = churn_predictor.train_model(sample_data)
    
#     # Get top churn risks
#     top_risks = churn_predictor.get_top_churn_risks(sample_data)
#     print("Top 10 Customers with Highest Churn Risk:")
#     print(top_risks)
    
#     # Save the model
#     churn_predictor.save_model('churn_model.pkl')
#     print("Model saved successfully!")
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, file_path):
        """Load data from CSV or Excel file"""
        try:
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.lower().endswith('.xlsx'):
                # Use openpyxl for .xlsx
                df = pd.read_excel(file_path, engine='openpyxl')
            elif file_path.lower().endswith('.xls'):
                # Use xlrd for legacy .xls
                try:
                    df = pd.read_excel(file_path, engine='xlrd')
                except ImportError as ie:
                    raise ImportError("Missing optional dependency 'xlrd'. Install xlrd>=2.0.1 for .xls support.") from ie
            else:
                raise ValueError("Unsupported file format. Please use CSV, .xlsx, or .xls files.")

            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def parse_date_flexible(self, date_str):
        """Parse dates in various formats"""
        if pd.isna(date_str) or date_str == '':
            return None
            
        # Convert to string if not already
        date_str = str(date_str).strip()
        
        # Common date formats to try
        date_formats = [
            '%m/%d/%Y',     # 2/21/2023
            '%m-%d-%Y',     # 2-21-2023
            '%d-%m-%Y',     # 21-2-2023
            '%Y-%m-%d',     # 2023-2-21
            '%d/%m/%Y',     # 21/2/2023
            '%m/%d/%y',     # 2/21/23
            '%d-%m-%y',     # 21-2-23
            '%Y/%m/%d',     # 2023/2/21
            '%d-%b-%Y',     # 21-Feb-2023
            '%b-%d-%Y',     # Feb-21-2023
            '%d %b %Y',     # 21 Feb 2023
            '%B %d, %Y',    # February 21, 2023
            '%m-%d-%Y',     # 01-07-2021
            '%d-%m-%Y',     # 07-01-2021
        ]
        
        for fmt in date_formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue
        
        # If none of the formats work, try pandas' automatic parsing
        try:
            return pd.to_datetime(date_str, infer_datetime_format=True)
        except:
            print(f"Could not parse date: {date_str}")
            return None
    
    def clean_and_preprocess(self, df):
        """Clean and preprocess the data"""
        df_clean = df.copy()
        
        # Handle date columns with flexible parsing
        date_columns = ['signup_date', 'last_purchase_date']
        for col in date_columns:
            if col in df_clean.columns:
                print(f"Processing {col}...")
                df_clean[col] = df_clean[col].apply(self.parse_date_flexible)
        
        # Create derived features
        if 'signup_date' in df_clean.columns and 'last_purchase_date' in df_clean.columns:
            df_clean['days_since_signup'] = (pd.Timestamp.now() - df_clean['signup_date']).dt.days
            df_clean['days_since_last_purchase'] = (pd.Timestamp.now() - df_clean['last_purchase_date']).dt.days
            df_clean['customer_lifetime_days'] = (df_clean['last_purchase_date'] - df_clean['signup_date']).dt.days
            # Ensure non-negative durations to avoid divide-by-zero or negative lifetimes
            df_clean['days_since_signup'] = df_clean['days_since_signup'].clip(lower=0)
            df_clean['days_since_last_purchase'] = df_clean['days_since_last_purchase'].clip(lower=0)
            df_clean['customer_lifetime_days'] = df_clean['customer_lifetime_days'].clip(lower=0)
        
        # Create churn label (customers who haven't purchased in 90+ days or have cancelled/paused status)
        df_clean['is_churned'] = 0
        if 'subscription_status' in df_clean.columns:
            df_clean['is_churned'] = np.where(
                (df_clean['subscription_status'].isin(['cancelled', 'paused'])) | 
                (df_clean['days_since_last_purchase'] > 90), 1, 0
            )
        
        # Calculate total revenue per customer
        if 'unit_price' in df_clean.columns and 'quantity' in df_clean.columns:
            df_clean['total_revenue'] = df_clean['unit_price'] * df_clean['quantity']
        
        # Handle missing values
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_columns] = df_clean[numeric_columns].fillna(df_clean[numeric_columns].median())
        
        categorical_columns = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col not in date_columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')
        
        return df_clean
    
    def prepare_features_for_churn(self, df):
        """Prepare features for churn prediction"""
        feature_columns = [
            'age', 'unit_price', 'quantity', 'purchase_frequency', 
            'cancellations_count', 'days_since_signup', 'days_since_last_purchase',
            'customer_lifetime_days', 'total_revenue', 'Ratings'
        ]
        
        # Select available numeric features
        available_features = [col for col in feature_columns if col in df.columns]
        X = df[available_features].copy()
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        # Encode categorical features
        categorical_features = ['gender', 'country', 'category', 'subscription_status']
        for col in categorical_features:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    X[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
        
        return X
    
    def prepare_sales_data(self, df):
        """Prepare data for sales forecasting"""
        # Group by date for time series analysis
        if 'last_purchase_date' in df.columns and 'total_revenue' in df.columns:
            sales_data = df.groupby(df['last_purchase_date'].dt.date).agg({
                'total_revenue': 'sum',
                'quantity': 'sum',
                'order_id': 'count'
            }).reset_index()
            
            sales_data.columns = ['date', 'revenue', 'quantity_sold', 'orders_count']
            sales_data['date'] = pd.to_datetime(sales_data['date'])
            sales_data = sales_data.sort_values('date')
            
            return sales_data
        
        return None

# Test the data processor
if __name__ == "__main__":
    processor = DataProcessor()
    
    # Create sample data for testing
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
        'purchase_frequency': [37, 35, 44],
        'cancellations_count': [0, 0, 3],
        'Ratings': [4.2, 4.0, 3.9],
        'category': ['Sports', 'Home', 'Clothing']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data:")
    print(df.head())
    
    # Test preprocessing
    df_clean = processor.clean_and_preprocess(df)
    print("\nCleaned data:")
    print(df_clean.head())
    
    # Test feature preparation
    X = processor.prepare_features_for_churn(df_clean)
    print("\nFeatures for churn prediction:")
    print(X.head())

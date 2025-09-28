import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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
        self.categorical_mappings = {}
        self.debug = False

    def set_debug(self, enabled: bool = True):
        """Enable/disable debug mode for verbose prints."""
        self.debug = enabled
    
    def _to_numeric(self, s):
        """Coerce strings like "1,834" or '"1,033.48"' to numbers.
        Leaves NaN on failure.
        """
        if self.debug:
            try:
                print("_to_numeric sample:", pd.Series(s).head().to_list())
            except Exception:
                pass
        return pd.to_numeric(
            pd.Series(s, copy=False).astype(str).str.replace(r'[^\d\.-]', '', regex=True),
            errors='coerce'
        )
        
    def prepare_churn_features(self, df, impute: str | None = None):
        """Enhanced feature engineering for churn prediction with numeric coercion.

        Parameters:
        - df: input DataFrame
        - impute: one of {'median', 'mean', 'drop', None}. If None, fall back to fillna(0) for backward-compat.
        """
        features_df = df.copy()

        # 1) Base numeric candidates
        numeric_features = [
            'age', 'unit_price', 'quantity', 'purchase_frequency',
            'cancellations_count', 'Ratings'
        ]

        # Time-based
        if 'days_since_signup' in features_df.columns:
            numeric_features.append('days_since_signup')
        if 'days_since_last_purchase' in features_df.columns:
            numeric_features.append('days_since_last_purchase')
        if 'customer_lifetime_days' in features_df.columns:
            numeric_features.append('customer_lifetime_days')
        if 'total_revenue' in features_df.columns:
            numeric_features.append('total_revenue')

        # Coerce BEFORE engineering features
        for col in list(set(numeric_features) & set(features_df.columns)):
            features_df[col] = self._to_numeric(features_df[col])

        # 2) Engineered features (safe now)
        if 'total_revenue' in features_df.columns and 'customer_lifetime_days' in features_df.columns:
            features_df['revenue_per_day'] = features_df['total_revenue'] / (features_df['customer_lifetime_days'] + 1)
            numeric_features.append('revenue_per_day')

        if 'purchase_frequency' in features_df.columns and 'customer_lifetime_days' in features_df.columns:
            features_df['purchase_intensity'] = features_df['purchase_frequency'] / (features_df['customer_lifetime_days'] + 1)
            numeric_features.append('purchase_intensity')

        if 'cancellations_count' in features_df.columns and 'days_since_last_purchase' in features_df.columns:
            features_df['risk_score'] = (
                features_df['cancellations_count'] * 0.3 + features_df['days_since_last_purchase'] * 0.01
            )
            numeric_features.append('risk_score')

        # Categorical encodings
        if 'gender' in features_df.columns:
            features_df['gender_encoded'] = features_df['gender'].map({'Male': 1, 'Female': 0})
            numeric_features.append('gender_encoded')

        if 'subscription_status' in features_df.columns:
            status_mapping = {'active': 0, 'paused': 1, 'cancelled': 2}
            features_df['status_encoded'] = features_df['subscription_status'].map(status_mapping)
            numeric_features.append('status_encoded')
            self.categorical_mappings['subscription_status'] = status_mapping

        if 'country' in features_df.columns and 'is_churned' in features_df.columns:
            country_churn_rates = features_df.groupby('country')['is_churned'].mean().to_dict()
            features_df['country_risk'] = features_df['country'].map(country_churn_rates)
            numeric_features.append('country_risk')
            self.categorical_mappings['country'] = country_churn_rates

        if 'category' in features_df.columns and 'is_churned' in features_df.columns:
            category_churn_rates = features_df.groupby('category')['is_churned'].mean().to_dict()
            features_df['category_risk'] = features_df['category'].map(category_churn_rates)
            numeric_features.append('category_risk')
            self.categorical_mappings['category'] = category_churn_rates

        # Build X
        available_features = [c for c in numeric_features if c in features_df.columns]
        X = features_df[available_features].copy()
        X = X.replace([np.inf, -np.inf], np.nan)

        # Handle missing values
        if impute in ('median', 'mean'):
            strategy = impute
            imp = SimpleImputer(strategy=strategy)
            X_imputed = imp.fit_transform(X)
            X = pd.DataFrame(X_imputed, columns=available_features, index=features_df.index)
            if self.debug:
                n_missing = X.isna().sum().sum()
                print(f"Imputed missing values using {strategy}. Remaining NaNs: {n_missing}")
        elif impute == 'drop':
            before = len(X)
            mask = ~X.isna().any(axis=1)
            X = X.loc[mask]
            if self.debug:
                print(f"Dropped rows with NaNs: {before - len(X)} of {before}")
        else:
            # Backward-compatible: fill remaining NaNs with 0
            X = X.fillna(0)

        # Clip extremes
        caps = {
            'total_revenue': 1e9, 'revenue_per_day': 1e7, 'purchase_intensity': 1e4,
            'days_since_signup': 36500, 'days_since_last_purchase': 36500, 'customer_lifetime_days': 36500,
            'unit_price': 1e6, 'quantity': 1e6, 'purchase_frequency': 1e5, 'cancellations_count': 1e4,
            'Ratings': 10, 'risk_score': 1e6, 'country_risk': 1.0, 'category_risk': 1.0
        }
        for col, cap in caps.items():
            if col in X.columns:
                X[col] = np.clip(X[col].astype(float), 0, cap)

        self.feature_names = available_features
        return X

    
    def train_and_evaluate_models(self, X, y, use_smote: bool = True, cv_splits: int = 5):
        """Train multiple models and select the best one.

        Parameters:
        - X: features DataFrame
        - y: target Series/array
        - use_smote: apply SMOTE to the training split if imbalance is detected and imblearn is available
        - cv_splits: number of stratified CV splits
        """
        # Ensure y has no NaNs and align X accordingly
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_series = y.squeeze()
        else:
            y_series = pd.Series(y)
        valid_mask = ~pd.isna(y_series)
        if valid_mask.sum() < len(y_series):
            X = X.loc[valid_mask]
            y_series = y_series.loc[valid_mask]
            if self.debug:
                print(f"Dropped {len(valid_mask) - valid_mask.sum()} rows due to NaN targets")

        if self.debug and hasattr(y_series, 'value_counts'):
            print("Target distribution:\n", y_series.value_counts())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_series, test_size=0.2, random_state=42, stratify=y_series
        )
        
        # Scale features for models that need it
        # Optionally apply SMOTE on training split only
        X_resampled, y_resampled = X_train, y_train
        if use_smote:
            try:
                import importlib
                module = importlib.import_module('imblearn.over_sampling')
                SMOTE = getattr(module, 'SMOTE')
                # Simple imbalance check
                class_counts = y_train.value_counts(normalize=True)
                min_prop = class_counts.min()
                if min_prop < 0.35:  # threshold; adjust as needed
                    smote = SMOTE(random_state=42)
                    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                    if self.debug:
                        print("Applied SMOTE. New class distribution:")
                        print(pd.Series(y_resampled).value_counts(normalize=True))
            except Exception as e:
                if self.debug:
                    print(f"SMOTE not applied ({e}). Proceeding without resampling.")

        # Prepare scaled versions for models that require scaling
        X_train_scaled = self.scaler.fit_transform(X_resampled)
        X_test_scaled = self.scaler.transform(X_test)
        
        model_results = {}
        cv_strategy = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Use scaled data for SVM and Logistic Regression
            if name in ['SVM', 'Logistic Regression']:
                model.fit(X_train_scaled, y_resampled)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_resampled, y_resampled)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = (y_pred == y_test).mean()
            auc_score = roc_auc_score(y_test, y_pred_proba)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            # Cross-validation score
            if name in ['SVM', 'Logistic Regression']:
                # Use a pipeline to avoid CV leakage of scaling
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model.__class__(**model.get_params()))
                ])
                cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv_strategy, scoring='roc_auc')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='roc_auc')
            
            model_results[name] = {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'model': model,
                'precision': report.get('1', {}).get('precision', 0.0),
                'recall': report.get('1', {}).get('recall', 0.0),
                'f1': report.get('1', {}).get('f1-score', 0.0),
                'classification_report': report
            }
            
            print(f"{name} - Accuracy: {accuracy:.3f}, AUC: {auc_score:.3f}, CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
            try:
                print(classification_report(y_test, y_pred, digits=3, zero_division=0))
            except Exception:
                pass
        
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
    
    def get_top_risk_customers(
        self, df, X, top_n: int = 10, high_only: bool = True,
        sort_by: list | None = None, threshold: float | None = None
    ):
        """Return the top N highest-risk customers, with optional probability threshold.

        Parameters:
        - df: Original DataFrame.
        - X: Feature DataFrame used for prediction (to align indices if rows were dropped).
        - top_n: Number of rows to return.
        - high_only: If True, filter to 'High' risk_level (ignored if threshold is provided).
        - sort_by: Custom sort columns; defaults to sensible tie-breakers.
        - threshold: If provided, filter rows with churn_probability >= threshold.
        """
        probs = self.predict_churn_risk(X)

        # Align df to X when X is a DataFrame with filtered rows (e.g., impute='drop')
        if isinstance(X, pd.DataFrame):
            base_df = df.loc[X.index]
        else:
            base_df = df

        risk_df = base_df.copy().reset_index(drop=True)
        risk_df['churn_probability'] = probs
        risk_df['risk_level'] = pd.cut(
            probs, bins=[0, 0.3, 0.7, 1.0], labels=['Low', 'Medium', 'High']
        )

        # Bring engineered tie-breaker if available
        if isinstance(X, pd.DataFrame) and 'risk_score' in X.columns:
            risk_df['risk_score'] = X['risk_score'].values

        # CRITICAL FIX: Make sure the columns used for sorting are numeric
        for col in ['days_since_last_purchase', 'cancellations_count', 'total_revenue', 'risk_score']:
            if col in risk_df.columns:
                risk_df[col] = self._to_numeric(risk_df[col])

        if threshold is not None:
            try:
                thr = float(threshold)
            except Exception:
                thr = None
            if thr is not None:
                risk_df = risk_df[risk_df['churn_probability'] >= thr]
        elif high_only:
            risk_df = risk_df[risk_df['risk_level'] == 'High']

        if sort_by is None:
            sort_by = [c for c in [
                'churn_probability', 'risk_score',
                'days_since_last_purchase', 'cancellations_count', 'total_revenue'
            ] if c in risk_df.columns]
        if not sort_by:
            sort_by = ['churn_probability']

        risk_df = risk_df.sort_values(by=sort_by, ascending=[False] * len(sort_by))

        # Friendly columns
        candidate_id_cols = ['customer_id', 'CustomerID', 'customerId', 'id']
        id_col = next((c for c in candidate_id_cols if c in risk_df.columns), None)
        cols = [c for c in [
            id_col, 'age', 'gender', 'country', 'subscription_status',
            'total_revenue', 'days_since_last_purchase',
            'cancellations_count', 'churn_probability', 'risk_level'
        ] if c is not None and c in risk_df.columns]

        out = (risk_df[cols] if cols else risk_df[['churn_probability', 'risk_level']]).head(top_n).reset_index(drop=True)
        if 'churn_probability' in out.columns:
            out['churn_probability'] = out['churn_probability'].round(3)
        return out
    
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
            # Derive customers_at_risk using available signals
            customers_at_risk = 0
            if 'churn_probability' in df.columns:
                try:
                    customers_at_risk = int((pd.to_numeric(df['churn_probability'], errors='coerce') >= 0.7).sum())
                except Exception:
                    customers_at_risk = 0
            elif 'risk_level' in df.columns:
                try:
                    customers_at_risk = int((df['risk_level'] == 'High').sum())
                except Exception:
                    customers_at_risk = 0
            elif 'is_churned' in df.columns:
                try:
                    customers_at_risk = int(pd.to_numeric(df['is_churned'], errors='coerce').fillna(0).sum())
                except Exception:
                    customers_at_risk = 0

            churned_revenue = df[df['is_churned'] == 1]['total_revenue'].sum()
            total_revenue = df['total_revenue'].sum()
            insights['revenue_at_risk'] = {
                'churned_revenue': churned_revenue,
                'total_revenue': total_revenue,
                'percentage_at_risk': (churned_revenue / total_revenue * 100) if total_revenue > 0 else 0,
                'customers_at_risk': customers_at_risk
            }
        
        return insights
    
    def plot_churn_trends(self, df, date_col=None, period='M'):
        """Plot time-series churn trend that's resilient to varied uploads.

        The function will:
        - Auto-detect a date column if 'date_col' is not provided (tries common names).
        - Use 'is_churned' if present, otherwise use 'churn_probability' if present.
        - Aggregate by the given period ('M' monthly, 'Q' quarterly, 'W' weekly).

        Returns: matplotlib Figure
        """
        # Determine signal column: actual labels or predicted probability
        signal_col = None
        if 'is_churned' in df.columns:
            signal_col = 'is_churned'
        elif 'churn_probability' in df.columns:
            signal_col = 'churn_probability'
        else:
            raise ValueError("DataFrame must contain 'is_churned' or 'churn_probability' to plot trends")

        # Determine date column if not provided
        if date_col is None:
            candidate_dates = ['signup_date', 'last_purchase_date', 'order_date', 'invoice_date', 'date']
            date_col = next((c for c in candidate_dates if c in df.columns), None)
            if date_col is None:
                raise ValueError("No date column found. Provide 'date_col' explicitly.")

        ts = df.copy()
        ts[date_col] = pd.to_datetime(ts[date_col], errors='coerce')
        ts = ts.dropna(subset=[date_col])
        if ts.empty:
            raise ValueError("No valid dates found after parsing; cannot compute churn trend")

        ts['period'] = ts[date_col].dt.to_period(period)
        trend = ts.groupby('period')[signal_col].mean().sort_index()

        # Plot
        x = trend.index.to_timestamp()
        y = trend.values
        title_suffix = 'Churn Rate' if signal_col == 'is_churned' else 'Avg Churn Probability'
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, y, marker='o', color='red')
        ax.set_title(f"{title_suffix} Trend Over Time")
        ax.set_ylabel(title_suffix)
        xlabel = {'M': 'Time (Monthly)', 'Q': 'Time (Quarterly)', 'W': 'Time (Weekly)'}\
            .get(period, f'Time (Period={period})')
        ax.set_xlabel(xlabel)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    def segment_customers(self, df, X, method='kmeans', by='probability', n_segments=3,
                          bins=(0.0, 0.3, 0.7, 1.0), labels=None,
                          eps=0.5, min_samples=5, scale_features=True, return_model=False):
        """Create customer segments flexibly for varied uploaded datasets.

        Parameters:
        - df: original DataFrame to attach segment labels (and churn_probability) to
        - X: feature matrix from prepare_churn_features
        - method: 'kmeans' | 'dbscan' | 'bins'
        - by: 'probability' (uses predicted churn probabilities; requires trained model)
              or 'features' (clusters on features; no labels required)
        - n_segments: kmeans clusters
        - bins, labels: used only for 'bins' with by='probability'
        - eps, min_samples: DBSCAN params
        - scale_features: scale features for 'features' clustering
        - return_model: also return fitted clustering model
        """
        work_df = df.copy()

        # Determine data to cluster
        if by == 'probability':
            if not self.is_trained:
                raise ValueError("Model must be trained before segmenting by probability")
            probs = self.predict_churn_risk(X)
            work_df['churn_probability'] = probs
            data = probs.reshape(-1, 1)
        elif by == 'features':
            data = X.values if isinstance(X, pd.DataFrame) else X
            if scale_features:
                data = self.scaler.fit_transform(data)
        else:
            raise ValueError("by must be 'probability' or 'features'")

        fitted_model = None
        if method == 'kmeans':
            km = KMeans(n_clusters=n_segments, n_init=10, random_state=42)
            segments = km.fit_predict(data)
            fitted_model = km

            # If segmenting by probability, order clusters by mean risk for intuitive labels
            if by == 'probability':
                order = pd.Series(work_df['churn_probability']).groupby(segments).mean()\
                    .sort_values().index.tolist()
                remap = {cluster: rank for rank, cluster in enumerate(order)}
                segments = np.vectorize(remap.get)(segments)

            # Create default labels
            if labels is None and by == 'probability' and n_segments in (2, 3):
                labels = ['Low', 'High'] if n_segments == 2 else ['Low', 'Medium', 'High']

            if labels is not None and len(labels) == len(np.unique(segments)):
                label_map = {i: labels[i] for i in range(len(labels))}
                segment_labels = [label_map[int(s)] for s in segments]
            else:
                segment_labels = [int(s) for s in segments]

        elif method == 'dbscan':
            db = DBSCAN(eps=eps, min_samples=min_samples)
            segments = db.fit_predict(data)
            fitted_model = db
            segment_labels = ['Noise' if s == -1 else f'Segment {int(s)+1}' for s in segments]

        elif method == 'bins':
            if by != 'probability':
                raise ValueError("'bins' method requires by='probability'")
            if labels is None and len(bins) - 1 == 3:
                labels = ['Low', 'Medium', 'High']
            segment_labels = pd.cut(work_df['churn_probability'], bins=bins, labels=labels, include_lowest=True)
        else:
            raise ValueError("method must be 'kmeans', 'dbscan', or 'bins'")

        work_df['segment'] = segment_labels
        if return_model:
            return work_df, fitted_model
        return work_df
    
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class CustomerSegmentation:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = None
        self.pca = None
        self.n_clusters = None
        self.is_fitted = False
        self.feature_names = []
    
    def _first_present_col(self, df, candidates):
        """Find the first column present from a list of candidate names (case-insensitive)."""
        lower_map = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        return None
    
    def _to_numeric(self, s, clip_min=None, clip_max=None):
        s = pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)
        if clip_min is not None or clip_max is not None:
            s = np.clip(s, a_min=clip_min if clip_min is not None else -np.inf,
                        a_max=clip_max if clip_max is not None else np.inf)
        return s
        
    def prepare_segmentation_features(self, df):
        """Prepare features for customer segmentation"""
        features = []
        feature_names = []
        
        # RFM: add whichever is available, using common synonyms
        recency_col = self._first_present_col(df, ['days_since_last_purchase', 'recency', 'days_since_last_order', 'days_since_last_txn'])
        if recency_col:
            features.append(self._to_numeric(df[recency_col], clip_min=0, clip_max=36500).values)
            feature_names.append('recency')
        
        frequency_col = self._first_present_col(df, ['purchase_frequency', 'frequency', 'orders_count', 'num_orders', 'purchase_count'])
        if frequency_col:
            features.append(self._to_numeric(df[frequency_col], clip_min=0, clip_max=1e5).values)
            feature_names.append('frequency')
        
        monetary_col = self._first_present_col(df, ['total_revenue', 'revenue', 'sales', 'amount', 'lifetime_value', 'clv', 'cltv'])
        if monetary_col:
            features.append(self._to_numeric(df[monetary_col], clip_min=0, clip_max=1e9).values)
            feature_names.append('monetary')
        
        # Customer lifetime and engagement
        lifetime_col = self._first_present_col(df, ['customer_lifetime_days', 'lifetime_days', 'tenure_days', 'tenure'])
        if lifetime_col:
            features.append(self._to_numeric(df[lifetime_col], clip_min=0, clip_max=36500).values)
            feature_names.append('lifetime_days')
        
        cancels_col = self._first_present_col(df, ['cancellations_count', 'cancellations', 'cancel_count', 'refunds_count', 'returns_count'])
        if cancels_col:
            features.append(self._to_numeric(df[cancels_col], clip_min=0, clip_max=1e4).values)
            feature_names.append('cancellations')
        
        rating_col = self._first_present_col(df, ['Ratings', 'ratings', 'rating', 'avg_rating'])
        if rating_col:
            features.append(self._to_numeric(df[rating_col], clip_min=0, clip_max=10).values)
            feature_names.append('ratings')
        
        # Demographics / price-quantity if present
        age_col = self._first_present_col(df, ['age', 'customer_age'])
        if age_col:
            features.append(self._to_numeric(df[age_col], clip_min=0, clip_max=120).values)
            feature_names.append('age')
        
        qty_col = self._first_present_col(df, ['quantity', 'qty', 'units'])
        if qty_col:
            features.append(self._to_numeric(df[qty_col], clip_min=0, clip_max=1e6).values)
            feature_names.append('quantity')
        
        price_col = self._first_present_col(df, ['unit_price', 'price'])
        if price_col:
            features.append(self._to_numeric(df[price_col], clip_min=0, clip_max=1e6).values)
            feature_names.append('unit_price')
        
        # If no features detected, attempt fallback: all numeric columns except known targets
        if not features:
            numeric_df = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)
            drop_candidates = [
                'is_churned', 'churn_probability', 'cluster', 'pca_1', 'pca_2'
            ]
            numeric_df = numeric_df[[c for c in numeric_df.columns if c not in drop_candidates]]
            if numeric_df.shape[1] == 0:
                raise ValueError("No suitable numeric features found for segmentation")
            features = [numeric_df[c].values for c in numeric_df.columns]
            feature_names = list(numeric_df.columns)
        
        X = np.column_stack(features)
        self.feature_names = feature_names
        return X
    
    def find_optimal_clusters(self, X, max_clusters=10):
        """Find optimal number of clusters using elbow method and silhouette score"""
        n_samples = len(X)
        if n_samples < 2:
            raise ValueError("At least 2 samples required to estimate clusters")
        X_scaled = self.scaler.fit_transform(X)
        
        inertias = []
        sil_scores = []
        K_max = min(max_clusters, n_samples - 1)
        if K_max < 2:
            K_max = 2
        K_range = list(range(2, K_max + 1))
        
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertias.append(km.inertia_)
            try:
                sil = silhouette_score(X_scaled, labels)
            except Exception:
                sil = np.nan
            sil_scores.append(sil)
        
        # Prefer best silhouette score if available; otherwise fallback to min inertia elbow-ish
        if np.all(np.isnan(sil_scores)):
            optimal_k = K_range[int(np.argmin(inertias))] if K_range else 2
        else:
            # Replace NaNs with -inf to ignore them
            sil_array = np.array([(-np.inf if np.isnan(s) else s) for s in sil_scores])
            optimal_k = K_range[int(np.argmax(sil_array))]
        
        return optimal_k, inertias, sil_scores, K_range
    
    def segment_customers(self, df, n_clusters=None):
        """Segment customers using K-means clustering"""
        X = self.prepare_segmentation_features(df)
        
        if n_clusters is None:
            try:
                n_clusters, _, _, _ = self.find_optimal_clusters(X)
            except Exception:
                n_clusters = min(3, max(2, len(X)))  # fallback
        
        # Ensure n_clusters valid
        self.n_clusters = max(2, min(n_clusters, len(X)))
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply K-means
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        
        # Apply PCA for visualization
        n_components = 2 if X_scaled.shape[1] >= 2 else 1
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Create results dataframe
        results_df = df.copy()
        results_df['cluster'] = cluster_labels
        results_df['pca_1'] = X_pca[:, 0]
        if n_components == 2:
            results_df['pca_2'] = X_pca[:, 1]
        
        self.is_fitted = True
        
        return results_df
    
    def analyze_segments(self, segmented_df):
        """Analyze characteristics of each customer segment"""
        if not self.is_fitted:
            raise ValueError("Customer segmentation must be performed first")
        
        segment_analysis = {}
        
        for cluster in range(self.n_clusters):
            cluster_data = segmented_df[segmented_df['cluster'] == cluster]
            
            analysis = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(segmented_df) * 100
            }
            
            # Analyze key metrics for each segment
            numeric_cols = ['age', 'total_revenue', 'purchase_frequency', 'days_since_last_purchase', 
                          'customer_lifetime_days', 'cancellations_count', 'Ratings']
            
            for col in numeric_cols:
                if col in cluster_data.columns:
                    analysis[f'{col}_mean'] = cluster_data[col].mean()
                    analysis[f'{col}_median'] = cluster_data[col].median()
            
            # Churn rate for this segment
            if 'is_churned' in cluster_data.columns:
                analysis['churn_rate'] = cluster_data['is_churned'].mean()
            
            # Most common categories
            if 'category' in cluster_data.columns:
                analysis['top_categories'] = cluster_data['category'].value_counts().head(3).to_dict()
            
            if 'country' in cluster_data.columns:
                analysis['top_countries'] = cluster_data['country'].value_counts().head(3).to_dict()
            
            segment_analysis[f'Segment_{cluster}'] = analysis
        
        return segment_analysis
    
    def create_segment_profiles(self, segment_analysis):
        """Create business-friendly segment profiles"""
        profiles = {}
        
        for segment, analysis in segment_analysis.items():
            # Determine segment characteristics
            avg_revenue = analysis.get('total_revenue_mean', 0)
            avg_frequency = analysis.get('purchase_frequency_mean', 0)
            avg_recency = analysis.get('days_since_last_purchase_mean', 0)
            churn_rate = analysis.get('churn_rate', 0)
            
            # Create profile based on RFM characteristics
            if avg_revenue > 500 and avg_frequency > 30 and avg_recency < 30:
                profile = "Champions"
                description = "High-value, frequent, recent customers"
                strategy = "Reward and retain with VIP treatment"
            elif avg_revenue > 300 and avg_frequency > 20:
                profile = "Loyal Customers"
                description = "Regular customers with good value"
                strategy = "Upsell and cross-sell opportunities"
            elif avg_recency < 30 and avg_frequency < 10:
                profile = "New Customers"
                description = "Recent customers with low frequency"
                strategy = "Nurture and onboard effectively"
            elif avg_recency > 60 and churn_rate > 0.5:
                profile = "At Risk"
                description = "Customers likely to churn"
                strategy = "Win-back campaigns and retention offers"
            elif avg_revenue < 100 and avg_frequency < 10:
                profile = "Price Sensitive"
                description = "Low-value, infrequent customers"
                strategy = "Cost-effective marketing and promotions"
            else:
                profile = "Potential Loyalists"
                description = "Customers with growth potential"
                strategy = "Engagement campaigns to increase frequency"
            
            profiles[segment] = {
                'profile_name': profile,
                'description': description,
                'strategy': strategy,
                'size': analysis['size'],
                'percentage': analysis['percentage'],
                'key_metrics': {
                    'avg_revenue': avg_revenue,
                    'avg_frequency': avg_frequency,
                    'avg_recency': avg_recency,
                    'churn_rate': churn_rate
                }
            }
        
        return profiles
    
    def plot_segments(self, segmented_df):
        """Visualize customer segments"""
        if not self.is_fitted:
            raise ValueError("Customer segmentation must be performed first")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # PCA visualization
        scatter = axes[0, 0].scatter(segmented_df['pca_1'], segmented_df['pca_2'], 
                                   c=segmented_df['cluster'], cmap='viridis', alpha=0.6)
        axes[0, 0].set_title('Customer Segments (PCA Visualization)')
        axes[0, 0].set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[0, 0].set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # Segment size distribution
        segment_counts = segmented_df['cluster'].value_counts().sort_index()
        axes[0, 1].pie(segment_counts.values, labels=[f'Segment {i}' for i in segment_counts.index], 
                      autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Segment Size Distribution')
        
        # Revenue by segment
        if 'total_revenue' in segmented_df.columns:
            segmented_df.boxplot(column='total_revenue', by='cluster', ax=axes[1, 0])
            axes[1, 0].set_title('Revenue Distribution by Segment')
            axes[1, 0].set_xlabel('Segment')
            axes[1, 0].set_ylabel('Total Revenue')
        
        # Churn rate by segment
        if 'is_churned' in segmented_df.columns:
            churn_by_segment = segmented_df.groupby('cluster')['is_churned'].mean()
            axes[1, 1].bar(churn_by_segment.index, churn_by_segment.values, color='coral', alpha=0.7)
            axes[1, 1].set_title('Churn Rate by Segment')
            axes[1, 1].set_xlabel('Segment')
            axes[1, 1].set_ylabel('Churn Rate')
        
        plt.tight_layout()
        return fig

# Example usage
if __name__ == "__main__":
    print("Customer Segmentation Model Ready!")
    print("Use this class to segment customers based on their behavior and characteristics.")

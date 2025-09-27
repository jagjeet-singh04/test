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
        
    def prepare_segmentation_features(self, df):
        """Prepare features for customer segmentation"""
        features = []
        feature_names = []
        
        # RFM Analysis features
        if all(col in df.columns for col in ['total_revenue', 'purchase_frequency', 'days_since_last_purchase']):
            # Recency (lower is better)
            features.append(df['days_since_last_purchase'].values)
            feature_names.append('recency')
            
            # Frequency (higher is better)
            features.append(df['purchase_frequency'].values)
            feature_names.append('frequency')
            
            # Monetary (higher is better)
            features.append(df['total_revenue'].values)
            feature_names.append('monetary')
        
        # Customer lifetime and engagement
        if 'customer_lifetime_days' in df.columns:
            features.append(df['customer_lifetime_days'].values)
            feature_names.append('lifetime_days')
            
        if 'cancellations_count' in df.columns:
            features.append(df['cancellations_count'].values)
            feature_names.append('cancellations')
            
        if 'Ratings' in df.columns:
            features.append(df['Ratings'].values)
            feature_names.append('ratings')
            
        # Demographic features
        if 'age' in df.columns:
            features.append(df['age'].values)
            feature_names.append('age')
            
        # Create feature matrix
        if features:
            X = np.column_stack(features)
            self.feature_names = feature_names
            return X
        else:
            raise ValueError("No suitable features found for segmentation")
    
    def find_optimal_clusters(self, X, max_clusters=10):
        """Find optimal number of clusters using elbow method and silhouette score"""
        X_scaled = self.scaler.fit_transform(X)
        
        inertias = []
        silhouette_scores = []
        K_range = range(2, min(max_clusters + 1, len(X)))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        
        # Find elbow point (simplified)
        optimal_k = K_range[np.argmax(silhouette_scores)]
        
        return optimal_k, inertias, silhouette_scores, K_range
    
    def segment_customers(self, df, n_clusters=None):
        """Segment customers using K-means clustering"""
        X = self.prepare_segmentation_features(df)
        
        if n_clusters is None:
            n_clusters, _, _, _ = self.find_optimal_clusters(X)
        
        self.n_clusters = n_clusters
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply K-means
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        
        # Apply PCA for visualization
        self.pca = PCA(n_components=2)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Create results dataframe
        results_df = df.copy()
        results_df['cluster'] = cluster_labels
        results_df['pca_1'] = X_pca[:, 0]
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

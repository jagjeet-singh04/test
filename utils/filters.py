import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class FlexibleFilters:
    """Advanced filtering system for the dashboard"""
    
    def __init__(self):
        self.active_filters = {}
    
    def create_sidebar_filters(self, df):
        """Create comprehensive sidebar filters"""
        st.sidebar.markdown("### 🔍 Data Filters")
        
        filters = {}
        
        # Date range filter
        if 'last_purchase_date' in df.columns:
            st.sidebar.markdown("#### Date Range")
            date_col = pd.to_datetime(df['last_purchase_date'])
            min_date = date_col.min().date()
            max_date = date_col.max().date()
            
            date_range = st.sidebar.date_input(
                "Purchase Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) == 2:
                filters['date_range'] = date_range
        
        # Customer demographics
        st.sidebar.markdown("#### Customer Demographics")
        
        if 'age' in df.columns:
            age_range = st.sidebar.slider(
                "Age Range",
                min_value=int(df['age'].min()),
                max_value=int(df['age'].max()),
                value=(int(df['age'].min()), int(df['age'].max()))
            )
            filters['age_range'] = age_range
        
        if 'gender' in df.columns:
            gender_options = ['All'] + list(df['gender'].unique())
            selected_gender = st.sidebar.multiselect(
                "Gender",
                options=gender_options[1:],  # Exclude 'All'
                default=gender_options[1:]
            )
            if selected_gender:
                filters['gender'] = selected_gender
        
        if 'country' in df.columns:
            country_options = list(df['country'].unique())
            selected_countries = st.sidebar.multiselect(
                "Countries",
                options=country_options,
                default=country_options[:min(5, len(country_options))]
            )
            if selected_countries:
                filters['country'] = selected_countries
        
        # Product filters
        st.sidebar.markdown("#### Product Filters")
        
        if 'category' in df.columns:
            category_options = list(df['category'].unique())
            selected_categories = st.sidebar.multiselect(
                "Product Categories",
                options=category_options,
                default=category_options
            )
            if selected_categories:
                filters['category'] = selected_categories
        
        if 'product_name' in df.columns:
            # Show top products by frequency
            top_products = df['product_name'].value_counts().head(20).index.tolist()
            selected_products = st.sidebar.multiselect(
                "Top Products",
                options=top_products,
                default=[]
            )
            if selected_products:
                filters['product_name'] = selected_products
        
        # Financial filters
        st.sidebar.markdown("#### Financial Filters")
        
        if 'total_revenue' in df.columns:
            revenue_range = st.sidebar.slider(
                "Revenue Range ($)",
                min_value=float(df['total_revenue'].min()),
                max_value=float(df['total_revenue'].max()),
                value=(float(df['total_revenue'].min()), float(df['total_revenue'].max())),
                format="$%.2f"
            )
            filters['revenue_range'] = revenue_range
        
        if 'unit_price' in df.columns:
            price_range = st.sidebar.slider(
                "Unit Price Range ($)",
                min_value=float(df['unit_price'].min()),
                max_value=float(df['unit_price'].max()),
                value=(float(df['unit_price'].min()), float(df['unit_price'].max())),
                format="$%.2f"
            )
            filters['price_range'] = price_range
        
        # Behavioral filters
        st.sidebar.markdown("#### Customer Behavior")
        
        if 'subscription_status' in df.columns:
            status_options = list(df['subscription_status'].unique())
            selected_status = st.sidebar.multiselect(
                "Subscription Status",
                options=status_options,
                default=status_options
            )
            if selected_status:
                filters['subscription_status'] = selected_status
        
        if 'purchase_frequency' in df.columns:
            freq_range = st.sidebar.slider(
                "Purchase Frequency",
                min_value=int(df['purchase_frequency'].min()),
                max_value=int(df['purchase_frequency'].max()),
                value=(int(df['purchase_frequency'].min()), int(df['purchase_frequency'].max()))
            )
            filters['frequency_range'] = freq_range
        
        if 'Ratings' in df.columns:
            rating_range = st.sidebar.slider(
                "Customer Ratings",
                min_value=float(df['Ratings'].min()),
                max_value=float(df['Ratings'].max()),
                value=(float(df['Ratings'].min()), float(df['Ratings'].max())),
                step=0.1,
                format="%.1f"
            )
            filters['rating_range'] = rating_range
        
        # Churn-specific filters
        if 'is_churned' in df.columns:
            st.sidebar.markdown("#### Churn Analysis")
            churn_filter = st.sidebar.selectbox(
                "Customer Status",
                options=['All', 'Active Only', 'Churned Only']
            )
            if churn_filter != 'All':
                filters['churn_status'] = churn_filter
        
        # Advanced filters
        with st.sidebar.expander("🔧 Advanced Filters"):
            if 'days_since_last_purchase' in df.columns:
                days_since_purchase = st.slider(
                    "Days Since Last Purchase",
                    min_value=0,
                    max_value=int(df['days_since_last_purchase'].max()),
                    value=(0, int(df['days_since_last_purchase'].max()))
                )
                filters['days_since_purchase'] = days_since_purchase
            
            if 'customer_lifetime_days' in df.columns:
                lifetime_range = st.slider(
                    "Customer Lifetime (Days)",
                    min_value=0,
                    max_value=int(df['customer_lifetime_days'].max()),
                    value=(0, int(df['customer_lifetime_days'].max()))
                )
                filters['lifetime_range'] = lifetime_range
        
        # Filter summary
        if filters:
            st.sidebar.markdown("#### Active Filters")
            filter_count = len([f for f in filters.values() if f])
            st.sidebar.info(f"🔍 {filter_count} filters active")
            
            if st.sidebar.button("🗑️ Clear All Filters"):
                st.experimental_rerun()
        
        return filters
    
    def apply_filters(self, df, filters):
        """Apply all selected filters to the dataframe"""
        filtered_df = df.copy()
        
        # Date range filter
        if 'date_range' in filters and len(filters['date_range']) == 2:
            start_date, end_date = filters['date_range']
            date_col = pd.to_datetime(filtered_df['last_purchase_date'])
            filtered_df = filtered_df[
                (date_col.dt.date >= start_date) & 
                (date_col.dt.date <= end_date)
            ]
        
        # Age range filter
        if 'age_range' in filters:
            min_age, max_age = filters['age_range']
            filtered_df = filtered_df[
                (filtered_df['age'] >= min_age) & 
                (filtered_df['age'] <= max_age)
            ]
        
        # Categorical filters
        categorical_filters = ['gender', 'country', 'category', 'product_name', 'subscription_status']
        for filter_name in categorical_filters:
            if filter_name in filters and filters[filter_name]:
                filtered_df = filtered_df[filtered_df[filter_name].isin(filters[filter_name])]
        
        # Revenue range filter
        if 'revenue_range' in filters:
            min_rev, max_rev = filters['revenue_range']
            filtered_df = filtered_df[
                (filtered_df['total_revenue'] >= min_rev) & 
                (filtered_df['total_revenue'] <= max_rev)
            ]
        
        # Price range filter
        if 'price_range' in filters:
            min_price, max_price = filters['price_range']
            filtered_df = filtered_df[
                (filtered_df['unit_price'] >= min_price) & 
                (filtered_df['unit_price'] <= max_price)
            ]
        
        # Frequency range filter
        if 'frequency_range' in filters:
            min_freq, max_freq = filters['frequency_range']
            filtered_df = filtered_df[
                (filtered_df['purchase_frequency'] >= min_freq) & 
                (filtered_df['purchase_frequency'] <= max_freq)
            ]
        
        # Rating range filter
        if 'rating_range' in filters:
            min_rating, max_rating = filters['rating_range']
            filtered_df = filtered_df[
                (filtered_df['Ratings'] >= min_rating) & 
                (filtered_df['Ratings'] <= max_rating)
            ]
        
        # Churn status filter
        if 'churn_status' in filters:
            if filters['churn_status'] == 'Active Only':
                filtered_df = filtered_df[filtered_df['is_churned'] == 0]
            elif filters['churn_status'] == 'Churned Only':
                filtered_df = filtered_df[filtered_df['is_churned'] == 1]
        
        # Days since purchase filter
        if 'days_since_purchase' in filters:
            min_days, max_days = filters['days_since_purchase']
            filtered_df = filtered_df[
                (filtered_df['days_since_last_purchase'] >= min_days) & 
                (filtered_df['days_since_last_purchase'] <= max_days)
            ]
        
        # Lifetime range filter
        if 'lifetime_range' in filters:
            min_lifetime, max_lifetime = filters['lifetime_range']
            filtered_df = filtered_df[
                (filtered_df['customer_lifetime_days'] >= min_lifetime) & 
                (filtered_df['customer_lifetime_days'] <= max_lifetime)
            ]
        
        return filtered_df
    
    def create_dynamic_table_display(self, df, page_name="Data"):
        """Create dynamic table display with pagination and column selection"""
        st.markdown(f"### 📊 {page_name} Table View")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Column selection
            available_columns = df.columns.tolist()
            default_columns = available_columns[:min(8, len(available_columns))]
            selected_columns = st.multiselect(
                "Select columns to display",
                options=available_columns,
                default=default_columns
            )
        
        with col2:
            # Rows per page
            rows_per_page = st.selectbox(
                "Rows per page",
                options=[10, 25, 50, 100, 500],
                index=1
            )
        
        with col3:
            # Sort options
            sort_column = st.selectbox(
                "Sort by",
                options=selected_columns if selected_columns else available_columns
            )
            sort_order = st.selectbox("Order", ["Descending", "Ascending"])
        
        if selected_columns:
            # Apply sorting
            ascending = sort_order == "Ascending"
            if sort_column in df.columns:
                df_display = df[selected_columns].sort_values(sort_column, ascending=ascending)
            else:
                df_display = df[selected_columns]
            
            # Pagination
            total_rows = len(df_display)
            total_pages = (total_rows - 1) // rows_per_page + 1
            
            if total_pages > 1:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    page_number = st.selectbox(
                        f"Page (1-{total_pages})",
                        options=list(range(1, total_pages + 1)),
                        format_func=lambda x: f"Page {x} of {total_pages}"
                    )
                
                start_idx = (page_number - 1) * rows_per_page
                end_idx = start_idx + rows_per_page
                df_page = df_display.iloc[start_idx:end_idx]
            else:
                df_page = df_display.head(rows_per_page)
            
            # Display table
            st.dataframe(df_page, use_container_width=True)
            
            # Table summary
            st.info(f"Showing {len(df_page)} of {total_rows} records")
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                csv_data = df_display.to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered Data (CSV)",
                    data=csv_data,
                    file_name=f"filtered_{page_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Excel download would require additional libraries
                st.info("💡 Tip: Use CSV download for Excel compatibility")
        
        else:
            st.warning("Please select at least one column to display.")
    
    def create_summary_metrics(self, original_df, filtered_df):
        """Create summary metrics comparing original vs filtered data"""
        st.markdown("### 📈 Filter Impact Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            original_count = len(original_df)
            filtered_count = len(filtered_df)
            retention_rate = (filtered_count / original_count) * 100 if original_count > 0 else 0
            
            st.metric(
                "Records Retained",
                f"{filtered_count:,}",
                f"{retention_rate:.1f}% of original"
            )
        
        with col2:
            if 'total_revenue' in filtered_df.columns:
                original_revenue = original_df['total_revenue'].sum()
                filtered_revenue = filtered_df['total_revenue'].sum()
                revenue_retention = (filtered_revenue / original_revenue) * 100 if original_revenue > 0 else 0
                
                st.metric(
                    "Revenue Retained",
                    f"${filtered_revenue:,.2f}",
                    f"{revenue_retention:.1f}% of original"
                )
        
        with col3:
            if 'is_churned' in filtered_df.columns:
                original_churn = original_df['is_churned'].mean() * 100
                filtered_churn = filtered_df['is_churned'].mean() * 100 if len(filtered_df) > 0 else 0
                churn_change = filtered_churn - original_churn
                
                st.metric(
                    "Churn Rate",
                    f"{filtered_churn:.1f}%",
                    f"{churn_change:+.1f}pp"
                )
        
        with col4:
            if 'customer_id' in filtered_df.columns:
                original_customers = original_df['customer_id'].nunique()
                filtered_customers = filtered_df['customer_id'].nunique()
                customer_retention = (filtered_customers / original_customers) * 100 if original_customers > 0 else 0
                
                st.metric(
                    "Unique Customers",
                    f"{filtered_customers:,}",
                    f"{customer_retention:.1f}% of original"
                )
    
    def create_quick_filter_buttons(self, df):
        """Create quick filter buttons for common scenarios"""
        st.markdown("### ⚡ Quick Filters")
        
        col1, col2, col3, col4 = st.columns(4)
        
        quick_filters = {}
        
        with col1:
            if st.button("🔥 High Value Customers"):
                if 'total_revenue' in df.columns:
                    threshold = df['total_revenue'].quantile(0.8)
                    quick_filters['revenue_range'] = (threshold, df['total_revenue'].max())
        
        with col2:
            if st.button("⚠️ At Risk Customers"):
                if 'days_since_last_purchase' in df.columns:
                    quick_filters['days_since_purchase'] = (30, df['days_since_last_purchase'].max())
        
        with col3:
            if st.button("⭐ High Rated Products"):
                if 'Ratings' in df.columns:
                    quick_filters['rating_range'] = (4.0, df['Ratings'].max())
        
        with col4:
            if st.button("🆕 New Customers"):
                if 'days_since_signup' in df.columns:
                    quick_filters['days_since_signup'] = (0, 90)
        
        return quick_filters

def create_advanced_search(df):
    """Create advanced search functionality"""
    st.markdown("### 🔍 Advanced Search")
    
    search_column = st.selectbox(
        "Search in column",
        options=df.select_dtypes(include=['object']).columns.tolist()
    )
    
    search_term = st.text_input("Search term")
    
    if search_term and search_column:
        mask = df[search_column].astype(str).str.contains(search_term, case=False, na=False)
        return df[mask]
    
    return df

def export_filtered_data(df, filename_prefix="filtered_data"):
    """Export filtered data in multiple formats"""
    st.markdown("### 📤 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV Export
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV",
            data=csv_data,
            file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # JSON Export
        json_data = df.to_json(orient='records', date_format='iso')
        st.download_button(
            label="📋 Download JSON",
            data=json_data,
            file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col3:
        # Summary Report
        summary_report = generate_summary_report(df)
        st.download_button(
            label="📊 Download Summary",
            data=summary_report,
            file_name=f"{filename_prefix}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

def generate_summary_report(df):
    """Generate a text summary report of the filtered data"""
    report = f"""
DATA SUMMARY REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

BASIC STATISTICS:
- Total Records: {len(df):,}
- Total Columns: {len(df.columns)}

NUMERICAL COLUMNS SUMMARY:
"""
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        report += f"\n{col}:"
        report += f"\n  - Mean: {df[col].mean():.2f}"
        report += f"\n  - Median: {df[col].median():.2f}"
        report += f"\n  - Min: {df[col].min():.2f}"
        report += f"\n  - Max: {df[col].max():.2f}"
    
    report += f"\n\nCATEGORICAL COLUMNS SUMMARY:\n"
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        unique_count = df[col].nunique()
        report += f"\n{col}: {unique_count} unique values"
        if unique_count <= 10:
            top_values = df[col].value_counts().head(5)
            report += f"\n  Top values: {dict(top_values)}"
    
    return report

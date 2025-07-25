import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# --- Configuration ---
st.set_page_config(layout="wide", page_title="HealthKart Influencer Dashboard", initial_sidebar_state="collapsed")

# --- Custom CSS Injection ---
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Error loading CSS file: {file_name}. Please ensure it exists in the 'static' folder.")

load_css("static/style.css") # Assuming style.css is in 'static' for Streamlit version

# --- Global DataFrames (initialized as None or empty) ---
df_influencers = None
df_posts = None
df_tracking_data = None
df_payouts = None

# --- Data Loading Function ---
@st.cache_data(show_spinner=False)
def load_and_process_data():
    try:
        df_influencers = pd.read_csv('influencers.csv')
        df_posts = pd.read_csv('posts.csv')
        df_tracking_data = pd.read_csv('tracking_data.csv')
        df_payouts = pd.read_csv('payouts.csv')

        # Convert IDs to string for consistent merging
        df_influencers['ID'] = df_influencers['ID'].astype(str)
        df_posts['influencer_id'] = df_posts['influencer_id'].astype(str)
        df_tracking_data['influencer_id'] = df_tracking_data['influencer_id'].astype(str)
        df_payouts['influencer_id'] = df_payouts['influencer_id'].astype(str)

        # Add Influencer Tier
        def get_influencer_tier(followers):
            if followers < 10000: return "Nano (<10k)"
            elif 10000 <= followers < 100000: return "Micro (10k-100k)"
            elif 100000 <= followers < 1000000: return "Mid (100k-1M)"
            elif 1000000 <= followers < 5000000: return "Macro (1M-5M)"
            else: return "Mega (5M+)"
        df_influencers['tier'] = df_influencers['follower_count'].apply(get_influencer_tier)

        # Convert date columns to datetime objects
        df_posts['date'] = pd.to_datetime(df_posts['date'])
        df_tracking_data['date'] = pd.to_datetime(df_tracking_data['date'])
        
        # Add brand column if not present (for both uploaded and default data)
        if 'brand' not in df_tracking_data.columns:
            products_brands = {
                'Whey Protein (MB)': 'MuscleBlaze', 'Creatine (MB)': 'MuscleBlaze', 'BCAA (MB)': 'MuscleBlaze',
                'Multivitamin (HK)': 'HKVitals', 'Fish Oil (HK)': 'HKVitals', 'Biotin (HK)': 'HKVitals',
                'Kids Nutrition (GZ)': 'Gritzo', 'Kids Immunity (GZ)': 'Gritzo'
            }
            df_tracking_data['brand'] = df_tracking_data['product'].map(products_brands)

        return df_influencers, df_posts, df_tracking_data, df_payouts
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        st.info("Please ensure your CSV files are correctly formatted and have the required columns. If you just recreated venv, remember to run simulate_data.py again!")
        st.stop()

df_influencers_global, df_posts_global, df_tracking_data_global, df_payouts_global = load_and_process_data()

# --- Helper Functions for Formatting ---
def format_rupee_lakh(value):
    if pd.isna(value) or value == 0:
        return "₹0.0L"
    return f"₹{value / 100000:,.1f}L"

def format_roas(value):
    if pd.isna(value) or value == 0:
        return "0.0x"
    return f"{value:,.1f}x"

# --- Title and Description ---
st.markdown("""
<div class="card-container header-card">
    <h1>🚀 HealthKart Influencer ROI Dashboard</h1>
    <p>Track and optimize your influencer marketing campaigns across MuscleBlaze, HKVitals & Gritzo</p>
</div>
""", unsafe_allow_html=True)

# --- Filters Section (NOW IN MAIN DASHBOARD) ---
st.markdown("""
<div class="card-container">
    <h2 style='color: white;'>Campaign Filters</h2>
</div>
""", unsafe_allow_html=True)

# Use st.columns to arrange filters horizontally in the main dashboard
filter_col1, filter_col2, filter_col3 = st.columns(3)
filter_col4, filter_col5 = st.columns(2) # Separate row for Date Range or more filters

# Date Range Filter
current_date = datetime.now().date()
min_date_available = df_posts_global['date'].min().date() if not df_posts_global.empty else current_date
    
date_range_options = {
    "Last 30 Days": (current_date - timedelta(days=30)),
    "Last 90 Days": (current_date - timedelta(days=90)),
    "Last 180 Days": (current_date - timedelta(days=180)),
    "Last 365 Days": (current_date - timedelta(days=365)),
    "All Time": min_date_available
}

default_date_range_label = "All Time" # Ensure this matches a key in date_range_options
selected_date_range_label = filter_col5.selectbox("Select Date Range", list(date_range_options.keys()), index=list(date_range_options.keys()).index(default_date_range_label)) # Moved to filter_col5

start_date_filter = date_range_options[selected_date_range_label]
end_date_filter = current_date

# Filter dataframes based on date range initially
filtered_posts_base = df_posts_global[(df_posts_global['date'].dt.date >= start_date_filter) & (df_posts_global['date'].dt.date <= end_date_filter)]
filtered_tracking_data_base = df_tracking_data_global[(df_tracking_data_global['date'].dt.date >= start_date_filter) & (df_tracking_data_global['date'].dt.date <= end_date_filter)]

# Cascading Filters - Now in main dashboard
# Brand Filter
all_brands = ['All Brands'] + sorted(filtered_tracking_data_base['brand'].unique().tolist()) if not filtered_tracking_data_base.empty else ['All Brands']
selected_brands = filter_col1.selectbox("Brand", all_brands, index=0) 

# Platform Filter
all_platforms = ['All Platforms'] + sorted(df_influencers_global['platform'].unique().tolist()) if not df_influencers_global.empty else ['All Platforms']
selected_platforms = filter_col2.selectbox("Platform", all_platforms, index=0) 

# Influencer Category Filter
all_categories = ['All Categories'] + sorted(df_influencers_global['category'].unique().tolist()) if not df_influencers_global.empty else ['All Categories']
selected_categories = filter_col3.selectbox("Influencer Category", all_categories, index=0) 

# Influencer Tier Filter (moved to separate column if needed, or remove filter_col4 for 3x1 layout + date)
all_tiers = ['All Tiers'] + sorted(df_influencers_global['tier'].unique().tolist(), key=lambda x: ['Nano', 'Micro', 'Mid', 'Macro', 'Mega'].index(x.split(' ')[0])) if not df_influencers_global.empty else ['All Tiers']
selected_tiers = filter_col4.selectbox("Influencer Tier", all_tiers, index=0)


# --- Apply Filters to create final dataframes for display ---
# Start with the base filtered by date
filtered_posts_final = filtered_posts_base.copy()
filtered_tracking_data_final = filtered_tracking_data_base.copy()
filtered_influencers_final = df_influencers_global.copy() # Start with all influencers and filter down

# Apply Brand filter
if selected_brands != 'All Brands':
    filtered_tracking_data_final = filtered_tracking_data_final[filtered_tracking_data_final['brand'] == selected_brands]
    relevant_influencer_ids_by_brand = filtered_tracking_data_final['influencer_id'].unique()
    filtered_influencers_final = filtered_influencers_final[filtered_influencers_final['ID'].isin(relevant_influencer_ids_by_brand)]
    filtered_posts_final = filtered_posts_final[filtered_posts_final['influencer_id'].isin(relevant_influencer_ids_by_brand)]

# Apply Platform filter
if selected_platforms != 'All Platforms':
    filtered_influencers_final = filtered_influencers_final[filtered_influencers_final['platform'] == selected_platforms]
    filtered_posts_final = filtered_posts_final[filtered_posts_final['influencer_id'].isin(filtered_influencers_final['ID'])]
    filtered_tracking_data_final = filtered_tracking_data_final[filtered_tracking_data_final['influencer_id'].isin(filtered_influencers_final['ID'])]

# Apply Category filter
if selected_categories != 'All Categories':
    filtered_influencers_final = filtered_influencers_final[filtered_influencers_final['category'] == selected_categories]
    filtered_posts_final = filtered_posts_final[filtered_influencers_final['ID'].isin(filtered_influencers_final['ID'])]
    filtered_tracking_data_final = filtered_tracking_data_final[filtered_tracking_data_final['influencer_id'].isin(filtered_influencers_final['ID'])]

# Apply Tier filter
if selected_tiers != 'All Tiers':
    filtered_influencers_final = filtered_influencers_final[filtered_influencers_final['tier'] == selected_tiers]
    filtered_posts_final = filtered_posts_final[filtered_influencers_final['ID'].isin(filtered_influencers_final['ID'])]
    filtered_tracking_data_final = filtered_tracking_data_final[filtered_tracking_data_final['influencer_id'].isin(filtered_influencers_final['ID'])]

# Adjust payouts based on final filtered influencers
relevant_influencer_ids_final = filtered_tracking_data_final['influencer_id'].unique()
filtered_payouts_final = df_payouts_global[df_payouts_global['influencer_id'].isin(relevant_influencer_ids_final)]

# --- CRITICAL FIX: MERGE PLATFORM/CATEGORY/TIER INTO filtered_tracking_data_final HERE ---
if not filtered_tracking_data_final.empty and not filtered_influencers_final.empty:
    filtered_tracking_data_final = pd.merge(
        filtered_tracking_data_final,
        filtered_influencers_final[['ID', 'platform', 'category', 'tier']], # Merge these columns
        left_on='influencer_id',
        right_on='ID',
        how='left'
    )
    filtered_tracking_data_final.drop(columns='ID', inplace=True)
elif filtered_tracking_data_final.empty: # If tracking data is empty after filtering, ensure it still has these columns
    filtered_tracking_data_final = pd.DataFrame(columns=[
        'source', 'campaign', 'influencer_id', 'user_id', 'product', 'date', 'orders', 'revenue', 'brand', 'platform', 'category', 'tier'
    ])
# --- END CRITICAL FIX ---


# --- Section: KPI Cards ---
st.header("📊 Campaign Performance Overview")
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

total_revenue = filtered_tracking_data_final['revenue'].sum()
total_payout = filtered_payouts_final['total_payout'].sum()
total_orders = filtered_tracking_data_final['orders'].sum()
    
overall_roas = total_revenue / total_payout if total_payout > 0 else 0
avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
unique_active_influencers = filtered_tracking_data_final['influencer_id'].nunique()

with col1:
    st.metric(label="Average ROAS", value=format_roas(overall_roas))
with col2:
    st.metric(label="Total Revenue", value=format_rupee_lakh(total_revenue))
with col3:
    st.metric(label="Total Ad Spend", value=format_rupee_lakh(total_payout))
with col4:
    st.metric(label="Total Orders", value=f"{int(total_orders):,}")
with col5:
    st.metric(label="Avg Order Value", value=f"₹{avg_order_value:,.0f}")
with col6:
    st.metric(label="Active Influencers", value=f"{unique_active_influencers:,}")

st.markdown("---")

# --- Section: Conversion Funnel ---
st.header("📈 Conversion Funnel")
total_posts_funnel = filtered_posts_final.shape[0]
total_orders_funnel = filtered_tracking_data_final['orders'].sum()
total_revenue_funnel = filtered_tracking_data_final['revenue'].sum()

funnel_data = dict(
    number=[total_posts_funnel, total_orders_funnel, total_revenue_funnel],
    stage=["Total Posts", "Total Orders", "Total Revenue"],
    value_display=[
        f"{total_posts_funnel:,} posts",
        f"{int(total_orders_funnel):,} orders",
        f"₹{total_revenue_funnel:,.0f}"
    ]
)
conversion_funnel_fig = go.Figure().update_layout(template="plotly_dark", title="No Data for Funnel")
if total_posts_funnel > 0 or total_orders_funnel > 0 or total_revenue_funnel > 0:
    conversion_funnel_fig = go.Figure(go.Funnel(
        y=funnel_data["stage"],
        x=funnel_data["number"],
        text=funnel_data["value_display"],
        textinfo="text",
        marker={"color": ["#8B5CFF", "#A99DE0", "#6C4EC7"]},
        connector={"line": {"color": "white", "dash": "solid", "width": 2}},
        hoverinfo="y+text"
    ))
    conversion_funnel_fig.update_layout(
        title_text='Influencer Conversion Path',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        margin=dict(l=50, r=50, t=50, b=50)
    )
st.plotly_chart(conversion_funnel_fig, use_container_width=True)

st.markdown("---")

# --- Section: Charts ---
st.header("Dashboard Charts")
chart_col1, chart_col2 = st.columns(2)
chart_col3, chart_col4 = st.columns(2)

# Chart 1: ROAS Trend Over Time
with chart_col1:
    st.subheader('📈 ROAS Trend Over Time')
    min_date_for_chart = filtered_posts_final['date'].min() if not filtered_posts_final.empty else start_date_filter
    max_date_for_chart = filtered_posts_final['date'].max() if not filtered_posts_final.empty else end_date_filter

    daily_payout_sum = filtered_payouts_final.groupby(filtered_posts_final['date'].dt.date)['total_payout'].sum().reindex(pd.date_range(min_date_for_chart, max_date_for_chart).date, fill_value=0).rename('Daily Payout')
    daily_revenue_sum = filtered_tracking_data_final.groupby(filtered_tracking_data_final['date'].dt.date)['revenue'].sum().reindex(pd.date_range(min_date_for_chart, max_date_for_chart).date, fill_value=0).rename('Daily Revenue')
    
    daily_roas_df = pd.DataFrame({'Date': daily_revenue_sum.index, 'Daily Revenue': daily_revenue_sum.values, 'Daily Payout': daily_payout_sum.values})
    daily_roas_df['ROAS'] = daily_roas_df.apply(lambda row: row['Daily Revenue'] / row['Daily Payout'] if row['Daily Payout'] > 0 else 0, axis=1)

    roas_trend_fig = go.Figure().update_layout(template="plotly_dark", title="No Data for ROAS Trend")
    if not daily_roas_df.empty:
        roas_trend_fig = px.line(daily_roas_df, x='Date', y='ROAS', title='',
                                labels={'ROAS': 'ROAS (x)', 'Date': 'Date'},
                                color_discrete_sequence=[px.colors.qualitative.Plotly[0]])
        # Set x-axis range to actual data limits
        if not daily_roas_df['Date'].empty:
            roas_trend_fig.update_xaxes(range=[daily_roas_df['Date'].min(), daily_roas_df['Date'].max()])
        roas_trend_fig.update_layout(plot_bgcolor='#A99DE0', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(roas_trend_fig, use_container_width=True)

# Chart 2: Top Performing Platforms
with chart_col2:
    st.subheader('🧑‍💻 Top Performing Platforms')
    platform_revenue_df = filtered_tracking_data_final.groupby('platform')['revenue'].sum().reset_index()
    platform_performance_fig = go.Figure().update_layout(template="plotly_dark", title="No Data for Platforms")
    if not platform_revenue_df.empty:
        platform_performance_fig = px.pie(platform_revenue_df, values='revenue', names='platform', hole=0.5,
                                    title='',
                                    color_discrete_sequence=px.colors.sequential.Agsunset)
        platform_performance_fig.update_traces(textinfo='percent+label', pull=[0.05, 0, 0, 0])
        platform_performance_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white', showlegend=True)
    st.plotly_chart(platform_performance_fig, use_container_width=True)

# Chart 3: Revenue by Brand
with chart_col3:
    st.subheader('💰 Revenue by Brand')
    brand_revenue_df = filtered_tracking_data_final.groupby('brand')['revenue'].sum().reset_index().sort_values('revenue', ascending=False)
    revenue_by_brand_fig = go.Figure().update_layout(template="plotly_dark", title="No Data for Brands")
    if not brand_revenue_df.empty:
        revenue_by_brand_fig = px.bar(brand_revenue_df, x='brand', y='revenue', title='',
                                labels={'revenue': 'Revenue (₹)', 'brand': 'Brand'},
                                color='revenue', color_continuous_scale=px.colors.sequential.Plasma)
        revenue_by_brand_fig.update_layout(plot_bgcolor='#A99DE0', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(revenue_by_brand_fig, use_container_width=True)

# Chart 4: Influencer Performance Distribution (Scatter Plot)
with chart_col4:
    st.subheader('👥 Influencer Performance Distribution')
    influencer_scatter_summary = filtered_tracking_data_final.groupby('influencer_id').agg(
        TotalRevenue=('revenue', 'sum'),
        TotalOrders=('orders', 'sum')
    ).reset_index()
    influencer_scatter_summary = pd.merge(influencer_scatter_summary, filtered_influencers_final[['ID', 'name', 'platform', 'category', 'follower_count', 'tier']],
                                left_on='influencer_id', right_on='ID', how='left')
    payouts_for_merge_scatter = filtered_payouts_final.groupby('influencer_id')['total_payout'].sum().reset_index()
    influencer_scatter_summary = pd.merge(influencer_scatter_summary, payouts_for_merge_scatter,
                                on='influencer_id', how='left')
    influencer_scatter_summary['total_payout'] = influencer_scatter_summary['total_payout'].fillna(0)
    influencer_scatter_summary['ROAS_Val'] = influencer_scatter_summary.apply(
        lambda row: row['TotalRevenue'] / row['total_payout'] if row['total_payout'] > 0 else np.inf, axis=1
    )
    influencer_scatter_summary['ROAS_Val'] = influencer_scatter_summary['ROAS_Val'].replace([np.inf, -np.inf], np.nan).fillna(0)
    if not influencer_scatter_summary.empty and influencer_scatter_summary['ROAS_Val'].max() > 0:
        influencer_scatter_summary.loc[(influencer_scatter_summary['total_payout'] == 0) & (influencer_scatter_summary['TotalRevenue'] > 0), 'ROAS_Val'] = influencer_scatter_summary['ROAS_Val'].max() * 1.2 + 10
    else:
        influencer_scatter_summary.loc[(filtered_tracking_data_final['revenue'].sum() > 0 and filtered_payouts_final['total_payout'].sum() == 0), 'ROAS_Val'] = 1000 # Default high if total revenue but 0 payout for this influencer
        
    scatter_fig = go.Figure().update_layout(template="plotly_dark", title="No Data for Scatter Plot")
    if not influencer_scatter_summary.empty:
        scatter_fig = px.scatter(influencer_scatter_summary, x='TotalRevenue', y='ROAS_Val',
                                hover_name='name',
                                hover_data=['platform', 'category', 'total_payout', 'tier'],
                                labels={'Revenue': 'Revenue (₹)', 'ROAS_Val': 'ROAS (x)'},
                                color_discrete_sequence=[px.colors.qualitative.Plotly[1]])
        scatter_fig.update_layout(plot_bgcolor='#A99DE0', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(scatter_fig, use_container_width=True)

st.markdown("---")

# --- Section: Key Insights ---
st.header("💡 Key Insights")
overall_performance_insight = f"📊 Total Performance: {format_roas(overall_roas)} ROAS with {format_rupee_lakh(total_revenue)} revenue generated"

top_platform_name = "N/A"
top_platform_revenue = 0
if not platform_revenue_df.empty:
    top_platform_name = platform_revenue_df.loc[platform_revenue_df['revenue'].idxmax(), 'platform']
    top_platform_revenue = platform_revenue_df['revenue'].max()
top_platform_insight = f"🏆 Top Platform: {top_platform_name} generates {format_rupee_lakh(top_platform_revenue)} revenue"

best_brand_name = "N/A"
best_brand_revenue = 0
if not brand_revenue_df.empty:
    best_brand_name = brand_revenue_df.loc[brand_revenue_df['revenue'].idxmax(), 'brand']
    best_brand_revenue = brand_revenue_df['revenue'].max()
best_brand_insight = f"💪 Best Brand: {best_brand_name} leads with {format_rupee_lakh(best_brand_revenue)} in sales"

high_roas_influencers_count = 0
if not influencer_scatter_summary.empty and not filtered_payouts_final.empty:
    high_roas_influencers_count = influencer_scatter_summary[influencer_scatter_summary['ROAS_Val'] >= 5].shape[0]
high_performers_insight = f"🌟 High Performers: {high_roas_influencers_count} influencers achieving 5x+ ROAS"

st.markdown(f"""
<div class="card-container">
    <div class="insight-box"><p>{overall_performance_insight}</p></div>
    <div class="insight-box"><p>{top_platform_insight}</p></div>
    <div class="insight-box"><p>{best_brand_insight}</p></div>
    <div class="insight-box"><p>{high_performers_insight}</p></div>
    <div class="insight-box"><p>📈 Recommendation: Excellent performance! Scale successful campaigns</p></div>
</div>
""", unsafe_allow_html=True)


st.markdown("---")

# --- Section: Top Performing Influencers Table ---
st.header("⭐ Top Performing Influencers")
influencer_table_summary = influencer_scatter_summary.copy()
    
post_counts = filtered_posts_final.groupby('influencer_id').size().rename('Posts').reset_index()
influencer_table_summary = pd.merge(influencer_table_summary, post_counts,
                                    left_on='influencer_id', right_on='influencer_id', how='left')
influencer_table_summary['Posts'] = influencer_table_summary['Posts'].fillna(0).astype(int)

influencer_table_summary = influencer_table_summary.rename(columns={
    'name': 'Influencer',
    'platform': 'Platform',
    'category': 'Category',
    'follower_count': 'Followers',
    'TotalOrders': 'Orders',
    'TotalRevenue': 'Revenue',
    'total_payout': 'Spend',
    'tier': 'Tier'
})

influencer_table_summary['Revenue'] = influencer_table_summary['Revenue'].apply(lambda x: f"₹{x:,.0f}")
influencer_table_summary['Spend'] = influencer_table_summary['Spend'].apply(lambda x: f"₹{x:,.0f}")

table_max_roas = influencer_table_summary['ROAS_Val'].max() if not influencer_table_summary.empty else 0
influencer_table_summary['ROAS_Display'] = influencer_table_summary['ROAS_Val'].apply(
    lambda val: "Inf" if (isinstance(val, (int, float)) and val > table_max_roas * 1.1) else f"{val:.1f}x"
)

# This function determines the color based on ROAS_Val (for text color)
def get_roas_text_color_html(roas_val, max_roas):
    color = "#333" # Default dark text color
    if isinstance(roas_val, (int, float)):
        if roas_val > max_roas * 1.1: # "Inf" case
            color = "#66FF66" # Bright green
        elif roas_val >= 1: # Good ROAS
            color = "#66FF66" # Green
        elif roas_val > 0 and roas_val < 1: # Sub-optimal ROAS
            color = "#999999" # Grey
        else: # 0 or negative ROAS
            color = "#FF6666" # Red
    # Include padding, border-radius, and display: inline-block for a "pill" look, if desired by CSS
    return f'color: {color}; font-weight: bold; padding: 4px 8px; border-radius: 4px; display: inline-block;' 

if not influencer_table_summary.empty:
    influencer_table_summary = influencer_table_summary.sort_values(by='ROAS_Val', ascending=False)
    
    display_df_for_table = influencer_table_summary.head(10).copy()
    
    # --- FINAL DECISION ON ROAS TABLE STYLING ---
    # Streamlit's st.dataframe does NOT render HTML within cells.
    # To avoid displaying raw <span> tags, we must provide plain text.
    # The ROAS column will be displayed as a simple formatted string.
    
    display_df_for_table['ROAS'] = display_df_for_table['ROAS_Display'] # Use the plain formatted string (e.g., "14.1x", "Inf")
    
    # Define columns to be displayed to the user
    final_display_columns = ['Influencer', 'Platform', 'Category', 'Tier', 'Followers', 'Posts', 'Orders', 'Revenue', 'Spend', 'ROAS']
    
    # Display the DataFrame with the plain ROAS column. NO HTML embedding here.
    st.dataframe(display_df_for_table[final_display_columns], hide_index=True)

else:
    st.info("No top performing influencers to display with the current filters.")

st.markdown("---")

# --- Optional: Export Data ---
st.header("⬇️ Export Data")
st.markdown("Download aggregated data based on current filters.")

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

if not influencer_table_summary.empty:
    csv_export_df = influencer_table_summary.copy()
    csv_export_df['ROAS'] = csv_export_df['ROAS_Val'] # Use the raw value for CSV export
    
    csv_export_cols = ['Influencer', 'Platform', 'Category', 'Tier', 'Followers', 'Posts', 'Orders', 'Revenue', 'Spend', 'ROAS']
    
    st.download_button(
        label="Export Influencer Summary to CSV",
        data=convert_df_to_csv(csv_export_df[csv_export_cols]),
        file_name='influencer_summary.csv',
        mime='text/csv',
        help="Download the current table as CSV."
    )
else:
    st.info("No data available to export based on current filters.")

st.markdown("---")
st.info("💡 Note on Incremental ROAS: This calculation relies on a user-defined 'baseline' percentage, which is an assumption of organic sales. For more accurate incremental ROAS, A/B testing or more sophisticated statistical models would be required.")
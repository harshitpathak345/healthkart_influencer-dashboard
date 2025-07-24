import dash
from dash import html, dcc, Input, Output, State, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

# --- Data Loading and Preprocessing ---
def load_all_data():
    try:
        df_influencers = pd.read_csv('influencers.csv')
        df_posts = pd.read_csv('posts.csv')
        df_tracking_data = pd.read_csv('tracking_data.csv')
        df_payouts = pd.read_csv('payouts.csv')

        df_influencers['ID'] = df_influencers['ID'].astype(str)
        
        def get_influencer_tier(followers):
            if followers < 10000: return "Nano (<10k)"
            elif 10000 <= followers < 100000: return "Micro (10k-100k)"
            elif 100000 <= followers < 1000000: return "Mid (100k-1M)"
            elif 1000000 <= followers < 5000000: return "Macro (1M-5M)"
            else: return "Mega (5M+)"
        df_influencers['tier'] = df_influencers['follower_count'].apply(get_influencer_tier)

        df_posts['date'] = pd.to_datetime(df_posts['date'])
        df_posts['influencer_id'] = df_posts['influencer_id'].astype(str)

        df_tracking_data['date'] = pd.to_datetime(df_tracking_data['date'])
        df_tracking_data['influencer_id'] = df_tracking_data['influencer_id'].astype(str)
        if 'brand' not in df_tracking_data.columns:
            products_brands = {
                'Whey Protein (MB)': 'MuscleBlaze', 'Creatine (MB)': 'MuscleBlaze', 'BCAA (MB)': 'MuscleBlaze',
                'Multivitamin (HK)': 'HKVitals', 'Fish Oil (HK)': 'HKVitals', 'Biotin (HK)': 'HKVitals',
                'Kids Nutrition (GZ)': 'Gritzo', 'Kids Immunity (GZ)': 'Gritzo'
            }
            df_tracking_data['brand'] = df_tracking_data['product'].map(products_brands)

        df_payouts['influencer_id'] = df_payouts['influencer_id'].astype(str)

        # --- Create a comprehensive tracking_data_full DataFrame for efficient filtering ---
        df_tracking_data_full = pd.merge(
            df_tracking_data,
            df_influencers[['ID', 'platform', 'category', 'tier']],
            left_on='influencer_id',
            right_on='ID',
            how='left'
        )
        df_tracking_data_full.drop(columns='ID', inplace=True)

        return df_influencers, df_posts, df_tracking_data, df_payouts, df_tracking_data_full
    except Exception as e:
        raise type(e)(f"Error loading or processing data. Ensure CSVs are present and correct in the root folder: {e}")

df_influencers_global, df_posts_global, df_tracking_data_raw_global, df_payouts_raw_global, df_tracking_data_full_global = load_all_data()

# --- Helper Functions for Formatting ---
def format_rupee_lakh(value):
    if pd.isna(value) or value == 0:
        return "₹0.0L"
    return f"₹{value / 100000:,.1f}L"

def format_roas(value):
    if pd.isna(value) or value == 0:
        return "0.0x"
    return f"{value:,.1f}x"

def calculate_kpis(filtered_df_tracking, filtered_df_payouts):
    total_revenue = filtered_df_tracking['revenue'].sum()
    total_payout = filtered_df_payouts['total_payout'].sum()
    total_orders = filtered_df_tracking['orders'].sum()
    
    overall_roas = total_revenue / total_payout if total_payout > 0 else 0
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
    unique_active_influencers = filtered_df_tracking['influencer_id'].nunique()

    return total_revenue, total_payout, total_orders, overall_roas, avg_order_value, unique_active_influencers

# --- Initialize the Dash app ---
app = dash.Dash(__name__, title="HealthKart Influencer Dashboard")
server = app.server # For gunicorn deployment

# --- Define the App Layout ---
app.layout = html.Div(id='main-app-container', children=[
    # Header Section
    html.Div(className='card-container header-card', children=[
        html.H1(children='🚀 HealthKart Influencer ROI Dashboard'),
        html.P(children='Track and optimize your influencer marketing campaigns across MuscleBlaze, HKVitals & Gritzo')
    ]),

    # Filters Section
    html.Div(className='card-container', children=[
        html.H3('Filters'),
        html.Div(className='filter-grid', children=[ # Using CSS grid for layout
            html.Div(children=[
                html.Label('Brand', className='filter-label'),
                dcc.Dropdown(
                    id='brand-filter',
                    options=[{'label': 'All Brands', 'value': 'All Brands'}] +
                            [{'label': brand, 'value': brand} for brand in sorted(df_tracking_data_raw_global['brand'].unique().tolist())],
                    value='All Brands',
                    clearable=False, # Make sure 'All Brands' can't be cleared
                    className='dash-dropdown' # For custom CSS
                )
            ]),
            html.Div(children=[
                html.Label('Platform', className='filter-label'),
                dcc.Dropdown(
                    id='platform-filter',
                    options=[{'label': 'All Platforms', 'value': 'All Platforms'}] +
                            [{'label': platform, 'value': platform} for platform in sorted(df_influencers_global['platform'].unique().tolist())],
                    value='All Platforms',
                    clearable=False,
                    className='dash-dropdown'
                )
            ]),
            html.Div(children=[
                html.Label('Influencer Category', className='filter-label'),
                dcc.Dropdown(
                    id='category-filter',
                    options=[{'label': 'All Categories', 'value': 'All Categories'}] +
                            [{'label': category, 'value': category} for category in sorted(df_influencers_global['category'].unique().tolist())],
                    value='All Categories',
                    clearable=False,
                    className='dash-dropdown'
                )
            ]),
            html.Div(children=[
                html.Label('Influencer Tier', className='filter-label'),
                dcc.Dropdown(
                    id='tier-filter',
                    options=[{'label': 'All Tiers', 'value': 'All Tiers'}] +
                            [{'label': tier, 'value': tier} for tier in sorted(df_influencers_global['tier'].unique().tolist(), key=lambda x: ['Nano', 'Micro', 'Mid', 'Macro', 'Mega'].index(x.split(' ')[0]))],
                    value='All Tiers',
                    clearable=False,
                    className='dash-dropdown'
                )
            ]),
            html.Div(children=[
                html.Label('Date Range', className='filter-label'),
                dcc.Dropdown(
                    id='date-range-filter',
                    options=[
                        {'label': 'Last 30 Days', 'value': '30'},
                        {'label': 'Last 90 Days', 'value': '90'},
                        {'label': 'Last 180 Days', 'value': '180'},
                        {'label': 'Last 365 Days', 'value': '365'},
                        {'label': 'All Time', 'value': 'All'}
                    ],
                    value='All', # Default to 'All Time'
                    clearable=False,
                    className='dash-dropdown'
                )
            ])
        ])
    ]),

    # KPI Cards Section - Output of callbacks will update these
    html.Div(className='card-container', children=[
        html.H3('Key Performance Indicators'),
        html.Div(className='kpi-grid', children=[
            html.Div(className='kpi-card', children=[html.P('Average ROAS', className='kpi-label'), html.H2(id='avg-roas-kpi', className='kpi-value')]),
            html.Div(className='kpi-card', children=[html.P('Total Revenue', className='kpi-label'), html.H2(id='total-revenue-kpi', className='kpi-value')]),
            html.Div(className='kpi-card', children=[html.P('Total Ad Spend', className='kpi-label'), html.H2(id='total-ad-spend-kpi', className='kpi-value')]),
            html.Div(className='kpi-card', children=[html.P('Total Orders', className='kpi-label'), html.H2(id='total-orders-kpi', className='kpi-value')]),
            html.Div(className='kpi-card', children=[html.P('Avg Order Value', className='kpi-label'), html.H2(id='avg-order-value-kpi', className='kpi-value')]),
            html.Div(className='kpi-card', children=[html.P('Active Influencers', className='kpi-label'), html.H2(id='active-influencers-kpi', className='kpi-value')])
        ])
    ]),

    # Conversion Funnel
    html.Div(className='card-container', children=[
        html.H3('📈 Conversion Funnel'),
        html.P('Visualize the journey from influencer posts to revenue generation.'),
        dcc.Graph(id='conversion-funnel-chart', className='chart-container', style={'height': '350px'}) # Added fixed height
    ]),

    # Charts Section
    html.Div(className='card-container', children=[
        html.H3('Dashboard Charts'),
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'}, children=[
            html.Div(className='chart-container', children=[
                html.H3('📈 ROAS Trend Over Time'),
                dcc.Graph(id='roas-trend-chart', style={'height': '350px'}) # Added fixed height
            ]),
            html.Div(className='chart-container', children=[
                html.H3('🧑‍💻 Top Performing Platforms'),
                dcc.Graph(id='platform-performance-chart', style={'height': '350px'}) # Added fixed height
            ]),
            html.Div(className='chart-container', children=[
                html.H3('💰 Revenue by Brand'),
                dcc.Graph(id='revenue-by-brand-chart', style={'height': '350px'}) # Added fixed height
            ]),
            html.Div(className='chart-container', children=[
                html.H3('👥 Influencer Performance Distribution'),
                dcc.Graph(id='influencer-scatter-chart', style={'height': '350px'}) # Added fixed height
            ])
        ])
    ]),
    
    # Key Insights
    html.Div(className='card-container', children=[
        html.H3('💡 Key Insights'),
        html.Div(className='insight-box', children=[html.P(id='overall-performance-insight')]),
        html.Div(className='insight-box', children=[html.P(id='top-platform-insight')]),
        html.Div(className='insight-box', children=[html.P(id='best-brand-insight')]),
        html.Div(className='insight-box', children=[html.P(id='high-performers-insight')]),
        html.Div(className='insight-box', children=[html.P('📈 Recommendation: Excellent performance! Scale successful campaigns')])
    ]),

    # Top Performing Influencers Table
    html.Div(className='card-container dash-table-container', children=[
        html.H3('⭐ Top Performing Influencers'),
        html.Div(style={'display': 'flex', 'gap': '10px', 'marginBottom': '15px'}, children=[
            html.Button('Export CSV', id='btn-export-csv', n_clicks=0, className='dash-button'),
            html.Button('Export PDF', id='btn-export-pdf', n_clicks=0, className='dash-button',
                        title="Direct PDF export is complex. Use browser's Print to PDF (Ctrl+P/Cmd+P).", disabled=True)
        ]),
        dash_table.DataTable(
            id='influencer-table',
            columns=[
                {"name": "Influencer", "id": "Influencer"},
                {"name": "Platform", "id": "Platform"},
                {"name": "Category", "id": "Category"},
                {"name": "Tier", "id": "Tier"},
                {"name": "Followers", "id": "Followers"},
                {"name": "Posts", "id": "Posts"},
                {"name": "Orders", "id": "Orders"},
                {"name": "Revenue", "id": "Revenue"},
                {"name": "Spend", "id": "Spend"},
                {"name": "ROAS", "id": "ROAS_Display"} # This uses the column name we just created
            ],
            data=[], # Data will be updated by callback
            style_header={
                'backgroundColor': '#8B5CFF',
                'color': 'white',
                'fontWeight': 'bold',
                'border': '1px solid #6C4EC7'
            },
            style_cell={
                'backgroundColor': '#A99DE0',
                'color': 'white',
                'border': '1px solid #7A68B8',
                'padding': '12px',
                'textAlign': 'left' # Default alignment
            },
            style_data_conditional=[], # This will be updated by callback
            style_as_list_view=True,
            page_size=10, # Number of rows per page
        )
    ])
])

# --- Callbacks for Interactivity (MAIN LOGIC) ---
@app.callback(
    [
        Output('avg-roas-kpi', 'children'),
        Output('total-revenue-kpi', 'children'),
        Output('total-ad-spend-kpi', 'children'),
        Output('total-orders-kpi', 'children'),
        Output('avg-order-value-kpi', 'children'),
        Output('active-influencers-kpi', 'children'),
        Output('roas-trend-chart', 'figure'),
        Output('platform-performance-chart', 'figure'),
        Output('revenue-by-brand-chart', 'figure'),
        Output('influencer-scatter-chart', 'figure'),
        Output('overall-performance-insight', 'children'),
        Output('top-platform-insight', 'children'),
        Output('best-brand-insight', 'children'),
        Output('high-performers-insight', 'children'),
        Output('influencer-table', 'data'), # For the table data
        Output('influencer-table', 'style_data_conditional'), # For table styling
        Output('conversion-funnel-chart', 'figure') # For the funnel chart
    ],
    [
        Input('brand-filter', 'value'),
        Input('platform-filter', 'value'),
        Input('category-filter', 'value'),
        Input('tier-filter', 'value'),
        Input('date-range-filter', 'value')
    ]
)
def update_dashboard_content(selected_brand, selected_platform, selected_category, selected_tier, selected_date_range):
    # --- Data Filtering Logic ---
    current_date = datetime.now().date()
    if selected_date_range == 'All':
        start_date_filter = df_posts_global['date'].min().date() if not df_posts_global.empty else current_date
    else:
        start_date_filter = current_date - timedelta(days=int(selected_date_range))
    end_date_filter = current_date

    # Filter posts for ROAS trend and post counts
    filtered_posts_df = df_posts_global[(df_posts_global['date'].dt.date >= start_date_filter) & (df_posts_global['date'].dt.date <= end_date_filter)]
    
    # Start with full tracking data and filter down
    filtered_tracking_data_current = df_tracking_data_full_global[
        (df_tracking_data_full_global['date'].dt.date >= start_date_filter) &
        (df_tracking_data_full_global['date'].dt.date <= end_date_filter)
    ].copy()

    # Apply Brand filter
    if selected_brand != 'All Brands':
        filtered_tracking_data_current = filtered_tracking_data_current[
            filtered_tracking_data_current['brand'] == selected_brand
        ]
    
    # Apply Platform filter
    if selected_platform != 'All Platforms':
        filtered_tracking_data_current = filtered_tracking_data_current[
            filtered_tracking_data_current['platform'] == selected_platform
        ]
    
    # Apply Category filter
    if selected_category != 'All Categories':
        filtered_tracking_data_current = filtered_tracking_data_current[
            filtered_tracking_data_current['category'] == selected_category
        ]

    # Apply Tier filter
    if selected_tier != 'All Tiers':
        filtered_tracking_data_current = filtered_tracking_data_current[
            filtered_tracking_data_current['tier'] == selected_tier
        ]

    # Filter payouts based on the influencers remaining after tracking data filtering
    relevant_influencer_ids_filtered = filtered_tracking_data_current['influencer_id'].unique()
    filtered_payouts_df = df_payouts_raw_global[df_payouts_raw_global['influencer_id'].isin(relevant_influencer_ids_filtered)]
    
    # Filter influencers based on all applied filters for specific influencer insights
    filtered_influencers_df = df_influencers_global[df_influencers_global['ID'].isin(relevant_influencer_ids_filtered)]


    # --- KPI Calculations ---
    total_revenue, total_payout, total_orders, overall_roas, avg_order_value, unique_active_influencers = \
        calculate_kpis(filtered_tracking_data_current, filtered_payouts_df)

    # --- Chart 1: ROAS Trend Over Time ---
    # Determine the actual min/max dates with data to limit the X-axis range
    min_date_with_data = filtered_posts_df['date'].min().date() if not filtered_posts_df.empty else start_date_filter
    max_date_with_data = filtered_posts_df['date'].max().date() if not filtered_posts_df.empty else end_date_filter
    
    daily_payout_sum = filtered_payouts_df.groupby(filtered_posts_df['date'].dt.date)['total_payout'].sum().reindex(pd.date_range(start_date_filter, end_date_filter).date, fill_value=0).rename('Daily Payout')
    daily_revenue_sum = filtered_tracking_data_current.groupby(filtered_tracking_data_current['date'].dt.date)['revenue'].sum().reindex(pd.date_range(start_date_filter, end_date_filter).date, fill_value=0).rename('Daily Revenue')
    
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

    # --- Chart 2: Top Performing Platforms ---
    platform_revenue_df = filtered_tracking_data_current.groupby('platform')['revenue'].sum().reset_index()
    platform_performance_fig = go.Figure().update_layout(template="plotly_dark", title="No Data for Platforms")
    if not platform_revenue_df.empty:
        platform_performance_fig = px.pie(platform_revenue_df, values='revenue', names='platform', hole=0.5,
                                    title='',
                                    color_discrete_sequence=px.colors.sequential.Agsunset)
        platform_performance_fig.update_traces(textinfo='percent+label', pull=[0.05, 0, 0, 0])
        platform_performance_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white', showlegend=True)

    # --- Chart 3: Revenue by Brand ---
    brand_revenue_df = filtered_tracking_data_current.groupby('brand')['revenue'].sum().reset_index().sort_values('revenue', ascending=False)
    revenue_by_brand_fig = go.Figure().update_layout(template="plotly_dark", title="No Data for Brands")
    if not brand_revenue_df.empty:
        revenue_by_brand_fig = px.bar(brand_revenue_df, x='brand', y='revenue', title='',
                                labels={'revenue': 'Revenue (₹)', 'brand': 'Brand'},
                                color='revenue', color_continuous_scale=px.colors.sequential.Plasma)
        revenue_by_brand_fig.update_layout(plot_bgcolor='#A99DE0', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    
    # --- Chart 4: Influencer Performance Distribution (Scatter Plot) ---
    influencer_scatter_summary = filtered_tracking_data_current.groupby('influencer_id').agg(
        TotalRevenue=('revenue', 'sum'),
        TotalOrders=('orders', 'sum')
    ).reset_index()
    influencer_scatter_summary = pd.merge(influencer_scatter_summary, filtered_influencers_df[['ID', 'name', 'platform', 'category', 'follower_count', 'tier']],
                                left_on='influencer_id', right_on='ID', how='left')
    payouts_for_merge_scatter = filtered_payouts_df.groupby('influencer_id')['total_payout'].sum().reset_index()
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
        influencer_scatter_summary.loc[(influencer_scatter_summary['total_payout'] == 0) & (influencer_scatter_summary['TotalRevenue'] > 0), 'ROAS_Val'] = 1000

    scatter_fig = go.Figure().update_layout(template="plotly_dark", title="No Data for Scatter Plot")
    if not influencer_scatter_summary.empty:
        scatter_fig = px.scatter(influencer_scatter_summary, x='TotalRevenue', y='ROAS_Val',
                                hover_name='name',
                                hover_data=['platform', 'category', 'total_payout', 'tier'],
                                labels={'TotalRevenue': 'Revenue (₹)', 'ROAS_Val': 'ROAS (x)'},
                                color_discrete_sequence=[px.colors.qualitative.Plotly[1]])
        scatter_fig.update_layout(plot_bgcolor='#A99DE0', paper_bgcolor='rgba(0,0,0,0)', font_color='white')

    # --- Conversion Funnel ---
    total_posts_funnel = filtered_posts_df.shape[0]
    total_orders_funnel = filtered_tracking_data_current['orders'].sum()
    total_revenue_funnel = filtered_tracking_data_current['revenue'].sum()

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


    # --- Key Insights ---
    overall_performance_insight = html.P(f"📊 Total Performance: {format_roas(overall_roas)} ROAS with {format_rupee_lakh(total_revenue)} revenue generated")

    top_platform_name = "N/A"
    top_platform_revenue = 0
    if not platform_revenue_df.empty:
        top_platform_name = platform_revenue_df.loc[platform_revenue_df['revenue'].idxmax(), 'platform']
        top_platform_revenue = platform_revenue_df['revenue'].max()
    top_platform_insight = html.P(f"🏆 Top Platform: {top_platform_name} generates {format_rupee_lakh(top_platform_revenue)} revenue")

    best_brand_name = "N/A"
    best_brand_revenue = 0
    if not brand_revenue_df.empty:
        best_brand_name = brand_revenue_df.loc[brand_revenue_df['revenue'].idxmax(), 'brand']
        best_brand_revenue = brand_revenue_df['revenue'].max()
    best_brand_insight = html.P(f"💪 Best Brand: {best_brand_name} leads with {format_rupee_lakh(best_brand_revenue)} in sales")

    high_roas_influencers_count = 0
    if not influencer_scatter_summary.empty and not filtered_payouts_df.empty:
        high_roas_influencers_count = influencer_scatter_summary[influencer_scatter_summary['ROAS_Val'] >= 5].shape[0]
    high_performers_insight = html.P(f"🌟 High Performers: {high_roas_influencers_count} influencers achieving 5x+ ROAS")


    # --- Top Performing Influencers Table ---
    influencer_table_summary = influencer_scatter_summary.copy()
    
    post_counts = filtered_posts_df.groupby('influencer_id').size().rename('Posts').reset_index()
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

    table_data = []
    style_data_conditional_list = []

    if not influencer_table_summary.empty:
        influencer_table_summary = influencer_table_summary.sort_values(by='ROAS_Val', ascending=False)
        
        # Take head(10) to display in the table
        display_df_for_table = influencer_table_summary.head(10)

        table_data = display_df_for_table[[
            'Influencer', 'Platform', 'Category', 'Tier', 'Followers', 'Posts', 'Orders',
            'Revenue', 'Spend', 'ROAS_Display'
        ]].to_dict('records')

        # Prepare style_data_conditional for the ROAS column based on its value
        # This iterates over the rows that will actually be displayed in the table
        for i, row in display_df_for_table.iterrows(): # Iterate over the top 10 displayed rows
            roas_val = row['ROAS_Val']
            
            # Use the actual row_index within the filtered (head(10)) data for styling
            # The 'if' condition matches the row_index within the `data` list passed to DataTable
            row_index_in_data = display_df_for_table.index.get_loc(i) 

            class_name = 'roas-neutral'
            if roas_val > table_max_roas * 1.1 or roas_val >= 1:
                class_name = 'roas-positive'
            elif roas_val <= 0:
                class_name = 'roas-negative'

            style_data_conditional_list.append({
                'if': {'row_index': row_index_in_data, 'column_id': 'ROAS_Display'},
                'class_name': class_name
            })


    # Return all outputs in the correct order
    return (
        format_roas(overall_roas),
        format_rupee_lakh(total_revenue),
        format_rupee_lakh(total_payout),
        f"{int(total_orders):,}",
        f"₹{avg_order_value:,.0f}",
        f"{unique_active_influencers:,}",
        roas_trend_fig,
        platform_performance_fig,
        revenue_by_brand_fig,
        scatter_fig,
        overall_performance_insight,
        top_platform_insight,
        best_brand_insight,
        high_performers_insight,
        table_data, # Updated table data
        style_data_conditional_list, # Updated table styling
        conversion_funnel_fig
    )

server = app.server  # <- Hugging Face needs this line

if __name__ == '__main__':
    app.run_server(host="0.0.0.0", port=8050)

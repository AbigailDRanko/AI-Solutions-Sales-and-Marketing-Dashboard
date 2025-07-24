# MUST BE THE FIRST STREAMLIT COMMAND (before any other code)
import streamlit as st
st.set_page_config(page_title="AI Sales & Marketing Dashboard", layout="wide")

# THEN other imports
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.graph_objects import Indicator
import plotly.graph_objects as go
import json
import os
import joblib
from pathlib import Path
import sys
print(sys.executable)
from urllib.parse import parse_qs
import uuid
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin

# Define SaleFeatureGenerator class (copied from model code, modified for prediction)
class SaleFeatureGenerator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Compute ROI using page_views and acquisition_cost (no dependency on y_sale)
        X['ROI'] = X['page_views'] / (X['acquisition_cost'] + 1e-5)
        return X

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .prediction-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin: 10px 0;
    }
    .metric-title {
        font-size: 14px;
        color: #555;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 20px;
        font-weight: bold;
        color: #4CAF50;
    }
    .form-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
    }
    .header {
        color: #4CAF50;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .subheader {
        color: R33;
        font-size: 18px;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .code-block {
        background-color: #1e1e1e;
        color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        font-size: 14px;
        max-height: 400px;
        overflow-y: auto;
    }
    .error-message {
        color: #F44336;
        font-size: 14px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

import os


# --- Custom CSS Styling ---
st.markdown(
    """
    <style>
    .main {
        background-color: #f4f4f4;
        padding: 0.5rem;
    }
    h1 {
        color: #4CAF50;
        font-size: 1.8rem !important;
    }
    .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    .stPlotlyChart {
        border-radius: 8px;
        box-shadow: 0 0 5px rgba(0,0,0,0.1);
        background: white;
        padding: 5px !important;
        margin-bottom: 10px !important;
    }
    .st-emotion-cache-1y4p8pa {
        padding: 0.5rem 1rem 0.5rem;
    }
    .metric-container {
        padding: 0.5rem;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 0 5px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-title {
        font-size: 12px;
        color: #666;
        margin: 0;
    }
    .metric-value {
        font-size: 18px;
        color: #4CAF50;
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_models():
    try:
        model_dir = os.path.join(os.getcwd(), 'model')
        sale_model_path = os.path.join(model_dir, 'sale_pipeline.pkl')
        
        # Debug: Verify path
        if not os.path.exists(sale_model_path):
            st.error(f"Model file not found at: {sale_model_path}")
            return None
            
        sale_pipeline = joblib.load(sale_model_path)

        # Extract feature names
        try:
            feature_names = sale_pipeline.named_steps['preprocessor'].get_feature_names_out()
            feature_names = feature_names.tolist()
        except Exception as e:
            st.warning(f"Could not extract feature names: {str(e)}")
            feature_names = None

        return {
            'sale_model': sale_pipeline,
            'features': feature_names
        }
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# load the data
@st.cache_data
def load_data():
    df = pd.read_csv(r'C:\Users\bida21-094\Desktop\cet333_project\notebook\enhanced_web_analytics_dataset3.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M')  # Specify the exact format
    df['month'] = df['timestamp'].dt.to_period('M').astype(str)
    df['year'] = df['timestamp'].dt.to_period('Y').astype(str)
    
    # Ensure numeric columns are properly typed
    numeric_cols = ['sale_amount', 'acquisition_cost', 'roi', 'ctr', 
                   'conversion_rate', 'page_views', 'session_duration_sec']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

df = load_data()


# --- Sidebar Filters ---
with st.sidebar:
    # logo/image at the top
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(r'C:\Users\bida21-094\Desktop\cet333_project\notebook\logocty.jpg', width=120)

    st.markdown("🔍 Filters")

    # Date filter
    start_date = st.date_input('Start Date', df['timestamp'].min().date())
    end_date = st.date_input('End Date', df['timestamp'].max().date())

    # Region filter
    region_filter = st.multiselect(
        'Select Regions', 
        df['region'].unique(), 
        default=list(df['region'].unique())
    )


# --- Filter Data Function ---
def filter_data(start_date, end_date, region_filter):
    filtered = df[
        (df['timestamp'] >= pd.to_datetime(start_date)) &
        (df['timestamp'] <= pd.to_datetime(end_date))
    ]
    filtered = filtered[filtered['region'].isin(region_filter)]
    return filtered

# Apply filters
filtered_df = filter_data(start_date, end_date, region_filter)

# --- KPI Metrics ---
def kpi_metrics(data):
    total_sales = data['sale_amount'].sum()
    retention_rate = round(data['retained_customer'].mean() * 100, 2)
    page_views_total = data['page_views'].sum()
    return total_sales, retention_rate, page_views_total

total_sales, retention_rate, page_views_total = kpi_metrics(filtered_df)

def display_metric_container(title, value, suffix="", small=False, full_width=False):
    width = "100%" if full_width else "auto"
    title_size = "12px" if small else "14px"
    value_size = "16px" if small else "18px"
    
    st.markdown(
        f"""
        <div class="metric-container" style="width:{width}">
            <p class="metric-title" style="font-size:{title_size}">{title}</p>
            <p class="metric-value" style="font-size:{value_size}">{value}{suffix}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# --- Charts ---
def create_charts(data):
    charts = []
    chart_height = 210  # Standard height for all charts
    margin = dict(l=5, r=5, t=30, b=5)
    
    # 1. Line Chart: Sales Over Time with Dynamic Coloring
    monthly_sales = data.groupby('month')['sale_amount'].sum().reset_index()
    target_value = 800000  # Your target value

    # Create figure
    fig1 = go.Figure()

    # all line segments with proper colors
    for i in range(len(monthly_sales)-1):
        current_val = monthly_sales.iloc[i]['sale_amount']
        next_val = monthly_sales.iloc[i+1]['sale_amount']
        
        # Determine segment color
        if current_val >= target_value and next_val >= target_value:
            segment_color = 'green'
        elif current_val < target_value and next_val < target_value:
            segment_color = 'red'
        else:
            segment_color = 'orange'  # Crossing target
        
        fig1.add_scatter(
            x=[monthly_sales.iloc[i]['month'], monthly_sales.iloc[i+1]['month']],
            y=[current_val, next_val],
            mode='lines',
            line=dict(color=segment_color, width=2),
            showlegend=False
        )

    # Add markers with matching colors
    marker_colors = []
    for val in monthly_sales['sale_amount']:
        if val >= target_value:
            marker_colors.append('green')
        else:
            marker_colors.append('red')

    fig1.add_scatter(
        x=monthly_sales['month'],
        y=monthly_sales['sale_amount'],
        mode='markers',
        marker=dict(size=8, color=marker_colors),
        name='Monthly Sales',
        showlegend=False
    )

    # Add target line
    fig1.add_hline(
        y=target_value,
        line_dash="dot",
        line_color="black",
        annotation_text=f"Target: $800,000",
        annotation_position="top left",
        name="Sales Target"
    )

    # Create custom legend
    fig1.add_scatter(
        x=[None], y=[None],
        mode='lines+markers',
        line=dict(color='green', width=2),
        marker=dict(size=8, color='green'),
        name='Above Target'
    )
    fig1.add_scatter(
        x=[None], y=[None],
        mode='lines+markers',
        line=dict(color='red', width=2),
        marker=dict(size=8, color='red'),
        name='Below Target'
    )
    fig1.add_scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='orange', width=2),
        name='Crossing Target'
    )

    # Update layout
    fig1.update_layout(
        title='Sales Over Time',
        xaxis_title='Month',
        yaxis_title='Sales Amount',
        margin=margin,
        height=150,
        title_font_size=14,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    charts.append(fig1)

    
    # 2. Treemap: Top 10 Products with Precise Color Grading
    top_products = data.groupby('product')['sale_amount'].sum().nlargest(10).reset_index()
    top_products = top_products.sort_values('sale_amount', ascending=False).reset_index(drop=True)

    # Assign specific colors by position (0-based index)
    color_assignments = {
        0: '#006400',    # Dark green (top performer)
        1: '#3CB371',    # Medium green (second top)
        len(top_products)-3: '#FFA500',  # Light amber (third lowest)
        len(top_products)-2: '#E74C3C',  # Light red (second lowest)
        len(top_products)-1: '#FF0000'    # Red (lowest)
    }

    # color sequence using only hex codes
    colors = []
    for i in range(len(top_products)):
        if i in color_assignments:
            colors.append(color_assignments[i])
        elif i < len(top_products)-3:  # Upper middle (green gradient)
            # Interpolate between #3CB371 and #90EE90
            r = int(60 + (144-60)*(i/(len(top_products)-3)))
            g = int(179 + (238-179)*(i/(len(top_products)-3)))
            b = int(113 + (144-113)*(i/(len(top_products)-3)))
            colors.append(f'#{r:02x}{g:02x}{b:02x}')
        else:  # Lower middle
            colors.append('#FFA500')  # Default to light amber

    # color mapping dictionary
    color_discrete_map = {product: color for product, color in zip(top_products['product'], colors)}

    # treemap
    fig2 = px.treemap(
        top_products,
        path=['product'],
        values='sale_amount',
        color='product',
        color_discrete_map=color_discrete_map,
        title='Sales performance by Products'
    )

    # Customize appearance
    fig2.update_traces(
        textinfo='label+value',
        texttemplate='<b>%{label}</b><br>$%{value:,.0f}',
        marker=dict(line=dict(width=1, color='#333333')),
        hovertemplate='<b>%{label}</b><br>Sales: $%{value:,.0f}<extra></extra>'
    )

    # Update layout
    fig2.update_layout(
        margin=margin,
        height=chart_height,
        title_font_size=14,
        uniformtext=dict(minsize=10, mode='hide'),
        showlegend=False
    )

    charts.append(fig2)


    # Aggregate sales by region and map to representative countries
    # 3. Choropleth Map: Sales by Region with Performance Coloring

    # Define mapping from country codes to full country names
    country_code_to_name = {
        # Your specific country codes
        'MX': 'Mexico',
        'CA': 'Canada',
        'US': 'United States',
        'AU': 'Australia',
        'PE': 'Peru',
        'AR': 'Argentina',
        'ES': 'Spain',
        'CN': 'China',
        'NL': 'Netherlands',
        'NZ': 'New Zealand',
        'GB': 'United Kingdom',
        'CL': 'Chile',
        'ZA': 'South Africa',
        'FJ': 'Fiji',
        'NG': 'Nigeria',
        'IN': 'India',
        'KE': 'Kenya',
        'FR': 'France',
        'KR': 'South Korea',
        'CO': 'Colombia',
        'CL': 'Chile',
        'DE': 'Germany',
        'MA': 'Morocco',

        
        # Your regional mappings
        'North America': 'United States',
        'Europe': 'Germany',
        'Asia': 'China',
        'South America': 'Brazil',
        'Africa': 'South Africa',
        'Oceania': 'Australia'
    }

    # 3. Choropleth Map: Sales by Location with Performance Coloring
    def get_location_data(data):
        """Process data to get locations, handling both country codes and region names"""
        # First try grouping by country code
        location_sales = data.groupby('country')['sale_amount'].sum().reset_index()
        location_sales['location_name'] = location_sales['country'].map(country_code_to_name)
        location_sales['location_type'] = 'country'
        
        # Then try grouping by region if country not found
        if 'region' in data.columns:
            region_sales = data.groupby('region')['sale_amount'].sum().reset_index()
            region_sales['location_name'] = region_sales['region'].map(country_code_to_name)
            region_sales['location_type'] = 'region'
            location_sales = pd.concat([location_sales, region_sales])
        
        # Clean up and return
        location_sales = location_sales.dropna(subset=['location_name'])
        return location_sales

    location_sales = get_location_data(data)

    if len(location_sales) > 0:
        # Create performance tiers
        sales_quantiles = location_sales['sale_amount'].quantile([0.33, 0.66])
        location_sales['performance'] = pd.cut(
            location_sales['sale_amount'],
            bins=[-1, sales_quantiles[0.33], sales_quantiles[0.66], float('inf')],
            labels=['low', 'medium', 'high']
        )

        # Create custom color scale
        custom_scale = [
            [0.0, '#FF0000'],    # Red for low
            [0.5, '#FFA500'],    # Amber for medium
            [1.0, '#008000']     # Green for high
        ]

        # Create hover text
        location_sales['hover_text'] = location_sales.apply(
            lambda x: f"{x['country'] if x['location_type'] == 'country' else x['region']}",
            axis=1
        )

        fig3 = px.choropleth(
            location_sales,
            locations='location_name',
            locationmode='country names',
            color='sale_amount',
            hover_name='hover_text',
            hover_data={'location_name': False, 'sale_amount': ':.0f'},
            color_continuous_scale=custom_scale,
            title='Regional Sales Performance'
        )

        fig3.update_geos(
            showcountries=True,
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth'
        )

        fig3.update_layout(
            margin={"r":0,"t":40,"l":0,"b":0},
            height=200,
            title_font_size=16,
            coloraxis_colorbar=dict(
                title="Sales Amount",
                thicknessmode="pixels",
                thickness=15,
                lenmode="pixels",
                len=300,
                yanchor="middle",
                y=0.5
            )
        )

        charts.append(fig3)
    else:
        print("Warning: No valid location mappings found")

    # 4. Box Plot: ROI by Campaign
    fig4 = px.box(data, x='campaign_id', y='roi', color='campaign_id', title='ROI by Campaign')
    fig4.update_layout(margin=margin, height=chart_height, title_font_size=14, showlegend=False)
    charts.append(fig4)

   # 5. Bubble Chart: Acquisition Cost vs ROI
    # Add performance category to the dataset
    data['performance'] = data.apply(
        lambda row: 'Poor' if row['roi'] < 0.6 and row['acquisition_cost'] > data['acquisition_cost'].median() else 'Good',
        axis=1
    )

    # Bubble chart with performance-based coloring
    fig5 = px.scatter(
        data,
        x='acquisition_cost',
        y='roi',
        size='sale_amount',
        color='campaign_id',
        hover_name='campaign_id',
        title='Acquisition Cost vs ROI',
        size_max=20 # Reduced from 30
    )
    fig5.update_layout(margin=margin, height=200, title_font_size=14)
    charts.append(fig5) 


    # 6. Horizontal Bar Chart: Customer Segments with Performance Coloring
    segment_sales = data.groupby('customer_segment')['sale_amount'].sum().reset_index()

    # Sort from highest to lowest sales
    segment_sales = segment_sales.sort_values('sale_amount', ascending=False)

    # Assign colors based on performance ranking
    segment_sales['color'] = 'lightgreen'  # Default color
    if len(segment_sales) >= 1:
        segment_sales.iloc[0, segment_sales.columns.get_loc('color')] = 'darkgreen'  # 1st place
    if len(segment_sales) >= 2:
        segment_sales.iloc[1, segment_sales.columns.get_loc('color')] = '#3CB371'  # 2nd place
    if len(segment_sales) >= 3:
        # Second lowest gets amber
        segment_sales.iloc[-2, segment_sales.columns.get_loc('color')] = 'orange'
    if len(segment_sales) >= 4:
        # Lowest gets red
        segment_sales.iloc[-1, segment_sales.columns.get_loc('color')] = 'red'

    fig6 = px.bar(
        segment_sales,
        x='sale_amount',
        y='customer_segment',
        orientation='h',
        color='customer_segment',
        color_discrete_map=dict(zip(segment_sales['customer_segment'], segment_sales['color'])),
        title='Customer Segmentation by Sales Performance'
    )

    fig6.update_layout(
        margin=margin,
        height=chart_height,
        title_font_size=14,
        yaxis_title='',
        xaxis_title='Sales',
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}  # Ensure proper ordering
    )

    # Add value labels
    fig6.update_traces(
        texttemplate='%{x:,.0f}',
        textposition='outside',
        textfont_size=7
    )

    charts.append(fig6)
        
    
    try:
        # First create the region_data DataFrame
        region_data = data.groupby(['region', 'campaign_id']).agg({
            'conversion_rate': 'mean',
            'roi': 'mean'
        }).reset_index()
        
        # Create subplots with custom height (350px for this specific chart)
        fig7 = make_subplots(
            rows=1, 
            cols=2,
            #subplot_titles=('Conversion Rate by Region', 'ROI by Region'),
            specs=[[{"type": "bar"}, {"type": "bar"}]],
            shared_yaxes=False,
            horizontal_spacing=0.1
        )
        
        campaigns = sorted(region_data['campaign_id'].unique())
        
        for i, campaign in enumerate(campaigns):
            campaign_df = region_data[region_data['campaign_id'] == campaign]
            
            # Conversion Rate subplot (left)
            fig7.add_trace(
                go.Bar(
                    x=campaign_df['region'],
                    y=campaign_df['conversion_rate'],
                    name=f'Campaign {campaign}',
                    marker_color=px.colors.qualitative.Plotly[i],
                    legendgroup=f'campaign_{campaign}',
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # ROI subplot (right)
            fig7.add_trace(
                go.Bar(
                    x=campaign_df['region'],
                    y=campaign_df['roi'],
                    name=f'Campaign {campaign}',
                    marker_color=px.colors.qualitative.Plotly[i],
                    legendgroup=f'campaign_{campaign}',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Custom height just for this chart (350px instead of standard 210px)
        fig7.update_layout(
            title_text='Campaign Performance Across Regions',
            margin=margin,
            height=450,  # Custom height for this specific chart
            barmode='group',
            legend_title_text='Campaigns',
            hovermode='closest'
        )
        
        # Update axes
        fig7.update_xaxes(title_text="Region", row=1, col=1)
        fig7.update_xaxes(title_text="Region", row=1, col=2)
        fig7.update_yaxes(title_text="Conversion Rate", row=1, col=1)
        fig7.update_yaxes(title_text="ROI", row=1, col=2)
        
        charts.append(fig7)
        
    except Exception as e:
        st.error(f"Could not create regional performance chart: {str(e)}")
        charts.append(go.Figure())
    

    # 8. Donut Chart: Customer Retention with Green/Red Colors
    # 8. Donut Chart: Customer Retention with Proper Green/Red Colors
    # Ensure boolean values are properly formatted as strings
    retention_data = data['retained_customer'].astype(str).value_counts().reset_index()
    retention_data.columns = ['retained', 'count']

    # Create the donut chart with explicit color mapping
    fig8 = px.pie(
        retention_data,
        values='count',
        names='retained',
        title='Customer Retention',
        hole=0.4,
        color='retained',
        color_discrete_map={'True': '#008000', 'False': '#F44336'}  # Green for True, Red for False
    )

    # Customize appearance
    fig8.update_traces(
        textinfo='percent+label',
        marker=dict(line=dict(color='white', width=1)),
        textfont_size=12,
        textposition='inside'
    )

    # Add center annotation and update layout
    fig8.update_layout(
        margin=margin,
        height=chart_height,
        title_font_size=14,
        showlegend=False,
        annotations=[dict(
            text=f"Retention<br>Rate",
            x=0.5, y=0.5,
            font_size=14,
            showarrow=False
        )]
    )

    charts.append(fig8)

    # 9. Area Chart: Page Views Over Time
    views_over_time = data.groupby('timestamp')['page_views'].sum().reset_index()
    fig9 = px.area(views_over_time, x='timestamp', y='page_views', title='Page Views Trend')
    fig9.update_layout(margin=margin, height=chart_height, title_font_size=14)
    charts.append(fig9)

    
    # 1. Sunburst Chart: Device/Website Analytics (Hierarchical)
   # 1. First, create and sort the aggregated data
    device_analytics = (data.groupby(['device', 'os', 'browser'])['page_views']
                        .sum()
                        .reset_index()
                        .sort_values('page_views', ascending=False))

    # 2. Then create the chart (Stacked Bar example)
    fig10 = px.bar(
        device_analytics,
        x='device',          # Devices on x-axis
        y='page_views',      # Traffic volume
        color='os',          # OS as stacked segments
        title='Page Views by Device & OS (Stacked - Highest to Lowest)',
        category_orders={"device": device_analytics['device'].unique()},  # Maintain sorted order
        color_discrete_map={
            'Windows': '#4CAF50',  # Green
            'MacOS': '#FFC107',    # Amber
            'iOS': '#F44336',      # Red
            'Android': '#9C27B0'   # Purple
        }
    )

    # 3. Update layout
    fig10.update_layout(
        barmode='stack',
        height=450,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis={'categoryorder':'total descending'}  # Ensures descending order
    )

    charts.append(fig10)

    # 2. Gauge Chart: Marketing Performance vs Target
    
    # 2. Interactive Gauge Chart: Sales Performance vs Target
    # Sales Performance Gauge Chart
    target_sales = 800000
    current_sales = data['sale_amount'].sum()
    sales_met = current_sales >= target_sales

    sales_gauge = go.Indicator(
        mode="gauge+number+delta",
        value=current_sales,
        delta={
            'reference': target_sales,
            'increasing': {'color': "#008000"},
            'decreasing': {'color': "#FF0000"},
            'relative': True,
            'valueformat': ".1%",
            'font': {'size': 10}
        },
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f"Sales Performance<br><span style='font-size:11px'>Target: ${target_sales:,.0f}</span>",
            'font': {'size': 8}
        },
        number={
            'prefix': "$",
            'valueformat': ",",
            'font': {'size': 10}
        },
        gauge={
            'axis': {
                'range': [0, target_sales*1.2],
                'tickfont': {'size': 6},
                'tickformat': ",",
                'tickcolor': "darkgray"
            },
            'bar': {'color': "rgba(0,0,0,0)"},  # Transparent bar (removes the black line)
            'steps': [
                {'range': [0, target_sales*0.7], 'color': "#FF0000", 'name': "Below Target"},
                {'range': [target_sales*0.7, target_sales], 'color': "#FFA500", 'name': "Approaching"},
                {'range': [target_sales, target_sales*1.2], 'color': "#008000", 'name': "Above Target"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 1},  # Keeps only the target line
                'thickness': 0.85,
                'value': target_sales
            }
        }
    )

    fig11 = go.Figure(sales_gauge)
    fig11.add_annotation(
        x=0.5, y=0.3,
        text=f"{'✓ Target Met' if sales_met else '✗ Target Missed'}",
        showarrow=False,
        font={'size': 10, 'color': "#008000" if sales_met else "#FF0000"}
    )
    fig11.update_layout(
        margin={'l': 30, 'r': 30, 't': 60, 'b': 30},
        height=chart_height,
        font={'family': "Arial"}
    )

    charts.append(fig11)


    # 3. Funnel Chart: Conversion Journey
    # 3. Funnel Chart: Conversion Journey (simplified)
    funnel_data = {
        'Stage': ['Visits', 'Product Views', 'Purchase'],
        'Count': [
            len(data),
            len(data[data['page_views'] > 1]),
            len(data[data['sale_amount'] > 0])
        ]
    }
    fig12 = px.funnel(funnel_data, x='Count', y='Stage', title='Conversion Funnel')
    fig12.update_layout(margin=margin, height=chart_height)
    charts.append(fig12)

    # In the create_charts function, replace the radar chart section with:

    # 4. Radar Chart: Campaign Performance Metrics (Fixed version)
    # In your create_charts() function, replace the existing radar chart code (fig13) with this:

    # 13. Radar Chart: Customer Segment Profiles
    try:
        # Aggregate metrics by customer segment
        segment_metrics = data.groupby('customer_segment').agg({
            'conversion_rate': 'mean',
            'sale_amount': 'mean',
            'session_duration_sec': 'mean',
            'page_views': 'mean',
            'roi': 'mean'
        }).reset_index()

        # Normalize metrics to 0-1 scale for better comparison
        metrics_to_normalize = ['conversion_rate', 'sale_amount', 
                              'session_duration_sec', 'page_views', 'roi']
        segment_metrics[metrics_to_normalize] = (
            segment_metrics[metrics_to_normalize] - 
            segment_metrics[metrics_to_normalize].min()
        ) / (
            segment_metrics[metrics_to_normalize].max() - 
            segment_metrics[metrics_to_normalize].min()
        )

        # Melt data for radar chart
        melted_data = segment_metrics.melt(
            id_vars=['customer_segment'],
            value_vars=metrics_to_normalize,
            var_name='metric',
            value_name='normalized_value'
        )

        # Create radar chart
        fig13 = px.line_polar(
            melted_data,
            r='normalized_value',
            theta='metric',
            color='customer_segment',
            line_close=True,
            title='<b>Customer Segment Profiles</b>',
            template='plotly_white',
            color_discrete_sequence=px.colors.qualitative.Vivid,
            hover_data={'customer_segment': True, 'metric': True, 'normalized_value': False}
        )

        # Enhance visual appearance
        fig13.update_traces(
            fill='toself',
            opacity=0.7,
            line=dict(width=2),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Metric: %{customdata[1]}<br>"
                "Normalized Value: %{r:.2f}<extra></extra>"
            )
        )

        fig13.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[0, 0.5, 1],
                    ticktext=['Low', 'Medium', 'High']
                ),
                angularaxis=dict(
                    direction="clockwise",
                    rotation=90
                )
            ),
            legend=dict(
                title='Customer Segments',
                orientation='h',
                yanchor='bottom',
                y=-0.2,
                xanchor='center',
                x=0.5
            ),
            margin=dict(l=50, r=50, t=80, b=50),
            height=450,
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        charts.append(fig13)
        
    except Exception as e:
        st.error(f"Error creating segment radar chart: {str(e)}")
        charts.append(go.Figure())
    # 5. Violin Plot: Session Duration by Device
    fig14 = px.violin(
        data,
        y='session_duration_sec',
        x='device',
        color='device',
        box=True,
        title='Session Duration Distribution by Device'
    )
    fig14.update_layout(margin=margin, height=chart_height, showlegend=False)
    charts.append(fig14)

    # 6. Scatter Matrix: Multivariate Analysis
    fig15 = px.scatter_matrix(
        data,
        dimensions=['sale_amount', 'acquisition_cost', 'page_views', 'session_duration_sec'],
        color='customer_segment',
        title='Multivariate Relationships'
    )
    fig15.update_layout(margin=margin, height=chart_height)
    charts.append(fig15)

    # 3. Gauge Chart: Marketing ROI Performance
   # ROI Gauge Chart
    roi_target = 2.0
    current_roi = (data['sale_amount'].sum() - data['acquisition_cost'].sum()) / data['acquisition_cost'].sum()
    target_met = current_roi >= roi_target

    fig16 = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_roi,
        delta={
            'reference': roi_target,
            'increasing': {'color': "#008000"},
            'decreasing': {'color': "#FF0000"},
            'relative': True,
            'valueformat': ".1%",
            'font': {'size': 8}
        },
        domain={'x': [0.1, 0.9], 'y': [0.1, 0.9]},
        title={
            'text': "Marketing ROI<br>Target: {:.1f}x".format(roi_target),
            'font': {'size': 10}
        },
        number={
            'suffix': "x",
            'valueformat': ".1f",
            'font': {'size': 10}
        },
        gauge={
            'axis': {
                'range': [0, roi_target*1.5],
                'tickfont': {'size': 8},
                'tickformat': ".1f",
                'tickcolor': "darkgray"
            },
            'bar': {'color': "rgba(0,0,0,0)"},  # Transparent bar (removes the black line)
            'steps': [
                {'range': [0, roi_target*0.7], 'color': "#FF0000", 'name': "Poor"},
                {'range': [roi_target*0.7, roi_target], 'color': "#FFA500", 'name': "Moderate"},
                {'range': [roi_target, roi_target*1.5], 'color': "#008000", 'name': "Excellent"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 2},  # Keeps only the target line
                'thickness': 0.85,
                'value': roi_target
            }
        }
    ))

    # Performance status annotation
    fig16.add_annotation(
        x=0.5, y=0.25,
        text="✓ Target Met" if target_met else "✗ Target Missed",
        showarrow=False,
        font={'size': 10, 'color': "#008000" if target_met else "#FF0000"}
    )

    # Update layout with chart height and other settings
    fig16.update_layout(
        margin={'l': 30, 'r': 30, 't': 60, 'b': 30},
        height=250,  # You can adjust this value or use a variable like chart_height=350
        font={'family': "Arial"},
        # Add interactivity
        hovermode="x unified",
        clickmode="event+select"
    )

    charts.append(fig16)


    # Prepare campaign performance data
    campaign_performance = data.groupby('campaign_id').agg({
        'sale_amount': 'sum',
        'roi': 'mean',
        'acquisition_cost': 'sum'
    }).reset_index().sort_values('roi', ascending=False)

    # Assign performance tiers
    campaign_performance['performance_tier'] = pd.qcut(
        campaign_performance['roi'],
        q=[0, 0.2, 0.4, 0.6, 0.8, 1],
        labels=['red', 'light_amber', 'amber', 'light_green', 'dark_green']
    )

    # Define color map
    color_map = {
        'dark_green': '#2e7d32',  # Dark green for top performers
        'light_green': '#4CAF50',  # Light green for second tier
        'amber': '#FF9800',       # Amber for middle performers#FFD700
        'light_amber': '#FFC107', # Light amber for low performers
        'red': '#F44336'          # Red for lowest performers
    }

    # Create interactive bar chart
    fig17 = px.bar(
        campaign_performance,
        x='campaign_id',
        y='roi',
        color='performance_tier',
        color_discrete_map=color_map,
        hover_data={
            'campaign_id': True,
            'sale_amount': ':.2f',
            'roi': ':.2f',
            'acquisition_cost': ':.2f',
            'performance_tier': False
        },
        labels={
            'sale_amount': 'Sales Amount ($)',
            'roi': 'Average ROI',
            'acquisition_cost': 'Total Acquisition Cost ($)'
        },
        title='Campaign Performance by ROI'
    )

    # Customize hover template
    fig17.update_traces(
        hovertemplate=(
            "<b>Campaign ID:</b> %{x}<br>"
            "<b>Sales:</b> $%{y:,.2f}<br>"
            "<b>ROI:</b> %{customdata[0]:.2f}<br>"
            "<b>Acquisition Cost:</b> $%{customdata[1]:,.2f}"
        ),
        marker_line_color='rgba(0,0,0,0.3)',
        marker_line_width=1
    )

    # Update layout
    fig17.update_layout(
        xaxis_title='Campaign ID',
        yaxis_title='ROI ($)',
        showlegend=False,
        hovermode='x unified',
        margin=margin,
        height=210,
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # Add interactivity
    fig17.update_layout(
        clickmode='event+select',
        xaxis={'categoryorder': 'total descending'}
    )

    # Add color scale explanatio

    charts.append(fig17)

    

    return charts  # ✅ Ensure this is properly indented inside the function
charts = create_charts(filtered_df)

# --- Layout Options ---
def grid_layout():
    """Compact 3x3 grid layout"""
    # KPI Row - more compact
    with st.container():
        cols = st.columns(3)
        with cols[0]:
            display_metric_container("Total Sales", f"${total_sales:,.2f}", small=True)
        with cols[1]:
            display_metric_container("Retention Rate", f"{retention_rate}%", small=True)
        with cols[2]:
            display_metric_container("Page Views", f"{page_views_total:,}", small=True)
    
    # Charts in compact grid
    with st.container():
        row1 = st.columns(3)
        for col, chart in zip(row1, charts[:3]):
            with col:
                st.plotly_chart(chart, use_container_width=True)
    
    with st.container():
        row2 = st.columns(3)
        for col, chart in zip(row2, charts[3:6]):
            with col:
                st.plotly_chart(chart, use_container_width=True)
    
    with st.container():
        row3 = st.columns(3)
        for col, chart in zip(row3, charts[6:9]):
            with col:
                st.plotly_chart(chart, use_container_width=True)

def two_column_layout():
    """Two-column layout with primary/secondary emphasis"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Main focus charts
        st.plotly_chart(charts[0], use_container_width=True)  # Sales trend
        st.plotly_chart(charts[4], use_container_width=True)  # Acquisition vs ROI
        st.plotly_chart(charts[2], use_container_width=True)  # Choropleth
    
    with col2:
        # Supporting metrics and smaller charts
        display_metric_container("Total Sales", f"${total_sales:,.2f}", full_width=True)
        display_metric_container("Retention Rate", f"{retention_rate}%", full_width=True)
        st.plotly_chart(charts[7], use_container_width=True)  # Donut
        st.plotly_chart(charts[5], use_container_width=True)  # Segments
        st.plotly_chart(charts[1], use_container_width=True)  # Treemap

def focus_supporting_layout():
    """One main focus chart with supporting charts"""
    # Main focus chart
    st.plotly_chart(charts[0], use_container_width=True)  # Sales trend
    
    # Supporting charts in 4 columns
    cols = st.columns(4)
    with cols[0]:
        st.plotly_chart(charts[1], use_container_width=True)  # Treemap
        st.plotly_chart(charts[7], use_container_width=True)  # Donut
    with cols[1]:
        st.plotly_chart(charts[3], use_container_width=True)  # ROI
        display_metric_container("Retention", f"{retention_rate}%", small=True)
    with cols[2]:
        st.plotly_chart(charts[5], use_container_width=True)  # Segments
        display_metric_container("Page Views", f"{page_views_total:,}", small=True)
    with cols[3]:
        st.plotly_chart(charts[6], use_container_width=True)  # Heatmap
        display_metric_container("Total Sales", f"${total_sales:,.2f}", small=True)

# --- Dashboard Tabs ---
# --- Dashboard Tabs ---
st.markdown("""
<style>
    /* Fix tab container spacing */
    div[data-testid="stTabs"] {
        margin-top: 1rem;
    }
    
    /* Make tabs more visible */
    button[data-baseweb="tab"] {
        padding: 8px 16px;
        margin: 0 4px;
        border-radius: 4px 4px 0 0;
    }
    
    /* Active tab styling */
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: white;
        border-bottom: 2px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Sales Dashboard", 
    "📢 Marketing Dashboard",
    "🌐 Website & Campaign Analytics",
    "📊 Insights",
    "📌 Dashboard Overview",
    "🔮 Predictions"
])

with tab1:  # Sales Dashboard
    

    # Add color key legend container
    with st.container():
        cols = st.columns([4, 1])  # 80% chart, 20% legend
        
        with cols[0]:
            st.plotly_chart(charts[0], use_container_width=True)  # Your existing chart
        
        with cols[1]:
            st.markdown("""
            <div style="background: #f8f9fa; padding: 12px; border-radius: 8px; 
                        border-left: 4px solid #f8f9fa; margin-bottom: 16px;">
                <div style="font-weight: bold; margin-bottom: 8px; font-size: 14px;">Dashboard
                    Performance Key
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                    <div style="width: 16px; height: 16px; background: #2e7d32; 
                                margin-right: 8px; border-radius: 2px;"></div>
                    <div style="font-size: 12px;">Top Performing</div>
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                    <div style="width: 16px; height: 16px; background: #4CAF50; 
                                margin-right: 8px; border-radius: 2px;"></div>
                    <div style="font-size: 12px;">High</div>
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                    <div style="width: 16px; height: 16px; background: #FF9800; 
                                margin-right: 8px; border-radius: 2px;"></div>
                    <div style="font-size: 12px;">Medium</div>
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                    <div style="width: 16px; height: 16px; background: #FFC107; 
                                margin-right: 8px; border-radius: 2px;"></div>
                    <div style="font-size: 12px;">Low Medium</div>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 16px; height: 16px; background: #F44336; 
                                margin-right: 8px; border-radius: 2px;"></div>
                    <div style="font-size: 12px;">Low</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with st.container():
    # Calculate metrics
        sales_target = 800000  # Set your sales target here
        sales_percentage = (total_sales / sales_target) * 100
        met_target = total_sales >= sales_target
        
        # Create the metric display
        cols = st.columns(1)
        with cols[0]:
            # Main metric container
            with st.container():
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; 
                                background: #f0f2f6; padding: 12px 16px; border-radius: 8px;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <div>
                            <div style="font-size: 14px; color: #555;">Total Sales</div>
                            <div style="font-size: 24px; font-weight: bold; color: #333;">
                                ${total_sales:,.2f}
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="display: flex; align-items: center; justify-content: flex-end;">
                                <span style="font-size: 14px; color: {'#4CAF50' if met_target else '#F44336'}; 
                                    margin-right: 4px;">
                                    {'▲' if met_target else '▼'} {abs(sales_percentage-100):.1f}%
                                </span>
                                <span style="font-size: 12px; color: #666;">
                                    vs target
                                </span>
                            </div>
                            <div style="font-size: 12px; color: {'#4CAF50' if met_target else '#F44336'}; 
                                background: {'#E8F5E9' if met_target else '#FFEBEE'}; 
                                padding: 2px 6px; border-radius: 4px; margin-top: 4px;">
                                {'Target met' if met_target else 'Target missed'}
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    chart_cols = st.columns(3)
    with chart_cols[0]:
        st.plotly_chart(charts[1], use_container_width=True)
    with chart_cols[1]:
        st.plotly_chart(charts[2], use_container_width=True)
    with chart_cols[2]:
        st.plotly_chart(charts[5], use_container_width=True)

with tab2:  # Marketing Dashboard
    # Define CSS for compact metric cards
    # Define CSS for the new layout
    st.markdown("""
        <style>
            .metric-card {
                background: #f8f9fa;
                padding: 10px;
                border-radius: 8px;
                margin-bottom: 10px;
            }
            .metric-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 4px;
            }
            .metric-title {
                font-size: 12px;
                color: #555;
            }
            .metric-target {
                font-size: 11px;
                color: #666;
            }
            .metric-value-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .metric-value {
                font-size: 20px;
                font-weight: bold;
                color: #333;
            }
            .metric-delta {
                font-size: 11px;
            }
            .metric-description {
                font-size: 10px;
                color: #2ecc71;  /* Green color */
                margin-left: 8px;
            }
        </style>
        """, unsafe_allow_html=True)

    with st.container():
        # Define targets
        RETENTION_TARGET = 60
        CONVERSION_TARGET = 0.10
        
        # Calculate metrics
        retention_rate = filtered_df['retained_customer'].mean() * 100
        conversion_rate = filtered_df['conversion_rate'].mean()
        total_campaigns = len(filtered_df['campaign_id'].unique())
        
        # Create three equal columns for metrics
        m1, m2, m3 = st.columns(3)
        
        # Retention Rate
        with m1:
            retention_met = retention_rate >= RETENTION_TARGET
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-header">
                    <div class="metric-title">Retention Rate</div>
                    <div class="metric-target">Target: {RETENTION_TARGET}%</div>
                </div>
                <div class="metric-value-row">
                    <div class="metric-value">{retention_rate:.1f}%</div>
                    <div class="metric-delta" style="color: {'#2ecc71' if retention_met else '#e74c3c'}">
                        {'✓' if retention_met else '✗'} {abs(retention_rate - RETENTION_TARGET):.1f}% {'above' if retention_met else 'below'} target
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Total Campaigns
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-header">
                    <div class="metric-title">Total Campaigns</div>
                </div>
                <div class="metric-value-row">
                    <div class="metric-value">{total_campaigns}</div>
                    <div class="metric-description">Active marketing</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Conversion Rate
        with m3:
            conversion_met = conversion_rate >= CONVERSION_TARGET
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-header">
                    <div class="metric-title">Conversion Rate</div>
                    <div class="metric-target">Target: {CONVERSION_TARGET:.0%}</div>
                </div>
                <div class="metric-value-row">
                    <div class="metric-value">{conversion_rate:.2%}</div>
                    <div class="metric-delta" style="color: {'#2ecc71' if conversion_met else '#e74c3c'}">
                        {'✓' if conversion_met else '✗'} {abs(conversion_rate - CONVERSION_TARGET):.2%} {'above' if conversion_met else 'below'} target
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Chart layout remains the same
    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([7, 3])  # 70%/30% split

    with col1:
        st.plotly_chart(charts[6], use_container_width=True)  # Main chart (heatmap)

    with col2:
        st.plotly_chart(charts[7], use_container_width=True)  # Top right chart
        st.plotly_chart(charts[16], use_container_width=True)  # Bottom right chart
            
                


with tab3:  # Website Analytics
    
    # --- Data Processing for Log Analysis ---
    if 'endpoint' in filtered_df.columns:
        # Safely extract path and parameters
        filtered_df['path'] = filtered_df['endpoint'].apply(
            lambda x: str(x).split('?')[0].lower() if pd.notnull(x) else 'unknown'
        )
        
        # Safely parse parameters (handle missing/empty cases)
        def safe_parse_qs(url):
            try:
                if pd.isnull(url) or '?' not in str(url):
                    return {}
                return parse_qs(str(url).split('?')[1])
            except:
                return {}
        
        filtered_df['params'] = filtered_df['endpoint'].apply(safe_parse_qs)
        filtered_df['params_str'] = filtered_df['params'].astype(str)  # Convert dicts to strings
        
        # Enhanced categorization
        def categorize_request(path):
            path = str(path).lower()
            if not path or path == 'unknown':
                return 'Other'
            if 'jobrequest' in path:
                return 'Job Request'
            elif 'event' in path:
                return 'Event'
            elif 'assistant' in path:
                return 'AI Assistant'
            elif 'campaign' in path:
                return 'Campaign'
            elif 'prototype' in path:
                return 'Prototype'
            else:
                return 'Other'
        
        filtered_df['request_type'] = filtered_df['path'].apply(categorize_request)
    
    # --- Key Metrics ---
    # First, you'll need to calculate previous period data (example - adjust based on your actual data)
    previous_period_df = df[(df['timestamp'] < pd.to_datetime(start_date))]  # Data before current filter period

    # Calculate comparison metrics
    cols = st.columns(4)
    with cols[0]:
        total_requests = len(filtered_df)
        prev_total = len(previous_period_df)
        
        st.metric(
            "Total Requests", 
            total_requests,
            delta_color="normal",
            help=f"Previous period: {prev_total}"
        )


    with cols[1]:
        current_jobs = len(filtered_df[filtered_df['request_type'] == 'Job Request']) if 'request_type' in filtered_df.columns else 0
        prev_jobs = len(previous_period_df[previous_period_df['request_type'] == 'Job Request']) if 'request_type' in previous_period_df.columns else 0
        delta_jobs = current_jobs - prev_jobs
        
        st.metric(
            "Job Requests", 
            current_jobs,
            help=f"Previous: {prev_jobs}"
        )
        st.progress(
            min(1.0, current_jobs/max(1, total_requests)), 
            text=f"{current_jobs/total_requests*100:.1f}% of total"
        )

    with cols[2]:
        current_events = len(filtered_df[filtered_df['request_type'] == 'Event']) if 'request_type' in filtered_df.columns else 0
        prev_events = len(previous_period_df[previous_period_df['request_type'] == 'Event']) if 'request_type' in previous_period_df.columns else 0
        delta_events = current_events - prev_events
        
        st.metric(
            "Event Requests", 
            current_events,
            delta=f"{delta_events}",
            delta_color="normal",
            help=f"Previous: 117,456"
        )

    with cols[3]:
        current_ai = len(filtered_df[filtered_df['request_type'] == 'AI Assistant']) if 'request_type' in filtered_df.columns else 0
        prev_ai = len(previous_period_df[previous_period_df['request_type'] == 'AI Assistant']) if 'request_type' in previous_period_df.columns else 0
        delta_ai = current_ai - prev_ai
        
        st.metric(
            "AI Assistant Queries", 
            current_ai,
            delta=f"{delta_ai}",
            delta_color="normal",
            help=f"Previous: 15009"
        )
        # Performance badge
        if delta_ai > (prev_ai * 0.1):  # More than 10% growth
            st.markdown("<span style='color:green;font-weight:bold'>Rapid Growth</span>", unsafe_allow_html=True)
        elif delta_ai > 0:
            st.markdown("<span style='color:green'>↑ Steady Growth</span>", unsafe_allow_html=True)
        elif delta_ai < 0:
            st.markdown("<span style='color:red'>↓ Declining</span>", unsafe_allow_html=True)
    
   # --- Main Analysis Sections ---
    tab1, tab2, tab3, subtab4 = st.tabs(["🌐 Website Request Patterns", "👔 Job Request Analysis", "🎟️ Event Request Analysis", "🤖 AI Assistant Usage"])

    # Chart height configuration
    CHART_HEIGHT = 300  # Set your default height here (in pixels)

    with tab1:  # Request Analysis
        # Row 1 - Always show these 3 charts
        cols = st.columns(2)
        with cols[0]:
            type_counts = filtered_df['request_type'].value_counts().reset_index()
            fig1 = px.pie(type_counts, names='request_type', values='count',
                        title='Request Type Distribution', hole=0.3)
            st.plotly_chart(fig1.update_layout(height=CHART_HEIGHT), 
                        use_container_width=True)
        
        
        # Your existing code with target line added
        TARGET = 12000  # Instead of the calculated value
        with cols[1]:
            filtered_df['hour'] = filtered_df['timestamp'].dt.hour
            hourly = filtered_df.groupby('hour').size().reset_index(name='count')
            
            # Calculate target (example: average + 20%)
            TARGET = hourly['count'].mean() * 1.2  
            
            fig3 = px.line(
                hourly, 
                x='hour', 
                y='count',
                title=f'Hourly Requests (Target: {TARGET:,.0f})',
                labels={'count': 'Requests', 'hour': 'Hour of Day'}
            )
            
            # Add target line and peak highlight
            fig3.add_hline(
                y=TARGET,
                line_dash="dot",
                line_color="red",
                annotation_text=f"Target: {TARGET:,.0f}",
                annotation_position="top right"
            )
            
            # Highlight hours exceeding target
            fig3.add_traces(
                px.scatter(
                    hourly[hourly['count'] >= TARGET],
                    x='hour',
                    y='count',
                    color_discrete_sequence=['limegreen']
                ).update_traces(
                    marker=dict(size=12, symbol='star'),
                    name='Above Target'
                ).data
            )
            
            # Formatting
            fig3.update_layout(
                height=CHART_HEIGHT,
                xaxis=dict(tickmode='linear', dtick=1),
                showlegend=False
            )
            
            st.plotly_chart(fig3, use_container_width=True)

    with tab2:  # Job Analytics
        if 'request_type' in filtered_df.columns and (filtered_df['request_type'] == 'Job Request').any():
            job_df = filtered_df[filtered_df['request_type'] == 'Job Request'].copy()

            # Extract job types from params
            job_types = []
            for params in job_df['params']:
                if isinstance(params, dict) and 'type' in params:
                    if isinstance(params['type'], list):
                        job_types.extend([t for t in params['type'] if pd.notnull(t)])
                    elif pd.notnull(params['type']):
                        job_types.append(params['type'])

            # Prepare daily job requests
            job_df['date'] = job_df['timestamp'].dt.date
            daily_jobs = job_df.groupby('date').size().reset_index(name='count')

            # Prepare status code counts
            status_counts = job_df['status_code'].value_counts().reset_index()
            status_counts.columns = ['status_code', 'count']

            cols = st.columns(2)

            with cols[0]:
                # Sunburst Chart for Job Types Hierarchy
                if job_types:
                    type_counts = pd.Series(job_types).value_counts().reset_index()
                    type_counts.columns = ['Job Type', 'Count']
                    fig4 = px.sunburst(type_counts, path=['Job Type'], values='Count',
                                    title='Job Type Hierarchy',
                                    color='Count', color_continuous_scale='thermal')
                else:
                    fig4 = px.pie(names=['No Data'], values=[1], title='No Job Types Found')

                st.plotly_chart(fig4.update_layout(height=CHART_HEIGHT, 
                                                margin=dict(t=40, b=20)), 
                                use_container_width=True)

           
            with cols[1]:
                # Bar chart for top status codes
                status_counts = job_df['status_code'].value_counts().reset_index()
                status_counts.columns = ['Status Code', 'Count']
                fig6 = px.bar(status_counts.head(10), x='Status Code', y='Count',
                            title='Top 10 Job Request Status Codes',
                            text='Count', color='Count',
                            color_continuous_scale='Blues')
                fig6.update_traces(textposition='outside')
                fig6.update_layout(xaxis_title='Status Code',
                                yaxis_title='Count',
                                height=CHART_HEIGHT)
                st.plotly_chart(fig6, use_container_width=True)



    with tab3:  # Event Analytics
        if 'request_type' in filtered_df.columns and (filtered_df['request_type'] == 'Event').any():
            event_df = filtered_df[filtered_df['request_type'] == 'Event']
            
            # Row 1 - Always show these 3 event-related charts
            cols = st.columns(2)
            
            with cols[0]:
                status_counts = event_df['status_code'].value_counts().reset_index()
                fig8 = px.pie(status_counts, names='status_code', values='count',
                            title='Event Status Codes')
                st.plotly_chart(fig8.update_layout(height=CHART_HEIGHT),
                            use_container_width=True)
            
            with cols[1]:
                event_df['hour'] = event_df['timestamp'].dt.hour
                hourly_events = event_df.groupby('hour').size().reset_index(name='count')
                fig9 = px.bar(hourly_events, x='hour', y='count',
                            title='Event Requests by Hour')
                st.plotly_chart(fig9.update_layout(height=CHART_HEIGHT),
                            use_container_width=True)
        
        else:
            fig = px.bar(x=['Event Requests'], y=[0], title='No Event Requests Found')
            st.plotly_chart(fig.update_layout(height=CHART_HEIGHT),
                        use_container_width=True)

    with subtab4:  # AI Assistant
        if 'request_type' in filtered_df.columns and (filtered_df['request_type'] == 'AI Assistant').any():
            ai_df = filtered_df[filtered_df['request_type'] == 'AI Assistant']
            
            # Row 1 - Always show these 3 AI-related charts
            cols = st.columns(3)
            with cols[0]:
                ai_df['date'] = ai_df['timestamp'].dt.date
                daily_ai = ai_df.groupby('date').size().reset_index(name='count')
                fig10 = px.area(daily_ai, x='date', y='count',
                            title='Daily AI Queries')
                st.plotly_chart(fig10.update_layout(height=CHART_HEIGHT),
                            use_container_width=True)
            
            with cols[1]:
                status_counts = ai_df['status_code'].value_counts().reset_index()
                fig11 = px.pie(status_counts, names='status_code', values='count',
                            title='AI Query Status Codes')
                st.plotly_chart(fig11.update_layout(height=CHART_HEIGHT),
                            use_container_width=True)
            
            with cols[2]:
                ai_df['status'] = ai_df['status_code'].apply(lambda x: 'Success' if x == 200 else 'Failure')
                status_counts = ai_df['status'].value_counts().reset_index()
                fig12 = px.bar(status_counts, x='status', y='count',
                            title='AI Query Success Rate')
                st.plotly_chart(fig12.update_layout(height=CHART_HEIGHT),
                            use_container_width=True)
        
        else:
            fig = px.bar(x=['AI Queries'], y=[0], title='No AI Queries Found')
            st.plotly_chart(fig.update_layout(height=CHART_HEIGHT),
                        use_container_width=True)

# This should be a separate tab, not part of tab3
with tab4:  # Descriptive Analytics
    st.header("📊 Descriptive Analytics")
    
    # Create two columns for layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Select only numeric columns
        numeric_cols = filtered_df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numeric_cols) > 0:
            # Calculate basic stats
            stats_df = filtered_df[numeric_cols].describe().T
            stats_df = stats_df[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
            stats_df.columns = ['Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max']
            
            # Format numbers
            st.dataframe(
                stats_df.style.format({
                    'Count': '{:,.0f}',
                    'Mean': '{:,.2f}',
                    'Std Dev': '{:,.2f}',
                    'Min': '{:,.2f}',
                    '25%': '{:,.2f}',
                    'Median': '{:,.2f}',
                    '75%': '{:,.2f}',
                    'Max': '{:,.2f}'
                }),
                height=400
            )
        else:
            st.warning("No numeric columns found for statistical analysis")

    with col2:
        # --- Enhanced Categorical Data Analysis ---
        cat_cols = filtered_df.select_dtypes(include=['object', 'category']).columns
        
        if len(cat_cols) > 0:
            with st.expander("View Categorical Distributions", expanded=False):
                cat_stats = []
                for col in cat_cols:
                    try:
                        # Safely convert to string and count unique values
                        unique_count = filtered_df[col].astype(str).nunique()
                        if unique_count < 20:
                            counts = filtered_df[col].astype(str).value_counts().reset_index()
                            counts.columns = ['Value', 'Count']
                            counts['Percentage'] = (counts['Count'] / counts['Count'].sum() * 100).round(1)
                            counts['Column'] = col
                            cat_stats.append(counts)
                    except Exception as e:
                        st.warning(f"Could not process column {col}: {str(e)}")
                        continue
                
                if cat_stats:
                    cat_stats_df = pd.concat(cat_stats)
                    
                    # Limit to top 5 columns if many
                    display_cols = cat_stats_df['Column'].unique()[:5]
                    tabs = st.tabs([str(col) for col in display_cols])
                    
                    for tab, col in zip(tabs, display_cols):
                        with tab:
                            col_data = cat_stats_df[cat_stats_df['Column'] == col]
                            
                            # Show top values as progress bars
                            for _, row in col_data.head(5).iterrows():
                                st.progress(row['Percentage']/100, 
                                          text=f"{row['Value']}: {row['Count']} ({row['Percentage']}%)")
                            
                            # Show full table
                            st.dataframe(
                                col_data[['Value', 'Count', 'Percentage']]
                                .sort_values('Count', ascending=False)
                                .style.format({'Percentage': '{:.1f}%'}),
                                height=min(200, 35 * len(col_data)),
                                use_container_width=True
                            )
                else:
                    st.info("No categorical columns with <20 unique values found")
        else:
            st.warning("No categorical columns found")
        
        # --- Enhanced Distribution Visualizations ---
        with st.expander("📊 Distribution Visualizations", expanded=False):
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Select column:", numeric_cols, key="dist_col")
                
                # Dynamic visualization options
                viz_options = ["Histogram", "Box Plot"]
                if 'timestamp' in filtered_df.columns:
                    viz_options.append("Time Trend")
                viz_options.append("Violin Plot")
                
                plot_type = st.radio("Visualization type:", viz_options, key="plot_type")
                
                # Create the selected visualization
                try:
                    if plot_type == "Histogram":
                        fig = px.histogram(filtered_df, x=selected_col, nbins=20,
                                         title=f"Distribution of {selected_col}")
                    elif plot_type == "Box Plot":
                        fig = px.box(filtered_df, y=selected_col,
                                   title=f"Box Plot of {selected_col}")
                    elif plot_type == "Time Trend":
                        time_df = filtered_df.set_index('timestamp').resample('D')[selected_col].mean().reset_index()
                        fig = px.line(time_df, x='timestamp', y=selected_col,
                                    title=f"Daily Trend of {selected_col}")
                    else:
                        fig = px.violin(filtered_df, y=selected_col,
                                      title=f"Violin Plot of {selected_col}")
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not create visualization: {str(e)}")
            else:
                st.warning("No numeric columns available")
                

with tab5:  # Dashboard Overview
    with st.container():
        cols = st.columns(4)
        
        # Calculate metrics
        sales_target = 800000  # Set your sales target
        retention_target = 75  # Set your retention target
        prev_page_views = 50000  # Previous period page views for comparison
        
        sales_percentage = (total_sales / sales_target) * 100
        sales_met = total_sales >= sales_target
        retention_met = retention_rate >= retention_target
        page_views_change = ((page_views_total - prev_page_views) / prev_page_views) * 100
        page_views_up = page_views_change >= 0
        
        with cols[0]:  # Total Sales
            st.markdown(f"""
            <div style="background:#f8f9fa;padding:12px;border-radius:8px;margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:12px;color:#555;">Total Sales</span>
                    <span style="font-size:11px;color:#666;">Target: ${sales_target:,.0f}</span>
                </div>
                <div style="display:flex;justify-content:space-between;align-items:center;margin-top:4px;">
                    <span style="font-size:18px;font-weight:bold;">${total_sales:,.2f}</span>
                    <span style="font-size:11px;color:{'#4CAF50' if sales_met else '#F44336'}">
                        {'▲' if sales_met else '▼'} {abs(sales_percentage-100):.1f}%
                    </span>
                </div>
                <div style="font-size:11px;color:{'#4CAF50' if sales_met else '#F44336'};text-align:right;">
                    {'Target met' if sales_met else 'Target missed'}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[1]:  # Retention Rate
            st.markdown(f"""
            <div style="background:#f8f9fa;padding:12px;border-radius:8px;margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:12px;color:#555;">Retention Rate</span>
                    <span style="font-size:11px;color:#666;">Target: {retention_target}%</span>
                </div>
                <div style="display:flex;justify-content:space-between;align-items:center;margin-top:4px;">
                    <span style="font-size:18px;font-weight:bold;">{retention_rate}%</span>
                    <span style="font-size:11px;color:{'#4CAF50' if retention_met else '#F44336'}">
                        {'▲' if retention_met else '▼'} {abs(retention_rate-retention_target):.1f}%
                    </span>
                </div>
                <div style="font-size:11px;color:{'#4CAF50' if retention_met else '#F44336'};text-align:right;">
                    {'Target met' if retention_met else 'Target missed'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with cols[2]:  # Page Views
            st.markdown(f"""
            <div style="background:#f8f9fa;padding:12px;border-radius:8px;margin-bottom:10px;">
                <div style="font-size:12px;color:#555;">Page Views</div>
                <div style="display:flex;justify-content:space-between;align-items:center;margin:4px 0;">
                    <span style="font-size:18px;font-weight:bold;">{page_views_total:,}</span>
                    <span style="font-size:11px;color:#4CAF50;">
                    </span>
                </div>
                <div style="font-size:11px;color:{'#4CAF50' if page_views_up else '#F44336'};">
                    {'▲' if page_views_up else '▼'} {abs(page_views_change):.1f}% from last period
                </div>
            </div>
            """, unsafe_allow_html=True)

        
        with cols[3]:  # Avg ROI
            st.markdown(f"""
            <div style="background:#f8f9fa;padding:12px;border-radius:8px;margin-bottom:10px;">
                <div style="font-size:12px;color:#555;">Avg ROI</div>
                <div style="font-size:18px;font-weight:bold;margin:4px 0;">{filtered_df['roi'].mean():.2f}</div>
                <div style="font-size:11px;color:#666;">Per campaign average</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Create a 2-column layout
    col1, col2 = st.columns([0.7, 1])  # Left column wider than right
    
    with col1:
        # Display the funnel chart (chart 12) full height on left
        if len(charts) > 9:
            st.plotly_chart(charts[9], use_container_width=True, height=500)
    
    with col2:
        # Create a 2x2 grid for the other charts
        grid = st.columns(2)  # Creates 2 columns
        
        # First row
        with grid[0]:
            if len(charts) > 8:
                st.plotly_chart(charts[8], use_container_width=True)  # Choropleth
            if len(charts) > 15:
                st.plotly_chart(charts[15], use_container_width=True)  # Sunburst
        
        # Second row
        with grid[1]:
            if len(charts) > 10:
                st.plotly_chart(charts[10], use_container_width=True)  # Gauge
            if len(charts) > 9:
                st.plotly_chart(charts[11], use_container_width=True)  # Radar
print(sys.executable)



with tab6:  # Predictions Tab
    st.header("Sales Prediction")
    st.markdown("Use the form below to predict sales based on campaign and website metrics.")

    # Load the model
    try:
        models = load_models()
        if models is None:
            st.error("Prediction functionality is unavailable due to model loading issues.")
        else:
            sale_model = models['sale_model']

            # Create input form
            with st.form(key="prediction_form"):
                st.markdown('<div class="form-container">', unsafe_allow_html=True)

                # Input fields
                col1, col2 = st.columns(2)
                with col1:
                    page_views = st.number_input(
                        "Page Views",
                        min_value=0,
                        value=15,
                        step=100,
                        help="Enter the number of page views for the campaign."
                             '<span class="help-tooltip">?'
                             '<span class="tooltip-text">'
                             'The total number of page views for the campaign or website. '
                             'Higher page views typically indicate greater user engagement.'
                             '</span></span>',
                        format="%d"
                    )
                    acquisition_cost = st.number_input(
                        "Acquisition Cost ($)",
                        min_value=0.0,
                        value=50.0,
                        step=100.0,
                        help="Enter the total acquisition cost for the campaign."
                             '<span class="help-tooltip">?'
                             '<span class="tooltip-text">'
                             'The total cost spent on acquiring traffic, such as ad spend. '
                             'Used to calculate ROI.'
                             '</span></span>',
                        format="%.2f"
                    )
                    session_duration_sec = st.number_input(
                        "Session Duration (seconds)",
                        min_value=0,
                        value=300,
                        step=10,
                        help="Enter the average session duration in seconds."
                             '<span class="help-tooltip">?'
                             '<span class="tooltip-text">'
                             'The average time users spend on the website per session. '
                             'Longer durations may indicate higher engagement.'
                             '</span></span>',
                        format="%d"
                    )

                with col2:
                    region = st.selectbox(
                        "Region",
                        options=['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America'],
                        help="Select the region for the campaign."
                             '<span class="help-tooltip">?'
                             '<span class="tooltip-text">'
                             'The geographical region where the campaign is targeted. '
                             'This affects sales performance.'
                             '</span></span>'
                    )
                    device = st.selectbox(
                        "Device",
                        options=['Desktop', 'Mobile', 'Tablet', 'Other'],
                        help="Select the device type used for the campaign."
                             '<span class="help-tooltip">?'
                             '<span class="tooltip-text">'
                             'The type of device users access the campaign from. '
                             'Different devices may influence user behavior.'
                             '</span></span>'
                    )
                    product = st.selectbox(
                        "Product",
                        options=['AI Assistant', 'Custom Package', 'Demo Booking', 'Event Pass', 'Prototyping Suite'],
                        help="Select the product associated with the campaign."
                             '<span class="help-tooltip">?'
                             '<span class="tooltip-text">'
                             'The product being marketed in the campaign. '
                             'Different products may have varying sales outcomes.'
                             '</span></span>'
                    )

                # Submit button
                submit_button = st.form_submit_button("Predict Sales")
                st.markdown('</div>', unsafe_allow_html=True)

            # Process prediction and display line graph
            if submit_button:
                try:
                    # Create raw input DataFrame for the model pipeline
                    raw_input = pd.DataFrame({
                        'page_views': [page_views],
                        'acquisition_cost': [acquisition_cost],
                        'session_duration_sec': [session_duration_sec],
                        'roi': [page_views / (acquisition_cost + 1e-5)],
                        'region': [region],
                        'device': [device],
                        'product': [product]
                    })

                    # Generate predictions for future months (2 years from last month)
                    last_month = pd.to_datetime(filtered_df['month'].max(), format='%Y-%m')
                    future_dates = pd.date_range(
                        start=last_month + pd.offsets.MonthEnd(1) + pd.offsets.MonthBegin(1),
                        periods=24,  # 2 years
                        freq='MS'
                    )
                    # Apply 10% monthly increase to page_views and 2% to session_duration_sec
                    future_inputs = pd.DataFrame({
                        'page_views': [page_views * (1.10 ** i) for i in range(len(future_dates))],
                        'acquisition_cost': [acquisition_cost] * len(future_dates),
                        'session_duration_sec': [session_duration_sec * (1.02 ** i) for i in range(len(future_dates))],
                        'region': [region] * len(future_dates),
                        'device': [device] * len(future_dates),
                        'product': [product] * len(future_dates)
                    })
                    future_inputs['roi'] = future_inputs['page_views'] / (future_inputs['acquisition_cost'] + 1e-5)

                    # Make predictions for each future month
                    predictions = sale_model.predict(future_inputs)
                    # Align predictions with last actual sales and apply 3% monthly growth
                    last_actual_sales = filtered_df.groupby('month')['sale_amount'].sum().max() if 'sale_amount' in filtered_df.columns else 800000
                    first_pred = predictions[0] if predictions[0] > 0 else last_actual_sales
                    scale_factor = last_actual_sales / first_pred if first_pred != 0 else 1.0
                    adjusted_predictions = [max(pred * scale_factor * (1.03 ** i), 0) for i, pred in enumerate(predictions)]

                    # Prepare data for line graph
                    if 'month' not in filtered_df.columns:
                        st.error("The 'month' column is missing in the filtered dataset.")
                    else:
                        monthly_sales = filtered_df.groupby('month')['sale_amount'].sum().reset_index()
                        try:
                            monthly_sales['month'] = pd.to_datetime(monthly_sales['month'], format='%Y-%m')
                        except Exception as e:
                            st.error(f"Failed to parse 'month' column: {str(e)}")
                            raise

                        # Create predicted sales DataFrame
                        pred_df = pd.DataFrame({
                            'month': future_dates,
                            'sale_amount': adjusted_predictions,
                            'type': ['Predicted'] * len(future_dates)
                        })
                        monthly_sales['type'] = 'Actual'
                        combined_df = pd.concat([monthly_sales, pred_df], ignore_index=True)

                        # Create line graph
                        fig = go.Figure()

                        # Actual sales (solid line with circles)
                        actual_data = combined_df[combined_df['type'] == 'Actual']
                        fig.add_trace(
                            go.Scatter(
                                x=actual_data['month'],
                                y=actual_data['sale_amount'],
                                mode='lines+markers',
                                name='Actual Sales',
                                line=dict(color='#4CAF50', width=2),
                                marker=dict(size=8, symbol='circle'),
                                hovertemplate='<b>Month</b>: %{x|%Y-%m}<br><b>Sales</b>: $%{y:,.2f}<extra></extra>'
                            )
                        )

                        # Predicted sales (dashed line with circles)
                        pred_data = combined_df[combined_df['type'] == 'Predicted']
                        last_actual = actual_data.iloc[-1]
                        fig.add_trace(
                            go.Scatter(
                                x=[last_actual['month']] + list(pred_data['month']),
                                y=[last_actual['sale_amount']] + list(pred_data['sale_amount']),
                                mode='lines+markers',
                                name='Predicted Sales',
                                line=dict(color='#F44336', width=2, dash='dash'),
                                marker=dict(size=8, symbol='circle', color='#F44336'),
                                hovertemplate='<b>Month</b>: %{x|%Y-%m}<br><b>Predicted Sales</b>: $%{y:,.2f}<extra></extra>'
                            )
                        )

                        # Add target line
                        sales_target = 800000
                        fig.add_hline(
                            y=sales_target,
                            line_dash="dot",
                            line_color="black",
                            annotation_text=f"Target: ${sales_target:,.0f}",
                            annotation_position="top left"
                        )

                        # Update layout
                        fig.update_layout(
                            title='Sales Trend with Prediction (2 Years)',
                            xaxis_title='Month',
                            yaxis_title='Sales Amount ($)',
                            height=400,
                            margin=dict(l=20, r=20, t=60, b=20),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ),
                            hovermode='x unified'
                        )

                        # Display results
                        with st.container():
                            st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
                            st.subheader("Prediction Results")
                            st.plotly_chart(fig, use_container_width=True)

                            # Insights
                            st.markdown("### Insights")
                            avg_sales = filtered_df['sale_amount'].mean() if 'sale_amount' in filtered_df.columns else 0
                            avg_pred = pred_df['sale_amount'].mean()
                            if avg_sales > 0:
                                if avg_pred > avg_sales:
                                    st.success(f"The average predicted sales (${avg_pred:,.2f}) are above the historical average (${avg_sales:,.2f})!")
                                else:
                                    st.warning(f"The average predicted sales (${avg_pred:,.2f}) are below the historical average (${avg_sales:,.2f}). Consider optimizing campaign parameters.")
                            if avg_pred >= sales_target:
                                st.success(f"The average predicted sales meet or exceed the target of ${sales_target:,.0f}!")
                            else:
                                st.warning(f"The average predicted sales are below the target of ${sales_target:,.0f}.")

                            # Input explanation
                            with st.expander("View Input Parameters"):
                                st.markdown("""
                                    **Input Parameters Explained:**
                                    - **Page Views**: Number of views for the campaign or website, indicating user engagement. Predictions assume a 10% monthly increase.
                                    - **Acquisition Cost**: Total cost to acquire traffic (e.g., ad spend), used to calculate ROI.
                                    - **Session Duration**: Average time users spend on the website per session (in seconds), with a 2% monthly increase assumed.
                                    - **ROI**: Calculated as page views divided by acquisition cost, reflecting campaign efficiency.
                                    - **Region**: The geographical region targeted by the campaign, affecting sales performance.
                                    - **Device**: The type of device used (e.g., Desktop, Mobile), influencing user behavior.
                                    - **Product**: The product marketed, impacting sales outcomes.
                                """)
                                st.write(f"**Selected Inputs:**")
                                st.write(f"Page Views: {page_views}")
                                st.write(f"Acquisition Cost: ${acquisition_cost:,.2f}")
                                st.write(f"Session Duration: {session_duration_sec} seconds")
                                st.write(f"ROI: {raw_input['roi'].iloc[0]:,.2f}")
                                st.write(f"Region: {region}")
                                st.write(f"Device: {device}")
                                st.write(f"Product: {product}")

                            st.markdown('</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
                    st.markdown('<div class="error-message">Please check your inputs and try again.</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Failed to load model or initialize form: {str(e)}")

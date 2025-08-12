import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
import io
import zipfile
from pathlib import Path
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="SEO Traffic Projection by Aayush",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1.5rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        letter-spacing: -0.5px;
    }
    
    /* Subtitle styling */
    .subtitle {
        font-size: 1.1rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #34495e;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    
    /* Metric containers */
    .metric-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.2rem;
        border-radius: 8px;
        margin: 0.8rem 0;
        border-left: 4px solid #3498db;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Success message styling */
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #28a745;
        border-radius: 6px;
        padding: 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Warning message styling */
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #ffc107;
        border-radius: 6px;
        padding: 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Error message styling */
    .error-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 1px solid #dc3545;
        border-radius: 6px;
        padding: 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Info message styling */
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 1px solid #17a2b8;
        border-radius: 6px;
        padding: 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2980b9 0%, #1f5f8b 100%);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        transform: translateY(-1px);
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        border: 2px dashed #bdc3c7;
        border-radius: 8px;
        padding: 2rem;
        background-color: #f8f9fa;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #3498db;
        background-color: #ecf0f1;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-collapse: collapse;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Divider styling */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #3498db 50%, transparent 100%);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions from the original script
def month_floor(d: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=d.year, month=d.month, day=1)

def baseline_from_gsc(gsc_df: pd.DataFrame, months: int = 3) -> float:
    if gsc_df.empty:
        return 0.0
    gsc_df = gsc_df.sort_values("date")
    last_date = gsc_df["date"].max()
    last_month_start = month_floor(last_date)
    start_month = last_month_start - pd.DateOffset(months=months-1)
    mask = (gsc_df["date"] >= start_month) & (gsc_df["date"] < last_month_start + pd.offsets.MonthEnd(1))
    window = gsc_df.loc[mask]
    if window.empty:
        window = gsc_df[gsc_df["date"] >= (last_date - pd.Timedelta(days=90))]
    clicks = window.groupby(window["date"].dt.to_period("M"))["clicks"].sum().astype(float)
    if clicks.empty:
        return float(window["clicks"].mean()) if "clicks" in window else 0.0
    return float(clicks.mean())

def compute_projection(baseline_clicks: float, kw_df: pd.DataFrame,
                       ctr_curves: dict, ramp_curve: dict) -> pd.DataFrame:
    required_cols = ["keyword", "target_rank", "serp_share", "seasonality", "search_volume"]
    for c in required_cols:
        if c not in kw_df.columns:
            raise ValueError(f"Keywords DataFrame missing required column: {c}")
    
    results = []
    for scenario, ctr_map in ctr_curves.items():
        ss_clicks = kw_df.apply(
            lambda r: float(r["search_volume"]) * (ctr_map.get(int(r["target_rank"]), 0)/100.0) * float(r["serp_share"]) * float(r["seasonality"]),
            axis=1
        )
        kw_df[f"steady_clicks_{scenario}"] = ss_clicks

        for month, pct in ramp_curve.items():
            new_clicks = ss_clicks.sum() * (pct/100.0)
            total = baseline_clicks + new_clicks
            inc_pct = 0.0 if baseline_clicks == 0 else (total - baseline_clicks)/baseline_clicks * 100.0
            results.append({
                "Scenario": scenario,
                "Month": month,
                "New Clicks": round(new_clicks, 0),
                "Total Clicks": round(total, 0),
                "% Increase": round(inc_pct, 1)
            })
    return pd.DataFrame(results)

def create_excel_output(projection_df: pd.DataFrame, baseline_clicks: float, kw_df: pd.DataFrame, baseline_months: int) -> bytes:
    """Create Excel file in memory and return bytes"""
    output = io.BytesIO()
    
    try:
        # Try xlsxwriter first, then fall back to openpyxl
        engine = "xlsxwriter"
        try:
            import xlsxwriter
        except ImportError:
            engine = "openpyxl"
            
        with pd.ExcelWriter(output, engine=engine) as writer:
            kw_df.to_excel(writer, index=False, sheet_name="Keywords_Input")
            projection_df.to_excel(writer, index=False, sheet_name="Projection")
            pd.DataFrame({
                "Setting": ["Baseline Monthly Clicks", "Baseline Months Used"],
                "Value": [baseline_clicks, baseline_months]
            }).to_excel(writer, index=False, sheet_name="Settings")
        
        return output.getvalue()
    except ImportError:
        return None

# Main app
def main():
    st.markdown('<h1 class="main-header">SEO Traffic Projection by Aayush</h1>', unsafe_allow_html=True)
    
    st.markdown('<p class="subtitle">Professional SEO traffic forecasting based on keyword rankings and historical performance data</p>', unsafe_allow_html=True)

    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Baseline months
    baseline_months = st.sidebar.slider(
        "Baseline Months", 
        min_value=1, 
        max_value=12, 
        value=3, 
        help="Number of months to average for baseline traffic"
    )
    
    # CTR Curves
    st.sidebar.subheader("CTR Curves by Position")
    
    # Default CTR curves
    default_ctr = {
        "Conservative": {1: 20, 2: 12, 3: 8, 4: 6, 5: 5, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3},
        "Realistic":    {1: 28, 2: 15, 3: 11, 4: 8, 5: 8, 6: 4, 7: 4, 8: 4, 9: 4, 10: 4},
        "Optimistic":   {1: 32, 2: 18, 3: 14, 4: 10, 5: 9, 6: 5, 7: 5, 8: 5, 9: 5, 10: 5}
    }
    
    # Allow users to modify CTR for position 1
    pos1_conservative = st.sidebar.number_input("Position 1 - Conservative CTR (%)", value=20, min_value=1, max_value=50)
    pos1_realistic = st.sidebar.number_input("Position 1 - Realistic CTR (%)", value=28, min_value=1, max_value=50)
    pos1_optimistic = st.sidebar.number_input("Position 1 - Optimistic CTR (%)", value=32, min_value=1, max_value=50)
    
    # Update CTR curves
    ctr_curves = default_ctr.copy()
    ctr_curves["Conservative"][1] = pos1_conservative
    ctr_curves["Realistic"][1] = pos1_realistic
    ctr_curves["Optimistic"][1] = pos1_optimistic
    
    # Ramp curve
    st.sidebar.subheader("Ramp Curve")
    ramp_curve = {}
    for month in range(1, 7):
        default_values = {1: 25, 2: 50, 3: 75, 4: 90, 5: 100, 6: 100}
        ramp_curve[month] = st.sidebar.slider(
            f"Month {month} (%)", 
            min_value=0, 
            max_value=100, 
            value=default_values.get(month, 100)
        )

    # Main content area
    col1, col2 = st.columns(2)
    
    # Google Search Console Data Upload
    with col1:
        st.markdown('<h3 class="section-header">Google Search Console Data</h3>', unsafe_allow_html=True)
        gsc_file = st.file_uploader(
            "Upload GSC CSV file",
            type=['csv'],
            help="CSV should contain columns: date, clicks (optional: impressions)",
            key="gsc_upload"
        )
        
        if gsc_file is not None:
            try:
                gsc_df = pd.read_csv(gsc_file)
                gsc_df["date"] = pd.to_datetime(gsc_df["date"])
                
                st.markdown(f'<div class="success-box">Successfully loaded {len(gsc_df)} rows of GSC data</div>', unsafe_allow_html=True)
                
                # Show data preview
                if st.checkbox("Show GSC data preview", key="gsc_preview"):
                    st.dataframe(gsc_df.head(10))
                
                # Calculate baseline
                baseline = baseline_from_gsc(gsc_df, months=baseline_months)
                st.metric("Baseline Monthly Clicks", f"{baseline:,.0f}")
                
            except Exception as e:
                st.markdown(f'<div class="error-box">Error reading GSC file: {str(e)}</div>', unsafe_allow_html=True)
                gsc_df = None
                baseline = 0
        else:
            gsc_df = None
            baseline = 0
    
    # Keywords Data Upload
    with col2:
        st.markdown('<h3 class="section-header">Keywords & Target Rankings</h3>', unsafe_allow_html=True)
        keywords_file = st.file_uploader(
            "Upload Keywords CSV file",
            type=['csv'],
            help="CSV should contain columns: keyword, target_rank, serp_share, seasonality, search_volume",
            key="keywords_upload"
        )
        
        if keywords_file is not None:
            try:
                kw_df = pd.read_csv(keywords_file)
                # Normalise column names
                kw_df.columns = [c.strip().lower().replace(" ", "_") for c in kw_df.columns]
                
                # Validate required columns
                required_cols = ["keyword", "target_rank", "serp_share", "seasonality", "search_volume"]
                missing_cols = [c for c in required_cols if c not in kw_df.columns]
                
                if missing_cols:
                    st.markdown(f'<div class="error-box">Missing required columns: {", ".join(missing_cols)}</div>', unsafe_allow_html=True)
                    kw_df = None
                else:
                    st.markdown(f'<div class="success-box">Successfully loaded {len(kw_df)} keywords</div>', unsafe_allow_html=True)
                    
                    # Show data preview
                    if st.checkbox("Show Keywords data preview", key="kw_preview"):
                        st.dataframe(kw_df.head(10))
                    
                    # Show summary stats
                    col2_1, col2_2 = st.columns(2)
                    with col2_1:
                        st.metric("Total Search Volume", f"{kw_df['search_volume'].sum():,.0f}")
                    with col2_2:
                        st.metric("Average Target Rank", f"{kw_df['target_rank'].mean():.1f}")
                        
            except Exception as e:
                st.markdown(f'<div class="error-box">Error reading Keywords file: {str(e)}</div>', unsafe_allow_html=True)
                kw_df = None
        else:
            kw_df = None
    
    # Sample data option
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    if st.checkbox("Use Sample Data (for testing)", key="sample_data"):
        # Generate sample GSC data
        dates = pd.date_range(end=pd.Timestamp.today(), periods=180, freq="D")
        clicks = (np.sin(np.linspace(0, 6*np.pi, len(dates)))+2.5)*120 + np.random.normal(0, 30, len(dates)) + 8000/180
        gsc_df = pd.DataFrame({"date": dates, "clicks": np.maximum(0, clicks).round(0)})
        baseline = baseline_from_gsc(gsc_df, months=baseline_months)
        
        # Generate sample keywords data
        kw_df = pd.DataFrame({
            "keyword": ["seo tools australia", "digital marketing melbourne", "content marketing sydney"],
            "target_rank": [3, 5, 2],
            "serp_share": [0.9, 0.85, 0.8],
            "seasonality": [1.0, 1.0, 1.1],
            "search_volume": [2000, 1500, 4000]
        })
        
        st.markdown('<div class="info-box">Using sample data for demonstration purposes</div>', unsafe_allow_html=True)
    
    # Run analysis
    if st.button("Run Traffic Projection Analysis", type="primary"):
        if gsc_df is not None and kw_df is not None and baseline > 0:
            try:
                # Compute projections
                projection_df = compute_projection(
                    baseline_clicks=float(baseline), 
                    kw_df=kw_df.copy(),
                    ctr_curves=ctr_curves, 
                    ramp_curve=ramp_curve
                )
                
                # Display results
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown('<h3 class="section-header">Projection Results</h3>', unsafe_allow_html=True)
                
                # Summary metrics
                scenarios = projection_df['Scenario'].unique()
                col_metrics = st.columns(len(scenarios))
                
                for i, scenario in enumerate(scenarios):
                    with col_metrics[i]:
                        month_6_data = projection_df[(projection_df['Scenario'] == scenario) & (projection_df['Month'] == 6)]
                        if not month_6_data.empty:
                            final_clicks = month_6_data['Total Clicks'].iloc[0]
                            increase_pct = month_6_data['% Increase'].iloc[0]
                            st.metric(
                                f"{scenario} (Month 6)",
                                f"{final_clicks:,.0f} clicks",
                                f"+{increase_pct:.1f}%"
                            )
                
                # Interactive chart
                st.markdown('<h3 class="section-header">Traffic Growth Over Time</h3>', unsafe_allow_html=True)
                
                # Debug info (can be removed in production)
                if st.checkbox("Show chart debug info", key="debug_chart"):
                    st.write("**Chart Values Debug:**")
                    st.write(f"Baseline: {baseline:.0f} clicks")
                    st.write("Sample projection values:")
                    for scenario in scenarios[:1]:  # Show first scenario only
                        scenario_data = projection_df[projection_df['Scenario'] == scenario].head(3)
                        st.write(f"{scenario}: {scenario_data[['Month', 'Total Clicks']].to_dict('records')}")
                
                # Create plotly chart
                fig = go.Figure()
                
                for scenario in scenarios:
                    scenario_data = projection_df[projection_df['Scenario'] == scenario]
                    fig.add_trace(go.Scatter(
                        x=scenario_data['Month'],
                        y=scenario_data['Total Clicks'],
                        mode='lines+markers',
                        name=scenario,
                        line=dict(width=3),
                        marker=dict(size=8),
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                    'Month: %{x}<br>' +
                                    'Total Clicks: %{y:.0f}<br>' +
                                    '<extra></extra>'
                    ))
                
                # Add baseline line
                fig.add_hline(
                    y=baseline, 
                    line_dash="dash", 
                    line_color="gray",
                    annotation_text=f"Baseline: {baseline:.0f} clicks",
                    annotation_position="top right"
                )
                
                fig.update_layout(
                    title="Projected Organic Traffic Over 6 Months",
                    xaxis_title="Month",
                    yaxis_title="Total Clicks",
                    height=500,
                    template="plotly_white",
                    yaxis=dict(
                        tickformat=".0f",  # Show as whole numbers without thousands separators
                        tickmode="auto",
                        nticks=10,
                        separatethousands=False  # Disable thousands separators
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Data table
                st.markdown('<h3 class="section-header">Detailed Projections</h3>', unsafe_allow_html=True)
                
                # Pivot table for better display
                pivot_df = projection_df.pivot(index="Month", columns="Scenario", values=["Total Clicks", "% Increase"])
                
                st.dataframe(pivot_df, use_container_width=True)
                
                # Keywords breakdown
                st.markdown('<h3 class="section-header">Keywords Analysis</h3>', unsafe_allow_html=True)
                
                # Show keywords with steady state clicks for each scenario
                kw_display = kw_df.copy()
                for scenario in scenarios:
                    if f"steady_clicks_{scenario}" in kw_display.columns:
                        kw_display[f"{scenario}_steady_clicks"] = kw_display[f"steady_clicks_{scenario}"].round(0)
                
                # Remove the temporary columns and display
                display_cols = ["keyword", "target_rank", "search_volume"] + [f"{s}_steady_clicks" for s in scenarios if f"steady_clicks_{s}" in kw_df.columns]
                if all(col in kw_display.columns for col in display_cols):
                    st.dataframe(kw_display[display_cols], use_container_width=True)
                
                # Download section
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown('<h3 class="section-header">Download Results</h3>', unsafe_allow_html=True)
                
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    # Excel download
                    excel_data = create_excel_output(projection_df, baseline, kw_df, baseline_months)
                    if excel_data:
                        st.download_button(
                            label="Download Excel Report",
                            data=excel_data,
                            file_name="traffic_projection_report.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.markdown('<div class="warning-box">Excel libraries not available. Install xlsxwriter or openpyxl for Excel export.</div>', unsafe_allow_html=True)
                
                with col_dl2:
                    # CSV download
                    csv_data = projection_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV Data",
                        data=csv_data,
                        file_name="traffic_projection_data.csv",
                        mime="text/csv"
                    )
                
                st.markdown('<div class="success-box">Analysis completed successfully!</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f'<div class="error-box">Error running analysis: {str(e)}</div>', unsafe_allow_html=True)
                st.markdown('<div class="error-box">Please check your data format and try again.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">Please upload both GSC data and Keywords data to run the analysis.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

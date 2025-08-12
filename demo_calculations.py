#!/usr/bin/env python3
"""
SEO Traffic Projection Demo
Demonstrates the mathematical calculations with sample data
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta

# Import our functions
from streamlit_app import month_floor, baseline_from_gsc, compute_projection

def create_sample_data():
    """Create realistic sample data for demonstration"""
    
    # Sample GSC data - 6 months of daily clicks
    dates = pd.date_range(end=pd.Timestamp.today(), periods=180, freq="D")
    # Create realistic click pattern with weekly seasonality and growth trend
    base_clicks = 80
    weekly_pattern = np.sin(np.linspace(0, 26*np.pi, len(dates))) * 15  # Weekly variation
    growth_trend = np.linspace(0, 20, len(dates))  # Gradual growth
    noise = np.random.normal(0, 10, len(dates))
    clicks = base_clicks + weekly_pattern + growth_trend + noise
    clicks = np.maximum(0, clicks).round(0)
    
    gsc_df = pd.DataFrame({
        "date": dates,
        "clicks": clicks
    })
    
    # Sample keywords data
    kw_df = pd.DataFrame({
        "keyword": [
            "melbourne seo consultant",
            "digital marketing agency sydney", 
            "content marketing services",
            "local seo specialist",
            "ppc management brisbane"
        ],
        "target_rank": [3, 5, 2, 4, 6],
        "serp_share": [0.9, 0.85, 0.95, 0.8, 0.75],
        "seasonality": [1.0, 1.0, 1.1, 1.0, 0.9],
        "search_volume": [2400, 1800, 3200, 1500, 980]
    })
    
    return gsc_df, kw_df

def demonstrate_calculations():
    """Demonstrate the mathematical calculations step by step"""
    
    print("=" * 70)
    print("SEO TRAFFIC PROJECTION CALCULATIONS DEMO")
    print("=" * 70)
    
    # Create sample data
    gsc_df, kw_df = create_sample_data()
    
    print("\n1. BASELINE TRAFFIC CALCULATION")
    print("-" * 40)
    
    # Calculate baseline from GSC data
    baseline = baseline_from_gsc(gsc_df, months=3)
    print(f"Current baseline traffic: {baseline:,.0f} clicks/month")
    print(f"Based on last 3 months of GSC data ({len(gsc_df)} days)")
    
    print("\n2. KEYWORD ANALYSIS")
    print("-" * 40)
    
    # Show keyword details
    print("Target Keywords and Parameters:")
    for _, row in kw_df.iterrows():
        print(f"  • {row['keyword']}")
        print(f"    - Target Rank: {row['target_rank']}")
        print(f"    - Search Volume: {row['search_volume']:,}/month")
        print(f"    - SERP Share: {row['serp_share']:.1%}")
        print(f"    - Seasonality: {row['seasonality']:.1f}x")
    
    print("\n3. CTR CURVES (Click-Through Rates by Position)")
    print("-" * 40)
    
    ctr_curves = {
        "Conservative": {1: 20, 2: 12, 3: 8, 4: 6, 5: 5, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3},
        "Realistic":    {1: 28, 2: 15, 3: 11, 4: 8, 5: 8, 6: 4, 7: 4, 8: 4, 9: 4, 10: 4},
        "Optimistic":   {1: 32, 2: 18, 3: 14, 4: 10, 5: 9, 6: 5, 7: 5, 8: 5, 9: 5, 10: 5}
    }
    
    for scenario, ctr_map in ctr_curves.items():
        print(f"\n{scenario} Scenario:")
        for pos in range(1, 6):
            print(f"  Position {pos}: {ctr_map[pos]}% CTR")
    
    print("\n4. STEADY-STATE CALCULATIONS")
    print("-" * 40)
    
    # Calculate steady-state clicks for each keyword in each scenario
    for scenario, ctr_map in ctr_curves.items():
        print(f"\n{scenario} Scenario - Monthly Clicks per Keyword:")
        total_steady = 0
        
        for _, row in kw_df.iterrows():
            ctr = ctr_map.get(int(row['target_rank']), 0) / 100.0
            steady_clicks = row['search_volume'] * ctr * row['serp_share'] * row['seasonality']
            total_steady += steady_clicks
            
            print(f"  • {row['keyword']}: {steady_clicks:.0f} clicks/month")
            print(f"    (Formula: {row['search_volume']:,} × {ctr:.1%} × {row['serp_share']:.1%} × {row['seasonality']:.1f})")
        
        print(f"  Total Steady-State: {total_steady:.0f} clicks/month")
    
    print("\n5. RAMP CURVE (Growth Timeline)")
    print("-" * 40)
    
    ramp_curve = {1: 25, 2: 50, 3: 75, 4: 90, 5: 100, 6: 100}
    print("Traffic Growth Schedule:")
    for month, percentage in ramp_curve.items():
        print(f"  Month {month}: {percentage}% of steady-state potential")
    
    print("\n6. FINAL PROJECTIONS")
    print("-" * 40)
    
    # Run the full projection
    projection_df = compute_projection(baseline, kw_df, ctr_curves, ramp_curve)
    
    # Display results for each scenario
    for scenario in ["Conservative", "Realistic", "Optimistic"]:
        scenario_data = projection_df[projection_df['Scenario'] == scenario].sort_values('Month')
        
        print(f"\n{scenario} Scenario:")
        print(f"  Month 1: {scenario_data.iloc[0]['Total Clicks']:,.0f} clicks (+{scenario_data.iloc[0]['% Increase']:.1f}%)")
        print(f"  Month 3: {scenario_data.iloc[2]['Total Clicks']:,.0f} clicks (+{scenario_data.iloc[2]['% Increase']:.1f}%)")
        print(f"  Month 6: {scenario_data.iloc[5]['Total Clicks']:,.0f} clicks (+{scenario_data.iloc[5]['% Increase']:.1f}%)")
    
    print("\n" + "=" * 70)
    print("CALCULATION SUMMARY")
    print("=" * 70)
    
    print(f"• Baseline Traffic: {baseline:,.0f} clicks/month")
    print(f"• Target Keywords: {len(kw_df)} keywords")
    print(f"• Total Search Volume: {kw_df['search_volume'].sum():,.0f} searches/month")
    print(f"• Projection Period: 6 months")
    print(f"• Scenarios: Conservative, Realistic, Optimistic")
    
    # Show the most optimistic final result
    optimistic_final = projection_df[(projection_df['Scenario'] == 'Optimistic') & (projection_df['Month'] == 6)].iloc[0]
    print(f"• Best Case (Optimistic Month 6): {optimistic_final['Total Clicks']:,.0f} clicks (+{optimistic_final['% Increase']:.1f}%)")
    
    print("\n" + "=" * 70)
    print("All calculations completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    demonstrate_calculations()

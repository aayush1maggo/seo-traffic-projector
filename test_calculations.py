import pandas as pd
import numpy as np
from datetime import date, timedelta
import sys
import os

# Add the current directory to path to import functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions from streamlit app
from streamlit_app import month_floor, baseline_from_gsc, compute_projection

def test_month_floor():
    """Test the month_floor function"""
    print("Testing month_floor function...")
    
    test_dates = [
        pd.Timestamp('2024-01-15'),
        pd.Timestamp('2024-02-28'),
        pd.Timestamp('2024-12-31')
    ]
    
    expected_results = [
        pd.Timestamp('2024-01-01'),
        pd.Timestamp('2024-02-01'),
        pd.Timestamp('2024-12-01')
    ]
    
    for i, test_date in enumerate(test_dates):
        result = month_floor(test_date)
        expected = expected_results[i]
        assert result == expected, f"month_floor failed for {test_date}: got {result}, expected {expected}"
        print(f"✓ month_floor({test_date}) = {result}")
    
    print("✓ month_floor function passed all tests\n")

def test_baseline_from_gsc():
    """Test the baseline_from_gsc function"""
    print("Testing baseline_from_gsc function...")
    
    # Create test data - 6 months of daily data
    dates = pd.date_range(end=pd.Timestamp.today(), periods=180, freq="D")
    # Create realistic click data with some variation
    base_clicks = 100
    clicks = base_clicks + np.random.normal(0, 20, len(dates))
    clicks = np.maximum(0, clicks)  # No negative clicks
    
    gsc_df = pd.DataFrame({
        "date": dates,
        "clicks": clicks.round(0)
    })
    
    # Test with 3 months baseline
    baseline = baseline_from_gsc(gsc_df, months=3)
    
    # The baseline should be roughly around the base_clicks (100) per day * 30 days = 3000 per month
    expected_min = 2000  # Allow for some variation
    expected_max = 4000
    
    assert expected_min <= baseline <= expected_max, f"Baseline {baseline} outside expected range [{expected_min}, {expected_max}]"
    print(f"✓ baseline_from_gsc returned {baseline:.0f} clicks/month (expected ~3000)")
    
    # Test with empty dataframe
    empty_df = pd.DataFrame(columns=["date", "clicks"])
    baseline_empty = baseline_from_gsc(empty_df, months=3)
    assert baseline_empty == 0.0, f"Empty dataframe should return 0, got {baseline_empty}"
    print("✓ baseline_from_gsc handles empty data correctly")
    
    print("✓ baseline_from_gsc function passed all tests\n")

def test_compute_projection():
    """Test the compute_projection function"""
    print("Testing compute_projection function...")
    
    # Test data
    baseline_clicks = 5000.0
    
    kw_df = pd.DataFrame({
        "keyword": ["test keyword 1", "test keyword 2", "test keyword 3"],
        "target_rank": [3, 5, 2],
        "serp_share": [0.9, 0.85, 0.8],
        "seasonality": [1.0, 1.0, 1.1],
        "search_volume": [2000, 1500, 4000]
    })
    
    ctr_curves = {
        "Conservative": {1: 20, 2: 12, 3: 8, 4: 6, 5: 5, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3},
        "Realistic":    {1: 28, 2: 15, 3: 11, 4: 8, 5: 8, 6: 4, 7: 4, 8: 4, 9: 4, 10: 4},
        "Optimistic":   {1: 32, 2: 18, 3: 14, 4: 10, 5: 9, 6: 5, 7: 5, 8: 5, 9: 5, 10: 5}
    }
    
    ramp_curve = {1: 25, 2: 50, 3: 75, 4: 90, 5: 100, 6: 100}
    
    # Run projection
    projection_df = compute_projection(baseline_clicks, kw_df, ctr_curves, ramp_curve)
    
    # Verify structure
    assert len(projection_df) == 18, f"Expected 18 rows (3 scenarios × 6 months), got {len(projection_df)}"
    assert list(projection_df.columns) == ["Scenario", "Month", "New Clicks", "Total Clicks", "% Increase"], f"Unexpected columns: {list(projection_df.columns)}"
    
    # Verify scenarios
    scenarios = projection_df['Scenario'].unique()
    assert set(scenarios) == {"Conservative", "Realistic", "Optimistic"}, f"Unexpected scenarios: {scenarios}"
    
    # Verify months
    months = projection_df['Month'].unique()
    assert set(months) == {1, 2, 3, 4, 5, 6}, f"Unexpected months: {months}"
    
    # Verify calculations for one specific case
    conservative_month_1 = projection_df[(projection_df['Scenario'] == 'Conservative') & (projection_df['Month'] == 1)]
    assert len(conservative_month_1) == 1, "Should have exactly one row for Conservative Month 1"
    
    # Manual calculation verification
    # Keyword 1: 2000 * (8/100) * 0.9 * 1.0 = 144 clicks/month
    # Keyword 2: 1500 * (5/100) * 0.85 * 1.0 = 63.75 clicks/month  
    # Keyword 3: 4000 * (12/100) * 0.8 * 1.1 = 422.4 clicks/month
    # Total steady state: 144 + 63.75 + 422.4 = 630.15 clicks/month
    # Month 1 (25%): 630.15 * 0.25 = 157.54 clicks
    # Total: 5000 + 157.54 = 5157.54 clicks
    
    row = conservative_month_1.iloc[0]
    expected_new_clicks = 158  # Rounded
    expected_total = 5158      # Rounded
    expected_increase = 3.2    # Rounded percentage
    
    assert abs(row['New Clicks'] - expected_new_clicks) <= 5, f"New clicks calculation error: got {row['New Clicks']}, expected ~{expected_new_clicks}"
    assert abs(row['Total Clicks'] - expected_total) <= 5, f"Total clicks calculation error: got {row['Total Clicks']}, expected ~{expected_total}"
    assert abs(row['% Increase'] - expected_increase) <= 1, f"Percentage increase calculation error: got {row['% Increase']}, expected ~{expected_increase}"
    
    print(f"✓ Conservative Month 1: {row['New Clicks']} new clicks, {row['Total Clicks']} total, {row['% Increase']}% increase")
    
    # Verify that Optimistic > Realistic > Conservative for same month
    month_6_data = projection_df[projection_df['Month'] == 6]
    conservative_6 = month_6_data[month_6_data['Scenario'] == 'Conservative']['Total Clicks'].iloc[0]
    realistic_6 = month_6_data[month_6_data['Scenario'] == 'Realistic']['Total Clicks'].iloc[0]
    optimistic_6 = month_6_data[month_6_data['Scenario'] == 'Optimistic']['Total Clicks'].iloc[0]
    
    assert conservative_6 < realistic_6 < optimistic_6, f"Scenario ordering incorrect: Conservative({conservative_6}) < Realistic({realistic_6}) < Optimistic({optimistic_6})"
    print(f"✓ Scenario ordering correct: Conservative({conservative_6:.0f}) < Realistic({realistic_6:.0f}) < Optimistic({optimistic_6:.0f})")
    
    # Verify that later months have higher totals (due to ramp curve)
    conservative_data = projection_df[projection_df['Scenario'] == 'Conservative'].sort_values('Month')
    month_1_total = conservative_data.iloc[0]['Total Clicks']
    month_6_total = conservative_data.iloc[-1]['Total Clicks']
    
    assert month_6_total > month_1_total, f"Month 6 total ({month_6_total}) should be greater than Month 1 total ({month_1_total})"
    print(f"✓ Ramp curve working: Month 1 ({month_1_total:.0f}) < Month 6 ({month_6_total:.0f})")
    
    print("✓ compute_projection function passed all tests\n")

def test_error_handling():
    """Test error handling in functions"""
    print("Testing error handling...")
    
    # Test compute_projection with missing columns
    try:
        kw_df_invalid = pd.DataFrame({
            "keyword": ["test"],
            "target_rank": [3]
            # Missing required columns
        })
        
        ctr_curves = {"Conservative": {1: 20}}
        ramp_curve = {1: 25}
        
        compute_projection(1000, kw_df_invalid, ctr_curves, ramp_curve)
        assert False, "Should have raised ValueError for missing columns"
    except ValueError as e:
        print("✓ compute_projection correctly raises ValueError for missing columns")
    
    print("✓ Error handling tests passed\n")

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("SEO TRAFFIC PROJECTION CALCULATION TESTS")
    print("=" * 60)
    
    try:
        test_month_floor()
        test_baseline_from_gsc()
        test_compute_projection()
        test_error_handling()
        
        print("=" * 60)
        print("ALL TESTS PASSED! Calculations are working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"TEST FAILED: {str(e)}")
        raise

if __name__ == "__main__":
    run_all_tests()

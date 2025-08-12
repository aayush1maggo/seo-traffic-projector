# SEO Traffic Projection by Aayush

A professional Python tool for projecting organic traffic growth based on keyword rankings and historical performance data.

## Features

- **Traffic Projection**: Generate 6-month traffic projections using three scenarios (Conservative, Realistic, Optimistic)
- **Multiple Data Sources**: Support for Google Search Console API and CSV uploads
- **Keyword Analysis**: Calculate potential traffic from target keyword rankings
- **Professional Web Interface**: Streamlit app with modern, professional design
- **Excel/CSV Export**: Download detailed reports in multiple formats
- **Mathematical Modeling**: Based on CTR curves, SERP share, and seasonality factors

## Quick Start

### Option 1: Streamlit Web App (Recommended)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

3. Open your browser and go to `http://localhost:8501`

### Option 2: Command Line Tool

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the analysis:
   ```bash
   python final_integration.py
   ```

## Data Requirements

### Google Search Console Data
CSV file with columns:
- `date`: Date in YYYY-MM-DD format
- `clicks`: Number of clicks (required)
- `impressions`: Number of impressions (optional)

### Keywords Data
CSV file with columns:
- `keyword`: Target keyword
- `target_rank`: Target ranking position (1-10)
- `serp_share`: Expected share of SERP clicks (0.0-1.0)
- `seasonality`: Seasonal factor (0.5-2.0)
- `search_volume`: Monthly search volume

## Mathematical Model

The tool uses a sophisticated mathematical model:

1. **Baseline Calculation**: Average monthly clicks from historical GSC data
2. **Steady-State Clicks**: `Search Volume × CTR × SERP Share × Seasonality`
3. **Ramp Curve**: Gradual traffic growth over 6 months (25% → 100%)
4. **CTR Curves**: Position-based click-through rates for three scenarios

### CTR Curves (Default)
- **Conservative**: Position 1: 20%, Position 3: 8%, Position 5: 5%
- **Realistic**: Position 1: 28%, Position 3: 11%, Position 5: 8%
- **Optimistic**: Position 1: 32%, Position 3: 14%, Position 5: 9%

### Ramp Curve (Default)
- Month 1: 25% of steady-state potential
- Month 2: 50% of steady-state potential
- Month 3: 75% of steady-state potential
- Month 4: 90% of steady-state potential
- Month 5-6: 100% of steady-state potential

## Files Structure

- `streamlit_app.py`: Main Streamlit web application
- `final_integration.py`: Command-line version of the tool
- `requirements.txt`: Python dependencies
- `test_calculations.py`: Unit tests for calculations
- `demo_calculations.py`: Step-by-step calculation demonstration
- `README_STREAMLIT.md`: Detailed Streamlit app documentation

## Dependencies

Required:
- `streamlit`: Web application framework
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `matplotlib`: Basic plotting
- `plotly`: Interactive charts

Optional:
- `xlsxwriter` or `openpyxl`: Excel export functionality
- `google-auth`, `google-api-python-client`: Google Search Console API
- `requests`: SEMrush API integration

## Testing

Run the test suite to verify calculations:
```bash
python test_calculations.py
```

View step-by-step calculations:
```bash
python demo_calculations.py
```

## Configuration

The tool allows customization of:
- Baseline calculation period (1-12 months)
- CTR curves for different positions
- Ramp curve percentages
- Target keyword parameters

## Output

The tool generates:
- Interactive traffic growth charts
- Detailed projection tables
- Excel/CSV reports
- Keywords analysis breakdown

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Aayush Maggo** - Professional SEO Traffic Projection Tool

---

For detailed usage instructions, see `README_STREAMLIT.md`.

# SEO Traffic Projection by Aayush

A professional web application for projecting organic traffic growth based on keyword rankings and historical Google Search Console performance data.

## Features

- 📊 **File Upload Interface**: Upload CSV files for Google Search Console data and keyword targets
- 🎯 **Interactive Configuration**: Adjust CTR curves, ramp schedules, and baseline periods
- 📈 **Real-time Visualisation**: Interactive charts showing traffic projections across 3 scenarios
- 💾 **Multiple Export Options**: Download results as Excel or CSV files
- 🧪 **Sample Data Mode**: Test the tool with built-in sample data

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run streamlit_app.py
```

### 3. Open in Browser
The app will automatically open in your browser at `http://localhost:8501`

## Data Requirements

### Google Search Console Data (CSV)
Your CSV file should contain these columns:
- `date`: Date in YYYY-MM-DD format
- `clicks`: Number of clicks per day
- `impressions`: (optional) Number of impressions per day

### Keywords Data (CSV)
Your CSV file should contain these columns:
- `keyword`: The target keyword
- `target_rank`: Target ranking position (1-10)
- `serp_share`: Expected SERP feature share (0.0-1.0)
- `seasonality`: Seasonal multiplier (e.g., 1.0 = normal, 1.2 = 20% boost)
- `search_volume`: Monthly search volume

## How to Use

1. **Upload Your Data**: Use the file uploaders in the main interface
2. **Configure Settings**: Adjust parameters in the sidebar:
   - Baseline months for averaging
   - CTR curves for different scenarios
   - Ramp curve showing traffic growth over time
3. **Run Analysis**: Click "Run Traffic Projection Analysis"
4. **Review Results**: View charts, metrics, and detailed projections
5. **Download**: Export results as Excel or CSV files

## Configuration Options

### CTR Curves
Three scenarios with different click-through rates by position:
- **Conservative**: Lower CTR estimates
- **Realistic**: Moderate CTR estimates  
- **Optimistic**: Higher CTR estimates

### Ramp Curve
Controls how quickly traffic grows over 6 months:
- Month 1: 25% of potential (default)
- Month 2: 50% of potential
- Month 3: 75% of potential
- Month 4: 90% of potential
- Month 5-6: 100% of potential

## Sample Data

Click "Use Sample Data" to test the tool with built-in example data including:
- 180 days of synthetic GSC data
- 3 sample keywords with realistic search volumes

## Troubleshooting

### Missing Excel Libraries
If you see warnings about Excel export:
```bash
pip install xlsxwriter
# OR
pip install openpyxl
```

### File Format Issues
Ensure your CSV files:
- Use UTF-8 encoding
- Have proper column headers (case-insensitive)
- Contain numeric data in volume/clicks columns
- Use YYYY-MM-DD date format

## Technical Notes

- Built with Streamlit for the web interface
- Uses Plotly for interactive visualisations
- Supports both xlsxwriter and openpyxl for Excel export
- Automatically handles missing dependencies with graceful fallbacks
- All processing happens locally - no data sent to external servers

## Customisation

You can modify the tool by editing `streamlit_app.py`:
- Adjust default CTR curves
- Change ramp curve options
- Add new visualisation types
- Modify data validation rules

## Support

For issues or questions:
1. Check that your CSV files match the required format
2. Ensure all dependencies are installed
3. Try the sample data mode to verify the tool works
4. Check the Streamlit logs for detailed error messages

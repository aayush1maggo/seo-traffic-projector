# Build a reusable Python tool that supports BOTH CSV uploads and API pulls (GSC + SEMrush)
# It will generate:
# - A 6‑month projection table across 3 scenarios
# - Charts (PNG)
# - An Excel workbook with inputs, settings, and projections
#
# You can run this cell as-is (it uses sample data).
# Then replace the CSV paths or add API keys/credentials to run with your data.

import os
import json
import math
from datetime import date, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# import ace_tools as tools  # Commented out - not available in standard environments

# -----------------------------
# CONFIGURATION
# -----------------------------

OUTPUT_DIR = Path("/mnt/data/traffic_projection_tool")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Toggle data sources:
USE_CSV_FOR_GSC = True    # If False, will try API
USE_CSV_FOR_SEMRUSH = True  # If False, will try API

# CSV input paths (replace with your real files when ready)
CSV_GSC_CLICKS = None  # path to a CSV with columns: date, clicks  (optional: impressions)
CSV_KEYWORDS = None    # path to a CSV with columns: keyword, target_rank, serp_share, seasonality, search_volume (if using CSV for SEMrush)

# API credentials (leave as None to skip)
# --- Google Search Console ---
GSC_SITE_URL = "https://example.com/"     # must match verified property
GSC_START_DAYS_AGO = 180
GSC_END_DAYS_AGO = 1
GSC_SERVICE_ACCOUNT_JSON = None  # e.g. "/mnt/data/your-service-account.json" (user must grant this SA access to the GSC property)

# --- SEMrush ---
SEMRUSH_API_KEY = None
SEMRUSH_DATABASE = "au"  # country DB, e.g., "au", "us", "uk"
# :) we will show code that uses requests; won't run here without internet, but ready to use in your environment.

# Projection settings
BASELINE_MONTHS = 3  # average last N months for baseline
RAMP_CURVE = {1: 25, 2: 50, 3: 75, 4: 90, 5: 100, 6: 100}
CTR_CURVES = {
    "Conservative": {1: 20, 2: 12, 3: 8, 4: 6, 5: 5, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3},
    "Realistic":    {1: 28, 2: 15, 3: 11, 4: 8, 5: 8, 6: 4, 7: 4, 8: 4, 9: 4, 10: 4},
    "Optimistic":   {1: 32, 2: 18, 3: 14, 4: 10, 5: 9, 6: 5, 7: 5, 8: 5, 9: 5, 10: 5}
}

# -----------------------------
# HELPERS
# -----------------------------

def month_floor(d: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=d.year, month=d.month, day=1)

def load_gsc_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # expected: date, clicks (optional: impressions)
    df["date"] = pd.to_datetime(df["date"])
    return df

def baseline_from_gsc(gsc_df: pd.DataFrame, months: int = 3) -> float:
    # Use last N full months average clicks
    if gsc_df.empty:
        return 0.0
    gsc_df = gsc_df.sort_values("date")
    last_date = gsc_df["date"].max()
    last_month_start = month_floor(last_date)
    start_month = last_month_start - pd.DateOffset(months=months-1)
    mask = (gsc_df["date"] >= start_month) & (gsc_df["date"] < last_month_start + pd.offsets.MonthEnd(1))
    window = gsc_df.loc[mask]
    if window.empty:
        # fallback: use last 90 days average
        window = gsc_df[gsc_df["date"] >= (last_date - pd.Timedelta(days=90))]
    clicks = window.groupby(window["date"].dt.to_period("M"))["clicks"].sum().astype(float)
    if clicks.empty:
        return float(window["clicks"].mean()) if "clicks" in window else 0.0
    return float(clicks.mean())

def fetch_gsc_via_api(site_url: str, start_date: str, end_date: str, sa_json_path: str) -> pd.DataFrame:
    # NOTE: This function is ready to run in your local environment with internet access.
    # It won’t execute in this notebook environment.
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    scopes = ["https://www.googleapis.com/auth/webmasters.readonly"]
    credentials = service_account.Credentials.from_service_account_file(sa_json_path, scopes=scopes)
    service = build("searchconsole", "v1", credentials=credentials)  # new API name
    request = {
        "startDate": start_date,
        "endDate": end_date,
        "dimensions": ["date"],
        "rowLimit": 25000
    }
    resp = service.searchanalytics().query(siteUrl=site_url, body=request).execute()
    rows = resp.get("rows", [])
    data = []
    for r in rows:
        d = r["keys"][0]
        clicks = r.get("clicks", 0)
        impressions = r.get("impressions", None)
        data.append({"date": d, "clicks": clicks, "impressions": impressions})
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df

def load_keywords_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # columns: keyword, target_rank, serp_share, seasonality, search_volume
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def fetch_semrush_volumes(api_key: str, keywords: list, database: str = "au") -> pd.DataFrame:
    # Ready-to-run in your local env with internet. Uses batch calls (per keyword) for simplicity.
    import requests
    rows = []
    for kw in keywords:
        params = {
            "type": "phrase_kdi",
            "key": api_key,
            "phrase": kw,
            "database": database,
            "export_columns": "Ph,Nq,Kd"
        }
        r = requests.get("https://api.semrush.com/", params=params, timeout=30)
        if r.status_code == 200 and len(r.text.strip().splitlines()) >= 2:
            # CSV-like text, header on first line
            lines = r.text.strip().splitlines()
            header = lines[0].split(";")
            vals = lines[1].split(";")
            rec = dict(zip(header, vals))
            # Nq is volume
            volume = float(rec.get("Nq", 0) or 0)
            rows.append({"keyword": kw, "search_volume": volume})
        else:
            rows.append({"keyword": kw, "search_volume": 0})
    return pd.DataFrame(rows)

def compute_projection(baseline_clicks: float, kw_df: pd.DataFrame,
                       ctr_curves: dict, ramp_curve: dict) -> pd.DataFrame:
    # Ensure needed columns exist
    required_cols = ["keyword", "target_rank", "serp_share", "seasonality", "search_volume"]
    for c in required_cols:
        if c not in kw_df.columns:
            raise ValueError(f"Keywords DataFrame missing required column: {c}")
    results = []
    for scenario, ctr_map in ctr_curves.items():
        # steady-state per keyword
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
                "New Clicks": new_clicks,
                "Total Clicks": total,
                "% Increase": inc_pct
            })
    return pd.DataFrame(results)

def save_outputs(projection_df: pd.DataFrame, baseline_clicks: float, kw_df: pd.DataFrame):
    # Charts
    pivot = projection_df.pivot(index="Month", columns="Scenario", values="Total Clicks")
    plt.figure(figsize=(8,5))
    for scenario in pivot.columns:
        plt.plot(pivot.index, pivot[scenario], marker="o", label=scenario)
    plt.axhline(y=baseline_clicks, linestyle="--", label="Baseline")
    plt.title("Projected Organic Traffic Over 6 Months")
    plt.xlabel("Month")
    plt.ylabel("Total Clicks")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    chart_path = OUTPUT_DIR / "traffic_projection_over_time.png"
    plt.savefig(chart_path, dpi=160)
    plt.close()

    # Excel export
    xlsx_path = OUTPUT_DIR / "Traffic_Projection_Output.xlsx"
    try:
        # Try xlsxwriter first, then fall back to openpyxl
        engine = "xlsxwriter"
        try:
            import xlsxwriter
        except ImportError:
            engine = "openpyxl"
            
        with pd.ExcelWriter(xlsx_path, engine=engine) as writer:
            # Inputs
            kw_df.to_excel(writer, index=False, sheet_name="Keywords_Input")
            proj_copy = projection_df.copy()
            proj_copy.to_excel(writer, index=False, sheet_name="Projection")
            # Settings
            pd.DataFrame({
                "Setting": ["Baseline Monthly Clicks", "Baseline Months Used"],
                "Value": [baseline_clicks, BASELINE_MONTHS]
            }).to_excel(writer, index=False, sheet_name="Settings")
    except ImportError:
        # If neither engine is available, save as CSV files instead
        print("Warning: Neither xlsxwriter nor openpyxl available. Saving as CSV files instead.")
        csv_dir = OUTPUT_DIR / "csv_outputs"
        csv_dir.mkdir(exist_ok=True)
        kw_df.to_csv(csv_dir / "keywords_input.csv", index=False)
        projection_df.to_csv(csv_dir / "projection.csv", index=False)
        pd.DataFrame({
            "Setting": ["Baseline Monthly Clicks", "Baseline Months Used"],
            "Value": [baseline_clicks, BASELINE_MONTHS]
        }).to_csv(csv_dir / "settings.csv", index=False)
        xlsx_path = csv_dir  # Return directory path instead
    return chart_path, xlsx_path

def create_readme(path: Path):
    txt = f"""
Traffic Projection Tool — How to Use
====================================

Inputs
------
You can use either CSV uploads or APIs.

1) Google Search Console (GSC) Baseline
   - CSV: provide a file with columns: date, clicks (optional impressions)
   - API: provide SERVICE ACCOUNT JSON and grant it access to your GSC property.
     Update: GSC_SITE_URL, GSC_SERVICE_ACCOUNT_JSON, GSC_START_DAYS_AGO, GSC_END_DAYS_AGO.

2) SEMrush Keyword Volumes
   - CSV: your keywords file should include columns:
         keyword, target_rank, serp_share, seasonality, search_volume
   - API: set SEMRUSH_API_KEY and SEMRUSH_DATABASE. The tool will fetch volumes for the given keywords.

Projection Model
----------------
Total per keyword (steady-state) = Volume * CTR(rank) * SERP Share * Seasonality
Then a ramp curve applies over months: {RAMP_CURVE}

Three scenarios are included with different CTR curves:
- Conservative
- Realistic
- Optimistic

Outputs
-------
- traffic_projection_over_time.png
- Traffic_Projection_Output.xlsx (Keywords_Input, Projection, Settings)

Switching Modes
---------------
- USE_CSV_FOR_GSC / USE_CSV_FOR_SEMRUSH booleans at the top of the script.
- Provide file paths or API creds accordingly.

Dependencies
------------
Required: pandas, numpy, matplotlib
Optional: xlsxwriter OR openpyxl (for Excel output; will fall back to CSV if neither available)
For APIs: google-auth, google-auth-oauthlib, google-auth-httplib2, google-api-python-client, requests

Notes
-----
- For GSC API: service account must be added as a user to the property.
- For SEMrush API: volume source is 'phrase_kdi' endpoint. Consider API limits.
- Adjust CTR curves and ramp in the script to match your niche.
- If Excel libraries are missing, output will be saved as CSV files instead.

"""
    path.write_text(txt.strip(), encoding="utf-8")


# -----------------------------
# DEMO RUN (sample data or real, based on config)
# -----------------------------

# 1) GSC baseline clicks
if USE_CSV_FOR_GSC and CSV_GSC_CLICKS:
    gsc_df = load_gsc_from_csv(CSV_GSC_CLICKS)
    baseline = baseline_from_gsc(gsc_df, months=BASELINE_MONTHS)
else:
    # fall back to simple synthetic baseline if no CSV, or try API if enabled
    if USE_CSV_FOR_GSC:
        # no CSV provided; use synthetic example
        dates = pd.date_range(end=pd.Timestamp.today(), periods=180, freq="D")
        clicks = (np.sin(np.linspace(0, 6*np.pi, len(dates)))+2.5)*120 + np.random.normal(0, 30, len(dates)) + 8000/180
        gsc_df = pd.DataFrame({"date": dates, "clicks": np.maximum(0, clicks).round(0)})
        baseline = baseline_from_gsc(gsc_df, months=BASELINE_MONTHS)
    else:
        # API path (won't run in this environment, but code is ready for your machine)
        end = date.today() - timedelta(days=GSC_END_DAYS_AGO)
        start = date.today() - timedelta(days=GSC_START_DAYS_AGO)
        gsc_df = pd.DataFrame()
        baseline = 0.0
        if GSC_SERVICE_ACCOUNT_JSON:
            try:
                gsc_df = fetch_gsc_via_api(GSC_SITE_URL, start.isoformat(), end.isoformat(), GSC_SERVICE_ACCOUNT_JSON)
                baseline = baseline_from_gsc(gsc_df, months=BASELINE_MONTHS)
            except Exception as e:
                baseline = 0.0

# 2) Keywords + SEMrush volumes
if USE_CSV_FOR_SEMRUSH and CSV_KEYWORDS:
    kw_df = load_keywords_from_csv(CSV_KEYWORDS)
else:
    # create a sample keywords table. If using API, we still need keywords from CSV (without volume)
    kw_df = pd.DataFrame({
        "keyword": ["keyword a", "keyword b", "keyword c"],
        "target_rank": [3, 5, 2],
        "serp_share": [0.9, 0.85, 0.8],
        "seasonality": [1.0, 1.0, 1.1],
        "search_volume": [2000, 1500, 4000]  # sample volumes; API would replace this
    })

# If using SEMrush API and we only have keywords (no volume), fetch volumes (not executed here)
if not USE_CSV_FOR_SEMRUSH and SEMRUSH_API_KEY:
    # Expect kw_df to have keyword column but maybe no search_volume
    if "search_volume" not in kw_df.columns or kw_df["search_volume"].isna().all():
        try:
            vols = fetch_semrush_volumes(SEMRUSH_API_KEY, kw_df["keyword"].tolist(), database=SEMRUSH_DATABASE)
            kw_df = kw_df.drop(columns=[c for c in ["search_volume"] if c in kw_df.columns])
            kw_df = kw_df.merge(vols, on="keyword", how="left")
            kw_df["search_volume"] = kw_df["search_volume"].fillna(0)
        except Exception as e:
            pass

# 3) Compute projections
projection_df = compute_projection(baseline_clicks=float(baseline), kw_df=kw_df.copy(),
                                   ctr_curves=CTR_CURVES, ramp_curve=RAMP_CURVE)

chart_path, xlsx_path = save_outputs(projection_df, baseline_clicks=float(baseline), kw_df=kw_df.copy())
readme_path = OUTPUT_DIR / "README.txt"
create_readme(readme_path)

# Show primary table to user
print("Projection Results (All Scenarios):")
print(projection_df.to_string(index=False))

(str(chart_path), str(xlsx_path), str(readme_path), float(baseline))

# Inverter Data Analyzer (Browser App)

This is a no-code Streamlit web app for analyzing solar inverter data. You only need Python installed.

## Quick Start

1) **Install Python 3.10+**  
2) Open a terminal and install dependencies:
```bash
pip install -r requirements.txt
```
3) Run the app:
```bash
streamlit run app.py
```
4) Your browser will open automatically (or visit the URL shown in the terminal).

## What You Get

- Upload CSV/Excel inverter logs.
- Map your columns interactively (no strict format).
- Auto KPIs: daily energy, capacity factor, CUF (if capacity input is given).
- Anomaly detection: zero power in daylight, spikes, negative power, grid frequency/voltage extremes.
- Time series and inverter comparison charts.
- Download cleaned data, daily summary, and anomalies as CSV.

## Tip

Start with the provided **sample template** from within the app to understand the format, then upload your real data.
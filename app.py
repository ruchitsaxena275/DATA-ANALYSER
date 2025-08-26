import io
import math
import zipfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(page_title="Inverter Data Analyzer", layout="wide")

st.title("⚡ Inverter Data Analyzer (No-Setup, Browser App)")
st.caption("Upload your inverter CSV/Excel → map columns → get auto analysis, anomalies, KPIs, and downloads.")

# -------------------------------
# Sidebar Controls
# -------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    tz = st.selectbox(
        "Timezone",
        options=[
            "Asia/Kolkata","UTC","Asia/Dubai","Asia/Karachi","Asia/Bangkok","Europe/Berlin",
            "Europe/London","America/New_York","Australia/Sydney"
        ],
        index=0
    )

    plant_dccap_mwp = st.number_input("Plant DC Capacity (MWp) — optional", min_value=0.0, value=0.0, step=0.1, help="Used to estimate CUF if provided.")
    inv_rated_kw = st.number_input("Inverter Rated Power (kW) — optional", min_value=0.0, value=0.0, step=10.0, help="Used for capacity factor calculations.")
    expected_interval_min = st.number_input("Expected data interval (minutes)", min_value=1, value=15, step=1, help="Typical SCADA intervals: 1, 5, 10, 15 minutes")
    dayfirst = st.checkbox("Parse dates with day-first format (DD/MM/YYYY)", value=True)

    st.markdown("---")
    st.subheader("Anomaly thresholds")
    min_daylight_start = st.slider("Earliest sunrise hour", 4, 9, 6)
    max_daylight_end = st.slider("Latest sunset hour", 16, 22, 19)
    zero_power_threshold_kw = st.number_input("Zero power threshold (kW) considered 'zero'", min_value=0.0, value=0.1, step=0.1)
    spike_multiplier = st.slider("Spike detection sensitivity (x IQR)", 3.0, 10.0, 5.0)
    grid_f_min = st.number_input("Grid frequency min (Hz)", min_value=0.0, value=49.5, step=0.1)
    grid_f_max = st.number_input("Grid frequency max (Hz)", min_value=0.0, value=50.5, step=0.1)
    v_ll_max_pct = st.number_input("Grid voltage max (% of nominal)", min_value=80.0, max_value=130.0, value=110.0, step=1.0)
    v_ll_min_pct = st.number_input("Grid voltage min (% of nominal)", min_value=50.0, max_value=120.0, value=90.0, step=1.0)

    st.markdown("---")
    st.subheader("Exports")
    allow_download = st.checkbox("Allow CSV downloads", value=True)

st.markdown("### 1) Upload your data")
uploaded = st.file_uploader("CSV or Excel (.csv, .xlsx, .xls)", type=["csv", "xlsx", "xls"], accept_multiple_files=False)

def read_any(file, dayfirst=True) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

def guess_column(candidates: List[str], columns: List[str]) -> Optional[str]:
    cols = [c.lower() for c in columns]
    for cand in candidates:
        if cand.lower() in cols:
            idx = cols.index(cand.lower())
            return columns[idx]
    # partial contains
    for i, c in enumerate(cols):
        for cand in candidates:
            if cand.lower() in c:
                return columns[i]
    return None

if uploaded:
    df_raw = read_any(uploaded, dayfirst=dayfirst)
    st.success(f"Loaded {len(df_raw):,} rows and {len(df_raw.columns)} columns.")
    st.dataframe(df_raw.head(20), use_container_width=True)

    st.markdown("### 2) Map columns")
    cols = list(df_raw.columns)

    # Try to guess
    ts_col_guess = guess_column(["timestamp","time","datetime","date_time","logged","recorded"], cols)
    inv_col_guess = guess_column(["inverter","inverter_id","inv","name","string"], cols)
    p_kw_col_guess = guess_column(["active_power_kw","power_kw","p_kw","kw","pac","active power","power(ac)"], cols)
    e_kwh_col_guess = guess_column(["energy_kwh","e_kwh","kwh","wh","energy"], cols)
    vdc_col_guess = guess_column(["vdc","dc_voltage","u_dc"], cols)
    idc_col_guess = guess_column(["idc","dc_current","i_dc"], cols)
    vac_col_guess = guess_column(["vac","ac_voltage","v_ll","line_voltage"], cols)
    iac_col_guess = guess_column(["iac","ac_current","i_line"], cols)
    freq_col_guess = guess_column(["freq","frequency","hz","grid_freq"], cols)
    status_col_guess = guess_column(["status","state","mode","alarm"], cols)

    with st.form("mapping"):
        c1, c2, c3 = st.columns(3)
        with c1:
            ts_col = st.selectbox("Timestamp column *", options=["<none>"] + cols, index=(cols.index(ts_col_guess)+1) if ts_col_guess in cols else 0)
            inv_col = st.selectbox("Inverter column *", options=["<none>"] + cols, index=(cols.index(inv_col_guess)+1) if inv_col_guess in cols else 0)
            p_kw_col = st.selectbox("Active Power (kW)", options=["<none>"] + cols, index=(cols.index(p_kw_col_guess)+1) if p_kw_col_guess in cols else 0)
            e_kwh_col = st.selectbox("Energy (kWh) cumulative", options=["<none>"] + cols, index=(cols.index(e_kwh_col_guess)+1) if e_kwh_col_guess in cols else 0)
        with c2:
            vdc_col = st.selectbox("DC Voltage (V)", options=["<none>"] + cols, index=(cols.index(vdc_col_guess)+1) if vdc_col_guess in cols else 0)
            idc_col = st.selectbox("DC Current (A)", options=["<none>"] + cols, index=(cols.index(idc_col_guess)+1) if idc_col_guess in cols else 0)
            vac_col = st.selectbox("AC Voltage (V or V_LL)", options=["<none>"] + cols, index=(cols.index(vac_col_guess)+1) if vac_col_guess in cols else 0)
        with c3:
            iac_col = st.selectbox("AC Current (A)", options=["<none>"] + cols, index=(cols.index(iac_col_guess)+1) if iac_col_guess in cols else 0)
            freq_col = st.selectbox("Grid Frequency (Hz)", options=["<none>"] + cols, index=(cols.index(freq_col_guess)+1) if freq_col_guess in cols else 0)
            status_col = st.selectbox("Status/Alarm", options=["<none>"] + cols, index=(cols.index(status_col_guess)+1) if status_col_guess in cols else 0)

        submitted = st.form_submit_button("Apply Mapping")

    if submitted:
        required_missing = []
        if ts_col == "<none>": required_missing.append("Timestamp")
        if inv_col == "<none>": required_missing.append("Inverter")
        if required_missing:
            st.error(f"Please map required columns: {', '.join(required_missing)}")
            st.stop()

        # Build working dataframe
        df = pd.DataFrame()
        df["timestamp_raw"] = df_raw[ts_col]
        df["inverter"] = df_raw[inv_col].astype(str)
        if p_kw_col != "<none>":
            df["p_kw"] = pd.to_numeric(df_raw[p_kw_col], errors="coerce")
        if e_kwh_col != "<none>":
            df["e_kwh_cum"] = pd.to_numeric(df_raw[e_kwh_col], errors="coerce")
        if vdc_col != "<none>":
            df["vdc"] = pd.to_numeric(df_raw[vdc_col], errors="coerce")
        if idc_col != "<none>":
            df["idc"] = pd.to_numeric(df_raw[idc_col], errors="coerce")
        if vac_col != "<none>":
            df["vac"] = pd.to_numeric(df_raw[vac_col], errors="coerce")
        if iac_col != "<none>":
            df["iac"] = pd.to_numeric(df_raw[iac_col], errors="coerce")
        if freq_col != "<none>":
            df["freq"] = pd.to_numeric(df_raw[freq_col], errors="coerce")
        if status_col != "<none>":
            df["status"] = df_raw[status_col].astype(str)

        # Parse timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp_raw"], errors="coerce", dayfirst=dayfirst, utc=True)
        # localize to tz
        try:
            df["timestamp"] = df["timestamp"].dt.tz_convert(tz)
        except Exception:
            # If naive, localize then convert
            df["timestamp"] = pd.to_datetime(df["timestamp_raw"], errors="coerce", dayfirst=dayfirst)
            df["timestamp"] = df["timestamp"].dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")

        df = df.dropna(subset=["timestamp"])
        df = df.sort_values(["inverter","timestamp"]).reset_index(drop=True)

        # derive dc/ac power
        if "p_kw" not in df.columns and {"vdc","idc"}.issubset(df.columns):
            df["p_kw"] = (df["vdc"] * df["idc"]) / 1000.0

        # infer sample interval
        dt = df.groupby("inverter")["timestamp"].diff().dt.total_seconds().dropna()
        inferred = dt.mode().iloc[0] / 60 if not dt.empty else expected_interval_min
        sample_min = inferred if not math.isnan(inferred) else expected_interval_min
        st.info(f"Inferred sample interval ≈ {sample_min:.1f} min (you set {expected_interval_min} min)")

        # energy calculation
        if "e_kwh_cum" in df.columns:
            df["e_kwh_step"] = df.groupby("inverter")["e_kwh_cum"].diff().clip(lower=0)  # handle resets
        elif "p_kw" in df.columns:
            # integrate power over interval
            delta_h = (expected_interval_min if math.isnan(sample_min) else sample_min) / 60.0
            df["e_kwh_step"] = df["p_kw"] * delta_h
        else:
            df["e_kwh_step"] = np.nan

        # runtime flag
        df["is_runtime"] = (~df["e_kwh_step"].isna()) & (df["e_kwh_step"] > 0)

        # daylight mask (local time hour)
        df["hour"] = df["timestamp"].dt.hour
        daylight = (df["hour"] >= min_daylight_start) & (df["hour"] <= max_daylight_end)

        # anomalies
        anomalies = []
        # 1) zero generation during daylight
        if "p_kw" in df.columns:
            zero_day = daylight & (df["p_kw"].fillna(0) <= zero_power_threshold_kw)
            anomalies.append(df.loc[zero_day, ["timestamp","inverter","p_kw"]].assign(reason="Zero power in daylight"))
            # 2) spikes using IQR
            q1 = df["p_kw"].quantile(0.25)
            q3 = df["p_kw"].quantile(0.75)
            iqr = max(q3 - q1, 1e-6)
            hi_cut = q3 + spike_multiplier * iqr
            lo_cut = max(q1 - spike_multiplier * iqr, 0)
            spike_mask = (df["p_kw"] > hi_cut) | (df["p_kw"] < lo_cut)
            anomalies.append(df.loc[spike_mask, ["timestamp","inverter","p_kw"]].assign(reason="Power spike/outlier"))
            # 3) negative power
            neg_mask = df["p_kw"] < 0
            anomalies.append(df.loc[neg_mask, ["timestamp","inverter","p_kw"]].assign(reason="Negative power"))
        # 4) frequency out of range
        if "freq" in df.columns:
            f_bad = (df["freq"] < grid_f_min) | (df["freq"] > grid_f_max)
            anomalies.append(df.loc[f_bad, ["timestamp","inverter","freq"]].assign(reason="Grid frequency out of range"))
        # 5) voltage out of range (percentage unknown -> flag extreme absolute using percent-of-median)
        if "vac" in df.columns:
            med_v = df["vac"].median()
            if med_v and med_v > 0:
                pct = df["vac"] / med_v * 100
                v_bad = (pct < v_ll_min_pct) | (pct > v_ll_max_pct)
                anomalies.append(df.loc[v_bad, ["timestamp","inverter","vac"]].assign(reason="Grid voltage extreme vs median"))

        anomalies_df = pd.concat(anomalies, ignore_index=True) if anomalies else pd.DataFrame(columns=["timestamp","inverter","reason"])
        anomalies_df = anomalies_df.sort_values("timestamp")

        # daily summaries
        df["date"] = df["timestamp"].dt.date
        daily = df.groupby(["date","inverter"]).agg(
            energy_kwh=("e_kwh_step","sum"),
            max_power_kw=("p_kw","max"),
            runtime_points=("is_runtime","sum"),
            points=("is_runtime","size")
        ).reset_index()
        # estimates
        if inv_rated_kw and inv_rated_kw > 0:
            daily["capacity_factor_%"] = (daily["energy_kwh"] / (inv_rated_kw * 24)) * 100
        else:
            daily["capacity_factor_%"] = np.nan

        # plant summaries
        plant_daily = daily.groupby("date").agg(
            energy_kwh=("energy_kwh","sum"),
            max_power_kw=("max_power_kw","sum")
        ).reset_index()
        if plant_dccap_mwp and plant_dccap_mwp > 0:
            plant_daily["CUF_%"] = (plant_daily["energy_kwh"] / (plant_dccap_mwp * 1000 * 24)) * 100
        else:
            plant_daily["CUF_%"] = np.nan

        st.markdown("### 3) Results")
        tab1, tab2, tab3, tab4 = st.tabs(["Plant Overview","Inverter Comparison","Time Series","Anomalies"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Plant Daily Energy")
                fig = px.bar(plant_daily, x="date", y="energy_kwh", title="Daily Plant Energy (kWh)")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.subheader("Plant CUF (if capacity set)")
                fig2 = px.line(plant_daily, x="date", y="CUF_%", title="Daily CUF (%)")
                st.plotly_chart(fig2, use_container_width=True)

            st.dataframe(plant_daily, use_container_width=True)

        with tab2:
            st.subheader("Top Underperforming Inverters (energy)")
            latest_date = daily["date"].max()
            if pd.notna(latest_date):
                latest = daily[daily["date"] == latest_date].sort_values("energy_kwh")
                st.dataframe(latest, use_container_width=True)

                fig = px.bar(latest, x="inverter", y="energy_kwh", title=f"Energy by Inverter on {latest_date}")
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("Time Series")
            inv_list = sorted(df["inverter"].unique().tolist())
            pick_inv = st.multiselect("Pick inverters", options=inv_list, default=inv_list[: min(5, len(inv_list))])
            metric = st.selectbox("Metric", options=[c for c in ["p_kw","e_kwh_step","vdc","idc","vac","iac","freq"] if c in df.columns], index=0 if "p_kw" in df.columns else 1)
            if pick_inv and metric:
                plot_df = df[df["inverter"].isin(pick_inv)]
                fig = px.line(plot_df, x="timestamp", y=metric, color="inverter", title=f"{metric} over time")
                st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.subheader("Detected Anomalies")
            st.dataframe(anomalies_df.head(1000), use_container_width=True, height=400)
            st.caption("Tip: Use the download button below to export full anomaly list.")

        st.markdown("### 4) Downloads")
        if allow_download:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button(
                    "⬇️ Cleaned Data (CSV)",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="cleaned_inverter_data.csv",
                    mime="text/csv"
                )
            with c2:
                st.download_button(
                    "⬇️ Inverter Daily Summary (CSV)",
                    data=daily.to_csv(index=False).encode("utf-8"),
                    file_name="inverter_daily_summary.csv",
                    mime="text/csv"
                )
            with c3:
                st.download_button(
                    "⬇️ Anomalies (CSV)",
                    data=anomalies_df.to_csv(index=False).encode("utf-8"),
                    file_name="anomalies.csv",
                    mime="text/csv"
                )

        st.markdown("---")
        st.success("Done. Adjust thresholds on the left to refine results.")
else:
    st.info("Upload a CSV/Excel file to begin. You can use the sample template below.")

# -------------------------------
# Sample template download
# -------------------------------
sample = pd.DataFrame({
    "Timestamp": pd.date_range("2025-01-01 06:00", periods=10, freq="15min"),
    "Inverter": ["INV-1"] * 10,
    "Active Power (kW)": np.linspace(0, 500, 10),
    "Energy (kWh)": np.cumsum(np.linspace(0, 500, 10) * (15/60)),
    "AC Voltage (V)": np.random.normal(400, 5, 10),
    "Grid Frequency (Hz)": np.random.normal(50, 0.1, 10)
})
st.download_button("⬇️ Download Sample Template (CSV)", data=sample.to_csv(index=False).encode("utf-8"),
                   file_name="inverter_sample_template.csv", mime="text/csv")

st.caption("Made by Ruchit ❤️")

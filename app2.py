import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import io
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
import json
import importlib
from fpdf import FPDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

import quant_risk_kit as erk
importlib.reload(erk)

# --- PAGE CONFIGURATION ---
st.markdown(
    '<meta name="color-scheme" content="light dark">',
    unsafe_allow_html=True,
)
st.set_page_config(
    page_title="WIRATOR",
    page_icon="🛡️",
    layout="wide",
)

# --- SESSION STATE NAVIGATION & WORKSPACE MANAGEMENT ---
if "page" not in st.session_state:
    st.session_state["page"] = "landing"
if "exchange" not in st.session_state:
    st.session_state["exchange"] = None
if "use_demo" not in st.session_state:
    st.session_state["use_demo"] = False
if "pdf_data" not in st.session_state:
    st.session_state["pdf_data"] = None
if "batch_audit_log" not in st.session_state:
    st.session_state["batch_audit_log"] = []


def navigate_to(page_name, exchange=None):
    st.session_state["page"] = page_name
    st.session_state["exchange"] = exchange


CHART_CONFIG = {
    "displayModeBar": False,
    "modeBarButtonsToRemove": [
        "2d", "pan2d", "select2d", "lasso2d",
        "In2d", "Out2d", "autoScale2d", "resetScale2d",
    ],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "wirator_risk_audit",
        "height": 600,
        "width": 1000,
        "scale": 2,
    },
    "displaylogo": False,
}


def resolve_yf_tickers(raw_tickers, current_cfg):
    """Resolves exchange suffixes (including EU multi-exchange fallbacks) for Yahoo Finance."""
    all_eu_suffixes = [".AS", ".DE", ".PA", ".MI", ".MC", ".SW", ".ST", ".BR"]
    resolved = []
    for raw_ticker in raw_tickers:
        rt = str(raw_ticker).strip().upper()
        if not rt or rt == "NAN":
            resolved.append("")
            continue
        if current_cfg.get("name") == "STOXX Europe 600" and not any(rt.endswith(sfx) for sfx in all_eu_suffixes):
            found_valid = False
            for sfx in all_eu_suffixes:
                test_symbol = f"{rt}{sfx}"
                try:
                    test_data = yf.download(test_symbol, period="5d", progress=False)
                    if not test_data.empty:
                        resolved.append(test_symbol)
                        found_valid = True
                        break
                except Exception:
                    continue
            if not found_valid:
                resolved.append(f"{rt}{current_cfg.get('suffix', '')}")
        else:
            if any(rt.endswith(sfx) for sfx in all_eu_suffixes) or current_cfg.get("name") != "STOXX Europe 600":
                sfx_to_add = "" if any(rt.endswith(sfx) for sfx in all_eu_suffixes) else current_cfg.get("suffix", "")
                resolved.append(f"{rt}{sfx_to_add}")
            else:
                resolved.append(rt)
    return resolved


def parse_uploaded_portfolio(uploaded_file, selected_workspace, is_demo=False):
    """Parses portfolio CSVs, filters out 'INF' non-equities, matches masked ISINs, and returns holdings + cost flag."""
    try:
        if not is_demo:
            uploaded_file.seek(0)
            bytes_data = uploaded_file.read()
            if not bytes_data:
                return pd.DataFrame(columns=["Ticker", "Quantity", "Cost Price"]), False
            safe_buffer = io.BytesIO(bytes_data)
            raw_df = pd.read_csv(safe_buffer, skipinitialspace=True, on_bad_lines="skip")
            raw_df = raw_df.dropna(how="all").dropna(how="all", axis=1)
            if raw_df.empty:
                return pd.DataFrame(columns=["Ticker", "Quantity", "Cost Price"]), False
        else:
            raw_df = uploaded_file

        raw_df.columns = raw_df.columns.astype(str).str.strip()
        possible_id_cols = ["ISIN No", "ISIN", "ISIN NUMBER", "ISIN Code", "Isin", "ISIN_NO", "CIK"]
        possible_ticker_cols = ["Stock Symbol", "Symbol", "Ticker", "Stock", "Security", "Company", "Scrip"]
        possible_qty_cols = ["Qty", "Quantity", "qty", "Shares", "Volume", "Units", "Holdings"]
        possible_cost_cols = ["Average Cost Price", "Cost Price", "Buy Price", "Average Price", "Avg Cost", "Avg_Cost", "Price", "Cost"]

        id_col = next((col for col in possible_id_cols if col in raw_df.columns), None)
        ticker_col = next((col for col in possible_ticker_cols if col in raw_df.columns), raw_df.columns[0])
        qty_col = next((col for col in possible_qty_cols if col in raw_df.columns), raw_df.columns[1] if len(raw_df.columns) > 1 else raw_df.columns[0])
        cost_col = next((col for col in possible_cost_cols if col in raw_df.columns), None)

        has_valid_cost_col = cost_col is not None

        workspace_json_mapping = {
            "NSE": "nse_isin_map.json",
            "BSE": "bse_isin_map.json",
            "US": "sp500_cik_map.json",
        }

        id_map = {}
        if selected_workspace in workspace_json_mapping:
            json_file = workspace_json_mapping[selected_workspace]
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    id_map = json.load(f)

        parsed_rows = []
        valid_cost_entries = 0
        for _, row in raw_df.iterrows():
            raw_id = str(row[id_col]).strip().upper() if id_col and pd.notna(row[id_col]) else ""
            raw_ticker = str(row[ticker_col]).strip().upper().replace("#", "") if pd.notna(row[ticker_col]) else ""

            if raw_id.startswith("INF") or raw_ticker.startswith("INF"):
                continue

            try:
                qty = float(row[qty_col])
            except (ValueError, TypeError):
                qty = 1.0

            cost_price = np.nan
            if cost_col and pd.notna(row[cost_col]):
                try:
                    val_str = str(row[cost_col]).replace(",", "").replace("₹", "").replace("$", "").strip()
                    parsed_c = float(val_str)
                    if parsed_c > 0:
                        cost_price = parsed_c
                        valid_cost_entries += 1
                except (ValueError, TypeError):
                    cost_price = np.nan

            base_symbol = ""
            matched = False
            if raw_id and id_map:
                clean_id = raw_id.split("X")[0].replace("*", "").strip()
                if raw_id in id_map:
                    base_symbol = id_map[raw_id]
                    matched = True
                else:
                    for full_key, sym in id_map.items():
                        if full_key.startswith(clean_id) and len(clean_id) >= 8:
                            base_symbol = sym
                            matched = True
                            break

            if not matched:
                base_symbol = raw_ticker

            if base_symbol and base_symbol != "NAN":
                parsed_rows.append({"Ticker": base_symbol, "Quantity": int(qty), "Cost Price": cost_price})

        holdings_df = pd.DataFrame(parsed_rows)
        has_cost = bool(has_valid_cost_col and valid_cost_entries > 0)
        if not holdings_df.empty:
            st.success("Portfolio successfully loaded!")
            return holdings_df, has_cost
    except Exception as e:
        st.warning(f"Could not parse custom CSV format: {e}. Using fallback portfolio.")
    return pd.DataFrame(columns=["Ticker", "Quantity", "Cost Price"]), False


def get_default_batch_df(current_cfg):
    """Provides a realistic sample multi-client batch for Model / Client Batch mode."""
    if current_cfg.get("default_curr") == "INR":
        return pd.DataFrame([
            {"Client_ID": "CLT-101", "PAN_Mask": "ABCPK1234M", "Risk_Tier": "Moderate", "Cash": 1500000.0, "Holdings_JSON": json.dumps({"HDFCBANK": 1200, "INFY": 1500, "RELIANCE": 900}), "Mandate_Vol_Cap": 22.0, "Max_Stock_Cap": 15.0, "Rationale": "Balanced core equity exposure aligned with 5Y moderate growth mandate."},
            {"Client_ID": "CLT-102", "PAN_Mask": "DEFPR5678Q", "Risk_Tier": "Conservative", "Cash": 2500000.0, "Holdings_JSON": json.dumps({"HDFCBANK": 800, "ITC": 3000, "SUNPHARMA": 700}), "Mandate_Vol_Cap": 14.0, "Max_Stock_Cap": 12.0, "Rationale": "Defensive cash-flow & pharma tilt with higher cash buffer for capital preservation."},
            {"Client_ID": "CLT-103", "PAN_Mask": "GHIST9012L", "Risk_Tier": "Aggressive", "Cash": 800000.0, "Holdings_JSON": json.dumps({"JSWENERGY": 3000, "TATASTEEL": 4000, "RELIANCE": 1200}), "Mandate_Vol_Cap": 32.0, "Max_Stock_Cap": 20.0, "Rationale": "High-beta cyclical & energy tilt suited for aggressive multi-year horizon."},
            {"Client_ID": "CLT-104", "PAN_Mask": "JKLUV3456N", "Risk_Tier": "Moderate", "Cash": 1000000.0, "Holdings_JSON": json.dumps({"ICICIBANK": 1800, "TCS": 500, "MARUTI": 200}), "Mandate_Vol_Cap": 20.0, "Max_Stock_Cap": 15.0, "Rationale": ""},
        ])
    return pd.DataFrame([
        {"Client_ID": "CLT-101", "PAN_Mask": "XXX-11-1234", "Risk_Tier": "Moderate", "Cash": 25000.0, "Holdings_JSON": json.dumps({"AAPL": 200, "MSFT": 150, "JPM": 180}), "Mandate_Vol_Cap": 22.0, "Max_Stock_Cap": 15.0, "Rationale": "Core large-cap allocation aligned with moderate growth risk profile."},
        {"Client_ID": "CLT-102", "PAN_Mask": "XXX-11-5678", "Risk_Tier": "Conservative", "Cash": 40000.0, "Holdings_JSON": json.dumps({"JNJ": 220, "PG": 250, "UNH": 100}), "Mandate_Vol_Cap": 14.0, "Max_Stock_Cap": 12.0, "Rationale": "Defensive healthcare & staples tilt with elevated cash reserve."},
        {"Client_ID": "CLT-103", "PAN_Mask": "XXX-11-9012", "Risk_Tier": "Aggressive", "Cash": 15000.0, "Holdings_JSON": json.dumps({"NVDA": 400, "META": 200, "CAT": 250}), "Mandate_Vol_Cap": 32.0, "Max_Stock_Cap": 20.0, "Rationale": ""},
    ])


def evaluate_mandate_gate(post_vol_pct, post_cvar_pct, max_stock_wt_pct, max_sector_wt_pct, vol_cap_pct, cvar_cap_pct, stock_cap_pct, sector_cap_pct, rationale_text):
    """Evaluates risk/concentration guardrails and returns status ('Pass', 'Warning', 'Breach') + reasons."""
    breaches = []
    warnings = []

    if post_vol_pct > vol_cap_pct:
        breaches.append(f"Vol {post_vol_pct:.1f}% > Cap {vol_cap_pct:.1f}%")
    elif post_vol_pct > vol_cap_pct * 0.90:
        warnings.append(f"Vol {post_vol_pct:.1f}% near Cap {vol_cap_pct:.1f}%")

    if post_cvar_pct > cvar_cap_pct:
        breaches.append(f"CVaR {post_cvar_pct:.1f}% > Cap {cvar_cap_pct:.1f}%")
    elif post_cvar_pct > cvar_cap_pct * 0.90:
        warnings.append(f"CVaR {post_cvar_pct:.1f}% near Cap {cvar_cap_pct:.1f}%")

    if max_stock_wt_pct > stock_cap_pct:
        breaches.append(f"Max Stock {max_stock_wt_pct:.1f}% > Cap {stock_cap_pct:.1f}%")
    elif max_stock_wt_pct > stock_cap_pct * 0.90:
        warnings.append(f"Max Stock {max_stock_wt_pct:.1f}% near Cap {stock_cap_pct:.1f}%")

    if max_sector_wt_pct > sector_cap_pct:
        breaches.append(f"Max Sector {max_sector_wt_pct:.1f}% > Cap {sector_cap_pct:.1f}%")
    elif max_sector_wt_pct > sector_cap_pct * 0.90:
        warnings.append(f"Max Sector {max_sector_wt_pct:.1f}% near Cap {sector_cap_pct:.1f}%")

    missing_rationale = not (rationale_text and str(rationale_text).strip())

    if breaches:
        status = "Breach"
    elif warnings:
        status = "Warning"
    else:
        status = "Pass"

    return {
        "status": status,
        "breaches": breaches,
        "warnings": warnings,
        "missing_rationale": missing_rationale,
    }


def sortino_ratio(returns_series, rfr=0.055, target_return=0.0, periods_per_year=252):
    """Computes annualized Sortino Ratio directly in app.py."""
    rfr_period = rfr / periods_per_year
    target_daily = (target_return / 100.0) / periods_per_year
    mean_excess_return = returns_series.mean() - rfr_period
    downside_returns = returns_series[returns_series < target_daily]
    downside_std = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(periods_per_year)
    annualized_excess_return = mean_excess_return * periods_per_year
    if isinstance(downside_std, (pd.Series, np.ndarray)):
        downside_std = downside_std.replace(0, np.nan)
    elif downside_std == 0 or np.isnan(downside_std):
        return np.nan
    return annualized_excess_return / downside_std


def compute_quant_metrics(ret_series, rfr_annual=0.065, target_return_pct=6.5, confidence_level=0.95, horizon_days=30, periods_per_year=252):
    """Uses erk.summary_stats, erk.var_historic, erk.cvar_historic, erk.var_gaussian & sortino_ratio."""
    s = pd.Series(ret_series).dropna()
    if s.empty or (s == 0).all():
        return {
            "ann_ret": 0.0, "ann_vol": 0.0, "sharpe": 0.0, "sortino": 0.0,
            "max_dd": 0.0, "hist_var": 0.0, "hist_cvar": 0.0, "cf_var": 0.0,
            "summary_df": pd.DataFrame(),
        }

    # erk.summary_stats for Annualized Return, Annualized Vol, Sharpe Ratio, Max Drawdown, Cornish-Fisher VaR, Historic CVaR
    stats_df = erk.summary_stats(s.to_frame("Portfolio"), riskfree_rate=rfr_annual, periods_per_year=periods_per_year)
    ann_ret = float(stats_df["Annualized Return"].iloc[0] * 100.0)
    ann_vol = float(stats_df["Annualized Vol"].iloc[0] * 100.0)
    sharpe_val = float(stats_df["Sharpe Ratio"].iloc[0])
    max_dd = float(abs(stats_df["Max Drawdown"].iloc[0]) * 100.0)

    # Tail risk at user-selected confidence level using erk functions scaled to horizon_days
    tail_pct = round((1.0 - confidence_level) * 100.0, 4)
    scale_h = np.sqrt(max(int(horizon_days), 1))
    hist_var_1d = float(erk.var_historic(s, level=tail_pct))
    hist_cvar_1d = float(erk.cvar_historic(s, level=tail_pct))
    cf_var_1d = float(erk.var_gaussian(s, level=tail_pct, modified=True))

    hist_var_h = hist_var_1d * scale_h * 100.0
    hist_cvar_h = hist_cvar_1d * scale_h * 100.0
    cf_var_h = cf_var_1d * scale_h * 100.0

    sort_val = sortino_ratio(s, rfr=rfr_annual, target_return=target_return_pct, periods_per_year=periods_per_year)
    sort_val = 0.0 if (sort_val is None or np.isnan(sort_val) or np.isinf(sort_val)) else float(sort_val)

    return {
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe_val if not np.isnan(sharpe_val) else 0.0,
        "sortino": sort_val,
        "max_dd": max_dd,
        "hist_var": hist_var_h,
        "hist_cvar": hist_cvar_h,
        "cf_var": cf_var_h,
        "summary_df": stats_df,
    }


def compute_risk_and_diversification(ret_matrix, weights, asset_labels):
    """Uses erk.risk_contribution and erk.enc for both Weight Diversification and Risk Diversification."""
    w = np.asarray(weights, dtype=float)
    n_assets = len(asset_labels)
    if n_assets == 0 or w.sum() <= 0 or ret_matrix.empty:
        zeros = np.zeros(n_assets)
        return {
            "weights_pct": zeros,
            "risk_contrib_pct": zeros,
            "weight_enc": 0.0,
            "risk_enc": 0.0,
            "table_df": pd.DataFrame({"Ticker": asset_labels, "Weight (%)": zeros, "Risk Contribution (%)": zeros}),
        }

    w_norm = w / w.sum()
    covmat = ret_matrix.cov() * 252.0
    # Regularize slightly if singular so portfolio_vola / risk_contribution never divide by zero
    cov_reg = covmat + np.eye(n_assets) * 1e-8

    rc_series = erk.risk_contribution(pd.Series(w_norm, index=cov_reg.index), cov_reg)
    rc_arr = np.asarray(rc_series, dtype=float)
    rc_sum = rc_arr.sum()
    rc_norm = (rc_arr / rc_sum) if rc_sum > 0 else w_norm

    weight_enc_val = float(erk.enc(pd.Series(w_norm)))
    risk_enc_val = float(erk.enc(pd.Series(np.maximum(rc_norm, 0.0))))

    table_df = pd.DataFrame({
        "Ticker": asset_labels,
        "Weight (%)": w_norm * 100.0,
        "Risk Contribution (%)": rc_norm * 100.0,
    })

    return {
        "weights_pct": w_norm * 100.0,
        "risk_contrib_pct": rc_norm * 100.0,
        "weight_enc": weight_enc_val,
        "risk_enc": risk_enc_val,
        "table_df": table_df,
    }


def run_cholesky_monte_carlo(price_matrix, weights, initial_val, horizon_days=30, n_sims=500):
    """Runs Cholesky Monte Carlo using erk.cholesky_returns(prices, num_steps=..., return_prices=False)."""
    w = np.asarray(weights, dtype=float)
    n_cols = price_matrix.shape[1]
    if n_cols == 0 or initial_val <= 0:
        arr = np.full((max(int(horizon_days), 1) + 1, 10), float(initial_val))
        return {
            "paths": arr, "p5": arr[:, 0], "p50": arr[:, 0], "p95": arr[:, 0],
            "mean_end": float(initial_val), "median_end": float(initial_val),
            "p5_end": float(initial_val), "p95_end": float(initial_val), "prob_loss": 0.0,
        }

    if w.sum() > 0:
        w = w / w.sum()
    else:
        w = np.ones(n_cols) / n_cols

    steps = max(int(horizon_days), 1)
    sims = max(int(n_sims), 50)
    # erk.cholesky_returns calls pct_change().dropna() on simulated prices, so request steps + 1 to get `steps` return rows
    sim_steps = steps + 1

    # Add tiny jitter to prices if any column is completely flat to ensure positive-definite covariance for Cholesky
    clean_prices = price_matrix.copy()
    for col in clean_prices.columns:
        if clean_prices[col].pct_change().dropna().std() == 0:
            clean_prices[col] = clean_prices[col] * (1.0 + np.linspace(0, 1e-5, len(clean_prices)))

    all_paths = np.zeros((steps + 1, sims), dtype=float)
    all_paths[0, :] = float(initial_val)

    for s_idx in range(sims):
        try:
            # Explicitly pass return_prices=False so erk.cholesky_returns returns simulated returns
            sim_rets_df = erk.cholesky_returns(clean_prices, num_steps=sim_steps, return_prices=False)
            port_step_rets = sim_rets_df.values @ w
            if len(port_step_rets) < steps:
                port_step_rets = np.pad(port_step_rets, (0, steps - len(port_step_rets)), constant_values=0.0)
            else:
                port_step_rets = port_step_rets[:steps]
            all_paths[1:, s_idx] = float(initial_val) * np.cumprod(1.0 + port_step_rets)
        except Exception:
            all_paths[1:, s_idx] = float(initial_val)

    terminal_vals = all_paths[-1, :]
    terminal_rets_pct = ((terminal_vals - float(initial_val)) / max(float(initial_val), 1.0)) * 100.0
    return {
        "paths": all_paths,
        "p5": np.percentile(all_paths, (1-confidence_level)*100, axis=1),
        "p50": np.percentile(all_paths, 50, axis=1),
        "p95": np.percentile(all_paths, confidence_level*100, axis=1),
        "mean_end": float(np.mean(terminal_vals)),
        "median_end": float(np.median(terminal_vals)),
        "p5_end": float(np.percentile(terminal_vals, (1-confidence_level)*100)),
        "p95_end": float(np.percentile(terminal_vals, confidence_level*100)),
        "prob_loss": float(np.mean(terminal_rets_pct < 0.0) * 100.0),
    }


def render_risk_diversification_and_mc_section(
    prefix_key,
    currency,
    time_horizon_days,
    confidence_level,
    pre_metrics,
    post_metrics,
    pre_div,
    post_div,
    pre_cum_pct,
    post_cum_pct,
    b_cum_pct,
    benchmark_name,
    price_matrix_valid,
    prop_wts,
    post_nav,
    turnover_pct,
    total_friction_amt,
    beta_val,
    pre_max_stock,
    post_max_stock,
    pre_max_sec,
    post_max_sec,
):
    """Renders identical comprehensive Pre vs Post metrics, Benchmark Graph, Weight/Risk Diversification & Monte Carlo for both Single Client & Batch."""
    # Row 1: Mandate & Tail Risk KPIs
    pm1, pm2, pm3, pm4, pm5 = st.columns(5)
    pm1.metric("Annualized Vol", f"{post_metrics['ann_vol']:.2f}%", delta=f"{post_metrics['ann_vol'] - pre_metrics['ann_vol']:+.2f}% vs pre", delta_color="inverse")
    pm2.metric(f"{time_horizon_days}D Hist CVaR ({int(confidence_level*100)}%)", f"{post_metrics['hist_cvar']:.2f}%", delta=f"{post_metrics['hist_cvar'] - pre_metrics['hist_cvar']:+.2f}% vs pre", delta_color="inverse")
    pm3.metric(f"{time_horizon_days}D Hist VaR ({int(confidence_level*100)}%)", f"{post_metrics['hist_var']:.2f}%", delta=f"{post_metrics['hist_var'] - pre_metrics['hist_var']:+.2f}% vs pre", delta_color="inverse")
    pm4.metric(f"{time_horizon_days}D Mod. CF VaR", f"{post_metrics['cf_var']:.2f}%", delta=f"{post_metrics['cf_var'] - pre_metrics['cf_var']:+.2f}% vs pre", delta_color="inverse")
    pm5.metric("Turnover / Friction", f"{turnover_pct:.2f}%", delta=f"{currency} {total_friction_amt:,.0f} (Beta {beta_val:.2f})", delta_color="off")

    # Row 2: Return, Risk-Adjusted Ratios, Drawdown & Concentration
    qm1, qm2, qm3, qm4, qm5 = st.columns(5)
    qm1.metric("Ann. Return (Post)", f"{post_metrics['ann_ret']:+.2f}%", delta=f"{post_metrics['ann_ret'] - pre_metrics['ann_ret']:+.2f}% vs pre")
    qm2.metric("Sharpe Ratio (erk)", f"{post_metrics['sharpe']:.2f}", delta=f"{post_metrics['sharpe'] - pre_metrics['sharpe']:+.2f} vs pre")
    qm3.metric("Sortino Ratio", f"{post_metrics['sortino']:.2f}", delta=f"{post_metrics['sortino'] - pre_metrics['sortino']:+.2f} vs pre")
    qm4.metric("Max Drawdown", f"{post_metrics['max_dd']:.2f}%", delta=f"{post_metrics['max_dd'] - pre_metrics['max_dd']:+.2f}% vs pre", delta_color="inverse")
    qm5.metric("Max Stock / Sector", f"{post_max_stock:.1f}% / {post_max_sec:.1f}%", delta=f"Stk {post_max_stock - pre_max_stock:+.1f}% | Sec {post_max_sec - pre_max_sec:+.1f}%", delta_color="off")

    # Row 3: Diversification (ENC on Weights & Risk Contributions via erk.enc)
    dv1, dv2, dv3, dv4 = st.columns(4)
    dv1.metric("Weight Diversification (ENC)", f"{post_div['weight_enc']:.1f}%", delta=f"{post_div['weight_enc'] - pre_div['weight_enc']:+.1f}% vs pre")
    dv2.metric("Risk Diversification (ENC)", f"{post_div['risk_enc']:.1f}%", delta=f"{post_div['risk_enc'] - pre_div['risk_enc']:+.1f}% vs pre")
    dv3.metric("Pre-Trade Weight ENC", f"{pre_div['weight_enc']:.1f}%")
    dv4.metric("Pre-Trade Risk ENC", f"{pre_div['risk_enc']:.1f}%")

    # Summary Stats Table from erk.summary_stats (Pre vs Post)
    if not pre_metrics["summary_df"].empty and not post_metrics["summary_df"].empty:
        with st.expander("📋 View Full Summary Table (Pre-Trade vs. Post-Trade)", expanded=False):
            comb_stats = pd.concat([
                pre_metrics["summary_df"].rename(index={"Portfolio": "Pre-Trade Portfolio"}),
                post_metrics["summary_df"].rename(index={"Portfolio": "Post-Trade Portfolio"}),
            ])
            st.dataframe(comb_stats, use_container_width=True)

    # Weight Distribution vs Risk Contribution Graph & Table (erk.risk_contribution)
    st.markdown("### ⚖️ Weight Distribution vs. Risk Contribution")
    rc_col1, rc_col2 = st.columns([1.8, 1.2])
    with rc_col1:
        rc_df = post_div["table_df"]
        fig_rc = go.Figure()
        fig_rc.add_trace(go.Bar(
            x=rc_df["Ticker"],
            y=rc_df["Weight (%)"],
            name="Portfolio Weight (%)",
            marker_color="#2563EB",
        ))
        fig_rc.add_trace(go.Bar(
            x=rc_df["Ticker"],
            y=rc_df["Risk Contribution (%)"],
            name="Risk Contribution (%)",
            marker_color="#F59E0B",
        ))
        fig_rc.update_layout(
            barmode="group",
            height=300,
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis_title="Share (%)",
            xaxis_title="Holding",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_rc, use_container_width=True, config=CHART_CONFIG)
    with rc_col2:
        st.dataframe(
            post_div["table_df"].style.format({
                "Weight (%)": "{:.2f}%",
                "Risk Contribution (%)": "{:.2f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # Pre-Trade vs Post-Trade vs Market Benchmark Graph
    st.markdown("### 📈 Portfolio vs. Market Benchmark (Pre-Trade vs. Post-Trade)")
    fig_bm = go.Figure()
    fig_bm.add_trace(go.Scatter(
        x=pre_cum_pct.index, y=pre_cum_pct.values,
        mode="lines", name="Current Portfolio (Pre-Trade)",
        line=dict(color="#2563EB", width=2)
    ))
    fig_bm.add_trace(go.Scatter(
        x=post_cum_pct.index, y=post_cum_pct.values,
        mode="lines", name="Proposed Portfolio (Post-Trade)",
        line=dict(color="#16A34A", width=2.5)
    ))
    if b_cum_pct is not None and not b_cum_pct.empty:
        fig_bm.add_trace(go.Scatter(
            x=b_cum_pct.index, y=b_cum_pct.values,
            mode="lines", name=f"Benchmark ({benchmark_name})",
            line=dict(color="#DC2626", width=2, dash="dash")
        ))
    fig_bm.update_layout(
        height=310,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis_title="Cumulative Return (%)",
        xaxis_title="Date",
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_bm, use_container_width=True, config=CHART_CONFIG)

    # Cholesky Monte Carlo Simulation (via erk.cholesky_returns(..., return_prices=False))
    st.markdown(f"### 🎲 Cholesky Monte Carlo Simulation ({time_horizon_days}D Horizon)")
    mc_c1, mc_c2 = st.columns([1.2, 2.8])
    with mc_c1:
        mc_sims = st.select_slider("Monte Carlo Paths", options=[200, 500, 1000, 2000], value=500, key=f"{prefix_key}_mc_sims")
        mc_res = run_cholesky_monte_carlo(
            price_matrix=price_matrix_valid,
            weights=prop_wts,
            initial_val=post_nav,
            horizon_days=int(time_horizon_days),
            n_sims=int(mc_sims),
        )
        st.metric("Median Projected Value", f"{currency} {mc_res['median_end']:,.0f}")
        st.metric("Tail Value", f"{currency} {mc_res['p5_end']:,.0f}", delta=f"Prob. Loss: {mc_res['prob_loss']:.1f}%", delta_color="inverse")
        st.metric("Upside Value", f"{currency} {mc_res['p95_end']:,.0f}")
    with mc_c2:
        day_axis = np.arange(0, mc_res["paths"].shape[0])
        fig_mc = go.Figure()
        # Plot sample individual simulated paths faintly for visual clarity
        sample_count = min(25, mc_res["paths"].shape[1])
        for p_idx in range(sample_count):
            fig_mc.add_trace(go.Scatter(
                x=day_axis,
                y=mc_res["paths"][:, p_idx],
                mode="lines",
                line=dict(color="rgba(148, 163, 184, 0.25)", width=1),
                showlegend=False,
                hoverinfo="skip",
            ))
        fig_mc.add_trace(go.Scatter(x=day_axis, y=mc_res["p95"], mode="lines", name="Upside", line=dict(color="#16A34A", dash="dot", width=2)))
        fig_mc.add_trace(go.Scatter(x=day_axis, y=mc_res["p50"], mode="lines", name="Median (50th Pct)", line=dict(color="#1A2B4C", width=2.5)))
        fig_mc.add_trace(go.Scatter(x=day_axis, y=mc_res["p5"], mode="lines", name="Downside", line=dict(color="#DC2626", dash="dash", width=2)))
        fig_mc.update_layout(
            height=290,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Trading Days Ahead",
            yaxis_title=f"Portfolio Value ({currency})",
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_mc, use_container_width=True, config=CHART_CONFIG)


def _fpdf_clean(val):
    """Sanitizes strings for FPDF core fonts (latin-1 safe without Unicode crashes)."""
    s = str(val if val is not None else "")
    replacements = {
        "\u2014": "-",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "*",
        "\u20b9": "INR ",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s.encode("latin-1", errors="replace").decode("latin-1")


def build_audit_packet_pdf(entity_type, workspace_mode, exchange, currency, client_label, reg_no, officer_or_advisor, mandate_limits, diff_summary, gate_result, rationale_text, timestamp):
    """Generates a timestamped Audit & Suitability Report PDF for Single Client or Batch."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, _fpdf_clean("WIRATOR - AUDIT & SUITABILITY REPORT"), ln=True, align="C")
    pdf.set_font("Arial", "I", 8.5)
    pdf.cell(0, 6, _fpdf_clean(f"Entity: {entity_type} | Workspace: {workspace_mode} | Exchange: {exchange} | {timestamp}"), ln=True, align="C")
    pdf.ln(6)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, _fpdf_clean("1. Mandate, Entity & Client Context"), ln=True)
    pdf.set_font("Arial", "", 9.5)
    pdf.cell(0, 5.5, _fpdf_clean(f"Entity Type: {entity_type} | Reg No.: {reg_no} | Signatory: {officer_or_advisor}"), ln=True)
    pdf.cell(0, 5.5, _fpdf_clean(f"Client / Batch Scope: {client_label} | Currency: {currency}"), ln=True)
    pdf.cell(0, 5.5, _fpdf_clean(f"Limits -> Vol Cap: {mandate_limits['vol_cap']:.1f}% | CVaR Cap: {mandate_limits['cvar_cap']:.1f}% | Stock Cap: {mandate_limits['stock_cap']:.1f}% | Sector Cap: {mandate_limits['sector_cap']:.1f}%"), ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, _fpdf_clean("2. Pre-Trade Diff & Mandate Gate Result"), ln=True)
    pdf.set_font("Arial", "", 9.5)
    pdf.cell(0, 5.5, _fpdf_clean(f"Gate Status: {gate_result['status'].upper()} | Missing Rationale: {'Yes' if gate_result['missing_rationale'] else 'No'}"), ln=True)
    pdf.cell(0, 5.5, _fpdf_clean(f"Turnover: {diff_summary.get('turnover_pct', 0.0):.2f}% | Est. Friction: {currency} {diff_summary.get('friction_cost', 0.0):,.2f}"), ln=True)
    pdf.cell(0, 5.5, _fpdf_clean(f"Volatility: {diff_summary.get('pre_vol', 0.0):.2f}% -> {diff_summary.get('post_vol', 0.0):.2f}% | CVaR: {diff_summary.get('pre_cvar', 0.0):.2f}% -> {diff_summary.get('post_cvar', 0.0):.2f}%"), ln=True)
    pdf.cell(0, 5.5, _fpdf_clean(f"Max Stock Wt: {diff_summary.get('pre_max_stock', 0.0):.2f}% -> {diff_summary.get('post_max_stock', 0.0):.2f}% | Max Sector Wt: {diff_summary.get('pre_max_sector', 0.0):.2f}% -> {diff_summary.get('post_max_sector', 0.0):.2f}%"), ln=True)
    if gate_result["breaches"]:
        pdf.multi_cell(0, 5, _fpdf_clean(f"Breaches: {'; '.join(gate_result['breaches'])}"))
    if gate_result["warnings"]:
        pdf.multi_cell(0, 5, _fpdf_clean(f"Warnings: {'; '.join(gate_result['warnings'])}"))
    pdf.ln(4)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, _fpdf_clean("3. Investment & Suitability Rationale"), ln=True)
    pdf.set_font("Arial", "", 9.5)
    rat_clean = rationale_text.strip() if rationale_text and rationale_text.strip() else "[No suitability/investment rationale logged]"
    pdf.multi_cell(0, 5.5, _fpdf_clean(rat_clean))
    pdf.ln(4)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, _fpdf_clean("4. Regulatory & Fiduciary Attestation"), ln=True)
    pdf.set_font("Arial", "", 8.5)
    pdf.multi_cell(0, 5, _fpdf_clean(
        "Generated via WIRATOR Risk & Compliance Engine for SEBI RIA / Boutique PMS governance and pre-trade suitability review. "
        "Escalation channels via SEBI SCORES and SMART ODR apply where applicable."
    ))
    raw_out = pdf.output(dest="S")
    if isinstance(raw_out, bytes):
        return raw_out
    return raw_out.encode("latin-1", errors="replace")


# ==============================================================================
# 1. LANDING PAGE
# ==============================================================================
if st.session_state["page"] == "landing":
    st.markdown(
        """
        <style>
        p, span, div, label { word-break: break-word; }
        input, textarea, select, div[data-baseweb="select"] > div {
            color: #111827 !important;
            background-color: #ffffff !important;
            -webkit-text-fill-color: #111827 !important;
        }
        li[role="option"] { color: #111827 !important; background-color: #ffffff !important; }
        @media (max-width: 768px) {
            .stButton button { width: 100% !important; }
            [data-testid="column"] { width: 100% !important; flex: 100% !important; min-width: 100% !important; }
            .hero-title { font-size: 2.5rem !important; }
            .hero-subtitle { font-size: 1.1rem !important; white-space: normal !important; letter-spacing: 1px !important; }
            .slogan-badge { font-size: 1.8rem !important; white-space: normal !important; letter-spacing: 1px !important; }
            .hero-description { font-size: 1.1rem !important; line-height: 1.5 !important; padding: 0 0.5rem; }
            .block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; padding-left: 0.8rem !important; padding-right: 0.8rem !important; }
        }
        .stApp {
            background-color: #1A2B4C;
            color: #FDFBD4;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        .block-container { padding-top: 3rem; padding-bottom: 5rem; max-width: 1000px; }
        .hero-container { text-align: center; padding: 2rem 1rem 1.5rem 1rem; }
        .hero-title { color: #FDFBD4; font-size: 3.8rem !important; font-weight: 600; letter-spacing: -1px; margin-bottom: 0.2rem; }
        .hero-subtitle { color: #FDFBD4; font-size: 2.4rem; font-weight: 500; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 1.2rem; opacity: 0.9; white-space: nowrap; display: flex; justify-content: center; align-items: center; text-align: center; width: 100%; }
        .slogan-badge { color: #FDFBD4; font-size: 4.25rem; font-weight: 800; letter-spacing: 3px; margin-bottom: 2rem; text-transform: uppercase; white-space: nowrap; }
        .hero-description { color: #E2E8F0; font-size: 2.2rem; line-height: 1.8; max-width: 800px; margin: 0 auto 2.5rem auto; text-align: center; }
        .workspace-header { text-align: center; color: #FFFFFF; font-size: 2rem; font-weight: 700; margin-top: 3rem; margin-bottom: 1.8rem; }
        div.stButton > button {
            width: 100%;
            background-color: #1E2E45 !important;
            color: #FDFBD4 !important;
            border: 2px solid #FDFBD4 !important;
            padding: 1.2rem 1rem !important;
            font-size: 1.15rem !important;
            font-weight: 700 !important;
            border-radius: 12px !important;
            transition: all 0.3s ease-in-out !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
        .slogan-badge {
            color: #FDFBD4;
            font-size: 3.0 rem;
            font-weight: 800;
            letter-spacing: 2px;
            margin: 0 auto 1.5rem auto;
            text-transform: uppercase;
            text-align: center;
            display: block;
            width: 100%;
            white-space: normal;
        }
        div.stButton > button:hover {
            background-color: #FDFBD4 !important;
            color: #0B192C !important;
            border-color: #FFFFFF !important;
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(253, 251, 212, 0.25);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero-container">
            <div style="font-size: 4rem; margin-bottom: 0.5rem;">🛡️</div>
            <h1 class="hero-title">WIRATOR</h1>
            <div class="hero-subtitle">Institutional Portfolio Risk Audit & Compliance Engine</div>
            <div class="slogan-badge">Sharper Risk. Stronger Proof.</div>
            <p class="hero-description">
                Turn every rebalance into a clearer and more defensible decision.
                See where real risk sits, test <b>forward</b> outcomes before committing capital.
                Leave with timestamped suitability records that back up your judgment across single clients and batches.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="workspace-header">Select Stock Exchange Workspace</div>', unsafe_allow_html=True)

    r1_col1, r1_col2 = st.columns(2, gap="medium")
    with r1_col1:
        if st.button("🇮🇳 National Stock Exchange (NSE)", use_container_width=True):
            navigate_to("analytics", exchange="NSE")
            st.rerun()
    with r1_col2:
        if st.button("🇮🇳 Bombay Stock Exchange (BSE)", use_container_width=True):
            navigate_to("analytics", exchange="BSE")
            st.rerun()
    # with r1_col3:
    #     if st.button("🇺🇸 US Markets (S&P 500 / Nasdaq)", use_container_width=True):
    #         navigate_to("analytics", exchange="US")
    #         st.rerun()

    # r2_col1, r2_col2, r2_col3 = st.columns(3, gap="medium")
    # with r2_col1:
    #     if st.button("🇬🇧 London Stock Exchange (FTSE 100)", use_container_width=True):
    #         navigate_to("analytics", exchange="UK")
    #         st.rerun()
    # with r2_col2:
    #     if st.button("🇪🇺 European Markets (STOXX 600)", use_container_width=True):
    #         navigate_to("analytics", exchange="EU")
    #         st.rerun()
    # with r2_col3:
    #     if st.button("🇿🇦 Johannesburg Stock Exchange (JSE)", use_container_width=True):
    #         navigate_to("analytics", exchange="JSE")
    #         st.rerun()


# ==============================================================================
# 2. ANALYTICS, PRE-TRADE MANDATE GATE & AUDIT WORKSPACE
# ==============================================================================
elif st.session_state["page"] == "analytics":
    exchange = st.session_state["exchange"]

    st.markdown(
        """
        <style>
        html { zoom: 98% !important; }
        .stApp { background-color: #FAF8ED !important; color: #1A2B4C !important; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        h1, h2, h3, h4, h5, h6 { color: #1A2B4C !important; font-weight: 700 !important; }
        .stMarkdown p:not([data-testid="stMetricValue"] p), .stCaption { color: #1A2B4C !important; font-weight: 500; }

        section[data-testid="stSidebar"] { background-color: #1A2B4C !important; border-right: 1px solid rgba(250, 248, 237, 0.1) !important; }
        section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] .stMarkdown p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
            color: #FAFAED !important; -webkit-text-fill-color: #FAFAED !important; font-weight: 500;
        }
        section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] textarea, section[data-testid="stSidebar"] div[data-baseweb="select"] span {
            color: #1A2B4C !important; -webkit-text-fill-color: #1A2B4C !important; font-weight: 600 !important;
        }
        .stSidebar [data-testid="stNumberInput"] button { background-color: #FAFAED !important; border: 1px solid #1A2B4C !important; }
        .stSidebar [data-testid="stNumberInput"] button svg { fill: #1A2B4C !important; }
        .stSidebar [data-testid="stNumberInput"] button:hover { background-color: #1A2B4C !important; }
        .stSidebar [data-testid="stNumberInput"] button:hover svg { fill: #FAFAED !important; }

        div[data-testid="stExpander"] { background-color: #1A2B4C !important; border-radius: 12px !important; border: 1px solid rgba(250, 248, 237, 0.15) !important; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); }
        div[data-testid="stExpander"] summary, div[data-testid="stExpander"] summary *, div[data-testid="stExpander"] summary span, div[data-testid="stExpander"] summary p {
            color: #FAFAED !important; -webkit-text-fill-color: #FAFAED !important; font-weight: 600 !important; opacity: 1 !important;
        }
        div[data-testid="stExpanderDetails"] { background-color: #1A2B4C !important; border-bottom-left-radius: 12px !important; border-bottom-right-radius: 12px !important; padding: 16px !important; }
        div[data-testid="stExpanderDetails"] *, div[data-testid="stExpanderDetails"] p, div[data-testid="stExpanderDetails"] span, div[data-testid="stExpanderDetails"] label, div[data-testid="stExpanderDetails"] div {
            color: #FAFAED !important; -webkit-text-fill-color: #FAFAED !important;
        }
        div[data-testid="stExpanderDetails"] input, div[data-testid="stExpanderDetails"] textarea, div[data-testid="stExpanderDetails"] div[data-baseweb="input"] input {
            color: #1A2B4C !important; -webkit-text-fill-color: #1A2B4C !important; background-color: #FAFAED !important; border-radius: 6px !important;
        }
        div[data-testid="stExpanderDetails"] [data-testid="stDownloadButton"] button {
            background-color: #FAFAED !important; color: #1A2B4C !important; -webkit-text-fill-color: #1A2B4C !important; border: 1px solid #1A2B4C !important; border-radius: 8px !important; font-weight: 600 !important;
        }
        div[data-testid="stExpanderDetails"] [data-testid="stDownloadButton"] button p, div[data-testid="stExpanderDetails"] [data-testid="stDownloadButton"] button span {
            color: #1A2B4C !important; -webkit-text-fill-color: #1A2B4C !important;
        }

        div[data-testid="stMetric"] {
            background-color: #1A2B4C !important; border-radius: 12px !important; padding: 12px !important;
            border: 1px solid rgba(250, 248, 237, 0.15) !important; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); text-align: center !important;
        }
        div[data-testid="stMetric"] > div { display: flex !important; flex-direction: column !important; align-items: center !important; }
        div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricValue"], div[data-testid="stMetric"] [data-testid="stMetricValue"] div, div[data-testid="stMetric"] [data-testid="stMetricValue"] span, div[data-testid="stMetric"] span, div[data-testid="stMetric"] p {
            color: #FAFAED !important; -webkit-text-fill-color: #FAFAED !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricDelta"], div[data-testid="stMetric"] div[data-testid="stMetricDelta"] * {
            background-color: #FAFAED !important; color: #1A2B4C !important; -webkit-text-fill-color: #1A2B4C !important; border-radius: 999px !important; padding: 2px 10px !important; font-weight: 700 !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricDelta"] svg { fill: #1A2B4C !important; stroke: #1A2B4C !important; }

        div.stRadio div[role="radiogroup"] label, div.stRadio div[role="radiogroup"] label p, div.stRadio span {
            color: #1A2B4C !important; -webkit-text-fill-color: #1A2B4C !important; font-weight: 500;
        }
        div.stTextArea textarea, div.stTextInput input { color: #1A2B4C !important; -webkit-text-fill-color: #1A2B4C !important; }

        div.stButton > button, div.stDownloadButton > button, section[data-testid="stSidebar"] div.stButton > button, div.stFormSubmitButton button {
            background-color: #FAFAED !important; color: #1A2B4C !important; -webkit-text-fill-color: #1A2B4C !important;
            border: 1px solid #1A2B4C !important; border-radius: 8px !important; font-weight: 700 !important;
        }
        div.stButton > button[kind="primary"], div.stDownloadButton > button[kind="primary"] {
            color: #FAFAED !important; -webkit-text-fill-color: #FAFAED !important; background-color: #1A2B4C !important;
        }
        div.stButton > button:hover, div.stFormSubmitButton button:hover {
            background-color: #1A2B4C !important; color: #FAFAED !important; -webkit-text-fill-color: #FAFAED !important; border-color: #FAFAED !important;
        }

        div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {
            background-color: #1A2B4C !important; border-radius: 8px !important; border: 1px solid rgba(250, 248, 237, 0.12) !important; width: 100% !important; overflow: visible !important;
        }
        div[data-testid="stDataFrame"] *, div[data-testid="stDataEditor"] * { color: #FAF8ED !important; }

        div[data-testid="stFileUploader"] { background-color: #FAFAED !important; border: 1px solid #1A2B4C !important; border-radius: 8px !important; padding: 8px !important; }
        div[data-testid="stFileUploader"] small, div[data-testid="stFileUploader"] span { color: #1A2B4C !important; -webkit-text-fill-color: #1A2B4C !important; }

        body, .stApp, div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] { background-color: #FAFAED !important; color: #1A2B4C !important; }
        div[data-baseweb="popover"] *, div[data-baseweb="menu"] *, ul[role="listbox"] * { color: #1A2B4C !important; -webkit-text-fill-color: #1A2B4C !important; }
        li[role="option"]:hover, div[role="option"]:hover { background-color: #1A2B4C !important; color: #FAFAED !important; -webkit-text-fill-color: #FAFAED !important; }

        .gate-strip {
            background-color: #1A2B4C;
            color: #FAFAED;
            border-radius: 10px;
            padding: 12px 16px;
            border-left: 6px solid #38BDF8;
            margin: 10px 0 16px 0;
        }
        .gate-pass { border-left-color: #22C55E !important; }
        .gate-warn { border-left-color: #F59E0B !important; }
        .gate-breach { border-left-color: #EF4444 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Top Navigation Header
    top_col1, top_col2 = st.columns([6, 1])
    with top_col1:
        st.title(f"📈 {exchange} Workspace — Portfolio Risk & Mandate Gate Engine")
    with top_col2:
        if st.button("🏠 Home"):
            navigate_to("landing")
            st.rerun()

    exchange_config = {
        "NSE": {"symbol": "^NSEI", "fallback": "NIFTYBEES.NS", "name": "Nifty 50 Index", "suffix": ".NS", "default_curr": "INR"},
        "BSE": {"symbol": "^BSESN", "fallback": "SETF.NS", "name": "SENSEX Index", "suffix": ".NS", "default_curr": "INR"},
        "JSE": {"symbol": "^JN00.JO", "fallback": "STX40.JO", "name": "JSE Top 40 Index", "suffix": ".JO", "default_curr": "ZAR"},
        "US":  {"symbol": "^GSPC", "fallback": "SPY", "name": "S&P 500 Index", "suffix": "", "default_curr": "USD"},
        "UK":  {"symbol": "^FTSE", "fallback": "ISF.L", "name": "FTSE 100 Index", "suffix": ".L", "default_curr": "GBP"},
        "EU":  {"symbol": "^STOXX", "fallback": "EXS1.DE", "name": "STOXX Europe 600", "suffix": ".DE", "default_curr": "EUR"},
    }
    current_cfg = exchange_config.get(exchange, exchange_config["NSE"])

    # ALL_EXCHANGE_CONFIGS = {
    #     "NSE": {
    #         "Nifty IT": {"indices": ["^CNXIT"], "stocks": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS", "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "LTTS.NS"]},
    #         "Nifty Bank": {"indices": ["^NSEBANK"], "stocks": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", "BANKBARODA.NS", "PNB.NS", "INDUSINDBK.NS", "AUBANK.NS", "IDFCFIRSTB.NS"]},
    #         "Nifty Auto": {"indices": ["^CNXAUTO"], "stocks": ["TATAMOTORS.NS", "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "TVSMOTOR.NS", "BHARATFORG.NS", "BOSCHLTD.NS", "MRF.NS"]},
    #         "Nifty FMCG": {"indices": ["^CNXFMCG"], "stocks": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS", "DABUR.NS", "GODREJCP.NS", "MARICO.NS", "COLPAL.NS", "VBL.NS"]},
    #         "Nifty Energy": {"indices": ["^CNXENERGY"], "stocks": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "NTPC.NS", "POWERGRID.NS", "IOC.NS", "GAIL.NS", "ADANIGREEN.NS", "TATAPOWER.NS", "COALINDIA.NS"]},
    #         "Nifty Pharma": {"indices": ["^CNXPHARMA"], "stocks": ["SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS", "LUPIN.NS", "APOLLOHOSP.NS", "BIOCON.NS", "TORNTPHARM.NS", "ZYDUSLIFE.NS", "MANKIND.NS"]},
    #     },
    #     "BSE": {
    #         "BSE IT": {"indices": ["^BSESIT.BO"], "stocks": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS", "PERSISTENT.NS", "COFORGE.NS"]},
    #         "BSE Bankex": {"indices": ["^BSEBANK.NS"], "stocks": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", "BANKBARODA.NS", "PNB.NS"]},
    #         "BSE Healthcare": {"indices": ["^BSEHC.NS"], "stocks": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS", "LUPIN.NS"]},
    #         "BSE Auto": {"indices": ["^BSEAUTO.NS"], "stocks": ["TATAMOTORS.NS", "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS"]},
    #         "BSE FMCG": {"indices": ["^BSEFMCG.NS"], "stocks": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS"]},
    #         "BSE Metal": {"indices": ["^BSEMT.NS"], "stocks": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS"]},
    #         "BSE Oil & Gas": {"indices": ["^BSEOIL.NS"], "stocks": ["RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS"]},
    #         "BSE Power & Utilities": {"indices": ["^BSEPOWER.NS"], "stocks": ["NTPC.NS", "POWERGRID.NS", "ADANIPOWER.NS", "TATAPOWER.NS"]},
    #         "BSE Realty": {"indices": ["^BSERLTY.NS"], "stocks": ["DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PRESTIGE.NS"]},
    #         "BSE Capital Goods & Industrials": {"indices": ["^BSECG.NS"], "stocks": ["LT.NS", "BEL.NS", "SIEMENS.NS", "ABB.NS"]},
    #     },
    #     "US": {
    #         "Technology": {"indices": ["^NDX", "^GSPC"], "stocks": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "AMD", "ORCL", "CRM", "ADBE"]},
    #         "Financials": {"indices": ["^IXFN", "^GSPC"], "stocks": ["JPM", "BAC", "WFC", "MS", "GS", "BLK", "C", "AXP", "SPGI", "SCHW", "V", "MA"]},
    #         "Healthcare": {"indices": ["^IXHC", "^GSPC"], "stocks": ["UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "AMGN", "DHR", "ISRG", "BMY"]},
    #         "Consumer Discretionary & Retail": {"indices": ["^GSPC"], "stocks": ["HD", "MCD", "NKE", "SBUX", "TJX", "LOW", "BKNG", "TGT", "CMG"]},
    #         "Consumer Staples": {"indices": ["^GSPC"], "stocks": ["WMT", "PG", "KO", "PEP", "COST", "PM", "MO", "CL"]},
    #         "Communication Services & Media": {"indices": ["^GSPC"], "stocks": ["NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS", "EA"]},
    #         "Energy & Industrials": {"indices": ["^IXID", "^GSPC"], "stocks": ["XOM", "CVX", "GE", "CAT", "UNP", "RTX", "HON", "BA", "LMT", "DE", "COP", "SLB"]},
    #         "Utilities & Real Estate": {"indices": ["^GSPC"], "stocks": ["NEE", "DUK", "SO", "PLD", "AMT", "EQIX", "SPG"]},
    #     },
    #     "UK": {
    #         "Energy & Commodities": {"indices": ["^FTSE"], "stocks": ["SHEL.L", "BP.L", "GLEN.L", "RIO.L", "AAL.L", "ANTO.L"]},
    #         "Banking & Finance": {"indices": ["^FTSE"], "stocks": ["HSBA.L", "LLOY.L", "BARC.L", "NWG.L", "PRU.L"]},
    #         "Healthcare & Consumer": {"indices": ["^FTSE"], "stocks": ["AZN.L", "GSK.L", "ULVR.L", "BATS.L", "RB.L", "DGE.L"]},
    #         "Consumer Goods & Retail": {"indices": ["^FTSE"], "stocks": ["ULVR.L", "BATS.L", "DGE.L", "ABF.L", "TSCO.L"]},
    #         "Technology & Telecoms": {"indices": ["^FTSE"], "stocks": ["REL.L", "SGE.L", "VOD.L", "AAF.L"]},
    #         "Aerospace, Industrials & Defense": {"indices": ["^FTSE"], "stocks": ["BA.L", "RR.L", "CRH.L", "SMIN.L"]},
    #     },
    #     "EU": {
    #         "Technology & Industrials": {"indices": ["^STOXX", "^GDAXI"], "stocks": ["ASML.AS", "SAP.DE", "SIE.DE", "AIR.PA", "IFX.DE", "AD.AS"]},
    #         "Luxury & Consumer Goods": {"indices": ["^STOXX", "^FCHI"], "stocks": ["MC.PA", "OR.PA", "RMS.PA", "KER.PA", "ABI.BR", "HEIA.AS"]},
    #         "Financials & Automakers": {"indices": ["^STOXX", "^GDAXI"], "stocks": ["ALV.DE", "MBG.DE", "BMW.DE", "VOW3.DE", "SAN.MC", "BNP.PA"]},
    #         "Healthcare & Pharmaceuticals": {"indices": ["^STOXX", "^SSMI"], "stocks": ["NOVN.SW", "ROG.SW", "NESN.SW", "SAN.PA", "BAYN.DE", "MRK.DE"]},
    #         "Energy & Utilities": {"indices": ["^STOXX", "^FCHI"], "stocks": ["TTE.PA", "ENI.MI", "IBE.MC", "ENEL.MI", "RWE.DE", "EOAN.DE"]},
    #         "Telecoms & Infrastructure": {"indices": ["^STOXX", "^GDAXI"], "stocks": ["DTE.DE", "ORA.PA", "TEF.MC", "VOD.L", "NOKIA.HE"]},
    #     },
    #     "JSE": {
    #         "JSE Mining & Resources": {"indices": ["^J210.JO"], "stocks": ["AGL.JO", "BHP.JO", "GFI.JO", "ANG.JO", "IMP.JO", "SSW.JO", "AMS.JO"]},
    #         "JSE Financials": {"indices": ["^J212.JO"], "stocks": ["FSR.JO", "SBK.JO", "ABG.JO", "NED.JO", "SLM.JO", "CPI.JO", "INP.JO"]},
    #         "JSE Tech & Telecoms": {"indices": ["^J211.JO"], "stocks": ["NPN.JO", "PRX.JO", "MTN.JO", "VOD.JO", "TEL.JO"]},
    #         "JSE Retail & Consumer": {"indices": ["^J213.JO"], "stocks": ["WHL.JO", "SHP.JO", "MRP.JO", "CLS.JO", "PIK.JO", "SPP.JO"]},
    #     },
    # }
        
    ALL_EXCHANGE_CONFIGS = "stock-sectors-nse-bse.json"
    ALL_EXCHANGE_CONFIGS = json.loads(ALL_EXCHANGE_CONFIGS)
    sector_config = ALL_EXCHANGE_CONFIGS.get(exchange, ALL_EXCHANGE_CONFIGS["NSE"])
    ticker_to_sector = {}
    for sec_name, sec_data in sector_config.items():
        for stk in sec_data["stocks"]:
            base_s = stk.split(".")[0].upper()
            ticker_to_sector[base_s] = sec_name
            ticker_to_sector[stk.upper()] = sec_name

    # --- LIVE INDEX CHART SECTION ---
    st.subheader(f"Live Market Context: {current_cfg['name']}")

    tf_options = {
        "1D": {"days": 1, "label": "past day"},
        "5D": {"days": 5, "label": "past 5 days"},
        "1M": {"days": 30, "label": "past month"},
        "6M": {"days": 180, "label": "past 6 months"},
        "YTD": {"days": "YTD", "label": "year to date"},
        "1Y": {"days": 365, "label": "past year"},
        "5Y": {"days": 1825, "label": "past 5 years"},
        "MAX": {"days": 3650, "label": "all time"},
    }

    col_tf, _ = st.columns([3, 1])
    with col_tf:
        selected_tf = st.segmented_control(
            "Timeframe",
            options=list(tf_options.keys()),
            default="6M",
            label_visibility="collapsed",
        )

    with st.spinner(f"Fetching live benchmark data for {current_cfg['name']}..."):
        try:
            end_d = datetime.datetime.now()
            tf_info = tf_options.get(selected_tf, tf_options["6M"])

            if tf_info["days"] == "YTD":
                start_d = datetime.datetime(end_d.year, 1, 1)
            else:
                start_d = end_d - datetime.timedelta(days=tf_info["days"])

            interval = "15m" if selected_tf == "1D" else "1d"
            bench_raw = yf.download(
                current_cfg["symbol"],
                start=start_d,
                end=end_d,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )

            if not bench_raw.empty:
                if "Close" in bench_raw.columns:
                    chart_series = bench_raw["Close"]
                else:
                    chart_series = bench_raw.iloc[:, 0]

                if isinstance(chart_series, pd.DataFrame):
                    chart_series = chart_series.squeeze()

                chart_df = chart_series.dropna().reset_index()
                chart_df.columns = ["Date", "Close"]

                start_price = chart_df["Close"].iloc[0]
                latest_price = chart_df["Close"].iloc[-1]
                abs_change = latest_price - start_price
                pct_change = (abs_change / start_price) * 100
                is_falling = abs_change < 0

                theme_color = "#D93025" if is_falling else "#188038"
                fill_color = "rgba(217, 48, 37, 0.10)" if is_falling else "rgba(24, 128, 56, 0.10)"
                arrow = "↓" if is_falling else "↑"
                sign = "" if is_falling else "+"

                st.markdown(
                    f"""
                    <div style="margin-top: -10px; margin-bottom: 10px;">
                        <span style="font-size: 2.2rem; font-weight: 700; color: #202124;">{latest_price:,.2f}</span>
                        <span style="background-color: {'#FCE8E6' if is_falling else '#E6F4EA'}; color: {theme_color}; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.95rem; margin-left: 10px;">
                            {arrow} {abs(pct_change):.2f}%
                        </span>
                        <span style="color: {theme_color}; font-weight: 600; font-size: 0.95rem; margin-left: 6px;">
                            {sign}{abs_change:,.2f}
                        </span>
                        <span style="color: #5f6368; font-size: 0.9rem; margin-left: 6px;">
                            {tf_info['label']}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                y_min = chart_df["Close"].min() * 0.998
                y_max = chart_df["Close"].max() * 1.002

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=chart_df["Date"],
                    y=chart_df["Close"],
                    mode="lines",
                    line=dict(color=theme_color, width=2),
                    fill="tonexty",
                    fillcolor=fill_color,
                    hovertemplate="%{x}<br><b>Price:</b> %{y:,.2f}<extra></extra>",
                ))

                fig.update_layout(
                    height=230,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_title=None,
                    yaxis_title=None,
                    xaxis=dict(fixedrange=True),
                    yaxis=dict(range=[y_min, y_max], fixedrange=True),
                    dragmode=False,
                    hovermode="x unified",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )

                st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
            else:
                st.warning("Market benchmark data temporarily unavailable.")
        except Exception as e:
            st.warning(f"Benchmark market index chart error: {str(e)}")

    # ==========================================================================
    # 0. TOP-LEVEL ENTITY, RISK PARAMETERS & WORKSPACE MODE SWITCH
    # ==========================================================================
    st.markdown("### 🏛️ Entity, Risk Parameters & Workspace")
    cfg_c1, cfg_c2, cfg_c3, cfg_c4 = st.columns([2.2, 1.3, 1.3, 1.4])
    with cfg_c1:
        entity_type = st.selectbox(
            "Entity Type",
            [
                "Boutique PMS (Principal Officer)",
                "Non-Individual SEBI RIA",
                "Individual SEBI RIA",
            ],
            key="top_entity_type",
        )
    with cfg_c2:
        confidence_level = st.selectbox(
            "VaR Confidence",
            [0.90, 0.95, 0.99],
            index=1,
            key="top_var_conf",
        )
    with cfg_c3:
        time_horizon_days = st.number_input(
            "Horizon (Days)",
            min_value=1,
            max_value=365 * 4,
            value=30,
            step=1,
            key="top_horizon_days",
        )
    with cfg_c4:
        target_return_pct = st.number_input(
            "Target / RFR (% p.a.)",
            min_value=-50.0,
            max_value=100.0,
            value=6.5 if exchange in ["NSE", "BSE"] else 4.5,
            step=0.5,
            key="top_target_return",
        )

    workspace_mode = st.radio(
        "Workspace",
        ["Single Client Audit", "Model Portfolio & Client Batch"],
        horizontal=True,
        key="workspace_mode_selector",
    )

    # Sidebar Configuration (Currency, Capital, CSV/Demo Upload, Feedback)
    st.sidebar.header(f"1. Configuration ({exchange})")
    currency = st.sidebar.selectbox(
        "Denominated Currency",
        ["INR", "ZAR", "USD", "EUR", "GBP"],
        index=["INR", "ZAR", "USD", "EUR", "GBP"].index(current_cfg["default_curr"]),
    )
    initial_capital = st.sidebar.number_input(
        label=f"Initial Capital / Cash Reference ({currency})",
        min_value=1000.0,
        value=25000000.0 if currency == "INR" else 250000.0,
        step=100000.0 if currency == "INR" else 5000.0,
        format="%.2f",
    )
    uploaded_file = st.sidebar.file_uploader(f"Upload {exchange} Portfolio CSV", type=["csv"])

    # if exchange in ["BSE", "NSE"]:
    #     if st.sidebar.button("📁 Load Demo Portfolio"):
    #         st.session_state["use_demo"] = True
    #     if uploaded_file is not None:
    #         st.session_state["use_demo"] = False
    #     elif st.session_state["use_demo"] and os.path.exists("Stocks _ Arghadeep  - 8511471544_PortFolioEqtSummary.csv"):
    #         uploaded_file = pd.read_csv("Stocks _ Arghadeep  - 8511471544_PortFolioEqtSummary.csv")

    rfr_dict = {"NSE": 0.065, "BSE": 0.065, "US": 0.045, "UK": 0.045, "EU": 0.030, "JSE": 0.0725}
    rfr_annual = rfr_dict.get(exchange, 0.05)

    # ==========================================================================
    # SECONDARY TABS
    # ==========================================================================
    st.markdown("---")
    sec_tab1, sec_tab2, sec_tab3 = st.tabs([
        "🧭 Stage 1–4: Rebalance, Mandate Gate & Audit",
        "📊 Market Context (Sectors & Leaders)",
        "📉 Fee & Tax Optimization",
    ])

    with sec_tab2:
        st.subheader(f"{exchange} Sector Performance & Stock Leaders")
        sector_config = ALL_EXCHANGE_CONFIGS.get(exchange, ALL_EXCHANGE_CONFIGS["BSE"])

        st.subheader("Select Time Horizon")
        timeframe = st.radio(
            "Choose period:",
            ["1 Month", "3 Months", "6 Months", "1 Year"],
            horizontal=True,
            key="sector_idx_tf",
        )

        timeframe_days_map = {
            "1 Month": 21,
            "3 Months": 63,
            "6 Months": 126,
            "1 Year": 252,
        }
        lookback_days = timeframe_days_map[timeframe]

        all_tickers_to_fetch = []
        stock_to_sector_map = {}

        for sector_name, data in sector_config.items():
            all_tickers_to_fetch.extend(data["indices"])
            all_tickers_to_fetch.extend(data["stocks"])
            for stock in data["stocks"]:
                stock_to_sector_map[stock] = sector_name

        all_tickers_to_fetch = list(set(all_tickers_to_fetch))

        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=400)

        @st.cache_data(ttl=3600)
        def fetch_sector_price_matrix(tickers, start, end):
            try:
                df = yf.download(tickers, start=start, end=end, progress=False)
                if isinstance(df, pd.DataFrame) and "Close" in df:
                    return df["Close"]
                return df
            except Exception:
                return pd.DataFrame()

        price_df = fetch_sector_price_matrix(all_tickers_to_fetch, start_date, end_date)

        sector_returns = {}
        stock_returns_dict = {}

        if not price_df.empty:
            for col in price_df.columns:
                series = price_df[col].dropna()
                if len(series) >= lookback_days:
                    s_start = series.iloc[-lookback_days]
                    s_end = series.iloc[-1]
                    if s_start > 0:
                        ret = ((s_end - s_start) / s_start) * 100
                        stock_returns_dict[col] = ret

            for sector_name, data in sector_config.items():
                sec_stock_rets = [
                    stock_returns_dict[stk]
                    for stk in data["stocks"]
                    if stk in stock_returns_dict
                ]
                if sec_stock_rets:
                    sector_returns[sector_name] = np.mean(sec_stock_rets)

        perf_df = pd.DataFrame(
            list(sector_returns.items()), columns=["Sector", "Growth (%)"]
        ).sort_values(by="Growth (%)", ascending=False)

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.subheader(f"🟢 Top Performing Sectors ({timeframe})")
            st.dataframe(
                perf_df.style.format({"Growth (%)": "{:+.2f}%"}),
                use_container_width=True,
                hide_index=True,
            )
            fig_top_sec = px.bar(
                perf_df.head(5),
                x="Growth (%)",
                y="Sector",
                orientation="h",
                title=f"Top Sectors Growth - {timeframe}",
                color="Growth (%)",
                color_continuous_scale="Greens",
            )
            fig_top_sec.update_layout(dragmode=False)
            st.plotly_chart(fig_top_sec, use_container_width=True)

        with col2:
            st.subheader(f"🔴 Worst Performing Sectors ({timeframe})")
            worst_sec_df = perf_df.tail(5).sort_values(by="Growth (%)", ascending=True)
            st.dataframe(
                worst_sec_df.style.format({"Growth (%)": "{:+.2f}%"}),
                use_container_width=True,
                hide_index=True,
            )
            fig_worst_sec = px.bar(
                worst_sec_df,
                x="Growth (%)",
                y="Sector",
                orientation="h",
                title=f"Worst Sectors Growth - {timeframe}",
                color="Growth (%)",
                color_continuous_scale="Reds_r",
            )
            fig_worst_sec.update_layout(dragmode=False)
            st.plotly_chart(fig_worst_sec, use_container_width=True)

        st.markdown("---")
        st.subheader(f"📌 Top 10 Best & Worst Stock Performers ({exchange})")

        all_stocks_in_exchange = set()
        for sector_data in sector_config.values():
            all_stocks_in_exchange.update(sector_data["stocks"])

        stock_perf_list = [
            {
                "Ticker": stk,
                "Sector": stock_to_sector_map.get(stk, "N/A"),
                "Return (%)": ret,
            }
            for stk, ret in stock_returns_dict.items()
            if stk in all_stocks_in_exchange
        ]
        if stock_perf_list:
            stock_perf_df = pd.DataFrame(stock_perf_list).sort_values(by="Return (%)", ascending=False)
        else:
            stock_perf_df = pd.DataFrame(columns=["Ticker", "Sector", "Return (%)"])

        stock_perf_df["Ticker"] = stock_perf_df["Ticker"].str.replace(r"\.(NS|BO|L|JO|DE|PA|AS|MI|MC)$", "", regex=True)
        st_col1, st_col2 = st.columns(2, gap="large")

        with st_col1:
            st.markdown("### 🚀 Top 10 Best Performing Stocks")
            top10_stocks = stock_perf_df.head(10).reset_index(drop=True)
            st.dataframe(
                top10_stocks.style.format({"Return (%)": "{:+.2f}%"}),
                use_container_width=True,
                hide_index=True,
            )
            fig_top_stocks = px.bar(
                top10_stocks,
                x="Return (%)",
                y="Ticker",
                color="Sector",
                orientation="h",
                title=f"Top 10 Stocks - {timeframe}",
            )
            fig_top_stocks.update_layout(dragmode=False)
            st.plotly_chart(fig_top_stocks, use_container_width=True)

        with st_col2:
            st.markdown("### 📉 Top 10 Worst Performing Stocks")
            worst10_stocks = stock_perf_df.tail(10).sort_values(by="Return (%)", ascending=True).reset_index(drop=True)
            st.dataframe(
                worst10_stocks.style.format({"Return (%)": "{:+.2f}%"}),
                use_container_width=True,
                hide_index=True,
            )
            fig_worst_stocks = px.bar(
                worst10_stocks,
                x="Return (%)",
                y="Ticker",
                color="Sector",
                orientation="h",
                title=f"Worst 10 Stocks - {timeframe}",
            )
            fig_worst_stocks.update_layout(dragmode=False)
            st.plotly_chart(fig_worst_stocks, use_container_width=True)

        st.markdown("---")
        st.subheader("🔎 Detailed Sector Drill-Down")
        available_sectors = list(sector_config.keys())
        selected_sector = st.selectbox("Choose a Sector to Analyze:", options=available_sectors)

        if selected_sector:
            sector_stocks = sector_config[selected_sector]["stocks"]
            sector_stocks_perf = [
                {"Ticker": stk, "Return (%)": stock_returns_dict.get(stk, np.nan)}
                for stk in sector_stocks
                if stk in stock_returns_dict
            ]
            sector_drill_df = pd.DataFrame(sector_stocks_perf).sort_values(by="Return (%)", ascending=False)
            if not sector_drill_df.empty:
                sector_drill_df["Ticker"] = sector_drill_df["Ticker"].str.replace(r"\.(NS|BO|L|JO|DE|PA|AS|MI|MC)$", "", regex=True)
                sd_col1, sd_col2 = st.columns([1, 1.5], gap="large")
                with sd_col1:
                    st.markdown(f"### Performance Breakdown: **{selected_sector}** ({timeframe})")
                    st.dataframe(
                        sector_drill_df.style.format({"Return (%)": "{:+.2f}%"}),
                        use_container_width=True,
                        hide_index=True,
                    )
                with sd_col2:
                    fig_sector_drill = px.bar(
                        sector_drill_df,
                        x="Return (%)",
                        y="Ticker",
                        orientation="h",
                        title=f"{selected_sector} Stock Returns — {timeframe}",
                        color="Return (%)",
                        color_continuous_scale="Viridis",
                    )
                    fig_sector_drill.update_layout(yaxis={"categoryorder": "total ascending"}, dragmode=False)
                    st.plotly_chart(fig_sector_drill, use_container_width=True)
            else:
                st.info(f"No price data available for stocks in {selected_sector}.")

    with sec_tab3:
        st.subheader("📉 Strategy Fee, Cash & Execution Drag Calculator")
        is_pms_role = "PMS" in entity_type
        d1, d2, d3 = st.columns(3)
        with d1:
            horizon_years = st.number_input("Investment Horizon (Years)", min_value=1, max_value=30, value=10, step=1, key="wd_horizon")
        with d2:
            expected_cagr = st.number_input("Expected Gross CAGR (%)", min_value=1.0, max_value=30.0, value=14.0, step=0.5, key="wd_cagr") / 100.0
        with d3:
            expense_diff = st.number_input("Annual Fee + Friction Drag (% p.a.)" if is_pms_role else "Expense Ratio Diff (% p.a.)", min_value=0.1, max_value=5.0, value=1.75 if is_pms_role else 1.0, step=0.1, key="wd_diff") / 100.0

        net_cagr = max(expected_cagr - expense_diff, -0.99)
        yrs = np.arange(0, int(horizon_years) + 1)
        net_w = initial_capital * ((1.0 + net_cagr) ** yrs)
        gross_w = initial_capital * ((1.0 + expected_cagr) ** yrs)
        drag_val = float(gross_w[-1] - net_w[-1])
        drag_pct = (drag_val / float(gross_w[-1]) * 100.0) if gross_w[-1] else 0.0

        dm1, dm2, dm3 = st.columns(3)
        dm1.metric("Net-of-Fee / Friction Value", f"{currency} {net_w[-1]:,.2f}")
        dm2.metric("Gross Strategy Potential", f"{currency} {gross_w[-1]:,.2f}")
        dm3.metric("Total Compounded Drag", f"{currency} {drag_val:,.2f}", delta=f"-{drag_pct:.1f}% drag", delta_color="inverse")
        st.line_chart(pd.DataFrame({"Year": yrs, "Net of Fee & Friction": net_w, "Gross Strategy Trajectory": gross_w}).set_index("Year"))

        st.markdown("---")
        st.subheader("🛡️ Tax Liabilities & Optimization (India)")
        if exchange not in ["NSE", "BSE"]:
            st.info("ℹ️ Tax audit optimization is currently available for NSE / BSE workspaces.")
        else:
            tax_cost_missing = False
            base_tax_df = pd.DataFrame([
                {"Instrument": "HDFCBANK", "Avg_Cost": 1520.0, "Qty": 100, "Holding_Days": 420},
                {"Instrument": "INFY", "Avg_Cost": 1480.0, "Qty": 120, "Holding_Days": 180},
                {"Instrument": "TATASTEEL", "Avg_Cost": 165.0, "Qty": 500, "Holding_Days": 90},
            ])
            if uploaded_file is not None:
                parsed_for_tax, has_cost_col = parse_uploaded_portfolio(uploaded_file, exchange, is_demo=st.session_state["use_demo"])
                if not parsed_for_tax.empty:
                    if not has_cost_col or parsed_for_tax["Cost Price"].isna().all():
                        tax_cost_missing = True
                    else:
                        base_tax_df = pd.DataFrame({
                            "Instrument": parsed_for_tax["Ticker"],
                            "Avg_Cost": parsed_for_tax["Cost Price"],
                            "Qty": parsed_for_tax["Quantity"].clip(lower=1),
                            "Holding_Days": int(time_horizon_days),
                        })

            if tax_cost_missing:
                st.error("❌ Cost cannot be read: Average Cost Price is missing in the uploaded CSV. Because only live market prices are available, capital gains tax liabilities cannot be calculated.")
            else:
                edited_tax_df = st.data_editor(base_tax_df, num_rows="dynamic", use_container_width=True, hide_index=True, key="tax_editor_tab")
                if not edited_tax_df.empty:
                    valid_cost_series = pd.to_numeric(edited_tax_df["Avg_Cost"], errors="coerce")
                    if valid_cost_series.isna().any() or (valid_cost_series <= 0).any():
                        st.error("❌ Cost cannot be read: One or more rows have missing or zero Average Cost. Tax liability cannot be calculated without valid cost basis.")
                    else:
                        tax_yf = [f"{str(t).strip().upper()}{current_cfg['suffix']}" for t in edited_tax_df["Instrument"] if str(t).strip()]
                        if tax_yf:
                            lp_raw = yf.download(tax_yf, period="5d", progress=False)
                            lp_series = lp_raw["Close"].ffill().bfill().iloc[-1] if isinstance(lp_raw, pd.DataFrame) and "Close" in lp_raw else pd.Series(dtype=float)
                            if isinstance(lp_series, (float, int)):
                                lp_series = pd.Series({tax_yf[0]: float(lp_series)})
                            edited_tax_df["Live Price"] = [float(lp_series.get(f"{str(t).strip().upper()}{current_cfg['suffix']}", np.nan)) for t in edited_tax_df["Instrument"]]
                            edited_tax_df["P&L"] = (edited_tax_df["Live Price"] - valid_cost_series) * edited_tax_df["Qty"]
                            edited_tax_df["Tax_Category"] = np.where(edited_tax_df["Holding_Days"] >= 365, "LTCG", "STCG")
                            edited_tax_df["Tax_Liability"] = np.where(edited_tax_df["Tax_Category"] == "LTCG", np.maximum(0.0, edited_tax_df["P&L"]) * 0.125, np.maximum(0.0, edited_tax_df["P&L"]) * 0.20)
                            tot_pnl = float(edited_tax_df["P&L"].sum())
                            tot_tax = float(edited_tax_df["Tax_Liability"].sum())
                            tx1, tx2, tx3 = st.columns(3)
                            tx1.metric("Total P&L", f"{currency} {tot_pnl:,.2f}")
                            tx2.metric("Estimated Tax Liability", "N/A" if tot_pnl <= 0 else f"{currency} {tot_tax:,.2f}")
                            tx3.metric("Effective Tax Share", "N/A" if tot_pnl <= 0 else f"{(tot_tax / tot_pnl) * 100:.2f}%")
                            st.dataframe(edited_tax_df, use_container_width=True, hide_index=True)

    # ==========================================================================
    # MAIN WORKFLOW TAB: STAGE 1 -> STAGE 2 -> STAGE 3 -> STAGE 4
    # ==========================================================================
    with sec_tab1:
        st.markdown("## Stage 1: Portfolio & Mandate")

        lim_c1, lim_c2, lim_c3, lim_c4 = st.columns(4)
        with lim_c1:
            vol_cap_pct = st.number_input("Max Volatility (% p.a.)", min_value=5.0, max_value=80.0, value=25.0, step=1.0, key="lim_vol_cap")
        with lim_c2:
            cvar_cap_pct = st.number_input("Max CVaR (% Horizon)", min_value=1.0, max_value=60.0, value=18.0, step=1.0, key="lim_cvar_cap")
        with lim_c3:
            stock_cap_pct = st.number_input("Max Single-Stock Wt (%)", min_value=2.0, max_value=100.0, value=15.0, step=1.0, key="lim_stock_cap")
        with lim_c4:
            sector_cap_pct = st.number_input("Max Sector Wt (%)", min_value=5.0, max_value=100.0, value=45.0, step=1.0, key="lim_sector_cap")

        mandate_limits = {
            "vol_cap": float(vol_cap_pct),
            "cvar_cap": float(cvar_cap_pct),
            "stock_cap": float(stock_cap_pct),
            "sector_cap": float(sector_cap_pct),
        }

        exchange_suggestions = sorted({
            stk.split(".")[0].upper()
            for sec_data in sector_config.values()
            for stk in sec_data.get("stocks", [])
        })

        # ======================================================================
        # WORKSPACE MODE A: SINGLE CLIENT AUDIT
        # ======================================================================
        if workspace_mode == "Single Client Audit":
            if uploaded_file is not None:
                init_holdings, _ = parse_uploaded_portfolio(uploaded_file, exchange, is_demo=st.session_state["use_demo"])
                if init_holdings.empty:
                    init_holdings = pd.DataFrame({"Ticker": ["HDFCBANK", "INFY", "RELIANCE"], "Quantity": [100, 120, 80], "Cost Price": [1500.0, 1450.0, 2600.0]})
            else:
                init_holdings = pd.DataFrame({
                    "Ticker": ["FEDERALBNK", "IDFCFIRSTB", "NCC", "JSWENERGY", "TATASTEEL"],
                    "Quantity": [1000, 2000, 1500, 800, 1200],
                    "Cost Price": [160.0, 80.0, 210.0, 520.0, 155.0],
                })

            # Fetch live prices for Stage 1 holdings table
            s1_raw_tickers = [str(t).strip().upper() for t in init_holdings["Ticker"] if str(t).strip()]
            s1_yf_syms = resolve_yf_tickers(s1_raw_tickers, current_cfg)
            s1_live_map = {}
            if s1_yf_syms:
                try:
                    lp_s1 = yf.download(s1_yf_syms, period="5d", progress=False)
                    lp_s1_close = lp_s1["Close"] if isinstance(lp_s1, pd.DataFrame) and "Close" in lp_s1 else lp_s1
                    if isinstance(lp_s1_close, pd.Series):
                        lp_s1_close = lp_s1_close.to_frame(name=s1_yf_syms[0])
                    if isinstance(lp_s1_close, pd.DataFrame) and not lp_s1_close.empty:
                        last_s1 = lp_s1_close.ffill().bfill().iloc[-1]
                        for bt, yt in zip(s1_raw_tickers, s1_yf_syms):
                            if yt in last_s1.index and pd.notna(last_s1[yt]):
                                s1_live_map[bt] = round(float(last_s1[yt]), 2)
                except Exception:
                    pass

            init_holdings_with_live = init_holdings.copy()
            init_holdings_with_live["Live Price"] = [
                s1_live_map.get(str(t).strip().upper(), np.nan) for t in init_holdings_with_live["Ticker"]
            ]

            sc_col1, sc_col2 = st.columns([3, 1.3])
            with sc_col1:
                st.caption(f"Edit client holdings (Ticker, Quantity, Cost Price) directly below — Live Price ({currency}) shown alongside:")
                client_holdings_df = st.data_editor(
                    init_holdings_with_live,
                    num_rows="dynamic",
                    use_container_width=True,
                    hide_index=True,
                    disabled=["Live Price"],
                    key="single_client_holdings_editor",
                )
            with sc_col2:
                client_cash = st.number_input(f"Client Cash Balance ({currency})", min_value=0.0, value=float(initial_capital) * 0.10, step=50000.0 if currency == "INR" else 1000.0, key="single_client_cash")
                client_profile_tag = st.selectbox("Client Risk Profile", ["Conservative", "Moderate", "Aggressive"], index=1, key="single_client_profile")
                client_id_tag = st.text_input("Client ID / PAN Mask", value="CLIENT-001 (XXXX9876X)", key="single_client_id")

            # ------------------------------------------------------------------
            # STAGE 2: PROPOSED REBALANCE (SINGLE CLIENT)
            # ------------------------------------------------------------------
            st.markdown("---")
            st.markdown("## Stage 2: Proposed Rebalance")
            r2_c1, r2_c2 = st.columns(2)
            with r2_c1:
                cash_flow_sim = st.number_input(f"Simulate Cash Inflow (+) / Outflow (-) ({currency})", value=0.0, step=50000.0 if currency == "INR" else 1000.0, key="sc_cash_flow")
            with r2_c2:
                friction_bps = st.slider("Est. Friction Cost (bps on traded notional)", min_value=0, max_value=100, value=15, step=5, key="sc_friction_bps")

            st.markdown("#### ➕ Add New Stock(s) to Rebalance")
            if "sc_added_tickers_list" not in st.session_state:
                st.session_state["sc_added_tickers_list"] = []

            add_c1, add_c2, add_c3, add_c4 = st.columns([2.0, 2.0, 1.3, 1.0])
            with add_c1:
                picked_from_list = st.multiselect(
                    "Pick from Exchange Universe",
                    options=exchange_suggestions,
                    default=[t for t in st.session_state["sc_added_tickers_list"] if t in exchange_suggestions],
                    placeholder="Select stocks (e.g. TCS, INFY)...",
                    key="sc_pick_universe",
                )
            with add_c2:
                custom_tickers_raw = st.text_input(
                    "Or Type Custom Ticker(s) (comma-separated)",
                    value="",
                    placeholder="e.g. ZOMATO, TRENT, BEL",
                    key="sc_added_stocks",
                ).strip().upper()
            with add_c3:
                default_added_wt = st.number_input(
                    "Initial Target Wt (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=10.0,
                    step=1.0,
                    key="sc_default_added_wt",
                )
            with add_c4:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                if st.button("Clear Added", key="sc_clear_added_btn"):
                    st.session_state["sc_added_tickers_list"] = []
                    st.rerun()

            custom_list = [p.strip().upper() for p in custom_tickers_raw.split(",") if p.strip()]
            combined_added = []
            for item in picked_from_list + custom_list:
                if item and item != "NAN" and item not in combined_added:
                    combined_added.append(item)
            st.session_state["sc_added_tickers_list"] = combined_added

            existing_tickers = []
            for t in client_holdings_df["Ticker"]:
                tu = str(t).strip().upper()
                if tu and tu != "NAN" and tu not in existing_tickers:
                    existing_tickers.append(tu)

            added_tickers = [
                pu for pu in st.session_state.get("sc_added_tickers_list", [])
                if pu and pu != "NAN" and pu not in existing_tickers
            ]

            clean_tickers = existing_tickers + added_tickers

            if not clean_tickers:
                st.warning("Add at least one valid stock ticker in Stage 1 or Stage 2 to compute risk and pre-trade diffs.")
                st.stop()

            yf_syms = resolve_yf_tickers(clean_tickers, current_cfg)
            with st.spinner("Fetching market prices & return histories..."):
                end_dt = datetime.datetime.now()
                start_dt = end_dt - datetime.timedelta(days=365 * 3)
                p_raw = yf.download(yf_syms, start=start_dt, end=end_dt, progress=False)
                if isinstance(p_raw, pd.DataFrame) and "Close" in p_raw:
                    p_df = p_raw["Close"]
                else:
                    p_df = p_raw
                if isinstance(p_df, pd.Series):
                    p_df = p_df.to_frame(name=yf_syms[0])
                p_df = p_df.dropna(how="all", axis=1).ffill().bfill()

            valid_pairs = [(base_t, yf_t) for base_t, yf_t in zip(clean_tickers, yf_syms) if yf_t in p_df.columns]
            if not valid_pairs:
                st.error("Unable to fetch market prices for the specified tickers.")
                st.stop()

            qty_map = {}
            for _, r in client_holdings_df.iterrows():
                bt = str(r.get("Ticker", "")).strip().upper()
                if bt:
                    qty_map[bt] = qty_map.get(bt, 0.0) + float(r.get("Quantity", 0.0) or 0.0)

            current_vals = np.array([qty_map.get(bt, 0.0) * float(p_df[yt].iloc[-1]) for bt, yt in valid_pairs], dtype=float)
            total_curr_equity = float(current_vals.sum())
            total_curr_nav = total_curr_equity + float(client_cash)
            curr_wts = (current_vals / total_curr_nav) if total_curr_nav > 0 else np.zeros(len(valid_pairs))

            st.write("**Adjust Target Weights (%) for Existing Holdings & Any Newly Added Stocks:**")
            tw_cols = st.columns(min(len(valid_pairs), 5))
            raw_target_pcts = []
            rem_existing_count = max(sum(1 for bt, _ in valid_pairs if bt not in added_tickers), 1)
            added_total_pct = sum(float(default_added_wt) for bt, _ in valid_pairs if bt in added_tickers)
            rem_pool_pct = max(0.0, 100.0 - min(added_total_pct, 95.0))
            default_ex_wt = rem_pool_pct / rem_existing_count

            for idx, (bt, _) in enumerate(valid_pairs):
                with tw_cols[idx % len(tw_cols)]:
                    is_added = bt in added_tickers
                    init_val = float(default_added_wt) if is_added else round(default_ex_wt, 2)
                    lbl = f"Target %: {bt} (NEW)" if is_added else f"Target %: {bt}"
                    val_pct = st.number_input(lbl, min_value=0.0, max_value=100.0, value=init_val, step=1.0, key=f"sc_tw_{bt}")
                    raw_target_pcts.append(val_pct)

            post_nav = max(total_curr_nav + float(cash_flow_sim), 1.0)
            sum_tw = sum(raw_target_pcts)
            if sum_tw > 0:
                prop_wts = np.array([x / sum_tw for x in raw_target_pcts], dtype=float) * (min(sum_tw, 100.0) / 100.0)
            else:
                prop_wts = np.zeros(len(valid_pairs), dtype=float)

            diff_wts = prop_wts - curr_wts
            curr_val_per_asset = curr_wts * total_curr_nav
            prop_val_per_asset = prop_wts * post_nav
            trade_notional_per_asset = prop_val_per_asset - curr_val_per_asset
            friction_rate = friction_bps / 10000.0

            unified_blotter_df = pd.DataFrame({
                "Ticker": [bt for bt, _ in valid_pairs],
                "Sector": [ticker_to_sector.get(bt, "Other") for bt, _ in valid_pairs],
                "Live Price": [float(p_df[yt].iloc[-1]) for _, yt in valid_pairs],
                "Current Wt (%)": curr_wts * 100.0,
                "Proposed Wt (%)": prop_wts * 100.0,
                "Delta Wt (%)": diff_wts * 100.0,
                "Current Val": curr_val_per_asset,
                "Proposed Val": prop_val_per_asset,
                "Action": [
                    "EXIT / SELL" if (prop_wts[i] <= 1e-6 and curr_wts[i] > 1e-6)
                    else ("NEW / BUY" if (curr_wts[i] <= 1e-6 and prop_wts[i] > 1e-6)
                    else ("BUY" if trade_notional_per_asset[i] > 1.0 else ("SELL" if trade_notional_per_asset[i] < -1.0 else "HOLD")))
                    for i in range(len(valid_pairs))
                ],
                "Trade Notional": np.abs(trade_notional_per_asset),
                "Est. Friction": np.abs(trade_notional_per_asset) * friction_rate,
            })

            st.write("**Pre-Trade vs. Post-Trade Allocation & Order Diff Table**")
            st.dataframe(
                unified_blotter_df.style.format({
                    "Live Price": f"{currency} {{:,.2f}}",
                    "Current Wt (%)": "{:.2f}%",
                    "Proposed Wt (%)": "{:.2f}%",
                    "Delta Wt (%)": "{:+.2f}%",
                    "Current Val": f"{currency} {{:,.2f}}",
                    "Proposed Val": f"{currency} {{:,.2f}}",
                    "Trade Notional": f"{currency} {{:,.2f}}",
                    "Est. Friction": f"{currency} {{:,.2f}}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            # ------------------------------------------------------------------
            # STAGE 3: PRE-TRADE DIFF, BENCHMARK GRAPH, QUANT METRICS & MONTE CARLO
            # ------------------------------------------------------------------
            st.markdown("---")
            st.markdown("## Stage 3: Pre-Trade Diff, Benchmark Comparison & Risk / Monte Carlo")

            valid_yf_ordered = [yt for _, yt in valid_pairs]
            valid_bt_ordered = [bt for bt, _ in valid_pairs]
            price_matrix_valid = p_df[valid_yf_ordered]
            ret_matrix = price_matrix_valid.pct_change().dropna()

            pre_port_ret = ret_matrix.dot(curr_wts) if curr_wts.sum() > 0 else ret_matrix.iloc[:, 0] * 0.0
            post_port_ret = ret_matrix.dot(prop_wts) if prop_wts.sum() > 0 else ret_matrix.iloc[:, 0] * 0.0

            pre_metrics = compute_quant_metrics(
                pre_port_ret, rfr_annual=rfr_annual, target_return_pct=target_return_pct,
                confidence_level=confidence_level, horizon_days=time_horizon_days
            )
            post_metrics = compute_quant_metrics(
                post_port_ret, rfr_annual=rfr_annual, target_return_pct=target_return_pct,
                confidence_level=confidence_level, horizon_days=time_horizon_days
            )

            pre_div = compute_risk_and_diversification(ret_matrix, curr_wts, valid_bt_ordered)
            post_div = compute_risk_and_diversification(ret_matrix, prop_wts, valid_bt_ordered)

            # Benchmark alignment
            beta_val = 1.0
            b_cum_pct = None
            try:
                b_raw = yf.download(current_cfg["symbol"], start=ret_matrix.index[0], end=ret_matrix.index[-1], progress=False)
                b_ser = b_raw["Close"].squeeze() if isinstance(b_raw, pd.DataFrame) and "Close" in b_raw else pd.Series(dtype=float)
                b_ret = b_ser.pct_change().dropna()
                aligned = pd.DataFrame({"pre": pre_port_ret, "post": post_port_ret, "m": b_ret}).dropna()
                if len(aligned) > 10 and np.var(aligned["m"]) > 0:
                    beta_val = float(np.cov(aligned["post"], aligned["m"])[0, 1] / np.var(aligned["m"]))
                if not aligned.empty:
                    pre_cum_pct = ((1.0 + aligned["pre"]).cumprod() - 1.0) * 100.0
                    post_cum_pct = ((1.0 + aligned["post"]).cumprod() - 1.0) * 100.0
                    b_cum_pct = ((1.0 + aligned["m"]).cumprod() - 1.0) * 100.0
                else:
                    pre_cum_pct = ((1.0 + pre_port_ret).cumprod() - 1.0) * 100.0
                    post_cum_pct = ((1.0 + post_port_ret).cumprod() - 1.0) * 100.0
            except Exception:
                pre_cum_pct = ((1.0 + pre_port_ret).cumprod() - 1.0) * 100.0
                post_cum_pct = ((1.0 + post_port_ret).cumprod() - 1.0) * 100.0

            pre_max_stock = float(curr_wts.max() * 100.0) if len(curr_wts) else 0.0
            post_max_stock = float(prop_wts.max() * 100.0) if len(prop_wts) else 0.0

            sec_series_curr = pd.Series(curr_wts, index=[ticker_to_sector.get(bt, "Other") for bt in valid_bt_ordered])
            sec_series_prop = pd.Series(prop_wts, index=[ticker_to_sector.get(bt, "Other") for bt in valid_bt_ordered])
            pre_max_sec = float(sec_series_curr.groupby(level=0).sum().max() * 100.0) if not sec_series_curr.empty else 0.0
            post_max_sec = float(sec_series_prop.groupby(level=0).sum().max() * 100.0) if not sec_series_prop.empty else 0.0

            total_turnover_amt = float(unified_blotter_df["Trade Notional"].sum())
            turnover_pct = (total_turnover_amt / post_nav * 100.0) if post_nav > 0 else 0.0
            total_friction_amt = float(unified_blotter_df["Est. Friction"].sum())

            render_risk_diversification_and_mc_section(
                prefix_key="sc",
                currency=currency,
                time_horizon_days=time_horizon_days,
                confidence_level=confidence_level,
                pre_metrics=pre_metrics,
                post_metrics=post_metrics,
                pre_div=pre_div,
                post_div=post_div,
                pre_cum_pct=pre_cum_pct,
                post_cum_pct=post_cum_pct,
                b_cum_pct=b_cum_pct,
                benchmark_name=current_cfg["name"],
                price_matrix_valid=price_matrix_valid,
                prop_wts=prop_wts,
                post_nav=post_nav,
                turnover_pct=turnover_pct,
                total_friction_amt=total_friction_amt,
                beta_val=beta_val,
                pre_max_stock=pre_max_stock,
                post_max_stock=post_max_stock,
                pre_max_sec=pre_max_sec,
                post_max_sec=post_max_sec,
            )

            # ------------------------------------------------------------------
            # STAGE 4: LOG RATIONALE & EXPORT AUDIT PACKET (SINGLE CLIENT)
            # ------------------------------------------------------------------
            st.markdown("---")
            st.markdown("## Stage 4: Rationale & Audit Export")

            rationale_input = st.text_area(
                "Investment & Suitability Rationale (why this stock/weight change fits the client mandate, horizon & risk profile)",
                value="",
                placeholder="Document suitability reasoning, drawdown tolerance alignment, sector/issuer concentration rationale, and fee/friction justification...",
                key="sc_rationale_text",
            )

            gate_res = evaluate_mandate_gate(
                post_vol_pct=post_metrics["ann_vol"],
                post_cvar_pct=post_metrics["hist_cvar"],
                max_stock_wt_pct=post_max_stock,
                max_sector_wt_pct=post_max_sec,
                vol_cap_pct=vol_cap_pct,
                cvar_cap_pct=cvar_cap_pct,
                stock_cap_pct=stock_cap_pct,
                sector_cap_pct=sector_cap_pct,
                rationale_text=rationale_input,
            )

            strip_cls = "gate-pass" if gate_res["status"] == "Pass" else ("gate-warn" if gate_res["status"] == "Warning" else "gate-breach")
            badge_icon = "🟢 PASS" if gate_res["status"] == "Pass" else ("🟡 WARNING" if gate_res["status"] == "Warning" else "🔴 BREACH")
            issues_txt = " | ".join(gate_res["breaches"] + gate_res["warnings"]) if (gate_res["breaches"] or gate_res["warnings"]) else "All volatility, CVaR, single-stock & sector limits within mandate."
            rat_flag = "Logged" if not gate_res["missing_rationale"] else "Missing Rationale"

            st.markdown(
                f"""<div class="gate-strip {strip_cls}">
                    <b>Mandate Gate Status: {badge_icon}</b> &nbsp;•&nbsp; <b>Suitability Log:</b> {rat_flag}<br/>
                    <span>{issues_txt}</span>
                </div>""",
                unsafe_allow_html=True,
            )

            audit_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            diff_summary_payload = {
                "turnover_pct": turnover_pct,
                "friction_cost": total_friction_amt,
                "pre_vol": pre_metrics["ann_vol"],
                "post_vol": post_metrics["ann_vol"],
                "pre_cvar": pre_metrics["hist_cvar"],
                "post_cvar": post_metrics["hist_cvar"],
                "pre_max_stock": pre_max_stock,
                "post_max_stock": post_max_stock,
                "pre_max_sector": pre_max_sec,
                "post_max_sector": post_max_sec,
            }
            pdf_packet = build_audit_packet_pdf(
                entity_type=entity_type,
                workspace_mode=workspace_mode,
                exchange=exchange,
                currency=currency,
                client_label=f"{client_id_tag} ({client_profile_tag})",
                reg_no="SEBI-REG-AUDIT",
                officer_or_advisor=entity_type,
                mandate_limits=mandate_limits,
                diff_summary=diff_summary_payload,
                gate_result=gate_res,
                rationale_text=rationale_input,
                timestamp=audit_ts,
            )
            st.download_button(
                label="📥 Export Timestamped Audit & Suitability Report (PDF)",
                data=pdf_packet,
                file_name=f"WIRATOR_Audit_{client_id_tag.replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                type="primary",
            )

        # ======================================================================
        # WORKSPACE MODE B: MODEL PORTFOLIO & CLIENT BATCH
        # ======================================================================
        else:
            st.caption("Define Model Target Weights, add stocks to the batch rebalance, and upload/edit a multi-client batch CSV/table.")

            b_col1, b_col2 = st.columns([1.4, 2.2])
            with b_col1:
                st.write("**Model Target Weights (%)**")
                default_model_df = pd.DataFrame([
                    {"Ticker": "HDFCBANK", "Target_Weight_Pct": 35.0},
                    {"Ticker": "INFY", "Target_Weight_Pct": 35.0},
                    {"Ticker": "RELIANCE", "Target_Weight_Pct": 30.0},
                ]) if current_cfg.get("default_curr") == "INR" else pd.DataFrame([
                    {"Ticker": "AAPL", "Target_Weight_Pct": 40.0},
                    {"Ticker": "MSFT", "Target_Weight_Pct": 35.0},
                    {"Ticker": "JPM", "Target_Weight_Pct": 25.0},
                ])
                model_target_df = st.data_editor(default_model_df, num_rows="dynamic", use_container_width=True, hide_index=True, key="batch_model_targets")
            with b_col2:
                st.write("**Client Batch (Upload CSV or Edit Rows)**")
                batch_csv = st.file_uploader("Upload Client Batch CSV", type=["csv"], key="batch_csv_uploader")
                if batch_csv is not None:
                    try:
                        batch_df = pd.read_csv(batch_csv)
                    except Exception as e:
                        st.warning(f"Could not read batch CSV ({e}). Using sample batch.")
                        batch_df = get_default_batch_df(current_cfg)
                else:
                    batch_df = get_default_batch_df(current_cfg)
                edited_batch_df = st.data_editor(batch_df, num_rows="dynamic", use_container_width=True, hide_index=True, key="batch_clients_editor")

            # Stage 2 (Batch): Apply Model Target Weights + Option to Add Stocks to Rebalance
            st.markdown("---")
            st.markdown("## Stage 2: Proposed Rebalance (Batch Model Application)")
            b2_c1, b2_c2, b2_c3 = st.columns(3)
            with b2_c1:
                deploy_cash = st.checkbox("Deploy Client Cash into Model Targets", value=True, key="batch_deploy_cash")
            with b2_c2:
                batch_friction_bps = st.slider("Est. Friction Cost (bps)", min_value=0, max_value=100, value=15, step=5, key="batch_friction_bps")
            with b2_c3:
                override_limits_per_client = st.checkbox("Use Per-Client Mandate Caps if Provided", value=True, key="batch_per_client_caps")

            st.markdown("#### ➕ Add New Stock(s) to Batch Model Rebalance")
            if "batch_added_tickers_list" not in st.session_state:
                st.session_state["batch_added_tickers_list"] = []

            b_add1, b_add2, b_add3, b_add4 = st.columns([2.0, 2.0, 1.3, 1.0])
            with b_add1:
                b_picked_universe = st.multiselect(
                    "Pick from Exchange Universe",
                    options=exchange_suggestions,
                    default=[t for t in st.session_state["batch_added_tickers_list"] if t in exchange_suggestions],
                    placeholder="Select stocks to add...",
                    key="batch_pick_universe",
                )
            with b_add2:
                b_custom_raw = st.text_input(
                    "Or Type Custom Ticker(s) (comma-separated)",
                    value="",
                    placeholder="e.g. TCS, AXISBANK, BEL",
                    key="batch_custom_stocks",
                ).strip().upper()
            with b_add3:
                b_added_wt_pct = st.number_input(
                    "Target Wt per Added Stock (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=10.0,
                    step=1.0,
                    key="batch_added_wt_pct",
                )
            with b_add4:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                if st.button("Clear Added", key="batch_clear_added_btn"):
                    st.session_state["batch_added_tickers_list"] = []
                    st.rerun()

            b_custom_list = [p.strip().upper() for p in b_custom_raw.split(",") if p.strip()]
            b_combined_added = []
            for item in b_picked_universe + b_custom_list:
                if item and item != "NAN" and item not in b_combined_added:
                    b_combined_added.append(item)
            st.session_state["batch_added_tickers_list"] = b_combined_added

            # Normalize Model Target Weights + Newly Added Stocks
            m_clean = model_target_df[model_target_df["Ticker"].astype(str).str.strip() != ""].copy()
            if m_clean.empty and not b_combined_added:
                st.warning("Specify at least one ticker in Model Target Weights or Add Stock(s) above.")
                st.stop()

            raw_model_map = {}
            if not m_clean.empty:
                m_clean["Ticker"] = m_clean["Ticker"].astype(str).str.strip().str.upper()
                m_clean["Target_Weight_Pct"] = pd.to_numeric(m_clean["Target_Weight_Pct"], errors="coerce").fillna(0.0)
                for _, r in m_clean.iterrows():
                    raw_model_map[r["Ticker"]] = float(r["Target_Weight_Pct"])

            for added_stk in b_combined_added:
                if added_stk not in raw_model_map:
                    raw_model_map[added_stk] = float(b_added_wt_pct)

            tot_m_pct = sum(raw_model_map.values())
            model_wt_map = {
                t: (wt / tot_m_pct if tot_m_pct > 0 else 1.0 / len(raw_model_map))
                for t, wt in raw_model_map.items()
            }

            if b_combined_added:
                st.info(f"Normalized Model Weights (including added stocks): " + ", ".join([f"**{k}**: {v*100:.1f}%" for k, v in model_wt_map.items()]))

            # Gather all tickers across model + batch clients
            all_batch_tickers = set(model_wt_map.keys())
            parsed_client_holdings = []
            for _, crow in edited_batch_df.iterrows():
                h_dict = {}
                raw_h = crow.get("Holdings_JSON", "{}")
                try:
                    parsed_obj = json.loads(raw_h) if isinstance(raw_h, str) else (raw_h if isinstance(raw_h, dict) else {})
                    for k, v in parsed_obj.items():
                        ku = str(k).strip().upper()
                        if ku:
                            h_dict[ku] = float(v)
                            all_batch_tickers.add(ku)
                except Exception:
                    pass
                parsed_client_holdings.append(h_dict)

            all_t_list = sorted(all_batch_tickers)
            yf_b_syms = resolve_yf_tickers(all_t_list, current_cfg)
            with st.spinner("Fetching market prices & return histories for batch universe..."):
                end_dt_b = datetime.datetime.now()
                start_dt_b = end_dt_b - datetime.timedelta(days=365 * 3)
                p_b_raw = yf.download(yf_b_syms, start=start_dt_b, end=end_dt_b, progress=False)
                p_b_df = p_b_raw["Close"] if isinstance(p_b_raw, pd.DataFrame) and "Close" in p_b_raw else p_b_raw
                if isinstance(p_b_df, pd.Series):
                    p_b_df = p_b_df.to_frame(name=yf_b_syms[0])
                p_b_df = p_b_df.dropna(how="all", axis=1).ffill().bfill()

            valid_b_pairs = [(bt, yt) for bt, yt in zip(all_t_list, yf_b_syms) if yt in p_b_df.columns]
            if not valid_b_pairs:
                st.error("Unable to fetch market prices for the batch universe tickers.")
                st.stop()

            valid_b_bt = [bt for bt, _ in valid_b_pairs]
            valid_b_yt = [yt for _, yt in valid_b_pairs]
            price_b_valid = p_b_df[valid_b_yt]
            ret_b_df = price_b_valid.pct_change().dropna()

            latest_px_map = {bt: float(p_b_df[yt].iloc[-1]) for bt, yt in valid_b_pairs}

            # Fetch benchmark series once for batch comparison
            b_ret_bench = pd.Series(dtype=float)
            try:
                bm_b_raw = yf.download(current_cfg["symbol"], start=ret_b_df.index[0], end=ret_b_df.index[-1], progress=False)
                bm_b_ser = bm_b_raw["Close"].squeeze() if isinstance(bm_b_raw, pd.DataFrame) and "Close" in bm_b_raw else pd.Series(dtype=float)
                b_ret_bench = bm_b_ser.pct_change().dropna()
            except Exception:
                pass

            # Compute per-client full quant metrics, risk contributions & gate status
            client_rows_detail = {}
            summary_records = []
            for idx, (crow, h_dict) in enumerate(zip(edited_batch_df.iterrows(), parsed_client_holdings)):
                _, crow_ser = crow
                cid = str(crow_ser.get("Client_ID", f"CLT-{idx+1:03d}")).strip()
                pmask = str(crow_ser.get("PAN_Mask", "XXXX0000X")).strip()
                rtier = str(crow_ser.get("Risk_Tier", "Moderate")).strip()
                cash_val = float(crow_ser.get("Cash", 0.0) or 0.0)
                rat_str = str(crow_ser.get("Rationale", "") or "")

                c_vol_cap = float(crow_ser.get("Mandate_Vol_Cap", vol_cap_pct)) if override_limits_per_client else vol_cap_pct
                c_stk_cap = float(crow_ser.get("Max_Stock_Cap", stock_cap_pct)) if override_limits_per_client else stock_cap_pct

                curr_pos_vals = np.array([h_dict.get(t, 0.0) * latest_px_map.get(t, 0.0) for t in valid_b_bt], dtype=float)
                eq_val = float(curr_pos_vals.sum())
                tot_val = eq_val + cash_val
                c_wts = (curr_pos_vals / tot_val) if tot_val > 0 else np.zeros(len(valid_b_bt))

                target_equity_base = tot_val if deploy_cash else eq_val
                prop_wts = np.array([(model_wt_map.get(t, 0.0) * target_equity_base / tot_val) if tot_val > 0 else 0.0 for t in valid_b_bt], dtype=float)

                r_pre = ret_b_df.dot(c_wts) if c_wts.sum() > 0 else ret_b_df.iloc[:, 0] * 0.0
                r_post = ret_b_df.dot(prop_wts) if prop_wts.sum() > 0 else ret_b_df.iloc[:, 0] * 0.0

                c_pre_metrics = compute_quant_metrics(
                    r_pre, rfr_annual=rfr_annual, target_return_pct=target_return_pct,
                    confidence_level=confidence_level, horizon_days=time_horizon_days
                )
                c_post_metrics = compute_quant_metrics(
                    r_post, rfr_annual=rfr_annual, target_return_pct=target_return_pct,
                    confidence_level=confidence_level, horizon_days=time_horizon_days
                )

                c_pre_div = compute_risk_and_diversification(ret_b_df, c_wts, valid_b_bt)
                c_post_div = compute_risk_and_diversification(ret_b_df, prop_wts, valid_b_bt)

                post_max_s = float(prop_wts.max() * 100.0) if len(prop_wts) else 0.0
                pre_max_s = float(c_wts.max() * 100.0) if len(c_wts) else 0.0
                sec_map_c = pd.Series(c_wts, index=[ticker_to_sector.get(t, "Other") for t in valid_b_bt])
                sec_map_p = pd.Series(prop_wts, index=[ticker_to_sector.get(t, "Other") for t in valid_b_bt])
                pre_max_sec = float(sec_map_c.groupby(level=0).sum().max() * 100.0) if not sec_map_c.empty else 0.0
                post_max_sec = float(sec_map_p.groupby(level=0).sum().max() * 100.0) if not sec_map_p.empty else 0.0

                turn_pct = float(np.abs(prop_wts - c_wts).sum() * 0.5 * 100.0)
                fric_amt = float(np.abs(prop_wts - c_wts).sum() * 0.5 * tot_val * (batch_friction_bps / 10000.0))

                g_res = evaluate_mandate_gate(
                    post_vol_pct=c_post_metrics["ann_vol"],
                    post_cvar_pct=c_post_metrics["hist_cvar"],
                    max_stock_wt_pct=post_max_s,
                    max_sector_wt_pct=post_max_sec,
                    vol_cap_pct=c_vol_cap,
                    cvar_cap_pct=cvar_cap_pct,
                    stock_cap_pct=c_stk_cap,
                    sector_cap_pct=sector_cap_pct,
                    rationale_text=rat_str,
                )

                summary_records.append({
                    "Client_ID": cid,
                    "PAN_Mask": pmask,
                    "Risk_Tier": rtier,
                    "Portfolio_Val": tot_val,
                    "Turnover (%)": turn_pct,
                    "Post_Ret (%)": c_post_metrics["ann_ret"],
                    "Post_Vol (%)": c_post_metrics["ann_vol"],
                    "Sharpe": c_post_metrics["sharpe"],
                    "Sortino": c_post_metrics["sortino"],
                    "Post_CVaR (%)": c_post_metrics["hist_cvar"],
                    "Max_DD (%)": c_post_metrics["max_dd"],
                    "Wt_ENC (%)": c_post_div["weight_enc"],
                    "Risk_ENC (%)": c_post_div["risk_enc"],
                    "Max_Stock (%)": post_max_s,
                    "Gate_Status": g_res["status"],
                    "Rationale_Status": "Logged" if not g_res["missing_rationale"] else "Missing",
                    "Flags": "; ".join(g_res["breaches"] + g_res["warnings"]) if (g_res["breaches"] or g_res["warnings"]) else "OK",
                })

                client_rows_detail[cid] = {
                    "cid": cid, "pmask": pmask, "rtier": rtier, "tot_val": tot_val, "cash_val": cash_val,
                    "c_wts": c_wts, "prop_wts": prop_wts, "r_pre": r_pre, "r_post": r_post,
                    "pre_metrics": c_pre_metrics, "post_metrics": c_post_metrics,
                    "pre_div": c_pre_div, "post_div": c_post_div,
                    "pre_max_s": pre_max_s, "post_max_s": post_max_s,
                    "pre_max_sec": pre_max_sec, "post_max_sec": post_max_sec,
                    "turn_pct": turn_pct, "fric_amt": fric_amt,
                    "rationale": rat_str, "gate": g_res,
                    "diff_df": pd.DataFrame({
                        "Ticker": valid_b_bt,
                        "Sector": [ticker_to_sector.get(t, "Other") for t in valid_b_bt],
                        "Live Price": [latest_px_map.get(t, 0.0) for t in valid_b_bt],
                        "Current Wt (%)": c_wts * 100.0,
                        "Proposed Wt (%)": prop_wts * 100.0,
                        "Delta Wt (%)": (prop_wts - c_wts) * 100.0,
                        "Trade Amt": np.abs(prop_wts - c_wts) * tot_val,
                    }),
                }

            # ------------------------------------------------------------------
            # STAGE 3 (BATCH): ROLL-UP SUMMARY TABLE + FULL DRILL-DOWN ANALYTICS
            # ------------------------------------------------------------------
            st.markdown("---")
            st.markdown("## Stage 3: Pre-Trade Diff, Benchmark Comparison & Risk / Monte Carlo (Batch)")

            sum_df = pd.DataFrame(summary_records)
            pass_cnt = int((sum_df["Gate_Status"] == "Pass").sum())
            warn_cnt = int((sum_df["Gate_Status"] == "Warning").sum())
            breach_cnt = int((sum_df["Gate_Status"] == "Breach").sum())
            miss_rat_cnt = int((sum_df["Rationale_Status"] == "Missing").sum())

            bk1, bk2, bk3, bk4 = st.columns(4)
            bk1.metric("Total Clients", f"{len(sum_df)}")
            bk2.metric("Pass / Warning", f"{pass_cnt} Pass / {warn_cnt} Warn")
            bk3.metric("Mandate / Conc. Breach", f"{breach_cnt}")
            bk4.metric("Missing Rationale", f"{miss_rat_cnt}")

            st.dataframe(
                sum_df.style.format({
                    "Portfolio_Val": f"{currency} {{:,.2f}}",
                    "Turnover (%)": "{:.2f}%",
                    "Post_Ret (%)": "{:+.2f}%",
                    "Post_Vol (%)": "{:.2f}%",
                    "Sharpe": "{:.2f}",
                    "Sortino": "{:.2f}",
                    "Post_CVaR (%)": "{:.2f}%",
                    "Max_DD (%)": "{:.2f}%",
                    "Wt_ENC (%)": "{:.1f}%",
                    "Risk_ENC (%)": "{:.1f}%",
                    "Max_Stock (%)": "{:.2f}%",
                }),
                use_container_width=True,
                hide_index=True,
            )

            selected_cid = st.selectbox("Drill-Down Inspector — Select Client Row", options=list(client_rows_detail.keys()), key="batch_drill_selector")
            d_item = client_rows_detail[selected_cid]
            st.markdown(f"**Client Drill-Down: {d_item['cid']} ({d_item['pmask']} | {d_item['rtier']}) — Gate: {d_item['gate']['status']}**")

            # Order Diff Table for Selected Client
            st.dataframe(
                d_item["diff_df"].style.format({
                    "Live Price": f"{currency} {{:,.2f}}",
                    "Current Wt (%)": "{:.2f}%",
                    "Proposed Wt (%)": "{:.2f}%",
                    "Delta Wt (%)": "{:+.2f}%",
                    "Trade Amt": f"{currency} {{:,.2f}}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            # Benchmark alignment for Selected Batch Client
            b_beta_val = 1.0
            b_cum_curve = None
            if not b_ret_bench.empty:
                aligned_b = pd.DataFrame({"pre": d_item["r_pre"], "post": d_item["r_post"], "m": b_ret_bench}).dropna()
                if len(aligned_b) > 10 and np.var(aligned_b["m"]) > 0:
                    b_beta_val = float(np.cov(aligned_b["post"], aligned_b["m"])[0, 1] / np.var(aligned_b["m"]))
                if not aligned_b.empty:
                    b_pre_cum = ((1.0 + aligned_b["pre"]).cumprod() - 1.0) * 100.0
                    b_post_cum = ((1.0 + aligned_b["post"]).cumprod() - 1.0) * 100.0
                    b_cum_curve = ((1.0 + aligned_b["m"]).cumprod() - 1.0) * 100.0
                else:
                    b_pre_cum = ((1.0 + d_item["r_pre"]).cumprod() - 1.0) * 100.0
                    b_post_cum = ((1.0 + d_item["r_post"]).cumprod() - 1.0) * 100.0
            else:
                b_pre_cum = ((1.0 + d_item["r_pre"]).cumprod() - 1.0) * 100.0
                b_post_cum = ((1.0 + d_item["r_post"]).cumprod() - 1.0) * 100.0

            # Full Parity Analytics: Metrics (Sharpe, Sortino, Return, Vol, VaR, CVaR, Drawdown), ENC, Risk Contribution, Benchmark Graph & Cholesky Monte Carlo
            render_risk_diversification_and_mc_section(
                prefix_key=f"batch_{selected_cid}",
                currency=currency,
                time_horizon_days=time_horizon_days,
                confidence_level=confidence_level,
                pre_metrics=d_item["pre_metrics"],
                post_metrics=d_item["post_metrics"],
                pre_div=d_item["pre_div"],
                post_div=d_item["post_div"],
                pre_cum_pct=b_pre_cum,
                post_cum_pct=b_post_cum,
                b_cum_pct=b_cum_curve,
                benchmark_name=current_cfg["name"],
                price_matrix_valid=price_b_valid,
                prop_wts=d_item["prop_wts"],
                post_nav=d_item["tot_val"],
                turnover_pct=d_item["turn_pct"],
                total_friction_amt=d_item["fric_amt"],
                beta_val=b_beta_val,
                pre_max_stock=d_item["pre_max_s"],
                post_max_stock=d_item["post_max_s"],
                pre_max_sec=d_item["pre_max_sec"],
                post_max_sec=d_item["post_max_sec"],
            )

            # ------------------------------------------------------------------
            # STAGE 4 (BATCH): LOG RATIONALE & EXPORT BATCH AUDIT PACKET
            # ------------------------------------------------------------------
            st.markdown("---")
            st.markdown("## Stage 4: Rationale & Audit Export (Batch)")
            batch_rationale_val = st.text_area(
                f"Suitability & Model Rebalance Rationale for {selected_cid}",
                value=d_item["rationale"],
                placeholder="Enter suitability rationale for this client's model rebalance...",
                key=f"batch_rat_input_{selected_cid}",
            )
            if st.button("Save Rationale to Selected Client", key="btn_save_batch_rat"):
                d_item["rationale"] = batch_rationale_val
                st.success(f"Rationale updated for {selected_cid}.")

            b_gate = evaluate_mandate_gate(
                post_vol_pct=d_item["post_metrics"]["ann_vol"],
                post_cvar_pct=d_item["post_metrics"]["hist_cvar"],
                max_stock_wt_pct=d_item["post_max_s"],
                max_sector_wt_pct=d_item["post_max_sec"],
                vol_cap_pct=vol_cap_pct,
                cvar_cap_pct=cvar_cap_pct,
                stock_cap_pct=stock_cap_pct,
                sector_cap_pct=sector_cap_pct,
                rationale_text=batch_rationale_val,
            )
            b_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            b_pdf = build_audit_packet_pdf(
                entity_type=entity_type,
                workspace_mode=workspace_mode,
                exchange=exchange,
                currency=currency,
                client_label=f"Batch ({len(sum_df)} clients) | Focus: {selected_cid}",
                reg_no="SEBI-BATCH-AUDIT",
                officer_or_advisor=entity_type,
                mandate_limits=mandate_limits,
                diff_summary={
                    "turnover_pct": d_item["turn_pct"],
                    "friction_cost": d_item["fric_amt"],
                    "pre_vol": d_item["pre_metrics"]["ann_vol"],
                    "post_vol": d_item["post_metrics"]["ann_vol"],
                    "pre_cvar": d_item["pre_metrics"]["hist_cvar"],
                    "post_cvar": d_item["post_metrics"]["hist_cvar"],
                    "pre_max_stock": d_item["pre_max_s"],
                    "post_max_stock": d_item["post_max_s"],
                    "pre_max_sector": d_item["pre_max_sec"],
                    "post_max_sector": d_item["post_max_sec"],
                },
                gate_result=b_gate,
                rationale_text=batch_rationale_val,
                timestamp=b_ts,
            )
            dl_c1, dl_c2 = st.columns(2)
            with dl_c1:
                st.download_button(
                    label="📥 Export Batch Summary Roll-Up (CSV)",
                    data=sum_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"WIRATOR_Batch_Summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
            with dl_c2:
                st.download_button(
                    label="📄 Export Timestamped Audit Report (PDF)",
                    data=b_pdf,
                    file_name=f"WIRATOR_Batch_Audit_{selected_cid}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    type="primary",
                )

    # Sidebar Feedback Form (Preserved)
    with st.sidebar:
        st.markdown("---")
        st.subheader("💬 Feedback & Feature Request")
        with st.form("feedback_form", clear_on_submit=True):
            user_email = st.text_input("Your Email (optional)")
            feedback_type = st.selectbox("Feedback Type", ["Bug Report", "Feature Request", "General Feedback"])
            user_message = st.text_area("Your Message / Suggestion")
            submitted = st.form_submit_button("Submit Feedback")
        if submitted:
            if user_message.strip():
                try:
                    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
                    creds_dict = dict(st.secrets["gcp_service_account"])
                    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
                    client = gspread.authorize(creds)
                    sheet = client.open("Wirator User Feedback").sheet1
                    sheet.append_row([
                        pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        user_email if user_email else "Anonymous",
                        feedback_type,
                        exchange if "exchange" in globals() else "NSE",
                        user_message,
                    ])
                    st.success("Thank you! Your feedback has been recorded.")
                except Exception as e:
                    st.error(f"Failed to record feedback: {e}")
            else:
                st.warning("Please enter a message before submitting.")

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import io
import re
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


OPTIMIZER_LABELS = {
    "EW": "Equally Weighted (EW)",
    "GMV": "Global Min Volatility (GMV)",
    "MSR": "Max Sharpe Ratio (MSR)",
    "ERC": "Equal Risk Contribution (ERC)",
}


def format_short_sector(sector_name):
    """Strips leading 'Nifty ' or 'BSE ' prefixes to save table column width."""
    if not sector_name or pd.isna(sector_name):
        return "Other"
    cleaned = re.sub(r"^(Nifty|BSE)\s+", "", str(sector_name).strip(), flags=re.IGNORECASE)
    return cleaned if cleaned else str(sector_name).strip()


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
        possible_days_cols = ["Holding_Days", "Holding Days", "Days Held", "HoldingPeriod", "Holding_Period", "Days", "Age_Days"]

        id_col = next((col for col in possible_id_cols if col in raw_df.columns), None)
        ticker_col = next((col for col in possible_ticker_cols if col in raw_df.columns), raw_df.columns[0])
        qty_col = next((col for col in possible_qty_cols if col in raw_df.columns), raw_df.columns[1] if len(raw_df.columns) > 1 else raw_df.columns[0])
        cost_col = next((col for col in possible_cost_cols if col in raw_df.columns), None)
        days_col = next((col for col in possible_days_cols if col in raw_df.columns), None)

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

            holding_days = np.nan
            if days_col and pd.notna(row[days_col]):
                try:
                    holding_days = int(float(str(row[days_col]).strip()))
                except (ValueError, TypeError):
                    holding_days = np.nan

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
                row_dict = {"Ticker": base_symbol, "Quantity": int(qty), "Cost Price": cost_price}
                if days_col is not None:
                    row_dict["Holding_Days"] = holding_days
                parsed_rows.append(row_dict)

        holdings_df = pd.DataFrame(parsed_rows)
        has_cost = bool(has_valid_cost_col and valid_cost_entries > 0)
        if not holdings_df.empty:
            st.success("Portfolio successfully loaded!")
            return holdings_df, has_cost
    except Exception as e:
        st.warning(f"Could not parse custom CSV format: {e}. Using fallback portfolio.")
    return pd.DataFrame(columns=["Ticker", "Quantity", "Cost Price"]), False


def compute_optimizer_target_weights(ret_df, ordered_tickers, ordered_yf_syms, method_key, rfr_annual=0.065):
    """Computes target weights (%) using quant_risk_kit functions (weight_ew, weight_gmv, msr, weight_erc)."""
    n_assets = len(ordered_tickers)
    if n_assets == 0:
        return {}
    if ret_df.empty or len(ordered_yf_syms) == 0:
        eq = round(100.0 / n_assets, 2)
        return {t: eq for t in ordered_tickers}

    sub_ret = ret_df[ordered_yf_syms].dropna()
    if sub_ret.empty or sub_ret.shape[0] < 5:
        eq = round(100.0 / n_assets, 2)
        return {t: eq for t in ordered_tickers}

    try:
        if method_key == "EW":
            w_arr = np.asarray(erk.weight_ew(sub_ret), dtype=float)
        elif method_key == "GMV":
            w_arr = np.asarray(erk.weight_gmv(sub_ret, cov_estimator=erk.shrinkage_cov), dtype=float)
        elif method_key == "MSR":
            er_ann = erk.annualized_returns(sub_ret, periods_per_year=252)
            cov_ann = erk.shrinkage_cov(sub_ret) * 252.0
            w_arr = np.asarray(erk.msr(er_ann, cov_ann, risk_free_rate=rfr_annual), dtype=float)
        elif method_key == "ERC":
            w_arr = np.asarray(erk.weight_erc(sub_ret, cov_estimator=erk.shrinkage_cov), dtype=float)
        else:
            w_arr = np.repeat(1.0 / n_assets, n_assets)
    except Exception:
        w_arr = np.repeat(1.0 / n_assets, n_assets)

    w_arr = np.clip(w_arr, 0.0, 1.0)
    if w_arr.sum() > 0:
        w_arr = w_arr / w_arr.sum()
    else:
        w_arr = np.repeat(1.0 / n_assets, n_assets)

    pcts = np.round(w_arr * 100.0, 2)
    diff_100 = round(100.0 - float(pcts.sum()), 2)
    if abs(diff_100) <= 0.10 and len(pcts) > 0:
        max_i = int(np.argmax(pcts))
        pcts[max_i] = round(max(0.0, pcts[max_i] + diff_100), 2)

    return {bt: float(pcts[i]) for i, bt in enumerate(ordered_tickers)}


def calculate_sell_tax_for_row(action_str, live_price, cost_price, current_qty, proposed_qty, tax_regime):
    """Calculates What-If Capital Gains Tax on realized P&L of the sold shares only; returns (tax_amount_or_nan, display_str)."""
    is_sell = "SELL" in str(action_str).upper()
    sold_qty = max(0.0, float(current_qty) - float(proposed_qty))
    if not is_sell or sold_qty <= 1e-6:
        return np.nan, "N/A"
    if cost_price is None or np.isnan(cost_price) or float(cost_price) <= 0:
        return np.nan, "N/A (No Cost)"
    realized_gain = (float(live_price) - float(cost_price)) * sold_qty
    if realized_gain <= 0:
        return 0.0, "N/A"
    rate = 0.20 if str(tax_regime).upper().startswith("STCG") else 0.125
    tax_amt = realized_gain * rate
    return float(tax_amt), f"{tax_amt:,.2f}"


def build_broker_execution_csv_df(
    client_id,
    exchange,
    tickers,
    live_prices,
    curr_qtys,
    prop_vals,
    friction_rate,
    starting_cash,
    cash_flow_delta=0.0,
):
    """Builds broker execution CSV with columns: Client_ID, Exchange, Ticker, Series, Action, Order_Qty, Limit_Price, Est_Notional, Est_Friction, Residual_Cash."""
    series_code = "EQ" if exchange in ["NSE", "BSE"] else "COMMON"
    rows = []
    running_cash = float(starting_cash) + float(cash_flow_delta)

    for i, tkr in enumerate(tickers):
        lp = float(live_prices[i])
        cq = float(curr_qtys[i])
        pv = float(prop_vals[i])
        target_qty = round(pv / lp) if lp > 0 else 0
        delta_qty = int(target_qty - round(cq))

        if delta_qty > 0:
            action = "BUY"
            order_qty = delta_qty
            est_notional = order_qty * lp
            est_friction = est_notional * friction_rate
            running_cash -= (est_notional + est_friction)
        elif delta_qty < 0:
            action = "SELL"
            order_qty = abs(delta_qty)
            est_notional = order_qty * lp
            est_friction = est_notional * friction_rate
            running_cash += (est_notional - est_friction)
        else:
            continue

        rows.append({
            "Client_ID": client_id,
            "Exchange": exchange,
            "Ticker": tkr,
            "Series": series_code,
            "Action": action,
            "Order_Qty": int(order_qty),
            "Limit_Price": round(lp, 2),
            "Est_Notional": round(est_notional, 2),
            "Est_Friction": round(est_friction, 2),
            "Residual_Cash": round(running_cash, 2),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "Client_ID", "Exchange", "Ticker", "Series", "Action",
            "Order_Qty", "Limit_Price", "Est_Notional", "Est_Friction", "Residual_Cash"
        ])
    return pd.DataFrame(rows)


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

    stats_df = erk.summary_stats(s.to_frame("Portfolio"), riskfree_rate=rfr_annual, periods_per_year=periods_per_year)
    ann_ret = float(stats_df["Annualized Return"].iloc[0] * 100.0)
    ann_vol = float(stats_df["Annualized Vol"].iloc[0] * 100.0)
    sharpe_val = float(stats_df["Sharpe Ratio"].iloc[0])
    max_dd = float(abs(stats_df["Max Drawdown"].iloc[0]) * 100.0)

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
    sim_steps = steps + 1

    clean_prices = price_matrix.copy()
    for col in clean_prices.columns:
        if clean_prices[col].pct_change().dropna().std() == 0:
            clean_prices[col] = clean_prices[col] * (1.0 + np.linspace(0, 1e-5, len(clean_prices)))

    all_paths = np.zeros((steps + 1, sims), dtype=float)
    all_paths[0, :] = float(initial_val)

    for s_idx in range(sims):
        try:
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
        "p5": np.percentile(all_paths, (1 - confidence_level) * 100, axis=1),
        "p50": np.percentile(all_paths, 50, axis=1),
        "p95": np.percentile(all_paths, confidence_level * 100, axis=1),
        "mean_end": float(np.mean(terminal_vals)),
        "median_end": float(np.median(terminal_vals)),
        "p5_end": float(np.percentile(terminal_vals, (1 - confidence_level) * 100)),
        "p95_end": float(np.percentile(terminal_vals, confidence_level * 100)),
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
    tax_impact_amt=None,
    tax_regime_label="Per-Row LTCG/STCG",
):
    """Renders identical comprehensive Pre vs Post metrics, Benchmark Graph, Weight/Risk Diversification & Monte Carlo for both Single Client & Batch."""
    pm1, pm2, pm3, pm4, pm5 = st.columns(5)
    pm1.metric("Annualized Vol", f"{post_metrics['ann_vol']:.2f}%", delta=f"{post_metrics['ann_vol'] - pre_metrics['ann_vol']:+.2f}% vs pre", delta_color="inverse")
    pm2.metric(f"{time_horizon_days}D Hist CVaR ({int(confidence_level*100)}%)", f"{post_metrics['hist_cvar']:.2f}%", delta=f"{post_metrics['hist_cvar'] - pre_metrics['hist_cvar']:+.2f}% vs pre", delta_color="inverse")
    pm3.metric(f"{time_horizon_days}D Hist VaR ({int(confidence_level*100)}%)", f"{post_metrics['hist_var']:.2f}%", delta=f"{post_metrics['hist_var'] - pre_metrics['hist_var']:+.2f}% vs pre", delta_color="inverse")
    pm4.metric(f"{time_horizon_days}D Mod. CF VaR", f"{post_metrics['cf_var']:.2f}%", delta=f"{post_metrics['cf_var'] - pre_metrics['cf_var']:+.2f}% vs pre", delta_color="inverse")
    pm5.metric("Turnover / Friction", f"{turnover_pct:.2f}%", delta=f"{currency} {total_friction_amt:,.0f} (Beta {beta_val:.2f})", delta_color="off")

    qm1, qm2, qm3, qm4, qm5 = st.columns(5)
    qm1.metric("Ann. Return (Post)", f"{post_metrics['ann_ret']:+.2f}%", delta=f"{post_metrics['ann_ret'] - pre_metrics['ann_ret']:+.2f}% vs pre")
    qm2.metric("Sharpe Ratio (erk)", f"{post_metrics['sharpe']:.2f}", delta=f"{post_metrics['sharpe'] - pre_metrics['sharpe']:+.2f} vs pre")
    qm3.metric("Sortino Ratio", f"{post_metrics['sortino']:.2f}", delta=f"{post_metrics['sortino'] - pre_metrics['sortino']:+.2f} vs pre")
    qm4.metric("Max Drawdown", f"{post_metrics['max_dd']:.2f}%", delta=f"{post_metrics['max_dd'] - pre_metrics['max_dd']:+.2f}% vs pre", delta_color="inverse")
    qm5.metric("Max Stock / Sector", f"{post_max_stock:.1f}% / {post_max_sec:.1f}%", delta=f"Stk {post_max_stock - pre_max_stock:+.1f}% | Sec {post_max_sec - pre_max_sec:+.1f}%", delta_color="off")

    dv1, dv2, dv3, dv4, dv5 = st.columns(5)
    dv1.metric("Weight Diversification (ENC)", f"{post_div['weight_enc']:.1f}%", delta=f"{post_div['weight_enc'] - pre_div['weight_enc']:+.1f}% vs pre")
    dv2.metric("Risk Diversification (ENC)", f"{post_div['risk_enc']:.1f}%", delta=f"{post_div['risk_enc'] - pre_div['risk_enc']:+.1f}% vs pre")
    dv3.metric("Pre-Trade Weight ENC", f"{pre_div['weight_enc']:.1f}%")
    dv4.metric("Pre-Trade Risk ENC", f"{pre_div['risk_enc']:.1f}%")
    if tax_impact_amt is None or np.isnan(tax_impact_amt) or tax_impact_amt <= 0:
        dv5.metric("Est. Rebalance Tax", "N/A", delta="No taxable sell", delta_color="off")
    else:
        dv5.metric("Est. Rebalance Tax", f"{currency} {tax_impact_amt:,.2f}", delta=f"{tax_regime_label}", delta_color="inverse")

    if not pre_metrics["summary_df"].empty and not post_metrics["summary_df"].empty:
        with st.expander("📋 View Full Summary Table (Pre-Trade vs. Post-Trade)", expanded=False):
            comb_stats = pd.concat([
                pre_metrics["summary_df"].rename(index={"Portfolio": "Pre-Trade Portfolio"}),
                post_metrics["summary_df"].rename(index={"Portfolio": "Post-Trade Portfolio"}),
            ])
            st.dataframe(comb_stats, use_container_width=True)

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
    tax_str = f"{currency} {diff_summary.get('tax_impact', 0.0):,.2f}" if diff_summary.get("tax_impact", 0.0) > 0 else "N/A"
    pdf.cell(0, 5.5, _fpdf_clean(f"Turnover: {diff_summary.get('turnover_pct', 0.0):.2f}% | Est. Friction: {currency} {diff_summary.get('friction_cost', 0.0):,.2f} | Est. Tax: {tax_str}"), ln=True)
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
            <div style="font-size: 4rem; margin-bottom: 0.25rem;">🛡️</div>
            <h1 class="hero-title">WIRATOR</h1>
            <div class="hero-subtitle">Quant PRE-Trade Optimization, Risk & Compliance Engine</div>
            <div class="slogan-badge">Test the Shift. Keep The Proof..</div>
            <p class="hero-description">
                Turn portfolio changes into clear, measurable decisions—tracking volatility, tail risk, 
                diversification, friction, and capital gains impact before execution, backed by a clean audit trail.
                Leave with timestamped suitability records that back up your judgment across single clients and batches.
                Includes four built-in portfolio optimizers.
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
        .opt-badge {
            background-color: #1A2B4C;
            color: #FAFAED !important;
            border-radius: 8px;
            padding: 8px 14px;
            margin: 8px 0 14px 0;
            font-weight: 600;
            border-left: 5px solid #22C55E;
        }
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

    try:
        with open("stock-sectors-nse-bse.json", "r", encoding="utf-8") as f:
            ALL_EXCHANGE_CONFIGS = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Could not find 'stock-sectors-nse-bse.json' in the app's working directory. "
            "Make sure it is committed to the root of your GitHub repo next to app2.py."
        )
    sector_config = ALL_EXCHANGE_CONFIGS.get(exchange, ALL_EXCHANGE_CONFIGS["NSE"])
    ticker_to_sector = {}
    for sec_name, sec_data in sector_config.items():
        short_sec = format_short_sector(sec_name)
        for stk in sec_data["stocks"]:
            base_s = stk.split(".")[0].upper()
            ticker_to_sector[base_s] = short_sec
            ticker_to_sector[stk.upper()] = short_sec

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
            short_sec_name = format_short_sector(sector_name)
            all_tickers_to_fetch.extend(data["indices"])
            all_tickers_to_fetch.extend(data["stocks"])
            for stock in data["stocks"]:
                stock_to_sector_map[stock] = short_sec_name

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
                short_sec_name = format_short_sector(sector_name)
                sec_stock_rets = [
                    stock_returns_dict[stk]
                    for stk in data["stocks"]
                    if stk in stock_returns_dict
                ]
                if sec_stock_rets:
                    sector_returns[short_sec_name] = np.mean(sec_stock_rets)

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
        selected_sector = st.selectbox(
            "Choose a Sector to Analyze:",
            options=available_sectors,
            format_func=format_short_sector,
        )

        if selected_sector:
            short_selected_sec = format_short_sector(selected_sector)
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
                    st.markdown(f"### Performance Breakdown: **{short_selected_sec}** ({timeframe})")
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
                        title=f"{short_selected_sec} Stock Returns — {timeframe}",
                        color="Return (%)",
                        color_continuous_scale="Viridis",
                    )
                    fig_sector_drill.update_layout(yaxis={"categoryorder": "total ascending"}, dragmode=False)
                    st.plotly_chart(fig_sector_drill, use_container_width=True)
            else:
                st.info(f"No price data available for stocks in {short_selected_sec}.")

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
                        detected_days = parsed_for_tax["Holding_Days"].fillna(int(time_horizon_days)) if "Holding_Days" in parsed_for_tax.columns else int(time_horizon_days)
                        base_tax_df = pd.DataFrame({
                            "Instrument": parsed_for_tax["Ticker"],
                            "Avg_Cost": parsed_for_tax["Cost Price"],
                            "Qty": parsed_for_tax["Quantity"].clip(lower=1),
                            "Holding_Days": detected_days,
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

            valid_yf_ordered = [yt for _, yt in valid_pairs]
            valid_bt_ordered = [bt for bt, _ in valid_pairs]
            price_matrix_valid = p_df[valid_yf_ordered]
            ret_matrix = price_matrix_valid.pct_change().dropna()

            # Portfolio Optimizer Presets (using erk.weight_ew, erk.weight_gmv, erk.msr, erk.weight_erc)
            st.markdown("#### ⚙️ Portfolio Optimizer Presets (Auto-Fill Target Weights)")
            if "sc_active_optimizer" not in st.session_state:
                st.session_state["sc_active_optimizer"] = None
            if "sc_last_opt_weights" not in st.session_state:
                st.session_state["sc_last_opt_weights"] = {}

            opt_c1, opt_c2, opt_c3, opt_c4 = st.columns(4)
            clicked_opt = None
            with opt_c1:
                if st.button("⚖️ Equally Weighted (EW)", use_container_width=True, key="sc_opt_ew"):
                    clicked_opt = "EW"
            with opt_c2:
                if st.button("🛡️ Global Min Variance (GMV)", use_container_width=True, key="sc_opt_gmv"):
                    clicked_opt = "GMV"
            with opt_c3:
                if st.button("🚀 Max Sharpe Ratio (MSR)", use_container_width=True, key="sc_opt_msr"):
                    clicked_opt = "MSR"
            with opt_c4:
                if st.button("🎯 Equal Risk Contribution (ERC)", use_container_width=True, key="sc_opt_erc"):
                    clicked_opt = "ERC"

            if clicked_opt is not None:
                opt_wts_map = compute_optimizer_target_weights(
                    ret_matrix, valid_bt_ordered, valid_yf_ordered, clicked_opt, rfr_annual=rfr_annual
                )
                for bt_opt, w_opt in opt_wts_map.items():
                    st.session_state[f"sc_tw_{bt_opt}"] = float(w_opt)
                st.session_state["sc_active_optimizer"] = clicked_opt
                st.session_state["sc_last_opt_weights"] = dict(opt_wts_map)
                st.rerun()

            qty_map = {}
            cost_val_sum_map = {}
            days_map = {}
            has_holding_days_col = "Holding_Days" in client_holdings_df.columns
            for _, r in client_holdings_df.iterrows():
                bt = str(r.get("Ticker", "")).strip().upper()
                if bt:
                    q_row = float(r.get("Quantity", 0.0) or 0.0)
                    c_row = pd.to_numeric(r.get("Cost Price", np.nan), errors="coerce")
                    qty_map[bt] = qty_map.get(bt, 0.0) + q_row
                    if pd.notna(c_row) and c_row > 0 and q_row > 0:
                        cost_val_sum_map[bt] = cost_val_sum_map.get(bt, 0.0) + (float(c_row) * q_row)
                    if has_holding_days_col and pd.notna(r.get("Holding_Days", np.nan)):
                        days_map[bt] = int(float(r.get("Holding_Days")))

            avg_cost_map = {
                bt: (cost_val_sum_map[bt] / qty_map[bt]) if (bt in cost_val_sum_map and qty_map.get(bt, 0.0) > 0) else np.nan
                for bt in valid_bt_ordered
            }

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

            # Display which Portfolio Optimizer Preset is currently active
            active_opt_code = st.session_state.get("sc_active_optimizer")
            if active_opt_code:
                last_opt_w = st.session_state.get("sc_last_opt_weights", {})
                is_exact_match = (
                    set(last_opt_w.keys()) == set(valid_bt_ordered)
                    and all(abs(last_opt_w[bt] - raw_target_pcts[i]) < 0.02 for i, bt in enumerate(valid_bt_ordered))
                )
                opt_full_name = OPTIMIZER_LABELS.get(active_opt_code, active_opt_code)
                if is_exact_match:
                    st.markdown(
                        f"""<div class="opt-badge">
                            📌 Active Optimizer Results: <b>{opt_full_name}</b> — Target weights and rebalance figures below reflect the <b>{active_opt_code}</b> optimization.
                        </div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""<div class="opt-badge" style="border-left-color: #F59E0B;">
                            ✏️ Active Allocation: <b>Custom / Manual</b> (started from <b>{opt_full_name}</b> preset)
                        </div>""",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    """<div class="opt-badge" style="border-left-color: #38BDF8;">
                        🧭 Active Allocation: <b>Manual / Custom Target Weights</b> (click an optimizer preset above to auto-fill)
                    </div>""",
                    unsafe_allow_html=True,
                )

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

            actions_list = [
                "EXIT / SELL" if (prop_wts[i] <= 1e-6 and curr_wts[i] > 1e-6)
                else ("NEW / BUY" if (curr_wts[i] <= 1e-6 and prop_wts[i] > 1e-6)
                else ("BUY" if trade_notional_per_asset[i] > 1.0 else ("SELL" if trade_notional_per_asset[i] < -1.0 else "HOLD")))
                for i in range(len(valid_pairs))
            ]

            has_any_sell_sc = any("SELL" in act for act in actions_list)
            if "sc_per_stock_tax_map" not in st.session_state:
                st.session_state["sc_per_stock_tax_map"] = {}
            if "sc_last_bulk_tax_regime" not in st.session_state:
                st.session_state["sc_last_bulk_tax_regime"] = "LTCG (12.5%)"

            if has_any_sell_sc:
                tx_tog_c1, tx_tog_c2 = st.columns([2.0, 2.0])
                with tx_tog_c1:
                    sc_bulk_tax_regime = st.radio(
                        "Set All Sellable Stocks at Once (Bulk Tax Regime)",
                        options=["LTCG (12.5%)", "STCG (20%)"],
                        index=0 if st.session_state["sc_last_bulk_tax_regime"] == "LTCG (12.5%)" else 1,
                        horizontal=True,
                        key="sc_whatif_tax_regime",
                    )
                with tx_tog_c2:
                    st.caption("Use the radio button to switch all sellable stocks at once, or change the **Tax Regime** dropdown on individual SELL rows in the table below.")

                # If user toggled the bulk radio button, apply that choice to all sellable stocks at once
                if sc_bulk_tax_regime != st.session_state["sc_last_bulk_tax_regime"]:
                    st.session_state["sc_last_bulk_tax_regime"] = sc_bulk_tax_regime
                    for i_s, (bt_s, _) in enumerate(valid_pairs):
                        if "SELL" in actions_list[i_s]:
                            st.session_state["sc_per_stock_tax_map"][bt_s] = sc_bulk_tax_regime
            else:
                sc_bulk_tax_regime = st.session_state["sc_last_bulk_tax_regime"]

            # Initialize default regime for each sellable stock if not yet set
            for i, (bt, _) in enumerate(valid_pairs):
                if "SELL" in actions_list[i]:
                    if bt not in st.session_state["sc_per_stock_tax_map"] or st.session_state["sc_per_stock_tax_map"][bt] not in ["LTCG (12.5%)", "STCG (20%)"]:
                        if bt in days_map:
                            st.session_state["sc_per_stock_tax_map"][bt] = "STCG (20%)" if days_map[bt] < 365 else "LTCG (12.5%)"
                        else:
                            st.session_state["sc_per_stock_tax_map"][bt] = sc_bulk_tax_regime
                else:
                    st.session_state["sc_per_stock_tax_map"][bt] = "N/A"

            row_regimes_initial = [st.session_state["sc_per_stock_tax_map"][bt] for bt in valid_bt_ordered]

            def compute_sc_tax_columns(regimes_seq):
                amts = []
                strs = []
                for idx_r, (bt_r, yt_r) in enumerate(valid_pairs):
                    lp_r = float(p_df[yt_r].iloc[-1])
                    cq_r = qty_map.get(bt_r, 0.0)
                    pq_r = (prop_val_per_asset[idx_r] / lp_r) if lp_r > 0 else 0.0
                    reg_r = regimes_seq[idx_r]
                    if "SELL" not in actions_list[idx_r] or reg_r not in ["LTCG (12.5%)", "STCG (20%)"]:
                        amts.append(np.nan)
                        strs.append("N/A")
                    else:
                        t_amt, t_disp = calculate_sell_tax_for_row(
                            actions_list[idx_r], lp_r, avg_cost_map.get(bt_r, np.nan), cq_r, pq_r, reg_r
                        )
                        amts.append(t_amt)
                        strs.append(f"{currency} {t_disp}" if pd.notna(t_amt) and t_amt > 0 else t_disp)
                return amts, strs

            row_tax_amts, row_tax_strs = compute_sc_tax_columns(row_regimes_initial)

            unified_blotter_df = pd.DataFrame({
                "Ticker": valid_bt_ordered,
                "Sector": [ticker_to_sector.get(bt, "Other") for bt in valid_bt_ordered],
                "Live Price": [f"{currency} {float(p_df[yt].iloc[-1]):,.2f}" for _, yt in valid_pairs],
                "Current Wt": [f"{w * 100.0:.2f}%" for w in curr_wts],
                "Proposed Wt": [f"{w * 100.0:.2f}%" for w in prop_wts],
                "Delta Wt": [f"{w * 100.0:+.2f}%" for w in diff_wts],
                "Current Val": [f"{currency} {v:,.2f}" for v in curr_val_per_asset],
                "Proposed Val": [f"{currency} {v:,.2f}" for v in prop_val_per_asset],
                "Action": actions_list,
                "Trade Notional": [f"{currency} {abs(v):,.2f}" for v in trade_notional_per_asset],
                "Est. Friction": [f"{currency} {abs(v) * friction_rate:,.2f}" for v in trade_notional_per_asset],
                "Tax Regime": row_regimes_initial,
                "What-If Tax": row_tax_strs,
            })

            st.write("**Pre-Trade vs. Post-Trade Allocation, Order Diff & What-If Tax Table** *(use Tax Regime dropdown on SELL rows)*")
            read_only_cols = [c for c in unified_blotter_df.columns if c != "Tax Regime"]
            edited_blotter_df = st.data_editor(
                unified_blotter_df,
                use_container_width=True,
                hide_index=True,
                disabled=read_only_cols,
                column_config={
                    "Tax Regime": st.column_config.SelectboxColumn(
                        "Tax Regime",
                        help="Select LTCG (12.5%) or STCG (20%) for each sellable stock",
                        options=["LTCG (12.5%)", "STCG (20%)", "N/A"],
                        required=True,
                    ),
                },
                key="sc_unified_blotter_editor",
            )

            # Sync any per-row dropdown overrides and recompute What-If Tax immediately
            updated_regimes = []
            regime_changed = False
            for idx_e, bt_e in enumerate(valid_bt_ordered):
                chosen_r = str(edited_blotter_df["Tax Regime"].iloc[idx_e])
                if "SELL" not in actions_list[idx_e]:
                    chosen_r = "N/A"
                elif chosen_r not in ["LTCG (12.5%)", "STCG (20%)"]:
                    chosen_r = sc_bulk_tax_regime
                if st.session_state["sc_per_stock_tax_map"].get(bt_e) != chosen_r:
                    st.session_state["sc_per_stock_tax_map"][bt_e] = chosen_r
                    regime_changed = True
                updated_regimes.append(chosen_r)

            if regime_changed:
                st.rerun()

            row_tax_amts, row_tax_strs = compute_sc_tax_columns(updated_regimes)
            total_sc_tax_impact = float(np.nansum(row_tax_amts)) if has_any_sell_sc else 0.0

            active_sell_regimes = {updated_regimes[i] for i in range(len(valid_pairs)) if "SELL" in actions_list[i]}
            if len(active_sell_regimes) > 1:
                sc_tax_summary_label = "Mixed LTCG / STCG"
            elif len(active_sell_regimes) == 1:
                sc_tax_summary_label = next(iter(active_sell_regimes))
            else:
                sc_tax_summary_label = sc_bulk_tax_regime

            # Broker Trade Execution CSV Export (Single Client)
            clean_cid_sc = client_id_tag.split("(")[0].strip() or "CLIENT-001"
            sc_exec_csv_df = build_broker_execution_csv_df(
                client_id=clean_cid_sc,
                exchange=exchange,
                tickers=valid_bt_ordered,
                live_prices=[float(p_df[yt].iloc[-1]) for _, yt in valid_pairs],
                curr_qtys=[qty_map.get(bt, 0.0) for bt in valid_bt_ordered],
                prop_vals=prop_val_per_asset,
                friction_rate=friction_rate,
                starting_cash=float(client_cash),
                cash_flow_delta=float(cash_flow_sim),
            )
            st.download_button(
                label="📥 Export Broker Trade Execution CSV",
                data=sc_exec_csv_df.to_csv(index=False).encode("utf-8"),
                file_name=f"WIRATOR_Trade_Execution_{clean_cid_sc}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="sc_download_exec_csv",
            )

            # ------------------------------------------------------------------
            # STAGE 3: PRE-TRADE DIFF, BENCHMARK GRAPH, QUANT METRICS & MONTE CARLO
            # ------------------------------------------------------------------
            st.markdown("---")
            st.markdown("## Stage 3: Pre-Trade Diff, Benchmark Comparison & Risk / Monte Carlo")

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

            total_turnover_amt = float(np.abs(trade_notional_per_asset).sum())
            turnover_pct = (total_turnover_amt / post_nav * 100.0) if post_nav > 0 else 0.0
            total_friction_amt = float((np.abs(trade_notional_per_asset) * friction_rate).sum())

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
                tax_impact_amt=total_sc_tax_impact if has_any_sell_sc else np.nan,
                tax_regime_label=sc_tax_summary_label,
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
                "tax_impact": total_sc_tax_impact,
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

            if "batch_model_df_state" not in st.session_state:
                st.session_state["batch_model_df_state"] = pd.DataFrame([
                    {"Ticker": "HDFCBANK", "Target_Weight_Pct": 35.0},
                    {"Ticker": "INFY", "Target_Weight_Pct": 35.0},
                    {"Ticker": "RELIANCE", "Target_Weight_Pct": 30.0},
                ]) if current_cfg.get("default_curr") == "INR" else pd.DataFrame([
                    {"Ticker": "AAPL", "Target_Weight_Pct": 40.0},
                    {"Ticker": "MSFT", "Target_Weight_Pct": 35.0},
                    {"Ticker": "JPM", "Target_Weight_Pct": 25.0},
                ])

            b_col1, b_col2 = st.columns([1.4, 2.2])
            with b_col1:
                st.write("**Model Target Weights (%)**")
                model_target_df = st.data_editor(
                    st.session_state["batch_model_df_state"],
                    num_rows="dynamic",
                    use_container_width=True,
                    hide_index=True,
                    key="batch_model_targets",
                )
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

            all_batch_tickers = set(raw_model_map.keys())
            parsed_client_holdings = []
            parsed_client_costs = []
            parsed_client_days = []
            for _, crow in edited_batch_df.iterrows():
                h_dict = {}
                c_dict = {}
                d_dict = {}
                raw_h = crow.get("Holdings_JSON", "{}")
                try:
                    parsed_obj = json.loads(raw_h) if isinstance(raw_h, str) else (raw_h if isinstance(raw_h, dict) else {})
                    for k, v in parsed_obj.items():
                        ku = str(k).strip().upper()
                        if ku:
                            if isinstance(v, dict):
                                h_dict[ku] = float(v.get("qty", v.get("Qty", 0.0)))
                                if "cost" in v or "Cost" in v:
                                    c_dict[ku] = float(v.get("cost", v.get("Cost", np.nan)))
                                if "days" in v or "Holding_Days" in v:
                                    d_dict[ku] = int(float(v.get("days", v.get("Holding_Days", 0))))
                            else:
                                h_dict[ku] = float(v)
                            all_batch_tickers.add(ku)
                except Exception:
                    pass

                for col_c in ["Cost_JSON", "Avg_Cost_JSON"]:
                    if col_c in crow and pd.notna(crow[col_c]):
                        try:
                            co = json.loads(crow[col_c]) if isinstance(crow[col_c], str) else crow[col_c]
                            if isinstance(co, dict):
                                for k, v in co.items():
                                    c_dict[str(k).strip().upper()] = float(v)
                        except Exception:
                            pass
                for col_d in ["Holding_Days_JSON", "Days_JSON"]:
                    if col_d in crow and pd.notna(crow[col_d]):
                        try:
                            do = json.loads(crow[col_d]) if isinstance(crow[col_d], str) else crow[col_d]
                            if isinstance(do, dict):
                                for k, v in do.items():
                                    d_dict[str(k).strip().upper()] = int(float(v))
                        except Exception:
                            pass

                parsed_client_holdings.append(h_dict)
                parsed_client_costs.append(c_dict)
                parsed_client_days.append(d_dict)

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

            # Portfolio Optimizer Presets for Batch Model Weights (erk.weight_ew, erk.weight_gmv, erk.msr, erk.weight_erc)
            st.markdown("#### ⚙️ Model Portfolio Optimizer Presets (Auto-Fill Model Target Weights)")
            if "batch_active_optimizer" not in st.session_state:
                st.session_state["batch_active_optimizer"] = None

            b_opt1, b_opt2, b_opt3, b_opt4 = st.columns(4)
            b_clicked_opt = None
            with b_opt1:
                if st.button("⚖️ Equally Weighted (EW)", use_container_width=True, key="batch_opt_ew"):
                    b_clicked_opt = "EW"
            with b_opt2:
                if st.button("🛡️ Global Min Volatility (GMV)", use_container_width=True, key="batch_opt_gmv"):
                    b_clicked_opt = "GMV"
            with b_opt3:
                if st.button("🚀 Max Sharpe Ratio (MSR)", use_container_width=True, key="batch_opt_msr"):
                    b_clicked_opt = "MSR"
            with b_opt4:
                if st.button("🎯 Equal Risk Contribution (ERC)", use_container_width=True, key="batch_opt_erc"):
                    b_clicked_opt = "ERC"

            model_universe_pairs = [(bt, yt) for bt, yt in valid_b_pairs if bt in raw_model_map]
            if b_clicked_opt is not None and model_universe_pairs:
                m_bt = [bt for bt, _ in model_universe_pairs]
                m_yt = [yt for _, yt in model_universe_pairs]
                opt_m_map = compute_optimizer_target_weights(ret_b_df, m_bt, m_yt, b_clicked_opt, rfr_annual=rfr_annual)
                st.session_state["batch_model_df_state"] = pd.DataFrame([
                    {"Ticker": k, "Target_Weight_Pct": v} for k, v in opt_m_map.items()
                ])
                st.session_state["batch_active_optimizer"] = b_clicked_opt
                st.rerun()

            if st.session_state.get("batch_active_optimizer"):
                b_opt_code = st.session_state["batch_active_optimizer"]
                b_opt_label = OPTIMIZER_LABELS.get(b_opt_code, b_opt_code)
                st.markdown(
                    f"""<div class="opt-badge">
                        📌 Active Model Optimizer: <b>{b_opt_label}</b> — Batch rebalance figures reflect <b>{b_opt_code}</b> weights.
                    </div>""",
                    unsafe_allow_html=True,
                )

            tot_m_pct = sum(raw_model_map.values())
            model_wt_map = {
                t: (wt / tot_m_pct if tot_m_pct > 0 else 1.0 / len(raw_model_map))
                for t, wt in raw_model_map.items()
            }

            if b_combined_added:
                st.info(f"Normalized Model Weights (including added stocks): " + ", ".join([f"**{k}**: {v*100:.1f}%" for k, v in model_wt_map.items()]))

            latest_px_map = {bt: float(p_b_df[yt].iloc[-1]) for bt, yt in valid_b_pairs}

            b_ret_bench = pd.Series(dtype=float)
            try:
                bm_b_raw = yf.download(current_cfg["symbol"], start=ret_b_df.index[0], end=ret_b_df.index[-1], progress=False)
                bm_b_ser = bm_b_raw["Close"].squeeze() if isinstance(bm_b_raw, pd.DataFrame) and "Close" in bm_b_raw else pd.Series(dtype=float)
                b_ret_bench = bm_b_ser.pct_change().dropna()
            except Exception:
                pass

            if "batch_last_bulk_tax" not in st.session_state:
                st.session_state["batch_last_bulk_tax"] = "LTCG (12.5%)"
            if "batch_per_stock_tax" not in st.session_state:
                st.session_state["batch_per_stock_tax"] = {}

            batch_tax_regime = st.radio(
                "Set All Sellable Stocks at Once — Batch Tax Regime (or override per stock in Drill-Down below)",
                options=["LTCG (12.5%)", "STCG (20%)"],
                index=0 if st.session_state["batch_last_bulk_tax"] == "LTCG (12.5%)" else 1,
                horizontal=True,
                key="batch_whatif_tax_regime",
            )
            if batch_tax_regime != st.session_state["batch_last_bulk_tax"]:
                st.session_state["batch_last_bulk_tax"] = batch_tax_regime
                for k_b in list(st.session_state["batch_per_stock_tax"].keys()):
                    if st.session_state["batch_per_stock_tax"][k_b] in ["LTCG (12.5%)", "STCG (20%)"]:
                        st.session_state["batch_per_stock_tax"][k_b] = batch_tax_regime

            client_rows_detail = {}
            summary_records = []
            batch_fric_rate = batch_friction_bps / 10000.0
            for idx, (crow, h_dict, c_dict, d_dict) in enumerate(zip(edited_batch_df.iterrows(), parsed_client_holdings, parsed_client_costs, parsed_client_days)):
                _, crow_ser = crow
                cid = str(crow_ser.get("Client_ID", f"CLT-{idx+1:03d}")).strip()
                pmask = str(crow_ser.get("PAN_Mask", "XXXX0000X")).strip()
                rtier = str(crow_ser.get("Risk_Tier", "Moderate")).strip()
                cash_val = float(crow_ser.get("Cash", 0.0) or 0.0)
                rat_str = str(crow_ser.get("Rationale", "") or "")
                row_holding_days = int(float(crow_ser["Holding_Days"])) if ("Holding_Days" in crow_ser and pd.notna(crow_ser["Holding_Days"])) else None

                c_vol_cap = float(crow_ser.get("Mandate_Vol_Cap", vol_cap_pct)) if override_limits_per_client else vol_cap_pct
                c_stk_cap = float(crow_ser.get("Max_Stock_Cap", stock_cap_pct)) if override_limits_per_client else stock_cap_pct

                curr_pos_vals = np.array([h_dict.get(t, 0.0) * latest_px_map.get(t, 0.0) for t in valid_b_bt], dtype=float)
                eq_val = float(curr_pos_vals.sum())
                tot_val = eq_val + cash_val
                c_wts = (curr_pos_vals / tot_val) if tot_val > 0 else np.zeros(len(valid_b_bt))

                target_equity_base = tot_val if deploy_cash else eq_val
                prop_wts = np.array([(model_wt_map.get(t, 0.0) * target_equity_base / tot_val) if tot_val > 0 else 0.0 for t in valid_b_bt], dtype=float)
                prop_pos_vals = prop_wts * tot_val

                c_actions = []
                c_regimes = []
                c_tax_amts = []
                c_tax_strs = []
                for i_t, t_sym in enumerate(valid_b_bt):
                    lp_t = latest_px_map.get(t_sym, 0.0)
                    cq_t = h_dict.get(t_sym, 0.0)
                    pq_t = (prop_pos_vals[i_t] / lp_t) if lp_t > 0 else 0.0
                    notional_diff_t = prop_pos_vals[i_t] - curr_pos_vals[i_t]
                    if prop_wts[i_t] <= 1e-6 and c_wts[i_t] > 1e-6:
                        act_t = "EXIT / SELL"
                    elif c_wts[i_t] <= 1e-6 and prop_wts[i_t] > 1e-6:
                        act_t = "NEW / BUY"
                    elif notional_diff_t > 1.0:
                        act_t = "BUY"
                    elif notional_diff_t < -1.0:
                        act_t = "SELL"
                    else:
                        act_t = "HOLD"
                    c_actions.append(act_t)

                    state_key_b = f"{cid}::{t_sym}"
                    if "SELL" in act_t:
                        if state_key_b not in st.session_state["batch_per_stock_tax"] or st.session_state["batch_per_stock_tax"][state_key_b] not in ["LTCG (12.5%)", "STCG (20%)"]:
                            if t_sym in d_dict:
                                st.session_state["batch_per_stock_tax"][state_key_b] = "STCG (20%)" if d_dict[t_sym] < 365 else "LTCG (12.5%)"
                            elif row_holding_days is not None:
                                st.session_state["batch_per_stock_tax"][state_key_b] = "STCG (20%)" if row_holding_days < 365 else "LTCG (12.5%)"
                            else:
                                st.session_state["batch_per_stock_tax"][state_key_b] = batch_tax_regime
                        eff_regime = st.session_state["batch_per_stock_tax"][state_key_b]
                    else:
                        st.session_state["batch_per_stock_tax"][state_key_b] = "N/A"
                        eff_regime = "N/A"

                    c_regimes.append(eff_regime)
                    cost_t = c_dict.get(t_sym, np.nan)
                    t_a, t_s = calculate_sell_tax_for_row(act_t, lp_t, cost_t, cq_t, pq_t, eff_regime)
                    c_tax_amts.append(t_a)
                    c_tax_strs.append(f"{currency} {t_s}" if pd.notna(t_a) and t_a > 0 else t_s)

                c_has_sell = any("SELL" in a for a in c_actions)
                c_tot_tax = float(np.nansum(c_tax_amts)) if c_has_sell else np.nan

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
                fric_amt = float(np.abs(prop_wts - c_wts).sum() * 0.5 * tot_val * batch_fric_rate)

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

                tax_roll_disp = f"{currency} {c_tot_tax:,.2f}" if (pd.notna(c_tot_tax) and c_tot_tax > 0) else "N/A"
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
                    "Est_Tax": tax_roll_disp,
                    "Gate_Status": g_res["status"],
                    "Rationale_Status": "Logged" if not g_res["missing_rationale"] else "Missing",
                    "Flags": "; ".join(g_res["breaches"] + g_res["warnings"]) if (g_res["breaches"] or g_res["warnings"]) else "OK",
                })

                c_exec_df = build_broker_execution_csv_df(
                    client_id=cid,
                    exchange=exchange,
                    tickers=valid_b_bt,
                    live_prices=[latest_px_map.get(t, 0.0) for t in valid_b_bt],
                    curr_qtys=[h_dict.get(t, 0.0) for t in valid_b_bt],
                    prop_vals=prop_pos_vals,
                    friction_rate=batch_fric_rate,
                    starting_cash=cash_val,
                    cash_flow_delta=0.0,
                )

                client_rows_detail[cid] = {
                    "cid": cid, "pmask": pmask, "rtier": rtier, "tot_val": tot_val, "cash_val": cash_val,
                    "c_wts": c_wts, "prop_wts": prop_wts, "r_pre": r_pre, "r_post": r_post,
                    "pre_metrics": c_pre_metrics, "post_metrics": c_post_metrics,
                    "pre_div": c_pre_div, "post_div": c_post_div,
                    "pre_max_s": pre_max_s, "post_max_s": post_max_s,
                    "pre_max_sec": pre_max_sec, "post_max_sec": post_max_sec,
                    "turn_pct": turn_pct, "fric_amt": fric_amt,
                    "tax_amt": c_tot_tax, "has_sell": c_has_sell,
                    "exec_df": c_exec_df,
                    "rationale": rat_str, "gate": g_res,
                    "diff_df": pd.DataFrame({
                        "Ticker": valid_b_bt,
                        "Sector": [ticker_to_sector.get(t, "Other") for t in valid_b_bt],
                        "Live Price": [f"{currency} {latest_px_map.get(t, 0.0):,.2f}" for t in valid_b_bt],
                        "Current Wt": [f"{w * 100.0:.2f}%" for w in c_wts],
                        "Proposed Wt": [f"{w * 100.0:.2f}%" for w in prop_wts],
                        "Delta Wt": [f"{w * 100.0:+.2f}%" for w in (prop_wts - c_wts)],
                        "Action": c_actions,
                        "Trade Amt": [f"{currency} {abs(v):,.2f}" for v in (np.abs(prop_wts - c_wts) * tot_val)],
                        "Tax Regime": c_regimes,
                        "What-If Tax": c_tax_strs,
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

            b_read_only_cols = [c for c in d_item["diff_df"].columns if c != "Tax Regime"]
            edited_b_diff_df = st.data_editor(
                d_item["diff_df"],
                use_container_width=True,
                hide_index=True,
                disabled=b_read_only_cols,
                column_config={
                    "Tax Regime": st.column_config.SelectboxColumn(
                        "Tax Regime",
                        help="Override LTCG (12.5%) or STCG (20%) for individual SELL rows",
                        options=["LTCG (12.5%)", "STCG (20%)", "N/A"],
                        required=True,
                    ),
                },
                key=f"batch_diff_editor_{selected_cid}",
            )

            b_reg_changed = False
            for idx_bd, bt_bd in enumerate(valid_b_bt):
                chosen_bd = str(edited_b_diff_df["Tax Regime"].iloc[idx_bd])
                act_bd = str(d_item["diff_df"]["Action"].iloc[idx_bd])
                s_key_bd = f"{selected_cid}::{bt_bd}"
                if "SELL" not in act_bd:
                    chosen_bd = "N/A"
                elif chosen_bd not in ["LTCG (12.5%)", "STCG (20%)"]:
                    chosen_bd = batch_tax_regime
                if st.session_state["batch_per_stock_tax"].get(s_key_bd) != chosen_bd:
                    st.session_state["batch_per_stock_tax"][s_key_bd] = chosen_bd
                    b_reg_changed = True
            if b_reg_changed:
                st.rerun()

            all_batch_exec_frames = [v["exec_df"] for v in client_rows_detail.values() if not v["exec_df"].empty]
            full_batch_exec_df = pd.concat(all_batch_exec_frames, ignore_index=True) if all_batch_exec_frames else pd.DataFrame(columns=[
                "Client_ID", "Exchange", "Ticker", "Series", "Action", "Order_Qty", "Limit_Price", "Est_Notional", "Est_Friction", "Residual_Cash"
            ])
            b_ex1, b_ex2 = st.columns(2)
            with b_ex1:
                st.download_button(
                    label=f"📥 Export Broker Execution CSV ({selected_cid})",
                    data=d_item["exec_df"].to_csv(index=False).encode("utf-8"),
                    file_name=f"WIRATOR_Trade_Execution_{selected_cid}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key=f"batch_dl_exec_{selected_cid}",
                )
            with b_ex2:
                st.download_button(
                    label="📥 Export All Clients Broker Execution CSV",
                    data=full_batch_exec_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"WIRATOR_Batch_Trade_Execution_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="batch_dl_exec_all",
                )

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
                tax_impact_amt=d_item["tax_amt"] if d_item["has_sell"] else np.nan,
                tax_regime_label="Per-Row LTCG/STCG",
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
                    "tax_impact": float(d_item["tax_amt"]) if pd.notna(d_item["tax_amt"]) else 0.0,
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

    # Sidebar Feedback Form
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

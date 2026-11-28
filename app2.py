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
    page_title="FolioLyn",
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

# Comprehensive NSE & BSE Indices Catalog (using .NS / Yahoo index tickers with safe fallbacks)
INDEX_CATALOG = {
    # NSE Broad Indices
    "Nifty 50": {"candidates": ["^NSEI", "NIFTYBEES.NS"], "group": "NSE Broad", "market": "NSE"},
    "Nifty Next 50": {"candidates": ["NIFTYNEXT50.NS", "JUNIORBEES.NS", "^NSEI"], "group": "NSE Broad", "market": "NSE"},
    "Nifty 100": {"candidates": ["^CNX100", "NIFTY100.NS", "^NSEI"], "group": "NSE Broad", "market": "NSE"},
    "Nifty 500": {"candidates": ["^CRSLDX", "NIFTY500.NS", "^NSEI"], "group": "NSE Broad", "market": "NSE"},
    "Nifty Midcap 150": {"candidates": ["^NSEMDCP150", "MID150BEES.NS", "^NSEI"], "group": "NSE Broad", "market": "NSE"},
    "Nifty Smallcap 250": {"candidates": ["^NSMALLCAP250", "SMALLCAP250.NS", "^NSEI"], "group": "NSE Broad", "market": "NSE"},
    # NSE Sectoral Indices
    "Nifty Bank": {"candidates": ["^NSEBANK", "BANKBEES.NS"], "group": "NSE Sectoral", "market": "NSE"},
    "Nifty IT": {"candidates": ["^CNXIT", "ITBEES.NS"], "group": "NSE Sectoral", "market": "NSE"},
    "Nifty Auto": {"candidates": ["^CNXAUTO", "AUTOBEES.NS"], "group": "NSE Sectoral", "market": "NSE"},
    "Nifty Pharma": {"candidates": ["^CNXPHARMA", "PHARMABEES.NS"], "group": "NSE Sectoral", "market": "NSE"},
    "Nifty FMCG": {"candidates": ["^CNXFMCG", "FMCGIETF.NS"], "group": "NSE Sectoral", "market": "NSE"},
    "Nifty Realty": {"candidates": ["^CNXREALTY", "REALTY.NS"], "group": "NSE Sectoral", "market": "NSE"},
    # BSE Broad Indices
    "BSE SENSEX": {"candidates": ["^BSESN", "SENSEX.NS"], "group": "BSE Broad", "market": "BSE"},
    "BSE 100": {"candidates": ["BSE-100.NS", "^BSESN"], "group": "BSE Broad", "market": "BSE"},
    "BSE 200": {"candidates": ["BSE-200.NS", "^BSESN"], "group": "BSE Broad", "market": "BSE"},
    "BSE 500": {"candidates": ["BSE-500.NS", "^BSESN"], "group": "BSE Broad", "market": "BSE"},
    "BSE MidCap": {"candidates": ["BSE-MIDCAP.NS", "^NSEMDCP150", "^BSESN"], "group": "BSE Broad", "market": "BSE"},
    "BSE SmallCap": {"candidates": ["BSE-SMLCAP.NS", "^NSMALLCAP250", "^BSESN"], "group": "BSE Broad", "market": "BSE"},
    # BSE Sectoral Indices
    "BSE BANKEX": {"candidates": ["BSE-BANK.NS", "^NSEBANK"], "group": "BSE Sectoral", "market": "BSE"},
    "BSE Auto": {"candidates": ["BSE-AUTO.NS", "^CNXAUTO"], "group": "BSE Sectoral", "market": "BSE"},
    "BSE Healthcare": {"candidates": ["BSE-HC.NS", "^CNXPHARMA"], "group": "BSE Sectoral", "market": "BSE"},
    "BSE FMCG": {"candidates": ["BSE-FMCG.NS", "^CNXFMCG"], "group": "BSE Sectoral", "market": "BSE"},
    "BSE Capital Goods": {"candidates": ["BSE-CG.NS", "^BSESN"], "group": "BSE Sectoral", "market": "BSE"},
    "BSE Energy": {"candidates": ["BSE-ENERGY.NS", "^BSESN"], "group": "BSE Sectoral", "market": "BSE"},
    # BSE Thematic Indices
    "BSE PSU": {"candidates": ["BSE-PSU.NS", "PSUBNKBEES.NS", "^BSESN"], "group": "BSE Thematic", "market": "BSE"},
    "BSE CPSE": {"candidates": ["BSE-CPSE.NS", "CPSEETF.NS", "^BSESN"], "group": "BSE Thematic", "market": "BSE"},
    "BSE India Infrastructure": {"candidates": ["BSE-INFRA.NS", "INFRA.NS", "^BSESN"], "group": "BSE Thematic", "market": "BSE"},
    "BSE India Defence": {"candidates": ["BSE-DEFENCE.NS", "DEFENCE.NS", "^BSESN"], "group": "BSE Thematic", "market": "BSE"},
    "BSE Internet Economy": {"candidates": ["BSE-INET.NS", "NETWEB.NS", "^BSESN"], "group": "BSE Thematic", "market": "BSE"},
}


def ensure_ns_suffix(symbol, exchange_code="NSE"):
    """Ensures NSE/BSE tickers/indices use .NS unless already carrying a Yahoo prefix (^) or suffix (.)."""
    s = str(symbol).strip().upper()
    if not s:
        return s
    if s.startswith("^") or "." in s:
        return s
    if exchange_code in ["NSE", "BSE"]:
        return f"{s}.NS"
    return s


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
        "filename": "foliolyn_risk_audit",
        "height": 600,
        "width": 1000,
        "scale": 2,
    },
    "displaylogo": False,
}


@st.cache_data(ttl=600, show_spinner=False)
def cached_yf_download(tickers_tuple, start_str, end_str, period=None, interval="1d"):
    """Cached wrapper around yf.download to reduce lag across tabs and reruns."""
    tickers_list = [t for t in tickers_tuple if t]
    if not tickers_list:
        return pd.DataFrame()
    try:
        if period:
            df = yf.download(tickers_list, period=period, interval=interval, progress=False, auto_adjust=True)
        else:
            df = yf.download(tickers_list, start=start_str, end=end_str, interval=interval, progress=False, auto_adjust=True)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def resolve_yf_tickers_cached(raw_tickers_tuple, exchange_name, suffix):
    """Resolves exchange suffixes (including EU multi-exchange fallbacks and .NS for NSE/BSE) with caching."""
    all_eu_suffixes = [".AS", ".DE", ".PA", ".MI", ".MC", ".SW", ".ST", ".BR"]
    resolved = []
    for raw_ticker in raw_tickers_tuple:
        rt = str(raw_ticker).strip().upper()
        if not rt or rt == "NAN":
            resolved.append("")
            continue
        if rt.startswith("^"):
            resolved.append(rt)
            continue
        if exchange_name == "STOXX Europe 600" and not any(rt.endswith(sfx) for sfx in all_eu_suffixes):
            found_valid = False
            for sfx in all_eu_suffixes:
                test_symbol = f"{rt}{sfx}"
                test_data = cached_yf_download((test_symbol,), "", "", period="5d", interval="1d")
                if not test_data.empty:
                    resolved.append(test_symbol)
                    found_valid = True
                    break
            if not found_valid:
                resolved.append(f"{rt}{suffix}")
        else:
            if any(rt.endswith(sfx) for sfx in all_eu_suffixes) or exchange_name != "STOXX Europe 600":
                eff_sfx = ".NS" if exchange_name in ["NSE", "BSE"] and not suffix else suffix
                sfx_to_add = "" if ("." in rt or any(rt.endswith(sfx) for sfx in all_eu_suffixes)) else eff_sfx
                resolved.append(f"{rt}{sfx_to_add}")
            else:
                resolved.append(rt)
    return resolved


def resolve_yf_tickers(raw_tickers, current_cfg):
    return resolve_yf_tickers_cached(
        tuple(str(t).strip().upper() for t in raw_tickers),
        current_cfg.get("name", ""),
        current_cfg.get("suffix", ".NS"),
    )


def extract_close_df(raw_df, expected_cols=None):
    """Extracts a clean Close price DataFrame from yf.download output."""
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()
    if isinstance(raw_df.columns, pd.MultiIndex):
        if "Close" in raw_df.columns.get_level_values(0):
            close_df = raw_df["Close"].copy()
        else:
            close_df = raw_df.xs(raw_df.columns.levels[0][0], axis=1, level=0).copy()
    else:
        if "Close" in raw_df.columns:
            col_name = expected_cols[0] if expected_cols and len(expected_cols) == 1 else "Close"
            close_df = raw_df[["Close"]].rename(columns={"Close": col_name}).copy()
        else:
            close_df = raw_df.copy()

    if isinstance(close_df, pd.Series):
        col_name = expected_cols[0] if expected_cols and len(expected_cols) == 1 else (close_df.name or "Asset")
        close_df = close_df.to_frame(name=col_name)

    close_df = close_df.dropna(how="all", axis=1).ffill().bfill()
    return close_df


@st.cache_data(ttl=900, show_spinner=False)
def fetch_benchmark_return_series(benchmark_label, start_str, end_str):
    """Fetches a benchmark series from INDEX_CATALOG using candidate tickers (.NS / Yahoo symbols)."""
    meta = INDEX_CATALOG.get(benchmark_label)
    candidates = meta["candidates"] if meta else [ensure_ns_suffix(benchmark_label, "NSE")]
    for sym in candidates:
        raw = cached_yf_download((sym,), start_str, end_str, interval="1d")
        cdf = extract_close_df(raw, expected_cols=[sym])
        if not cdf.empty and sym in cdf.columns:
            s = cdf[sym].dropna()
            if len(s) >= 15:
                return s.pct_change().dropna(), sym
        elif not cdf.empty and cdf.shape[1] >= 1:
            s = cdf.iloc[:, 0].dropna()
            if len(s) >= 15:
                return s.pct_change().dropna(), cdf.columns[0]
    return pd.Series(dtype=float), candidates[0]


def compute_annualized_alpha(port_ret_series, bench_ret_series, rfr_annual=0.065, periods_per_year=252):
    """Computes annualized Jensen's Alpha (%) and Beta against a benchmark return series."""
    aligned = pd.DataFrame({"p": port_ret_series, "b": bench_ret_series}).dropna()
    if len(aligned) < 10 or np.var(aligned["b"]) <= 1e-12:
        return 0.0, 1.0
    beta = float(np.cov(aligned["p"], aligned["b"])[0, 1] / np.var(aligned["b"]))
    p_ann = float(erk.annualized_returns(aligned["p"], periods_per_year=periods_per_year))
    b_ann = float(erk.annualized_returns(aligned["b"], periods_per_year=periods_per_year))
    rf = float(rfr_annual)
    alpha_ann_pct = ((p_ann - rf) - beta * (b_ann - rf)) * 100.0
    return alpha_ann_pct, beta


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


def evaluate_mandate_gate(
    post_vol_pct,
    post_cvar_pct

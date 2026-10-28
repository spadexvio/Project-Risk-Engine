import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import io
import plotly.io as pio
import plotly.express as px
import IPython.lib.deepreload # or built-in importlib
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Automatically reload imported local modules on every Streamlit run
import importlib
import quant_risk_kit as erk
importlib.reload(erk)

# --- PAGE CONFIGURATION ---
st.markdown(
    '<meta name="color-scheme" content="light dark">', 
    unsafe_allow_html=True
)
st.set_page_config(
    page_title="WIRATOR",
    page_icon="🛡️",
    layout="wide"
)

# --- SESSION STATE NAVIGATION MANAGEMENT ---
if 'page' not in st.session_state:
    st.session_state['page'] = 'landing'
if 'exchange' not in st.session_state:
    st.session_state['exchange'] = None

def navigate_to(page_name, exchange=None):
    st.session_state['page'] = page_name
    st.session_state['exchange'] = exchange

CHART_CONFIG = {
    'displayModeBar': False,
    'modeBarButtonsToRemove': [
        '2d', 'pan2d', 'select2d', 'lasso2d', 
        'In2d', 'Out2d', 'autoScale2d', 'resetScale2d'
    ],
    'toImageButtonOptions': {
        'format': 'png', 
        'filename': 'wirator_risk_audit',
        'height': 600,
        'width': 1000,
        'scale': 2
    },
    'displaylogo': False
}

# ==============================================================================
# 1. LANDING PAGE (Centered Logo, Description, Exchange Selection & Auth Wall)
# ==============================================================================
if st.session_state['page'] == 'landing':
    
    # Custom CSS for complete center alignment
    st.markdown(
    """
    <style>
    /* 1. Prevent text overflow and stretching on mobile */
    p, span, div, label {
        word-break: break-word;
    }

    /* 2. Force input text colors to remain visible (fix blue text blending into dark/black backgrounds) */
    input, textarea, select, div[data-baseweb="select"] > div {
        color: #111827 !important;
        background-color: #ffffff !important;
        -webkit-text-fill-color: #111827 !important;
    }

    li[role="option"] {
        color: #111827 !important;
        background-color: #ffffff !important;
    }

    /* 3. Mobile screen adjustments */
    @media (max-width: 768px) {
        .stButton button {
            width: 100% !important;
        }
        /* Ensure columns stack gracefully on mobile */
        [data-testid="column"] {
            width: 100% !important;
            flex: 100% !important;
            min-width: 100% !important;
        }
        /* Scale down heavy desktop typography for phones */
        .hero-title {
            font-size: 2.5rem !important;
        }
        .hero-subtitle {
            font-size: 1.1rem !important;
            white-space: normal !important;
            letter-spacing: 1px !important;
        }
        .slogan-badge {
            font-size: 1.8rem !important;
            white-space: normal !important;
            letter-spacing: 1px !important;
        }
        .hero-description {
            font-size: 1.1rem !important;
            line-height: 1.5 !important;
            padding: 0 0.5rem;
        }
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
        }
    }

    /* Main Background: Sophisticated Dark Midnight Blue */
    .stApp {
        background-color: #1A2B4C;
        color: #FDFBD4;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Hide default Streamlit header padding */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 5rem;
        max-width: 1000px;
    }

    /* Center-aligned Hero Header Styling */
    .hero-container {
        text-align: center;
        padding: 2rem 1rem 1.5rem 1rem;
    }

    .hero-title {
        color: #FDFBD4;
        font-size: 3.8rem !important;
        font-weight: 600;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        color: #FDFBD4;
        font-size: 2.4rem;
        font-weight: 500;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
        opacity: 0.9;
        white-space: nowrap;
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
        width: 100%;
    }

    .slogan-badge {
        color: #FDFBD4;
        font-size: 4.25rem;
        font-weight: 800;
        letter-spacing: 3px;
        margin-bottom: 2rem;
        text-transform: uppercase;
        white-space: nowrap;
    }

    .hero-description {
        color: #E2E8F0;
        font-size: 2.2rem;
        line-height: 1.8;
        max-width: 800px;
        margin: 0 auto 2.5rem auto;
        text-align: center;
    }

    /* Accordion / Login Styling */
    .stSelectbox, div[data-baseweb="accordion"] {
        background-color: #1E2E45 !important;
        border-radius: 10px !important;
        border: 1px solid rgba(253, 251, 212, 0.2) !important;
    }

    /* Workspace Section Header */
    .workspace-header {
        text-align: center;
        color: #FFFFFF;
        font-size: 2rem;
        font-weight: 700;
        margin-top: 3rem;
        margin-bottom: 1.8rem;
    }

    /* Custom Large Buttons (Midnight Blue Hover + Cream Accents) */
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

    div.stButton > button:hover {
        background-color: #FDFBD4 !important;
        color: #0B192C !important;
        border-color: #FFFFFF !important;
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(253, 251, 212, 0.25);
    }

    /* Center Align Expander Headers */
    .streamlit-expanderHeader {
        font-size: 2.1rem !important;
        color: #FDFBD4 !important;
        justify-content: center !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# ------------------------------------------------------------------------------
# HERO SECTION
# ------------------------------------------------------------------------------
    st.markdown(
    """
    <div class="hero-container">
        <div style="font-size: 4rem; margin-bottom: 0.5rem;">🛡️</div>
        <h1 class="hero-title">WIRATOR</h1>
        <div class="hero-subtitle">Institutional Portfolio Risk Audit & Monte Carlo Stress Testing</div>
        <div class="slogan-badge">Analyze. Build. Perform.</div>
        <p class="hero-description">
            <b>Wirator</b> is an open-access quantitative risk intelligence portal built specifically for emerging markets. 
            It transforms raw broker transaction data into audit-ready <b>Value at Risk (VaR)</b>, <b>Conditional VaR (CVaR)</b>, 
            and <b>10,000-Path Monte Carlo Stress Simulations</b> tailored to local benchmarks and currencies.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# CLIENT LOGIN EXPANDER
# ------------------------------------------------------------------------------
    # with st.expander("🔑 Client Access / Account Login", expanded=False):
    #   col_a, col_b, col_c = st.columns([1, 2, 1])
    #   with col_b:
    #     st.text_input("Username / Client ID", key="login_user")
    #     st.text_input("Password", type="password", key="login_pass")
    #     if st.button("Sign In to Workspace", key="btn_login"):
    #       st.success("Authenticated successfully.")

# ------------------------------------------------------------------------------
# STOCK EXCHANGE WORKSPACE SELECTION (LARGE BUTTONS)
# ------------------------------------------------------------------------------
    st.markdown(
        '<div class="workspace-header">Select Stock Exchange Workspace</div>',
        unsafe_allow_html=True,
    )
    
    # Exchange Selector Buttons — Row 1 (Emerging & US Anchors)
    r1_col1, r1_col2, r1_col3 = st.columns(3, gap="medium")
    
    with r1_col1:
        if st.button("🇮🇳 National Stock Exchange (NSE)", use_container_width=True):
            navigate_to('analytics', exchange='NSE')
            st.rerun()
    
    with r1_col2:
        if st.button("🇮🇳 Bombay Stock Exchange (BSE)", use_container_width=True):
            navigate_to('analytics', exchange='BSE')
            st.rerun()
    
    with r1_col3:
        if st.button("🇺🇸 US Markets (S&P 500 / Nasdaq)", use_container_width=True):
            navigate_to('analytics', exchange='US')
            st.rerun()
    
    # Exchange Selector Buttons — Row 2 (UK, Europe & Africa)
    r2_col1, r2_col2, r2_col3 = st.columns(3, gap="medium")
    
    with r2_col1:
        if st.button("🇬🇧 London Stock Exchange (FTSE 100)", use_container_width=True):
            navigate_to('analytics', exchange='UK')
            st.rerun()
    
    with r2_col2:
        if st.button("🇪🇺 European Markets (STOXX 600)", use_container_width=True):
            navigate_to('analytics', exchange='EU')
            st.rerun()
    
    with r2_col3:
        if st.button("🇿🇦 Johannesburg Stock Exchange (JSE)", use_container_width=True):
            navigate_to('analytics', exchange='JSE')
            st.rerun()
# ==============================================================================
# 2. ANALYTICS & AUDIT PAGE (Selected Exchange Workspace)
# ==============================================================================
elif st.session_state['page'] == 'analytics':
    exchange = st.session_state['exchange']

    # ------------------------------------------------------------------------------
# ANALYTICS WORKSPACE STYLING: MIDNIGHT BLUE (#1A2B4C) & CREAM ACCENTS
# ------------------------------------------------------------------------------
    st.markdown(
    """
    <style>
    /* ========================================================= */
    /* 1. GLOBAL APP & TYPOGRAPHY                              */
    /* ========================================================= */
    html {
        zoom: 98% !important;
    }
    
    .stApp {
        background-color: #FAF8ED !important;
        color: #1A2B4C !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #1A2B4C !important;
        font-weight: 700 !important;
    }

    /* General markdown paragraphs and captions (main area) */
    .stMarkdown p:not([data-testid="stMetricValue"] p), .stCaption {
        color: #1A2B4C !important;
        font-weight: 500;
    }


    /* ========================================================= */
    /* 2. SIDEBAR STYLING & INPUTS                               */
    /* ========================================================= */
    section[data-testid="stSidebar"] {
        background-color: #1A2B4C !important;
        border-right: 1px solid rgba(250, 248, 237, 0.1) !important;
    }
    
    /* All labels and text inside sidebar */
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] .stMarkdown p, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #FAFAED !important;
        -webkit-text-fill-color: #FAFAED !important;
        font-weight: 500;
    }

    /* Typed text inside sidebar input boxes */
    section[data-testid="stSidebar"] input, 
    section[data-testid="stSidebar"] textarea, 
    section[data-testid="stSidebar"] div[data-baseweb="select"] span {
        color: #1A2B4C !important;
        -webkit-text-fill-color: #1A2B4C !important;
        font-weight: 600 !important;
    }

    /* Sidebar Number Input (- / +) buttons */
    .stSidebar [data-testid="stNumberInput"] button {
        background-color: #FAFAED !important;
        border: 1px solid #1A2B4C !important;
    }
    
    .stSidebar [data-testid="stNumberInput"] button svg {
        fill: #1A2B4C !important;
    }

    .stSidebar [data-testid="stNumberInput"] button:hover {
        background-color: #1A2B4C !important;
    }

    .stSidebar [data-testid="stNumberInput"] button:hover svg {
        fill: #FAFAED !important;
    }


    /* ========================================================= */
    /* 3. CARDS, METRICS & EXPANDERS                             */
    /* ========================================================= */
    div[data-testid="stMetric"],  
    div[data-testid="stExpander"] {
        background-color: #1A2B4C !important;
        border-radius: 12px !important;
        padding: 12px !important;
        border: 1px solid rgba(250, 248, 237, 0.12) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        text-align: center !important;
    }

    div[data-testid="stMetric"] > div {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
    }

    /* Protect Metric Cards: Keep values and labels white/cream */
    div[data-testid="stMetric"] label,  
    div[data-testid="stMetric"] div[data-testid="stMetricValue"],  
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"],
    div[data-testid="stExpander"] *,
    div[data-testid="stExpander"] {
        color: #FAFAED !important;
        -webkit-text-fill-color: #FAFAED !important;
    }


    /* ========================================================= */
    /* 4. FORM ELEMENTS & BUTTONS                                */
    /* ========================================================= */
    /* Radio button options text visibility */
    div.stRadio div[role="radiogroup"] label,  
    div.stRadio div[role="radiogroup"] label p,  
    div.stRadio span {
        color: #1A2B4C !important;
        -webkit-text-fill-color: #1A2B4C !important;
        font-weight: 500;
    }

    /* Force text area and text input typed text to #1A2B4C */
    div.stTextArea textarea, div.stTextInput input {
        color: #1A2B4C !important;
        -webkit-text-fill-color: #1A2B4C !important;
    }

    /* Styled Home / Custom Buttons */
    div.stButton > button,
    .stSidebar button[kind="secondary"], 
    .stSidebar button[kind="primary"], 
    div.stFormSubmitButton button {
        background-color: #FAFAED !important;
        color: #1A2B4C !important;
        border: 1px solid #1A2B4C !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    div.stFormSubmitButton button p, 
    div.stFormSubmitButton button span {
        color: #1A2B4C !important;
        -webkit-text-fill-color: #1A2B4C !important;
        font-weight: 600 !important;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton button:hover {
        background-color: #1A2B4C !important;
        color: #FAFAED !important;
        border-color: #FAFAED !important;
    }

    div.stFormSubmitButton button:hover p, 
    div.stFormSubmitButton button:hover span {
        color: #FAFAED !important;
        -webkit-text-fill-color: #FAFAED !important;
    }


    /* ========================================================= */
    /* 5. DATAFRAME & DATA EDITOR TOOLBAR FIXES                  */
    /* ========================================================= */
    div[data-testid="stDataFrame"],
    div[data-testid="stDataEditor"] {
        background-color: #1A2B4C !important;
        border-radius: 8px !important;
        border: 1px solid rgba(250, 248, 237, 0.12) !important;
        box-shadow: 0 4px 2px rgba(0, 0, 0, 0.2);
        width: 100% !important;
        overflow: visible !important;
    }

    div[data-testid="stDataFrame"] *,
    div[data-testid="stDataEditor"] * {
        color: #FAF8ED !important;
    }

    div[data-testid="stElementToolbar"],
    div[data-testid="stDataEditor"] > div:first-child,
    div[data-testid="stDataFrame"] > div:first-child {
        background-color: #1A2B4C !important;
        border-top-left-radius: 10px !important;
        border-top-right-radius: 10px !important;
        padding: 4px 8px !important;
    }

    div[data-testid="stElementToolbar"] button,
    div[data-testid="stDataEditor"] button,
    button[aria-label="Add row"],
    button[title="Add row"],
    button[aria-label="Download as CSV"] {
        background-color: #0F1C33 !important;
        color: #FAF8ED !important;
        border: 1px solid #FAF8ED !important;
        border-radius: 6px !important;
        opacity: 1 !important;
        margin: 2px !important;
        visibility: visible !important;
    }

    div[data-testid="stElementToolbar"] button svg,
    div[data-testid="stDataEditor"] button svg,
    button[aria-label="Add row"] svg {
        fill: #FAF8ED !important;
        stroke: #FAF8ED !important;
        color: #FAF8ED !important;
    }

    div[role="tooltip"],
    div[data-baseweb="tooltip"] {
        background-color: #1A2B4C !important;
        color: #FAF8ED !important;
        border: 1px solid #FAF8ED !important;
    }
    
    div[role="tooltip"] *,
    div[data-baseweb="tooltip"] * {
        color: #FAF8ED !important;
    }
    /* ========================================================= */
    /* CLEAN FILE UPLOADER FIX                                   */
    /* ========================================================= */
    div[data-testid="stFileUploader"] {
        background-color: #FAFAED !important;
        border: 1px solid #1A2B4C !important;
        border-radius: 8px !important;
        padding: 8px !important;
    }

    /* Instruction text inside the uploader */
    div[data-testid="stFileUploader"] small,
    div[data-testid="stFileUploader"] span {
        color: #1A2B4C !important;
        -webkit-text-fill-color: #1A2B4C !important;
    }

    /* The actual 'Browse files' button */
    div[data-testid="stFileUploader"] button {
        background-color: #1A2B4C !important;
        color: #FAFAED !important;
        border: 1px solid #FAFAED !important;
        border-radius: 6px !important;
        visibility: visible !important;
        display: inline-block !important;
        opacity: 1 !important;
    }

    /* Text inside the browse button */
    div[data-testid="stFileUploader"] button * {
        background-color: #FAF8ED !important;
        color: #1A2B4C !important;
        /*-webkit-text-fill-color: #FAFAED !important;*/
    }

    /* Browse button hover state */
    div[data-testid="stFileUploader"] button:hover {
        background-color: #FAF8ED !important;
        color: #1A2B4C !important;
        border-color: #1A2B4C !important;
    }

    div[data-testid="stFileUploader"] button:hover * {
        color: #1A2B4C !important;
        -webkit-text-fill-color: #1A2B4C !important;
    }
    /* ========================================================= */
    /* FORCE GLOBAL LIGHT/WHITE BACKGROUND ON MOBILE POPOVERS    */
    /* ========================================================= */
    body, .stApp, div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
        background-color: #FAFAED !important;
        color: #1A2B4C !important;
    }

    /* Force all text inside popover selection lists to be dark navy */
    div[data-baseweb="popover"] *, 
    div[data-baseweb="menu"] *, 
    ul[role="listbox"] * {
        background-color: transparent !important;
        color: #1A2B4C !important;
        -webkit-text-fill-color: #1A2B4C !important;
    }

    /* Specific option styling with high contrast */
    li[role="option"], div[role="option"] {
        background-color: #FAFAED !important;
        color: #1A2B4C !important;
        -webkit-text-fill-color: #1A2B4C !important;
    }

    /* Hover/Selected state */
    li[role="option"]:hover, div[role="option"]:hover,
    li[role="option"][aria-selected="true"], div[role="option"][aria-selected="true"] {
        background-color: #1A2B4C !important;
        color: #FAFAED !important;
        -webkit-text-fill-color: #FAFAED !important;
    }
    /* Force chart container to expand 100% */
    div[data-testid="stPlotlyChart"] {
        width: 100% !important;
        max-width: 100% !important;
    }
    div[data-testid="stPlotlyChart"] > div {
        width: 100% !important;
    }
    /* 1. Reduce top padding of the sidebar to pull content up */
    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-top: 0.1rem !important; /* Reduces large top gap above "1. Configuration" */
    }
    
    /* 2. Tighten vertical spacing between sidebar widgets */
    [data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div {
        margin-bottom: -6px !important; /* Pulls elements closer together vertically */
    }
    
    /* 3. Make sidebar inputs slightly more compact */
    [data-testid="stSidebar"] div[data-baseweb="input"] {
        min-height: 36px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
    
    # Top Bar Navigation
    top_col1, top_col2 = st.columns([6, 1])
    with top_col1:
        st.title(f"📈 {exchange} Workspace — Portfolio Risk Engine")
    with top_col2:
        if st.button("🏠 Home"):
            navigate_to('landing')
            st.rerun()

    # Exchange Specific Configuration
    # Exchange Specific Configuration with robust benchmark symbols & fallbacks
    exchange_config = {
        'NSE': {'symbol': '^NSEI', 'fallback': 'NIFTYBEES.NS', 'name': 'Nifty 50 Index', 'suffix': '.NS', 'default_curr': 'INR'},
        'BSE': {'symbol': '^BSESN', 'fallback': 'SETF.NS', 'name': 'SENSEX Index', 'suffix': '.NS', 'default_curr': 'INR'},
        'JSE': {'symbol': '^JN00.JO', 'fallback': 'STX40.JO', 'name': 'JSE Top 40 Index', 'suffix': '.JO', 'default_curr': 'ZAR'},
        'US':  {'symbol': '^GSPC', 'fallback': 'SPY', 'name': 'S&P 500 Index', 'suffix': '', 'default_curr': 'USD'},
        'UK':  {'symbol': '^FTSE', 'fallback': 'ISF.L', 'name': 'FTSE 100 Index', 'suffix': '.L', 'default_curr': 'GBP'},
        'EU':  {'symbol': '^STOXX', 'fallback': 'EXS1.DE', 'name': 'STOXX Europe 600', 'suffix': '.DE', 'default_curr': 'EUR'}
    }
    
    current_cfg = exchange_config.get(exchange, exchange_config['NSE'])

   # --- LIVE INDEX CHART SECTION ---
    st.subheader(f"Live Market Context: {current_cfg['name']}")
    
    # Timeframe selector pill options
    tf_options = {
        "1D": {"days": 1, "label": "past day"},
        "5D": {"days": 5, "label": "past 5 days"},
        "1M": {"days": 30, "label": "past month"},
        "6M": {"days": 180, "label": "past 6 months"},
        "YTD": {"days": "YTD", "label": "year to date"},
        "1Y": {"days": 365, "label": "past year"},
        "5Y": {"days": 1825, "label": "past 5 years"},
        "MAX": {"days": 3650, "label": "all time"}
    }

    # Time period buttons bar
    col_tf, _ = st.columns([3, 1])
    with col_tf:
        selected_tf = st.segmented_control(
            "Timeframe", 
            options=list(tf_options.keys()), 
            default="6M", 
            label_visibility="collapsed"
        )

    with st.spinner(f"Fetching live benchmark data for {current_cfg['name']}..."):
        try:
            import plotly.graph_objects as go

            end_d = datetime.datetime.now()
            tf_info = tf_options.get(selected_tf, tf_options["6M"])
            
            # Determine start date based on selected timeframe
            if tf_info["days"] == "YTD":
                start_d = datetime.datetime(end_d.year, 1, 1)
            else:
                start_d = end_d - datetime.timedelta(days=tf_info["days"])
            
            # Fetch data with appropriate interval (1m/5m for 1D, daily for others)
            interval = "15m" if selected_tf == "1D" else "1d"
            bench_raw = yf.download(
                current_cfg['symbol'], 
                start=start_d, 
                end=end_d, 
                interval=interval,
                progress=False,
                auto_adjust=True
            )
            
            if not bench_raw.empty:
                if 'Close' in bench_raw.columns:
                    chart_series = bench_raw['Close']
                else:
                    chart_series = bench_raw.iloc[:, 0]
                
                if isinstance(chart_series, pd.DataFrame):
                    chart_series = chart_series.squeeze()
                
                chart_df = chart_series.dropna().reset_index()
                date_col = chart_df.columns[0]
                chart_df.columns = ['Date', 'Close']

                # Calculations for Price, Absolute Change & Percentage Change
                start_price = chart_df['Close'].iloc[0]
                latest_price = chart_df['Close'].iloc[-1]
                abs_change = latest_price - start_price
                pct_change = (abs_change / start_price) * 100
                is_falling = abs_change < 0

                # Format colors and arrows matching Google Finance
                theme_color = '#D93025' if is_falling else '#188038'
                fill_color = 'rgba(217, 48, 37, 0.10)' if is_falling else 'rgba(24, 128, 56, 0.10)'
                arrow = '↓' if is_falling else '↑'
                sign = '' if is_falling else '+'

                # Display Google Finance Header stats
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
                    unsafe_allow_html=True
                )

                # Chart bounds for clean dynamic zoom
                y_min = chart_df['Close'].min() * 0.998
                y_max = chart_df['Close'].max() * 1.002

                # Build Plotly Line Area Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=chart_df['Date'],
                    y=chart_df['Close'],
                    mode='lines',
                    line=dict(color=theme_color, width=2),
                    fill='tonexty',
                    fillcolor=fill_color,
                    hovertemplate="%{x}<br><b>Price:</b> %{y:,.2f}<extra></extra>"
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
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                
                st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
            else:
                st.warning("Market benchmark data temporarily unavailable.")
        except Exception as e:
            st.warning(f"Benchmark market index chart error: {str(e)}")
            
    # --- SIDEBAR CONTROL PANEL ---
    st.sidebar.header(f"1. Configuration ({exchange})")
    
    # Currency Dropdown Menu
    currency = st.sidebar.selectbox(
        "Denominated Currency", 
        ["INR", "ZAR", "USD", "EUR", "GBP"], 
        index=["INR", "ZAR", "USD", "EUR", "GBP"].index(current_cfg['default_curr'])
    )
    
    # Number Input for Capital (No slider)
    initial_capital = st.sidebar.number_input(
        label=f"Initial Capital ({currency})",
        min_value=1000.0,
        value=100000.0,
        step=10000.0,
        format="%.2f"
    )

    time_horizon_days = st.sidebar.number_input("Holding Period (Days)", min_value=1, max_value=365*4, value=30, step=1)
    confidence_level = st.sidebar.selectbox("VaR Confidence Level", [0.90, 0.95, 0.99], index=1)

    uploaded_file = st.sidebar.file_uploader(f"Upload {exchange} Portfolio CSV", type=["csv"]) 

    target_return_pct = st.sidebar.number_input(
    "Target Return (% p.a. for Sortino)",
    min_value=-50.0,
    max_value=100.0,
    value=0.0,
    step=0.5,
    help="Minimum acceptable return per annum used to calculate downside deviation.",
     )

# ==============================================================================
# DYNAMIC SECTOR PERFORMANCE & STOCK DRILL-DOWN (NSE / BSE / JSE)
# ==============================================================================

    st.markdown("---")
    st.info("Scroll down to add/edit the stocks")
    st.header(f"📊 {exchange} Sector Performance & Stock Drill-Down Analysis")
    
    # 1. Exchange-Specific Sector Configurations
    ALL_EXCHANGE_CONFIGS = {
        "NSE": {
            "Nifty IT": {
                "indices": ["^CNXIT"],
                "stocks": [
                    "TCS.NS",
                    "INFY.NS",
                    "WIPRO.NS",
                    "HCLTECH.NS",
                    "TECHM.NS",
                    "LTIM.NS",
                    "PERSISTENT.NS",
                    "COFORGE.NS",
                    "MPHASIS.NS",
                    "LTTS.NS",
                ],
            },
            "Nifty Bank": {
                "indices": ["^NSEBANK"],
                "stocks": [
                    "HDFCBANK.NS",
                    "ICICIBANK.NS",
                    "SBIN.NS",
                    "KOTAKBANK.NS",
                    "AXISBANK.NS",
                    "BANKBARODA.NS",
                    "PNB.NS",
                    "INDUSINDBK.NS",
                    "AUBANK.NS",
                    "IDFCFIRSTB.NS",
                ],
            },
            "Nifty Auto": {
                "indices": ["^CNXAUTO"],
                "stocks": [
                    "TATAMOTORS.NS",
                    "MARUTI.NS",
                    "M&M.NS",
                    "BAJAJ-AUTO.NS",
                    "HEROMOTOCO.NS",
                    "EICHERMOT.NS",
                    "TVSMOTOR.NS",
                    "BHARATFORG.NS",
                    "BOSCHLTD.NS",
                    "MRF.NS",
                ],
            },
            "Nifty FMCG": {
                "indices": ["^CNXFMCG"],
                "stocks": [
                    "HINDUNILVR.NS",
                    "ITC.NS",
                    "NESTLEIND.NS",
                    "BRITANNIA.NS",
                    "TATACONSUM.NS",
                    "DABUR.NS",
                    "GODREJCP.NS",
                    "MARICO.NS",
                    "COLPAL.NS",
                    "VBL.NS",
                ],
            },
            "Nifty Energy": {
                "indices": ["^CNXENERGY"],
                "stocks": [
                    "RELIANCE.NS",
                    "ONGC.NS",
                    "BPCL.NS",
                    "NTPC.NS",
                    "POWERGRID.NS",
                    "IOC.NS",
                    "GAIL.NS",
                    "ADANIGREEN.NS",
                    "TATAPOWER.NS",
                    "COALINDIA.NS",
                ],
            },
            "Nifty Pharma": {
                "indices": ["^CNXPHARMA"],
                "stocks": [
                    "SUNPHARMA.NS",
                    "CIPLA.NS",
                    "DRREDDY.NS",
                    "DIVISLAB.NS",
                    "LUPIN.NS",
                    "APOLLOHOSP.NS",
                    "BIOCON.NS",
                    "TORNTPHARM.NS",
                    "ZYDUSLIFE.NS",
                    "MANKIND.NS",
                ],
            },
        },
        "BSE": {
            "BSE IT": {
                "indices": ["^BSESIT.BO", "^BSESIT"],
                "stocks": [
                    "TCS.NS",
                    "INFY.NS",
                    "WIPRO.NS",
                    "HCLTECH.NS",
                    "TECHM.NS",
                    "LTIM.NS",
                    "PERSISTENT.NS",
                    "COFORGE.NS",
                ],
            },
            "BSE Bankex": {
                "indices": ["^BSEBANK.NS", "^BSEBANK"],
                "stocks": [
                    "HDFCBANK.NS",
                    "ICICIBANK.NS",
                    "SBIN.NS",
                    "KOTAKBANK.NS",
                    "AXISBANK.NS",
                    "BANKBARODA.NS",
                    "PNB.NS",
                ],
            },
            "BSE Healthcare": {
                "indices": ["^BSEHC.NS", "^BSEHC"],
                "stocks": [
                    "SUNPHARMA.NS",
                    "DRREDDY.NS",
                    "CIPLA.NS",
                    "DIVISLAB.NS",
                    "APOLLOHOSP.NS",
                    "LUPIN.NS",
                ],
            },
            "BSE Auto": {
                "indices": ["^BSEAUTO.NS", "^BSEAUTO"],
                "stocks": [
                    "TATAMOTORS.NS",
                    "MARUTI.NS",
                    "M&M.NS",
                    "BAJAJ-AUTO.NS",
                    "HEROMOTOCO.NS",
                    "EICHERMOT.NS",
                ],
            },
            "BSE FMCG": {
                "indices": ["^BSEFMCG.NS"],
                "stocks": [
                    "HINDUNILVR.NS",
                    "ITC.NS",
                    "NESTLEIND.NS",
                    "BRITANNIA.NS",
                    "DABUR.NS",
                ],
            },
            "BSE Metal": {
            "indices": ["^BSEMT.NS"],
            "stocks": [
                "TATASTEEL.NS",
                "JSWSTEEL.NS",
                "HINDALCO.NS",
                "VEDL.NS",
                ],
            },
            "BSE Oil & Gas": {
                "indices": ["^BSEOIL.NS"],
                "stocks": [
                    "RELIANCE.NS",
                    "ONGC.NS",
                    "IOC.NS",
                    "BPCL.NS",
                ],
            },
            "BSE Power & Utilities": {
                "indices": ["^BSEPOWER.NS"],
                "stocks": [
                    "NTPC.NS",
                    "POWERGRID.NS",
                    "ADANIPOWER.NS",
                    "TATAPOWER.NS",
                ],
            },
            "BSE Realty": {
                "indices": ["^BSERLTY.NS"],
                "stocks": [
                    "DLF.NS",
                    "GODREJPROP.NS",
                    "OBEROIRLTY.NS",
                    "PRESTIGE.NS",
                ],
            },
            "BSE Capital Goods & Industrials": {
                "indices": ["^BSECG.NS"],
                "stocks": [
                    "LT.NS",
                    "BEL.NS",
                    "SIEMENS.NS",
                    "ABB.NS",
                ],
            },
        },
        "US": {
            "Technology": {
                "indices": ["^NDX", "^GSPC"],
                "stocks": [
                    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "AMD", "ORCL"
                ]
            },
            "Financials": {
                "indices": ["^IXFN", "^GSPC"],
                "stocks": [
                    "JPM", "BAC", "WFC", "MS", "GS", "BLK", "C", "AXP", "SPGI", "SCHW"
                ]
            },
            "Healthcare": {
                "indices": ["^IXHC", "^GSPC"],
                "stocks": [
                    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "AMGN", "DHR"
                ]
            },
            "Energy & Industrials": {
                "indices": ["^IXID", "^GSPC"],
                "stocks": [
                    "XOM", "CVX", "GE", "CAT", "UNP", "RTX", "HON", "BA", "LMT", "DE"
                ]
            }
        },
        "UK": {
            "Energy & Commodities": {
                "indices": ["^FTSE"],
                "stocks": [
                    "SHEL.L", "BP.L", "GLEN.L", "RIO.L", "AAL.L", "ANTO.L"
                ]
            },
            "Banking & Finance": {
                "indices": ["^FTSE"],
                "stocks": [
                    "HSBA.L", "LLOY.L", "BARC.L", "NWG.L", "NWG.L", "PRU.L"
                ]
            },
            "Healthcare & Consumer": {
                "indices": ["^FTSE"],
                "stocks": [
                    "AZN.L", "GSK.L", "ULVR.L", "BATS.L", "RB.L", "DGE.L"
                ]
            }
        },
        "EU": {
            "Technology & Industrials": {
                "indices": ["^STOXX", "^GDAXI"],
                "stocks": [
                    "ASML.AS", "SAP.DE", "SIE.DE", "AIR.PA", "IFX.DE", "AD.AS"
                ]
            },
            "Luxury & Consumer Goods": {
                "indices": ["^STOXX", "^FCHI"],
                "stocks": [
                    "MC.PA", "OR.PA", "RMS.PA", "KER.PA", "ABI.BR", "HEIA.AS"
                ]
            },
            "Financials & Automakers": {
                "indices": ["^STOXX", "^GDAXI"],
                "stocks": [
                    "ALV.DE", "MBG.DE", "BMW.DE", "VOW3.DE", "SAN.MC", "BNP.PA"
                ]
            }
        },
        "JSE": {
            "JSE Mining & Resources": {
                "indices": ["^J210.JO"],
                "stocks": [
                    "AGL.JO",
                    "BHP.JO",
                    "GFI.JO",
                    "ANG.JO",
                    "IMP.JO",
                    "SSW.JO",
                    "AMS.JO",
                ],
            },
            "JSE Financials": {
                "indices": ["^J212.JO"],
                "stocks": [
                    "FSR.JO",
                    "SBK.JO",
                    "ABG.JO",
                    "NED.JO",
                    "SLM.JO",
                    "CPI.JO",
                    "INP.JO",
                ],
            },
            "JSE Tech & Telecoms": {
                "indices": ["^J211.JO"],
                "stocks": ["NPN.JO", "PRX.JO", "MTN.JO", "VOD.JO", "TEL.JO"],
            },
            "JSE Retail & Consumer": {
                "indices": ["^J213.JO"],
                "stocks": [
                    "WHL.JO",
                    "SHP.JO",
                    "MRP.JO",
                    "CLS.JO",
                    "PIK.JO",
                    "SPP.JO",
                ],
            },
        },
    }
    
    # Load chosen exchange sector configuration dynamically
    sector_config = ALL_EXCHANGE_CONFIGS.get(
        exchange, ALL_EXCHANGE_CONFIGS["BSE"]
    )
    
    # 2. Timeframe Selection
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
    
    # 3. Gather Tickers
    all_tickers_to_fetch = []
    stock_to_sector_map = {}
    
    for sector_name, data in sector_config.items():
        all_tickers_to_fetch.extend(data["indices"])
        all_tickers_to_fetch.extend(data["stocks"])
        for stock in data["stocks"]:
            stock_to_sector_map[stock] = sector_name
    
    # Dedup list
    all_tickers_to_fetch = list(set(all_tickers_to_fetch))
    
    # 4. Fetch Market Data Matrix
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
    
    # 5. Calculate Sector Returns & Individual Stock Performance
    sector_returns = {}
    stock_returns_dict = {}
    
    if not price_df.empty:
        # A. Calculate Individual Stock Growth Rates
        for col in price_df.columns:
            series = price_df[col].dropna()
            if len(series) >= lookback_days:
                s_start = series.iloc[-lookback_days]
                s_end = series.iloc[-1]
                if s_start > 0:
                    ret = ((s_end - s_start) / s_start) * 100
                    stock_returns_dict[col] = ret
    
        # B. Calculate Aggregate Sector Growth
        for sector_name, data in sector_config.items():
            sec_stock_rets = [
                stock_returns_dict[stk]
                for stk in data["stocks"]
                if stk in stock_returns_dict
            ]
            if sec_stock_rets:
                sector_returns[sector_name] = np.mean(sec_stock_rets)
    
    # 6. Sector Performance Tables & Charts
    perf_df = pd.DataFrame(
        list(sector_returns.items()), columns=["Sector", "Growth (%)"]
    )
    perf_df = perf_df.sort_values(by="Growth (%)", ascending=False)
    
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
        worst_sec_df = perf_df.tail(5).sort_values(
            by="Growth (%)", ascending=True
        )
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
    
    # 7. TOP 10 BEST & WORST INDIVIDUAL STOCKS ANALYSIS
    st.markdown("---")
    st.subheader(f"📌 Top 10 Best & Worst Stock Performers ({exchange})")
    
    # Convert individual stock returns into a DataFrame
    all_stocks_in_exchange = []
    for sector_data in sector_config.values():
        all_stocks_in_exchange.extend(sector_data["stocks"])
    
    all_stocks_in_exchange = set(all_stocks_in_exchange)
    
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
        stock_perf_df = pd.DataFrame(stock_perf_list).sort_values(
        by="Return (%)", ascending=False
        )
    else:
        stock_perf_df = pd.DataFrame(columns=["Ticker", "Sector", "Return (%)"])
    
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
        worst10_stocks = stock_perf_df.tail(10).sort_values(
            by="Return (%)", ascending=True
        ).reset_index(drop=True)
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

# ==============================================================================
# 8. INDIVIDUAL SECTOR DRILL-DOWN (DROPDOWN SELECTOR & GRAPH)
# ==============================================================================
    st.markdown("---")
    st.subheader("🔎 Detailed Sector Drill-Down")
    
    # Dropdown menu populated dynamically with available sectors for the chosen exchange
    available_sectors = list(sector_config.keys())
    selected_sector = st.selectbox(
        "Choose a Sector to Analyze:", options=available_sectors
    )
    
    if selected_sector:
        # Retrieve stocks configured for the selected sector
        sector_stocks = sector_config[selected_sector]["stocks"]
    
        # Filter stock performance results for the selected sector
        sector_stocks_perf = [
            {"Ticker": stk, "Return (%)": stock_returns_dict.get(stk, np.nan)}
            for stk in sector_stocks
            if stk in stock_returns_dict
        ]
    
        sector_drill_df = pd.DataFrame(sector_stocks_perf).sort_values(
            by="Return (%)", ascending=False
        )
    
        if not sector_drill_df.empty:
            sd_col1, sd_col2 = st.columns([1, 1.5], gap="large")
    
            with sd_col1:
                st.markdown(
                    f"### Performance Breakdown: **{selected_sector}** ({timeframe})"
                )
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
    
    # --- WORKFLOW: CSV PARSING & EDITABLE HOLDINGS TABLE ---
    flag = 0
    if uploaded_file is None:
        flag = 1
    if uploaded_file is not None or flag == 1:
        try:
            if uploaded_file is not None:
                try:
                    uploaded_file.seek(0)
                    bytes_data = uploaded_file.read()
                    
                    if bytes_data:
                        safe_buffer = io.BytesIO(bytes_data)
                        
                        # Try reading normally, skip bad lines/empty rows automatically
                        raw_df = pd.read_csv(safe_buffer, skipinitialspace=True, on_bad_lines='skip')
                        
                        # Drop completely empty rows or columns
                        raw_df = raw_df.dropna(how='all').dropna(axis=1, how='all')
                        
                        if not raw_df.empty and len(raw_df.columns) > 0:
                            raw_df.columns = raw_df.columns.astype(str).str.strip()
    
                            # Smart header search
                            possible_ticker_cols = ['Instrument', 'Symbol', 'Ticker', 'Stock', 'Security', 'Company', 'Scrip']
                            possible_qty_cols = ['Qty.', 'Quantity', 'Qty', 'Shares', 'Volume', 'Units']
    
                            ticker_col = next((col for col in possible_ticker_cols if col in raw_df.columns), raw_df.columns[0])
                            qty_col = next((col for col in possible_qty_cols if col in raw_df.columns), (raw_df.columns[1] if len(raw_df.columns) > 1 else raw_df.columns[0]))
                            
                            # Clean data and ensure no NaN or out-of-bounds indices
                            sub_df = raw_df[[ticker_col, qty_col]].copy()
                            sub_df.columns = ['Ticker', 'Quantity']
                            
                            sub_df['Ticker'] = sub_df['Ticker'].astype(str).str.replace(r'\.(NS|BO)$', '', regex=True).str.strip()
                            sub_df['Quantity'] = pd.to_numeric(sub_df['Quantity'], errors='coerce').fillna(1).astype(int)
                            
                            # Filter out invalid rows
                            sub_df = sub_df[sub_df['Ticker'] != 'nan']
                            if not sub_df.empty:
                                holdings_df = sub_df.reset_index(drop=True)
                                st.success("Portfolio CSV loaded successfully!")
                except Exception as e:
                    st.warning(f"Could not parse custom CSV format ({e}). Using default fallback portfolio.")
            else:
                holdings_df = pd.DataFrame({'Ticker': ['',],'Quantity': [0,],})

            st.write("### 📝 Interactive Holdings Adjuster")
            st.info("Add/Edit stocks & quantities directly in the table below to dynamically trigger stress-test calculations.")
            
            # EDITABLE DATA TABLE
            edited_df = st.data_editor(
                holdings_df,
                num_rows="dynamic",
                use_container_width=True,
                key="holdings_editor"
            )



            # Re-format Tickers with exchange suffix for Yahoo Finance
            # Re-format user tickers for display/mapping
            edited_df['yf_ticker'] = edited_df['Ticker'].astype(str).str.strip() + current_cfg['suffix']
            active_tickers = edited_df['yf_ticker'].tolist()

            if len(active_tickers) > 0 and edited_df['Quantity'].sum() > 0:
                with st.spinner("Fetching stock prices & building full portfolio risk matrix..."):
                    end_date = datetime.datetime.now()
                    start_date = end_date - datetime.timedelta(days=365 * 5)

        # 1. Download price data for all universe tickers
                    price_data = yf.download(active_tickers, start=start_date, end=end_date)['Close']

        # Handle single series edge cases
                    if isinstance(price_data, pd.Series):
                        price_data = price_data.to_frame()

        # Clean invalid or empty columns
                        price_data = price_data.dropna(how='all', axis=1)

        # Safety Check to prevent out-of-bounds iloc crash
                    if price_data.empty:
                        st.error("Price data download returned no valid records for the given tickers.")
                    else:
                        price_data = price_data.ffill().bfill()
                        latest_prices = price_data.iloc[-1]

            # 2. Map latest prices safely to user holdings
            # (Matches symbol or fallback if user CSV uses standard stock symbols)
                        edited_df['Latest Price'] = edited_df['yf_ticker'].map(latest_prices)
            
            # Fill missing prices with 0 to prevent downstream calculation explosions
                        edited_df['Latest Price'] = edited_df['Latest Price'].fillna(0.0)
                        edited_df['Position Value'] = edited_df['Quantity'] * edited_df['Latest Price']
                        total_portfolio_market_val = edited_df['Position Value'].sum()

                        st.dataframe(edited_df[["Ticker","Quantity","Latest Price"]], hide_index=True, use_container_width=True)

            # 3. Create full 516x1 universe matrix
                        universe_tickers = price_data.columns.tolist()
                        full_universe_df = pd.DataFrame(index=universe_tickers)

            # 4. Map user position values into the full universe
                        user_position_series = edited_df.groupby('yf_ticker')['Position Value'].sum()
                        full_universe_df['Position Value'] = user_position_series
                        full_universe_df['Position Value'] = full_universe_df['Position Value'].fillna(0.0)

            # 5. Compute full zero-filled vector w (516 x 1)
                        if total_portfolio_market_val > 0:
                            full_universe_df['Weight'] = full_universe_df['Position Value'] / total_portfolio_market_val
                        else:
                            full_universe_df['Weight'] = 0.0

                        weights = full_universe_df['Weight'].values
                        if len(active_tickers) == 1:
                          # Force single stock weight to 100% (1.0) if only 1 ticker is active
                          weights = np.array([1.0])

                    # Compute Daily Returns
                returns = price_data.pct_change().dropna()
                # portfolio_daily_returns = returns.dot(weights)
                # --- SAFE SINGLE-STOCK VS MULTI-STOCK PORTFOLIO METRICS ---
                if len(active_tickers) == 1:
                  # Single-stock fallback: 100% weight, direct series assignment
                  portfolio_daily_returns = returns[active_tickers[0]]
                  # portfolio_variance = portfolio_daily_returns.var() * 252
                  # portfolio_std = portfolio_daily_returns.std() * (252**0.5)
                else:
                  # Multi-stock matrix calculations
                  portfolio_daily_returns = returns.dot(weights)
                  # cov_matrix = returns.cov() * 252
                  # portfolio_variance = weights.T @ cov_matrix @ weights
                  # portfolio_std = np.sqrt(portfolio_variance)

                cum_returns = (1 + portfolio_daily_returns["2024-01-01":]).cumprod() - 1      #.rolling(window=63).apply(lambda x: (1 + x).prod() - 1)
                #cum_rolling_returns = cum_rolling_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)

# --- ADD BENCHMARK OVERLAY (RED LINE) ---
                benchmark_symbol = current_cfg.get("symbol", "^NSEI")
                bench_df = yf.download(
                    benchmark_symbol,
                    start=cum_returns.index[0],
                    end=cum_returns.index[-1],
                    progress=False,
                )
                
                if isinstance(bench_df, pd.DataFrame) and "Close" in bench_df:
                    bench_series = bench_df["Close"]
                    if isinstance(bench_series, pd.DataFrame):
                        bench_series = bench_series.squeeze()
                    bench_daily_ret = bench_series.pct_change().dropna()
                    bench_cum_returns = (1 + bench_daily_ret).cumprod() - 1
                else:
                    bench_cum_returns = pd.Series(0, index=cum_returns.index)
                
                # Build Combined Figure
                import plotly.graph_objects as go
                
                fig = go.Figure()
                
                # Add Portfolio Trace (Blue Line)
                fig.add_trace(
                    go.Scatter(
                        x=cum_returns.index,
                        y=cum_returns.values,
                        mode="lines",
                        name="Portfolio Return",
                        line=dict(color="#1f77b4", width=2),
                    )
                )
                
                # Add Live Benchmark Trace (Red Line)
                fig.add_trace(
                    go.Scatter(
                        x=bench_cum_returns.index,
                        y=bench_cum_returns.values,
                        mode="lines",
                        name=f"Market Index ({current_cfg.get('name', 'Benchmark')})",
                        line=dict(color="#FF0000", width=2, dash="solid"),
                    )
                )
                
                # Format Y-axis & Hover Timeline
                fig.update_layout(
                    title="Cumulative Portfolio vs. Market Performance",
                    yaxis_tickformat=".2%",
                    hovermode="x unified",
                    xaxis_title="Date",
                    yaxis_title="Return",
                    dragmode=False,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                
                # Render in Streamlit
                #st.plotly_chart(fig, use_container_width=True)
                #fig = px.line(cum_returns,title="Cumulative Portfolio Returns",labels={"value": "Cumulative Return", "index": "Date"},)

# 3. Format Y-axis as percentage and enable hover timeline
                fig.update_layout(
                #yaxis_tickformat=".2%",
                hovermode="x unified",
                xaxis_title="Date",
                yaxis_title="Return",
                dragmode=False
                )

# 4. Render in Streamlit
                st.plotly_chart(fig, use_container_width=True, key="portfolio_vs_market_chart")
                fig_bytes = pio.to_image(fig, format='png', width=700, height=450)

# --- STATISTICAL RISK CALCULATIONS ---

                # VaR Percentage & Monetary Loss
                var_percentile = (1 - confidence_level) * 100
                var_pct = np.percentile(portfolio_daily_returns, var_percentile) 
                var_amount = abs(var_pct) * initial_capital * np.sqrt(time_horizon_days)

                # CVaR (Expected Shortfall) Percentage & Monetary Loss
                cvar_pct = portfolio_daily_returns[portfolio_daily_returns <= var_pct].mean() * np.sqrt(time_horizon_days)
                cvar_amount = abs(cvar_pct) * initial_capital

                # Sharpe Ratio (6.5% baseline RFR)
                rfr_dict = {
                    'NSE': 0.065,  # India baseline ~6.5%
                    'BSE': 0.065,  # India baseline ~6.5%
                    'US':  0.045,  # US baseline ~4.5%
                    'UK':  0.045,  # UK baseline ~4.5%
                    'EU':  0.030,  # Eurozone baseline ~3.0%
                    'JSE': 0.0725  # South Africa baseline ~7.25%
                }
                rfr_annual = rfr_dict.get(exchange, 0.05)
                rfr_daily = rfr_annual / 252
                #sharpe = ((portfolio_daily_returns.mean() - rfr_daily) / portfolio_daily_returns.std()) * np.sqrt(252)
                if len(active_tickers) == 1:
                  sharpe = (
                      (portfolio_daily_returns.mean() - rfr_daily)
                      / portfolio_daily_returns.std()
                  ) * np.sqrt(252)
                  weight_entropy = 0.0  # Entropy of 100% weight in single asset is 0
                  risk_contri = np.array([1.0])
                  risk_entropy = 0.0
                else:
                    sharpe = erk.sharpe_ratio(portfolio_daily_returns, rfr_daily*252, 252)
                    weight_entropy = erk.enc(weights)
                    risk_contri = erk.risk_contribution(weights, np.cov(returns.T))
                    risk_entropy = erk.enc(risk_contri)
# Sortino
                #target_return_daily = (target_return_pct / 100.0) / 252.0
                
                def sortino_ratio(returns: pd.Series | pd.DataFrame, rfr: float = 0.055,target_return: float = 0.0,periods_per_year: int = 252,) -> float | pd.Series:
                      """Calculates the Annualized Sortino Ratio for a given returns series or DataFrame"""
  # Adjust risk-free rate to period frequency
                      rfr_period = rfr / periods_per_year
                      target_return = (target_return/100.0)/periods_per_year

  # Calculate excess returns above risk-free rate
                      mean_excess_return = returns.mean() - rfr_period

  # Isolate downside returns (returns below target threshold)
                      downside_returns = returns[returns < target_return]

  # Calculate Downside Deviation (annualized)
                      downside_std = np.sqrt((downside_returns**2).mean()) * np.sqrt(periods_per_year)

  # Annualize expected excess returns
                      annualized_excess_return = mean_excess_return * periods_per_year

  # Avoid division by zero if downside deviation is 0
                      if isinstance(downside_std, (pd.Series, np.ndarray)):
                            downside_std = downside_std.replace(0, np.nan)
                      elif downside_std == 0:
                            return np.nan

                      return annualized_excess_return / downside_std
                
                sortino = sortino_ratio(portfolio_daily_returns, rfr=rfr_daily*252, target_return=target_return_pct, periods_per_year=252)

# Expected Returns 
                res = erk.backtest_ws(returns, 252, erk.weight_fix, weights = weights)
                btr = pd.DataFrame({"Results":res})
                exp_ret = erk.summary_stats(btr.dropna(), rfr_daily, 252)["Annualized Return"]
                exp_vol = erk.summary_stats(btr.dropna(), rfr_daily, 252)["Annualized Vol"]
                exp_dd = erk.summary_stats(btr.dropna(), rfr_daily, 252)["Max Drawdown"]

                if isinstance(exp_dd, (pd.Series, np.ndarray)):
                  exp_dd = float(exp_dd.iloc[0])
                else:
                  exp_dd = float(exp_dd)

                if isinstance(exp_ret, (pd.Series, np.ndarray)):
                  exp_ret = float(exp_ret.iloc[0])
                else:
                  exp_ret = float(exp_ret)

                if isinstance(exp_vol, (pd.Series, np.ndarray)):
                  exp_vol = float(exp_vol.iloc[0])
                else:
                  exp_vol = float(exp_vol)

# --- CALCULATE PORTFOLIO BETA AGAINST BENCHMARK ---
                if "bench_daily_ret" in locals() and not bench_daily_ret.empty:
                    aligned_data = pd.DataFrame(
                        {"portfolio": portfolio_daily_returns, "market": bench_daily_ret}
                    ).dropna()
                    if len(aligned_data) > 5:
                        cov_matrix = np.cov(
                            aligned_data["portfolio"], aligned_data["market"]
                        )
                        portfolio_beta = cov_matrix[0, 1] / cov_matrix[1, 1]
                    else:
                        portfolio_beta = 1.0
                else:
                    portfolio_beta = 1.0
                
                # --- QUANTITATIVE PORTFOLIO SCORE (0-100) ---
                # 1. Sharpe Ratio component (Max 25 pts)
                norm_sharpe = min(max(sharpe / 2.0, 0), 1) * 25
                
                # 2. Sortino Ratio component (Max 25 pts)
                norm_sortino = min(max(sortino / 3.0, 0), 1) * 25
                
                # 3. Maximum Drawdown component (Max 25 pts - penalizing high drawdowns)
                norm_drawdown = max(1 - abs(exp_dd) / 0.5, 0) * 25
                
                # 4. Beta component (Max 25 pts - rewarding stability/close market tracking)
                beta_diff = abs(portfolio_beta - 1.0)
                norm_beta = max(1 - beta_diff / 1.5, 0) * 25
                
                portfolio_score = round(
                    norm_sharpe + norm_sortino + norm_drawdown + norm_beta, 1
                )

                
                # --- RENDER PORTFOLIO SCORE CARD ---
                _, center_col, _ = st.columns([1, 2, 1])

                with center_col:
                    st.metric(
                        label="⭐ Quantitative Portfolio Score",
                        value=f"{portfolio_score} / 100",
                        delta=f"Beta: {portfolio_beta:.2f}",
                    )
                
                st.markdown("---")
                st.write("### 🛡️ Returns")

                res3_col1, res3_col2, res3_col3 = st.columns(3)

                res3_col1.metric(
                    label="Annualized Expected Return",
                    value=f"{abs(exp_ret)*100:.2f}%", 
                    #delta=f"-{currency} {var_amount:,.2f}", 
                    #delta_color="inverse"
                )
                
                res3_col2.metric(
                    label="Annualized Volatility", 
                    value=f"{abs(exp_vol)*100:.2f}%", 
                    #delta=f"-{currency} {cvar_amount:,.2f}", 
                    #delta_color="inverse"
                )
                
                res3_col3.metric(
                    label="Maximum Drawdown", 
                    value=f"{abs(exp_dd)*100:.2f}%"
                )

                st.markdown("---")
                st.write("### 🛡️ Value at Risk (VaR) & Downside Metrics")
                
                res_col1, res_col2, res_col3 = st.columns(3)
                
                res_col1.metric(
                    label=f"Historical VaR ({int(confidence_level*100)}%) ({time_horizon_days} days)", 
                    value=f"{abs(var_pct)*100 * np.sqrt(time_horizon_days):.2f}%", 
                    delta=f"-{currency} {var_amount:,.2f}", 
                    delta_color="inverse"
                )
                
                res_col2.metric(
                    label="CVaR / Expected Shortfall", 
                    value=f"{abs(cvar_pct)*100:.2f}%", 
                    delta=f"-{currency} {cvar_amount:,.2f}", 
                    delta_color="inverse"
                )
                
                res_col3.metric(
                    label="Annualized Sharpe Ratio", 
                    value=f"{sharpe:.2f}"
                )

                res2_col1, res2_col2, res2_col3 = st.columns(3)

                res2_col1.metric(
                    label=f"Portfolio - Diversification ({int(confidence_level*100)}%)", 
                    value=f"{abs(weight_entropy):.2f}%", 
                    #delta=f"-{currency} {var_amount:,.2f}", 
                    #delta_color="inverse"
                )
                
                res2_col2.metric(
                    label="Risk - Diversification", 
                    value=f"{abs(risk_entropy):.2f}%", 
                    #delta=f"-{currency} {cvar_amount:,.2f}", 
                    #delta_color="inverse"
                )
                
                res2_col3.metric(
                    label="Sortino Ratio", 
                    value=f"{sortino:.2f}"
                )

                # --- WEIGHT & RISK CONTRIBUTION BAR PLOTS ---
                st.markdown("---")
                st.write("### 📊 Portfolio Weight vs. Risk Contribution Breakdown")
                
                # Extract non-zero positions for clean plotting
                active_tickers = [
                    t for t, w in zip(universe_tickers, weights) if w > 0.0001
                ]
                active_weights = [
                    w for w in weights if w > 0.0001
                ]
                active_risk_contri = [
                    rc for w, rc in zip(weights, risk_contri) if w > 0.0001
                ]
                
                # Create side-by-side columns
                r_col1, r_col2 = st.columns(2, gap="large")
                
                with r_col1:
                    fig_weights = px.bar(
                        x=active_tickers,
                        y=active_weights,
                        labels={"x": "Stock", "y": "Portfolio Weight"},
                        title="Asset Weight Distribution",
                        color=active_weights,
                        color_continuous_scale="Blues",
                    )
                    fig_weights.update_layout(
                        yaxis_tickformat=".2%",
                        showlegend=False,
                        hovermode="x unified",
                        dragmode=False
                    )
                    st.plotly_chart(
                        fig_weights,
                        use_container_width=True,
                        key="weight_entropy_bar_plot",
                    )
                
                with r_col2:
                    fig_risk = px.bar(
                        x=active_tickers,
                        y=active_risk_contri,
                        labels={"x": "Stock", "y": "Risk Contribution"},
                        title="Risk Contribution Breakdown",
                        color=active_risk_contri,
                        color_continuous_scale="Reds",
                    )
                    fig_risk.update_layout(
                        yaxis_tickformat=".2%",
                        showlegend=False,
                        hovermode="x unified",
                        dragmode=False
                    )
                    st.plotly_chart(
                        fig_risk,
                        use_container_width=True,
                        key="risk_contribution_bar_plot",
                    )
                fig_weights_bytes = pio.to_image(fig_weights, format='png', width=700, height=450)
                fig_risk_bytes = pio.to_image(fig_risk, format='png', width=700, height=450)

                # # --- MONTE CARLO RISK STRESS RESULTS (METRICS ONLY) ---
                # st.markdown("---")
                # st.write(f"### 🎲 Monte Carlo Risk Audit ({time_horizon_days}-Day Horizon)")

                # num_sims = 1000
                # mean_ret = portfolio_daily_returns.mean()
                # std_ret = portfolio_daily_returns.std()

                # # Calculate Monte Carlo scenarios without graph output
                # simulated_ending_values = np.zeros(num_sims)
                # for i in range(num_sims):
                #     daily_sims = np.random.normal(mean_ret, std_ret, time_horizon_days)
                #     simulated_ending_values[i] = initial_capital * np.exp(np.sum(daily_sims))

                # mc_var_95 = np.percentile(simulated_ending_values, 5)
                # max_expected_loss = initial_capital - mc_var_95
                # prob_loss_10pct = (np.sum(simulated_ending_values < (initial_capital * 0.90)) / num_sims) * 100

                # --- MONTE CARLO RISK STRESS RESULTS (USING QUANT_RISK_KIT GBM) ---
                st.markdown("---")
                st.write(f"### 🎲 Monte Carlo Risk Audit ({time_horizon_days}-Day Horizon)")
                
                # 1. Parameter setup for GBM
                num_sims = 10000
                n_years = time_horizon_days / 252.0  # Convert holding horizon days to annual fraction
                mean_ret = portfolio_daily_returns.mean()
                std_ret = portfolio_daily_returns.std()
                
                # Annualize daily parameters for Geometric Brownian Motion
                mu_annual = mean_ret * 252.0
                sigma_annual = std_ret * (252.0 ** 0.5)
                
                # 2. Call your custom quant_risk_kit GBM simulation function
                # Generates 10,000 scenario price paths starting at initial_capital
                gbm_paths = erk.gbm(
                    n_years=n_years,
                    n_scenarios=num_sims,
                    mu=mu_annual,
                    sigma=sigma_annual,
                    steps_per_year=252,
                    s_0=initial_capital,
                    prices=True
                )
                
                # 3. Extract ending values from the final step of all 10,000 simulated paths
                simulated_ending_values = gbm_paths.iloc[-1].values
                
                # 4. Compute Risk Metrics
                mc_var_95 = np.percentile(simulated_ending_values, (1 - confidence_level) * 100)
                max_expected_loss = initial_capital - mc_var_95
                prob_loss_10pct = (np.sum(simulated_ending_values < (initial_capital * 0.90)) / num_sims) * 100.0

                mc_col1, mc_col2, mc_col3 = st.columns(3)
                
                mc_col1.metric(
                    label=f"Max Expected Loss ({time_horizon_days} Days)", 
                    value=f"{currency} {max_expected_loss:,.2f}"
                )
                
                mc_col2.metric(
                    label="Projected Portfolio Tail Value", 
                    value=f"{currency} {mc_var_95:,.2f}"
                )
                
                mc_col3.metric(
                    label="Probability of >10% Capital Drawdown", 
                    value=f"{prob_loss_10pct:.1f}%"
                )

                import io
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
                
                
                def draw_cream_background(canvas, doc):
                    """Draws the cream theme background (#FAF8ED) across the entire PDF page."""
                    canvas.saveState()
                    canvas.setFillColor(colors.HexColor("#FAF8ED"))
                    canvas.rect(0, 0, doc.pagesize[0], doc.pagesize[1], fill=1, stroke=0)
                    canvas.restoreState()
                
                
                def generate_pdf_report(
                    portfolio_score,
                    portfolio_beta,
                    holdings_df,
                    metrics_dict,
                    mc_dict,
                    fig_bytes=None,fig_weights_bytes=None, fig_risk_bytes=None,
                ):
                    buffer = io.BytesIO()
                    doc = SimpleDocTemplate(
                        buffer,
                        pagesize=A4,
                        rightMargin=28,
                        leftMargin=28,
                        topMargin=28,
                        bottomMargin=28,
                    )
                    story = []
                
                    # --- THEME COLOR PALETTE ---
                    NAVY = colors.HexColor("#1A2B4C")
                    DARK_NAVY = colors.HexColor("#0F1C33")
                    TEXT_CREAM = colors.HexColor("#FAF8ED")
                    ACCENT_BLUE = colors.HexColor("#38BDF8")
                
                    styles = getSampleStyleSheet()
                    title_style = ParagraphStyle(
                        "TitleStyle",
                        parent=styles["Heading1"],
                        fontName="Helvetica-Bold",
                        fontSize=13,
                        textColor=TEXT_CREAM,
                        alignment=1,
                        spaceAfter=2,
                    )
                    subtitle_style = ParagraphStyle(
                        "SubTitleStyle",
                        parent=styles["Normal"],
                        fontName="Helvetica",
                        fontSize=7.5,
                        textColor=colors.white,
                        alignment=1,
                    )
                    section_style = ParagraphStyle(
                        "SectionStyle",
                        parent=styles["Heading2"],
                        fontName="Helvetica-Bold",
                        fontSize=9.5,
                        textColor=NAVY,
                        spaceBefore=6,
                        spaceAfter=2,
                    )
                    th_style = ParagraphStyle(
                        "TH",
                        fontName="Helvetica-Bold",
                        fontSize=7.5,
                        textColor=TEXT_CREAM,
                        alignment=0,
                    )
                    td_style = ParagraphStyle(
                        "TD", fontName="Helvetica", fontSize=7.5, textColor=TEXT_CREAM, alignment=0
                    )
                    conclusion_style = ParagraphStyle(
                        "Conclusion",
                        parent=styles["Normal"],
                        fontName="Helvetica",
                        fontSize=7.5,
                        textColor=TEXT_CREAM,
                        leading=10,
                    )
                
                    # 1. Masthead Header
                    header_data = [
                        [Paragraph("WIRATOR — Portfolio Audit", title_style)],
                        [
                            Paragraph(
                                "Comprehensive Asset Allocation, Risk Performance &amp; Monte Carlo Projections",
                                subtitle_style,
                            )
                        ],
                    ]
                    header_table = Table(header_data, colWidths=[536])
                    header_table.setStyle(
                        TableStyle(
                            [
                                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                                ("TOPPADDING", (0, 0), (-1, -1), 6),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                ("ROUNDEDCORNERS", [5, 5, 5, 5]),
                            ]
                        )
                    )
                    story.append(header_table)
                    story.append(Spacer(1, 4))
                
                    # 2. Portfolio Score Card
                    score_data = [
                        [Paragraph("⭐ QUANTITATIVE PORTFOLIO SCORE", subtitle_style)],
                        [
                            Paragraph(
                                f"<font size=18><b>{portfolio_score} / 100</b></font>",
                                ParagraphStyle("ScoreVal", parent=title_style, fontSize=18),
                            )
                        ],
                        [
                            Paragraph(
                                f"Market Benchmark Beta: {portfolio_beta:.2f} (Defensive Alignment)",
                                ParagraphStyle(
                                    "ScoreSub", parent=subtitle_style, textColor=ACCENT_BLUE
                                ),
                            )
                        ],
                    ]
                    score_table = Table(score_data, colWidths=[536])
                    score_table.setStyle(
                        TableStyle(
                            [
                                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                                ("TOPPADDING", (0, 0), (-1, -1), 5),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                ("ROUNDEDCORNERS", [5, 5, 5, 5]),
                            ]
                        )
                    )
                    story.append(score_table)
                    story.append(Spacer(1, 4))
                
                    # 3. Edited Holdings / Stocks Table Breakdown
                    story.append(Paragraph("Portfolio Holdings Breakdown", section_style))
                    h_table_data = [
                        [
                            Paragraph("Asset Symbol", th_style),
                            Paragraph("Quantity", th_style),
                            Paragraph("Latest Price (INR)", th_style),
                        ]
                    ]
                
                    if edited_df is not None and not edited_df.empty:
                        for _, row in edited_df.iterrows():
                            ticker = str(
                                row.get("Ticker", row.get("Symbol", row.get("asset", "-")))
                            )
                            qty = str(row.get("Quantity", row.get("qty", "0")))
                            price = float(
                                row.get(
                                    "Latest Price",
                                    row.get("Price", row.get("latest_price", 0.0)),
                                )
                            )
                            h_table_data.append(
                                [
                                    Paragraph(ticker, td_style),
                                    Paragraph(qty, td_style),
                                    Paragraph(f"{price:,.2f}", td_style),
                                ]
                            )
                
                    holdings_table = Table(h_table_data, colWidths=[206, 165, 165])
                    holdings_table.setStyle(
                        TableStyle(
                            [
                                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                                ("TOPPADDING", (0, 0), (-1, -1), 3),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                                ("GRID", (0, 0), (-1, -1), 0.5, DARK_NAVY),
                                ("ROUNDEDCORNERS", [5, 5, 5, 5]),
                            ]
                        )
                    )
                    story.append(holdings_table)
                    story.append(Spacer(1, 4))
                
                    # 4. Cumulative Portfolio vs. Market Comparison Graph
                    if fig_bytes:
                        story.append(
                            Paragraph("Cumulative Portfolio vs. Market Performance", section_style)
                        )
                        story.append(RLImage(io.BytesIO(fig_bytes), width=536, height=300))
                        story.append(Spacer(1, 4))
                
                    # 5. Core Return & Risk Metrics Table
                    story.append(Paragraph("Core Return &amp; Risk Metrics", section_style))
                    metrics_data = [
                        [
                            Paragraph(
                                f"<b>ANNUALIZED RETURN</b><br/><font size=10><b>{metrics_dict.get('ann_return', '0.00%')}</b></font>",
                                td_style,
                            ),
                            Paragraph(
                                f"<b>ANNUALIZED VOLATILITY</b><br/><font size=10><b>{metrics_dict.get('ann_vol', '0.00%')}</b></font>",
                                td_style,
                            ),
                            Paragraph(
                                f"<b>MAXIMUM DRAWDOWN</b><br/><font size=10><b>{metrics_dict.get('max_dd', '0.00%')}</b></font>",
                                td_style,
                            ),
                        ]
                    ]
                    m_table = Table(metrics_data, colWidths=[178, 179, 179])
                    m_table.setStyle(
                        TableStyle(
                            [
                                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                                ("TOPPADDING", (0, 0), (-1, -1), 4),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                                ("GRID", (0, 0), (-1, -1), 0.5, DARK_NAVY),
                                ("ROUNDEDCORNERS", [5, 5, 5, 5]),
                            ]
                        )
                    )
                    story.append(m_table)
                    story.append(Spacer(1, 4))
                
                    # 6. Value at Risk (VaR) & Downside Metrics Table
                    story.append(
                        Paragraph("Value at Risk (VaR) &amp; Downside Metrics", section_style)
                    )
                    var_data = [
                        [
                            Paragraph(
                                f"<b>HISTORICAL VaR (95%)</b><br/><font size=9><b>{metrics_dict.get('var', '0.00%')}</b></font>",
                                td_style,
                            ),
                            Paragraph(
                                f"<b>CVaR / SHORTFALL</b><br/><font size=9><b>{metrics_dict.get('cvar', '0.00%')}</b></font>",
                                td_style,
                            ),
                            Paragraph(
                                f"<b>SHARPE RATIO</b><br/><font size=9><b>{metrics_dict.get('sharpe', '0.00')}</b></font>",
                                td_style,
                            ),
                        ],
                        [
                            Paragraph(
                                f"<b>PORTFOLIO DIVERSIFICATION</b><br/><font size=9><b>{metrics_dict.get('weight_ent', '0.00%')}</b></font>",
                                td_style,
                            ),
                            Paragraph(
                                f"<b>RISK DIVERSIFICATION</b><br/><font size=9><b>{metrics_dict.get('risk_ent', '0.00%')}</b></font>",
                                td_style,
                            ),
                            Paragraph(
                                f"<b>SORTINO RATIO ({metrics_dict.get('target_return', '0.00%')})</b><br/><font size=9><b>{metrics_dict.get('sortino', '0.00')}</b></font>",
                                td_style,
                            ),
                        ],
                    ]
                    var_table = Table(var_data, colWidths=[178, 179, 179])
                    var_table.setStyle(
                        TableStyle(
                            [
                                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                                ("TOPPADDING", (0, 0), (-1, -1), 4),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                                ("GRID", (0, 0), (-1, -1), 0.5, DARK_NAVY),
                                ("ROUNDEDCORNERS", [5, 5, 5, 5]),
                            ]
                        )
                    )
                    story.append(var_table)
                    story.append(Spacer(1, 4))

                    # --- SIDE-BY-SIDE WEIGHT & RISK CONTRIBUTION GRAPHS ---
                    if fig_weights_bytes and fig_risk_bytes:
                        story.append(
                            Paragraph(
                                "Portfolio Weight vs. Risk Contribution Breakdown", section_style
                            )
                        )
                        img_w = RLImage(io.BytesIO(fig_weights_bytes), width= 400, height=250)
                        img_r = RLImage(io.BytesIO(fig_risk_bytes), width= 400, height=250)

                        story.append(img_w)
                        story.append(Spacer(1, 4))
                        story.append(Spacer(1, 4))
                        story.append(img_r)
                        story.append(Spacer(1, 4))
                        # charts_table = Table([[img_w, img_r]], colWidths=[268, 500])
                        # charts_table.setStyle(
                        #     TableStyle([
                        #         ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        #         ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        #         ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        #         ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                        #         ("TOPPADDING", (0, 0), (-1, -1), 2),
                        #         ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                        #     ])
                        
                        # story.append(charts_table)

  # --- 7. Monte Carlo Risk Audit & Conclusion Block ---
  # [Your existing Monte Carlo table code follows here]
                
                    # 7. Monte Carlo Risk Audit & Conclusion Block
                    story.append(
                        Paragraph("Monte Carlo Risk Audit (30-Day Horizon)", section_style)
                    )
                    mc_data = [
                        [
                            Paragraph(
                                f"<b>MAX EXPECTED LOSS</b><br/><font size=9.5><b>{mc_dict.get('max_loss', 'INR 0')}</b></font>",
                                td_style,
                            ),
                            Paragraph(
                                f"<b>PROJECTED TAIL VALUE</b><br/><font size=9.5><b>{mc_dict.get('tail_val', 'INR 0')}</b></font>",
                                td_style,
                            ),
                            Paragraph(
                                f"<b>PROBABILITY OF >10% DD</b><br/><font size=9.5><b>{mc_dict.get('prob_dd', '0.3%')}</b></font>",
                                td_style,
                            ),
                        ],
                        [
                            Paragraph(
                                "<b>Audit Conclusion:</b> The portfolio demonstrates a defensive market profile with low beta against the benchmark index. Rebalance underperforming weightings to optimize risk-adjusted returns.",
                                conclusion_style,
                            ),
                            "",
                            "",
                        ],
                    ]
                    mc_table = Table(mc_data, colWidths=[178, 179, 179])
                    mc_table.setStyle(
                        TableStyle(
                            [
                                ("BACKGROUND", (0, 0), (-1, -1), DARK_NAVY),
                                ("SPAN", (0, 1), (2, 1)),
                                ("TOPPADDING", (0, 0), (-1, -1), 4),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                                ("ROUNDEDCORNERS", [5, 5, 5, 5]),
                            ]
                        )
                    )
                    story.append(mc_table)
                
                    doc.build(
                        story,
                        onFirstPage=draw_cream_background,
                        onLaterPages=draw_cream_background,
                    )
                    buffer.seek(0)
                    return buffer.getvalue()
                
                
                # --- SIDEBAR DOWNLOAD BUTTON INTEGRATION ---
                st.sidebar.markdown("---")
                st.sidebar.subheader("📥 Export Reports")

                st.markdown(
                """
                <style>
                /* Fix button text visibility across standard and primary buttons */
                div.stButton > button, div.stDownloadButton > button {
                    color: #1A2B4C !important;
                    background-color: #FAF8ED !important;
                    font-weight: bold !important;
                    border: 1px solid #1A2B4C !important;
                }
                div.stButton > button[kind="primary"], div.stDownloadButton > button[kind="primary"] {
                    color: #FAF8ED !important;
                    background-color: #1A2B4C !important;
                }
                /* Style Streamlit status container (st.status) to match Midnight Blue theme */
                div[data-testid="stStatusWidget"] {
                    background-color: #1A2B4C !important;
                    color: #FFFFFF !important;
                    border: 1px solid #FAF8ED !important;
                }
                div[data-testid="stStatusWidget"] * {
                    color: #FAF8ED !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
                )

                # Initialize session state for the PDF data if it doesn't exist
                if "pdf_data" not in st.session_state:
                    st.session_state.pdf_data = None
                
                if st.sidebar.button("Generate PDF Audit Report", type="primary"):
                # Capture Plotly figure as static image bytes for PDF inclusion
                    with st.sidebar.status("🔄 Generating PDF Report...", expanded=True) as status:
                        st.write("Capturing Plotly charts & metrics...")
                        fig_bytes = None
                        try:
                            fig_bytes = fig.to_image(
                                format="png", width=700, height=250, scale=2
                            )  # 'fig' is your active cumulative performance plot
                        except Exception:
                            pass
    
                        # Capture both distribution charts as static image bytes
                        fig_weights_bytes = None
                        fig_risk_bytes = None
                        try:
                            fig_weights_bytes = fig_weights.to_image(
                                format="png", width=400, height=220, scale=2
                            )
                            fig_risk_bytes = fig_risk.to_image(
                                format="png", width=400, height=220, scale=2
                            )
                        except Exception:
                            pass
                                    
                        # Bundle dictionaries from your active session states
                        st.write("Compiling Report...")
                        metrics_summary = {
                            "ann_return": f"{portfolio_daily_returns.mean() * 252 * 100:.2f}%",
                            "ann_vol": f"{portfolio_daily_returns.std() * np.sqrt(252) * 100:.2f}%",
                            "max_dd": f"{exp_dd * 100:.2f}%",
                            "var": f"{abs(var_pct) * 100:.2f}%",
                            "cvar": f"{abs(cvar_pct) * 100:.2f}%",
                            "sharpe": f"{sharpe:.2f}",
                            "risk_ent": f"{abs(risk_entropy):.2f}%",
                            "weight_ent": f"{abs(weight_entropy):.2f}%",
                            "sortino": f"{abs(sortino):.2f}",
                            "target_return": f"{target_return_pct}%",
                        }
                        mc_summary = {
                            "max_loss": f"INR {abs(var_amount):,.2f}",
                            "tail_val": f"INR {initial_capital - abs(var_amount):,.2f}",
                            "prob_dd": "0.3%",
                        }
                    
                        st.session_state.pdf_data = generate_pdf_report(
                            portfolio_score,
                            portfolio_beta,
                            None,
                            metrics_summary,
                            mc_summary,
                            fig_bytes,
                            fig_weights_bytes,
                            fig_risk_bytes,
                        )
                        status.update(
                        label="✅ PDF Report Generated Successfully!",
                        state="complete",
                        expanded=False,
                        )
                if st.session_state.pdf_data is not None:
                    st.sidebar.download_button(
                        label="Download PDF Report",
                        data=st.session_state.pdf_data,
                        file_name="QuantRisk_Portfolio_Audit.pdf",
                        mime="application/pdf",
                        type = "primary",
                    )

      # -------------------------------------------------------------------
          # 🛡️️ TAX Liabilities & Optimization Section
          # -------------------------------------------------------------------
                st.markdown("---")
                st.header("🛡️ TAX Liabilities & Optimization")

          # 1. Base Summary Table (Static / Current Upload View)
                if exchange in ["US", "UK", "EU", "JSE"]:
                  st.info("ℹ️ Tax audit optimization is currently available only in India.")
                else:
                    if flag==1:  #No CSV is uploaded, stocks added on the portal
                        st.info(f"👈 Please upload a {exchange} portfolio CSV file from the left sidebar to generate the tax audit.")
                    else:
                        df_base = raw_df.copy()
                        ticker_col = (
                        'Instrument'
                        if 'Instrument' in df_base.columns
                        else df_base.columns[0]
                          )
                        cost_col = (
                        'Avg. cost'
                        if 'Avg. cost' in df_base.columns
                        else (
                          'Avg_Cost'
                        if 'Avg_Cost' in df_base.columns
                        else 'Buy_Price'
                        )
                        )
                        qty_col = (
                        'Qty.'
                        if 'Qty.' in df_base.columns
                        else ('Qty' if 'Qty' in df_base.columns else 'Quantity')
                        )
                        # Strip existing suffixes if re-uploading
                        #cleaned_tickers = df_base[ticker_col].astype(str).str.replace(r'\.(NS|BO|JO)$', '', regex=True).str.strip()
        
                  # Calculate initial P&L if not explicitly present
                        if 'P&L' in df_base.columns:
                            df_base['PnL_Val'] = df_base['P&L']
                        elif 'Latest Price' in df_base.columns:
                            df_base['PnL_Val'] = (
                            df_base['Latest Price'] - df_base[cost_col]
                            ) * df_base[qty_col]
                        else:
                            df_base['PnL_Val'] = 0.0
        
                        if 'Holding_Days' not in df_base.columns:
                            df_base['Holding_Days'] = time_horizon_days
        
                        df_base['Tax_Category'] = np.where(
                        df_base['Holding_Days'] >= 365, 'LTCG', 'STCG'
                        )
                        df_base['Tax_Liability'] = np.where(
                        df_base['Tax_Category'] == 'LTCG',
                        np.maximum(0, df_base['PnL_Val']) * 0.125,
                        np.maximum(0, df_base['PnL_Val']) * 0.20,
                        )
        
                  # Display initial static view table
                        st.dataframe(
                        df_base[
                          [
                              ticker_col,
                              'PnL_Val',
                              'Holding_Days',
                              'Tax_Category',
                              'Tax_Liability',
                          ]
                        ],
                        column_config={
                        ticker_col: st.column_config.TextColumn("Instrument"),
                          "PnL_Val": st.column_config.NumberColumn(
                              "P&L", format="₹%.2f"
                        ),
                          "Holding_Days": st.column_config.NumberColumn(
                              "Holding_Days"
                        ),
                          "Tax_Category": st.column_config.TextColumn("Tax_Category"),
                          "Tax_Liability": st.column_config.NumberColumn(
                              "Tax_Liability", format="₹%.2f"
                        ),
                        },
                        hide_index=True,
                        use_container_width=True,
                        )
        
                        total_base_tax = df_base['Tax_Liability'].sum()
                        total_book_profit = df_base['P&L'].sum()
        
                        res4_col1, res4_col2, res4_col3 = st.columns(3)
        
                        res4_col1.metric(
                            label="Total Book Profit", 
                            #value=f"{abs(total_profit):.2f}", 
                            value=f"{currency} {total_book_profit:,.2f}", 
                            #delta_color="inverse"
                            )
                        
                        res4_col2.metric(
                            label="Total Book Tax", 
                            #value=f"{abs(modified_tax):.2f}", 
                            value=f"{currency} {total_base_tax:,.2f}", 
                            #delta_color="inverse"
                            )
                        
                        res4_col3.metric(
                            label="Tax Share", 
                            #value=f"{abs(modified_tax):.2f}", 
                            value=f"{total_base_tax/total_book_profit*100:,.2f} %", 
                            #delta_color="inverse"
                            )
                            
        
                  # -------------------------------------------------------------------
                  # ✏️ Interactive & Editable Table (Supports Adding Stocks)
                  # -------------------------------------------------------------------
                        st.caption(
                      "📌 *You can edit individual stock holding days, prices, or add new rows directly in the table below:*"
                        )
        
                  # Prepare initial structure for editable grid
                        editable_init = pd.DataFrame({
                        'Instrument': df_base[ticker_col],
                        'Avg_Cost': (
                          df_base[cost_col]
                          if cost_col in df_base.columns
                          else 100.0
                            ),
                        'Qty': (
                         df_base[qty_col] if qty_col in df_base.columns else 1
                          ),
                        'Holding_Days': df_base['Holding_Days'],
                        })
        
                  # Render st.data_editor with num_rows="dynamic" to enable row additions
                        edited_tax_df = st.data_editor(
                        editable_init,
                        num_rows="dynamic",  # Enables (+ Add row) button
                        column_config={
                          "Instrument": st.column_config.TextColumn(
                              "Instrument", required=True
                          ),
                          "Avg_Cost": st.column_config.NumberColumn(
                              "Buy Price (₹)", min_value=0.01, format="₹%.2f"
                          ),
                          "Qty": st.column_config.NumberColumn(
                              "Quantity", min_value=1, step=1
                          ),
                          "Holding_Days": st.column_config.NumberColumn(
                              "Holding Days", min_value=1, step=1
                          ),
                        },
                        hide_index=True,
                        use_container_width=True,
                        key="tax_editor_grid_v2",
                        )
        
                  # Recompute live prices and dynamic tax on edited grid
                        # Recompute live prices and dynamic tax on edited grid
                        if not edited_tax_df.empty:
                      # Fetch/map live price safely
                            #cleaned_tickers = edited_tax_df['Instrument'].astype(str).str.replace(r'\.(NS|BO|JO)$', '', regex=True).str.strip()
                    
                            #edited_tax_df['Instrument'] = edited_tax_df[cleaned_tickers].astype(str).str.strip() + current_cfg['suffix']
                            edited_tax_df['yf_ticker'] = edited_tax_df['Instrument'].astype(str).str.strip() + current_cfg['suffix']
                            live_tickers = edited_tax_df['yf_ticker'].tolist()
                            #live_tickers = (cleaned_tickers + current_cfg["suffix"]).tolist()
                            end_date = datetime.datetime.now()
                            start_date = end_date - datetime.timedelta(days=365)
                            live_price_data = yf.download(live_tickers, start=start_date, end=end_date)['Close']
        
                # Handle single series edge cases
                            if isinstance(live_price_data, pd.Series):
                                live_price_data = live_price_data.to_frame()
        
                # Clean invalid or empty columns
                                live_price_data = live_price_data.dropna(how='all', axis=1)
        
                # Safety Check to prevent out-of-bounds iloc crash
                            if live_price_data.empty:
                                st.error("Price data download returned no valid records for the given tickers.")
                            else:
                                live_price_data = live_price_data.ffill().bfill()
                                live_prices = live_price_data.iloc[-1]
        
                    # 2. Map latest prices safely to user holdings
                    # (Matches symbol or fallback if user CSV uses standard stock symbols)
                                edited_tax_df['Live Price'] = edited_tax_df['yf_ticker'].map(live_prices)
                                edited_tax_df['Live Price'] = edited_tax_df['Live Price'].fillna(0.0)
        
                      # Compute Live P&L correctly
                            edited_tax_df['P&L'] = (edited_tax_df['Live Price'] - edited_tax_df['Avg_Cost']) * edited_tax_df['Qty']
        
                      # Tax calculation logic
                            edited_tax_df['Tax_Category'] = np.where(
                            edited_tax_df['Holding_Days'] >= 365, 'LTCG', 'STCG'
                            )
                            edited_tax_df['Tax_Liability'] = np.where(
                            edited_tax_df['Tax_Category'] == 'LTCG',
                            np.maximum(0, edited_tax_df['P&L']) * 0.125,
                            np.maximum(0, edited_tax_df['P&L']) * 0.20,
                            )
        
                      # Display metric
                            modified_tax = edited_tax_df['Tax_Liability'].sum()
                            total_profit = edited_tax_df['P&L'].sum()
        
                            res5_col1, res5_col2, res5_col3 = st.columns(3)
        
                            res5_col1.metric(
                            label="Total Profit", 
                            #value=f"{abs(total_profit):.2f}", 
                            value=f"{currency} {total_profit:,.2f}", 
                            #delta_color="inverse"
                            )
                        
                            res5_col2.metric(
                            label="Total Modified Tax", 
                            #value=f"{abs(modified_tax):.2f}", 
                            value=f"{currency} {modified_tax:,.2f}", 
                            #delta_color="inverse"
                            )
                            
                            res5_col3.metric(
                            label="Tax Share", 
                            #value=f"{abs(modified_tax):.2f}", 
                            #value=f"{currency} {modified_tax:,.2f}", 
                            #delta_color="inverse"
                            value=f"{modified_tax/total_profit*100:,.2f} %",
                            )
        
                      # Render summary table below metric
                            st.dataframe(
                              edited_tax_df[
                              [
                                  'Instrument',
                                  'Avg_Cost',
                                  'Live Price',
                                  'P&L',
                                  'Holding_Days',
                                  'Tax_Category',
                                  'Tax_Liability',
                              ]
                            ],
                            column_config={
                              "Instrument": st.column_config.TextColumn("Instrument"),
                              "Avg_Cost": st.column_config.NumberColumn(
                                  "Buy Price (₹)", format="₹%.2f"
                              ),    
                              "Live Price": st.column_config.NumberColumn(
                                  "Live Price (₹)", format="₹%.2f"
                              ),  
                              "P&L": st.column_config.NumberColumn(
                                  "P&L (₹)", format="₹%.2f"
                              ),
                              "Holding_Days": st.column_config.NumberColumn(
                                  "Holding_Days"
                              ),
                              "Tax_Category": st.column_config.TextColumn(
                                  "Tax_Category"
                              ),
                              "Tax_Liability": st.column_config.NumberColumn(
                                  "Tax_Liability (₹)", format="₹%.2f"
                              ),
                            },
                            hide_index=True,
                            use_container_width=True,
                            )
            if exchange == "BSE":
                st.caption(
                "Note: Market data sourced via NSE primary liquid feeds for dual-listed equities to ensure consistent pricing data availability and zero downtime across exchange views.")
        except Exception as e:
            st.error(f"Error reading CSV portfolio file: {str(e)}")

        # --- FEEDBACK & FEATURE REQUEST FORM ---
        with st.sidebar:
          st.markdown("---")
          st.subheader("💬 Feedback & Feature Request")
        
          with st.form("feedback_form", clear_on_submit=True):
            user_email = st.text_input("Your Email (optional)")
            feedback_type = st.selectbox(
                "Feedback Type", ["Bug Report", "Feature Request", "General Feedback"]
            )
            user_message = st.text_area("Your Message / Suggestion")
        
            submitted = st.form_submit_button("Submit Feedback")

          if submitted:
            if user_message.strip():
                try:
                    # 1. Setup credentials from Streamlit Secrets
                    scope = [
                        "https://spreadsheets.google.com/feeds",
                        "https://www.googleapis.com/auth/drive"
                    ]
                    creds_dict = dict(st.secrets["gcp_service_account"])
                    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
                    client = gspread.authorize(creds)

                    # 2. Open Google Sheet by name
                    sheet = client.open("Wirator User Feedback").sheet1

                    # 3. Build exact row payload matching your entries
                    row_data = [
                        pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),    # Timestamp
                        user_email if user_email else "Anonymous",           # Email
                        feedback_type,                                       # Type
                        exchange if "exchange" in globals() else "NSE",       # Exchange
                        user_message                                         # Message
                    ]

                    # 4. Append row directly to Google Sheet
                    sheet.append_row(row_data)
                    st.success("Thank you! Your feedback has been recorded.")

                except Exception as e:
                    st.error(f"Failed to record feedback: {e}")
            else:
                st.warning("Please enter a message before submitting.")
        

    # else:
    #     st.info(f"👈 Please upload a {exchange} portfolio CSV file from the left sidebar to generate the risk audit.")

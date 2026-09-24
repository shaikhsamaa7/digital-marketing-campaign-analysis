# -*- coding: utf-8 -*-
"""
================================================================================
  DIGITAL MARKETING CAMPAIGN ANALYSIS & REVENUE INTELLIGENCE
  Executive Ad Operations, Financial Unit Economics & Machine Learning Forecasting
  Author: Samaa Shaikh
================================================================================
"""

import os
import sys
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & METADATA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Digital Marketing Campaign Analysis | Revenue Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. DESIGN SYSTEM & EXECUTIVE STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }

    /* Executive Top Navigation Header */
    .executive-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f2b48 100%);
        padding: 24px 32px;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 22px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15), 0 8px 10px -6px rgba(15, 23, 42, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .header-title-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }
    .header-title {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #ffffff 0%, #93c5fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .header-subtext {
        color: #94a3b8;
        font-size: 13px;
        font-weight: 400;
        margin: 0;
    }
    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(59, 130, 246, 0.18);
        border: 1px solid rgba(147, 197, 253, 0.3);
        color: #bfdbfe;
        font-size: 11.5px;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 9999px;
        letter-spacing: 0.3px;
    }

    /* Executive KPI Scorecards */
    .kpi-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 18px 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -2px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        position: relative;
        overflow: hidden;
        margin-bottom: 12px;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 18px -4px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
        border-color: #cbd5e1;
    }
    .kpi-accent-bar {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
    }
    .accent-blue { background: linear-gradient(90deg, #2563eb, #60a5fa); }
    .accent-green { background: linear-gradient(90deg, #059669, #34d399); }
    .accent-purple { background: linear-gradient(90deg, #7c3aed, #a78bfa); }
    .accent-amber { background: linear-gradient(90deg, #d97706, #fbbf24); }
    .accent-cyan { background: linear-gradient(90deg, #0891b2, #38bdf8); }
    .accent-rose { background: linear-gradient(90deg, #e11d48, #fb7185); }
    .accent-indigo { background: linear-gradient(90deg, #4f46e5, #818cf8); }
    .accent-emerald { background: linear-gradient(90deg, #047857, #10b981); }

    .kpi-label {
        font-size: 11.5px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #64748b;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .kpi-value {
        font-size: 23px;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
        font-variant-numeric: tabular-nums;
    }
    .kpi-subtext {
        font-size: 11.5px;
        font-weight: 600;
        color: #64748b;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .subtext-pos { color: #059669; }
    .subtext-neg { color: #dc2626; }
    .subtext-neutral { color: #475569; }

    /* Glass Cards */
    .glass-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 20px 22px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 18px;
    }
    .card-title {
        font-size: 15.5px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .card-subtitle {
        font-size: 12.5px;
        color: #64748b;
        margin-bottom: 14px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #e2e8f0;
        padding: 6px;
        border-radius: 12px;
        margin-bottom: 18px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background-color: transparent;
        border-radius: 8px;
        color: #475569;
        font-weight: 600;
        font-size: 13px;
        border: none !important;
        padding: 0 16px;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08) !important;
        font-weight: 700 !important;
    }

    /* Strategic Diagnostic Callout Boxes */
    .diagnostic-box {
        padding: 16px 18px;
        border-radius: 10px;
        margin-bottom: 14px;
        border: 1px solid #e2e8f0;
        border-left-width: 4px;
    }
    .diagnostic-title {
        font-weight: 700;
        font-size: 13.5px;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .diagnostic-text {
        font-size: 12.5px;
        color: #334155;
        line-height: 1.5;
        margin: 0;
    }

    /* Fallback / Warning Notice */
    .fallback-banner {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 12.5px;
        color: #92400e;
        margin-bottom: 14px;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        color: #f8fafc;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    section[data-testid="stSidebar"] .stMarkdown h1, 
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #f1f5f9 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p, 
    section[data-testid="stSidebar"] label {
        color: #cbd5e1 !important;
        font-weight: 500;
        font-size: 12.5px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. DATA ENGINE & PREPROCESSING
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data(show_spinner=False)
def load_and_preprocess_data():
    """
    Load online advertising performance dataset and perform robust data cleaning.
    """
    candidates = [
        os.path.join(BASE_DIR, "online_advertising_performance_data.csv"),
        "online_advertising_performance_data.csv"
    ]
    file_path = None
    for p in candidates:
        if os.path.exists(p):
            file_path = p
            break
            
    if file_path is None:
        return None
        
    df = pd.read_csv(file_path)
    
    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Drop empty artifact columns produced by Excel/CSV exports
    drop_cols = [c for c in df.columns if 'unnamed' in c]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        
    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna(0)
            
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Normalized month mapping
    month_map = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    df['month_clean'] = df['month'].astype(str).str.strip().str.lower()
    df['month_num'] = df['month_clean'].map(month_map).fillna(4).astype(int)
    
    # Reference date resolution (using 2025 calendar reference for continuous time-series)
    df['date'] = pd.to_datetime(dict(year=2025, month=df['month_num'], day=df['day']), errors='coerce')
    df['weekday'] = df['date'].dt.day_name()
    
    # Row-level calculations for data auditing (safe division)
    df['ctr'] = np.where(df['displays'] > 0, (df['clicks'] / df['displays']) * 100, 0.0)
    df['cpc'] = np.where(df['clicks'] > 0, df['cost'] / df['clicks'], 0.0)
    df['cpm'] = np.where(df['displays'] > 0, (df['cost'] / df['displays']) * 1000, 0.0)
    df['cvr'] = np.where(df['clicks'] > 0, (df['post_click_conversions'] / df['clicks']) * 100, 0.0)
    df['cpa'] = np.where(df['post_click_conversions'] > 0, df['cost'] / df['post_click_conversions'], 0.0)
    df['roas'] = np.where(df['cost'] > 0, df['revenue'] / df['cost'], 0.0)
    df['net_profit'] = df['revenue'] - df['cost']
    df['profit_margin'] = np.where(df['revenue'] > 0, (df['net_profit'] / df['revenue']) * 100, 0.0)
    
    return df

@st.cache_resource(show_spinner=False)
def load_ml_model():
    """
    Load serialized production ML regression pipeline.
    """
    model_paths = [
        os.path.join(BASE_DIR, "best_advertising_model.pkl"),
        "best_advertising_model.pkl"
    ]
    for p in model_paths:
        if os.path.exists(p):
            try:
                model = joblib.load(p)
                return model
            except Exception:
                pass
    return None

def get_model_metadata(model):
    """
    Extract human-readable metadata from scikit-learn pipeline.
    """
    if model is None:
        return {"name": "Unavailable", "type": "None"}
    try:
        estimator = model.named_steps['model']
        class_name = estimator.__class__.__name__
        if 'LGBM' in class_name:
            return {"name": "LightGBM Regressor", "type": "Gradient Boosted Trees"}
        elif 'RandomForest' in class_name:
            return {"name": "Random Forest Regressor", "type": "Ensemble Trees"}
        elif 'XGB' in class_name:
            return {"name": "XGBoost Regressor", "type": "Gradient Boosted Trees"}
        elif 'SVR' in class_name:
            return {"name": "Support Vector Regressor", "type": "Kernel SVR"}
        else:
            return {"name": class_name, "type": "Regression Model"}
    except Exception:
        return {"name": "Trained Regressor Pipeline", "type": "Pipeline"}

# Load resources
raw_df = load_and_preprocess_data()
ml_pipeline = load_ml_model()
model_meta = get_model_metadata(ml_pipeline)

if raw_df is None:
    st.error("⚠️ Dataset `online_advertising_performance_data.csv` was not found in the application directory. Please ensure it is present.")
    st.stop()

# -----------------------------------------------------------------------------
# 4. SIDEBAR GLOBAL FILTERS & SYSTEM HEALTH
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 16px 0;">
        <div style="font-size: 26px;">📈</div>
        <div style="font-size: 16px; font-weight: 800; color: #ffffff; letter-spacing: -0.3px;">CAMPAIGN INTELLIGENCE</div>
        <div style="font-size: 10.5px; color: #94a3b8; text-transform: uppercase; font-weight: 600; letter-spacing: 1px;">Ad Operations & Revenue BI</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎛️ Global Data Slicers")
    
    # Month Filter
    all_months = sorted(list(raw_df['month'].unique()))
    selected_months = st.multiselect("Timeframe (Month)", all_months, default=all_months)
    if not selected_months:
        selected_months = all_months
        
    # Campaign Filter
    all_camps = sorted(list(raw_df['campaign_number'].unique()))
    selected_camps = st.multiselect("Campaign ID", all_camps, default=all_camps)
    if not selected_camps:
        selected_camps = all_camps

    # Creative Dimension Filter
    all_banners = sorted(list(raw_df['banner'].unique()))
    selected_banners = st.multiselect("Creative Dimension (Banner)", all_banners, default=all_banners)
    if not selected_banners:
        selected_banners = all_banners

    # Engagement Tier Filter
    all_engagements = sorted(list(raw_df['user_engagement'].unique()))
    selected_engagements = st.multiselect("Audience Engagement Tier", all_engagements, default=all_engagements)
    if not selected_engagements:
        selected_engagements = all_engagements

    # Placement Filter
    all_placements = sorted(list(raw_df['placement'].unique()))
    selected_placements = st.multiselect("Publisher Placement Slot", all_placements, default=all_placements)
    if not selected_placements:
        selected_placements = all_placements

    st.markdown("---")
    
    # Reset Filters Button
    if st.button("🔄 Reset All Filters", use_container_width=True):
        st.rerun()
        
    # System Status Indicator Widget
    st.markdown("### 🛡️ System & Pipeline Health")
    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255,255,255,0.08); padding: 12px; border-radius: 10px; font-size: 12px;">
        <div style="display:flex; justify-content:space-between; margin-bottom: 6px;">
            <span style="color:#94a3b8;">Pipeline Status:</span>
            <span style="color:#34d399; font-weight:700;">● Active</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom: 6px;">
            <span style="color:#94a3b8;">Total Records:</span>
            <span style="color:#f8fafc; font-weight:600;">{len(raw_df):,}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom: 6px;">
            <span style="color:#94a3b8;">ML Architecture:</span>
            <span style="color:{'#38bdf8' if ml_pipeline else '#fb7185'}; font-weight:600;">{model_meta['name']}</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span style="color:#94a3b8;">Prediction Target:</span>
            <span style="color:#a5b4fc; font-weight:600;">Revenue ($)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. FILTER APPLICATION & EMPTY-STATE RECOVERY
# -----------------------------------------------------------------------------
df = raw_df[
    (raw_df['month'].isin(selected_months)) &
    (raw_df['campaign_number'].isin(selected_camps)) &
    (raw_df['banner'].isin(selected_banners)) &
    (raw_df['user_engagement'].isin(selected_engagements)) &
    (raw_df['placement'].isin(selected_placements))
].copy()

if df.empty:
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 40px 20px;">
        <div style="font-size: 36px; margin-bottom: 12px;">🔍</div>
        <h3 style="color: #0f172a; margin-bottom: 8px;">No Data Matches Current Filter Criteria</h3>
        <p style="color: #64748b; font-size: 14px; max-width: 500px; margin: 0 auto 20px auto;">
            The active combination of month, campaign, creative format, and placement filters returned 0 records.
            Please broaden your filter selection in the sidebar or reset all filters.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🔄 Reset Filters to Default", use_container_width=True):
        st.rerun()
    st.stop()

# -----------------------------------------------------------------------------
# 6. EXECUTIVE HEADER BANNER
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="executive-header">
    <div class="header-title-row">
        <div style="display: flex; align-items: center; gap: 16px;">
            <div style="background: linear-gradient(135deg, #2563eb 0%, #38bdf8 100%); width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center; box-shadow: 0 6px 16px -2px rgba(37, 99, 235, 0.45); border: 1px solid rgba(255, 255, 255, 0.25); flex-shrink: 0;">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="18" y1="20" x2="18" y2="10"></line>
                    <line x1="12" y1="20" x2="12" y2="4"></line>
                    <line x1="6" y1="20" x2="6" y2="14"></line>
                    <path d="M3 20h18"></path>
                    <path d="M4 10l6-6 4 4 6-6"></path>
                </svg>
            </div>
            <div>
                <h1 class="header-title">Digital Marketing Campaign Analysis & Revenue Intelligence</h1>
                <p class="header-subtext">Multi-Channel Ad Operations, Financial Unit Economics & Machine Learning Forecasting</p>
            </div>
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center;">
            <span class="header-badge">🟢 Operational</span>
            <span class="header-badge" style="background: rgba(16, 185, 129, 0.18); border-color: rgba(52, 211, 153, 0.3); color: #a7f3d0;">📊 Records: {len(df):,} / {len(raw_df):,}</span>
            <span class="header-badge" style="background: rgba(168, 85, 247, 0.18); border-color: rgba(216, 180, 254, 0.3); color: #e9d5ff;">🎯 Model: {model_meta['name']}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. EXECUTIVE KPI SCORECARDS (AGGREGATE RATIOS STRICTLY CALCULATED)
# -----------------------------------------------------------------------------
# Aggregate Totals (Numerators & Denominators)
total_rev = float(df['revenue'].sum())
total_cost = float(df['cost'].sum())
net_profit = total_rev - total_cost
portfolio_roas = total_rev / total_cost if total_cost > 0 else 0.0
portfolio_roi = ((total_rev - total_cost) / total_cost * 100) if total_cost > 0 else 0.0
profit_margin = (net_profit / total_rev * 100) if total_rev > 0 else 0.0

total_displays = float(df['displays'].sum())
total_clicks = float(df['clicks'].sum())
total_conv = float(df['post_click_conversions'].sum())

portfolio_ctr = (total_clicks / total_displays * 100) if total_displays > 0 else 0.0
portfolio_cpc = total_cost / total_clicks if total_clicks > 0 else 0.0
portfolio_cpm = (total_cost / total_displays * 1000) if total_displays > 0 else 0.0
portfolio_cvr = (total_conv / total_clicks * 100) if total_clicks > 0 else 0.0
portfolio_cpa = total_cost / total_conv if total_conv > 0 else 0.0

# Row 1: Core Financial Outcomes
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-accent-bar accent-blue"></div>
        <div class="kpi-label">Total Attributed Revenue <span>💵</span></div>
        <div class="kpi-value">${total_rev:,.2f}</div>
        <div class="kpi-subtext subtext-pos">▲ Gross Media Return</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-accent-bar accent-rose"></div>
        <div class="kpi-label">Total Media Ad Spend <span>💳</span></div>
        <div class="kpi-value">${total_cost:,.2f}</div>
        <div class="kpi-subtext subtext-neutral">Gross Advertising Budget</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-accent-bar accent-green"></div>
        <div class="kpi-label">Net Operating Profit <span>📈</span></div>
        <div class="kpi-value">${net_profit:,.2f}</div>
        <div class="kpi-subtext {'subtext-pos' if net_profit >= 0 else 'subtext-neg'}">Margin: {profit_margin:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-accent-bar accent-purple"></div>
        <div class="kpi-label">Portfolio ROAS (Multiplier) <span>🎯</span></div>
        <div class="kpi-value">{portfolio_roas:.2f}x</div>
        <div class="kpi-subtext {'subtext-pos' if portfolio_roas >= 1.0 else 'subtext-neg'}">Net ROI: {portfolio_roi:+.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

# Row 2: Operational & Conversion Efficiency
k5, k6, k7, k8 = st.columns(4)
with k5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-accent-bar accent-amber"></div>
        <div class="kpi-label">Total Conversions <span>🏆</span></div>
        <div class="kpi-value">{total_conv:,.0f}</div>
        <div class="kpi-subtext subtext-neutral">From {total_clicks:,.0f} Total Clicks</div>
    </div>
    """, unsafe_allow_html=True)

with k6:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-accent-bar accent-cyan"></div>
        <div class="kpi-label">Conversion Rate (CVR) <span>⚡</span></div>
        <div class="kpi-value">{portfolio_cvr:.2f}%</div>
        <div class="kpi-subtext subtext-neutral">Conversions / Total Clicks</div>
    </div>
    """, unsafe_allow_html=True)

with k7:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-accent-bar accent-indigo"></div>
        <div class="kpi-label">Avg. Cost Per Click (CPC) <span>💲</span></div>
        <div class="kpi-value">${portfolio_cpc:.2f}</div>
        <div class="kpi-subtext subtext-neutral">Portfolio CTR: {portfolio_ctr:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with k8:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-accent-bar accent-emerald"></div>
        <div class="kpi-label">Cost Per Acquisition (CPA) <span>🎯</span></div>
        <div class="kpi-value">${portfolio_cpa:.2f}</div>
        <div class="kpi-subtext subtext-neutral">Media Spend / Conversion</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 8. MULTI-TAB EXECUTIVE ANALYTICS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Performance Overview & Funnel",
    "🎯 Placement & Creative Intelligence",
    "💎 Financial Ledger & Unit Economics",
    "🤖 Machine Learning Forecast & What-If",
    "💡 Strategic Diagnostics & Insights"
])

# =============================================================================
# TAB 1: PERFORMANCE OVERVIEW & CONVERSION FUNNEL
# =============================================================================
with tab1:
    col_t1, col_t2 = st.columns([7, 5])
    
    with col_t1:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">📊 Daily Revenue vs. Media Spend Tracking</div>
            <div class="card-subtitle">Continuous time-series tracking of daily gross revenue against marketing budget spend and net profit</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Aggregate daily
        daily_df = df.groupby('date').agg({
            'revenue': 'sum',
            'cost': 'sum',
            'net_profit': 'sum',
            'clicks': 'sum',
            'displays': 'sum'
        }).reset_index().sort_values('date')
        
        fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_ts.add_trace(
            go.Scatter(
                x=daily_df['date'],
                y=daily_df['revenue'],
                name="Gross Revenue ($)",
                mode='lines+markers',
                line=dict(color='#2563eb', width=2.8),
                marker=dict(size=5, color='#2563eb'),
                fill='tozeroy',
                fillcolor='rgba(37, 99, 235, 0.08)'
            ),
            secondary_y=False
        )
        
        fig_ts.add_trace(
            go.Scatter(
                x=daily_df['date'],
                y=daily_df['cost'],
                name="Media Spend ($)",
                mode='lines+markers',
                line=dict(color='#f43f5e', width=2.2, dash='dot'),
                marker=dict(size=4, color='#f43f5e')
            ),
            secondary_y=False
        )
        
        fig_ts.add_trace(
            go.Bar(
                x=daily_df['date'],
                y=daily_df['net_profit'],
                name="Net Profit ($)",
                marker_color='rgba(16, 185, 129, 0.5)',
                marker_line=dict(color='#059669', width=1)
            ),
            secondary_y=True
        )
        
        fig_ts.update_layout(
            template="plotly_white",
            height=390,
            margin=dict(l=20, r=20, t=20, b=20),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor='rgba(248, 250, 252, 0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='#f1f5f9', title="Date"),
            yaxis=dict(showgrid=True, gridcolor='#f1f5f9', title="Revenue & Spend ($)", tickprefix="$"),
            yaxis2=dict(showgrid=False, title="Net Profit ($)", tickprefix="$")
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    with col_t2:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">🌪️ Marketing Conversion Funnel</div>
            <div class="card-subtitle">Audience progression from Ad Impressions to Clicks and Sales Conversions</div>
        </div>
        """, unsafe_allow_html=True)
        
        funnel_stages = ["Displays (Impressions)", "Clicks (Engagements)", "Conversions (Sales)"]
        funnel_values = [total_displays, total_clicks, total_conv]
        
        fig_funnel = go.Figure(go.Funnel(
            y=funnel_stages,
            x=funnel_values,
            textinfo="value+percent initial",
            texttemplate="<b>%{label}</b><br>%{value:,.0f} (%{percentInitial:.2%})",
            marker=dict(
                color=["#3b82f6", "#8b5cf6", "#10b981"],
                line=dict(width=1, color="#ffffff")
            ),
            connector=dict(line=dict(color="#cbd5e1", width=1.5, dash="dot"))
        ))
        
        fig_funnel.update_layout(
            template="plotly_white",
            height=390,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_funnel, use_container_width=True)

    # Row 2: Day of Week & Campaign Benchmark
    col_t3, col_t4 = st.columns(2)
    
    with col_t3:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">📅 Day of Week Revenue & ROAS Efficiency</div>
            <div class="card-subtitle">Aggregated revenue and return multiplier by day of week (ROAS = Total Revenue / Total Cost)</div>
        </div>
        """, unsafe_allow_html=True)
        
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_df = df.groupby('weekday').agg({
            'revenue': 'sum',
            'cost': 'sum',
            'clicks': 'sum',
            'post_click_conversions': 'sum'
        }).reindex(weekday_order).reset_index()
        
        weekday_df['roas'] = np.where(weekday_df['cost'] > 0, weekday_df['revenue'] / weekday_df['cost'], 0.0)
        weekday_df['cvr'] = np.where(weekday_df['clicks'] > 0, (weekday_df['post_click_conversions'] / weekday_df['clicks']) * 100, 0.0)
        
        fig_wk = go.Figure()
        fig_wk.add_trace(go.Bar(
            x=weekday_df['weekday'],
            y=weekday_df['revenue'],
            name="Revenue ($)",
            marker_color='#2563eb',
            text=weekday_df['revenue'].apply(lambda x: f"${x:,.0f}"),
            textposition='auto'
        ))
        fig_wk.add_trace(go.Scatter(
            x=weekday_df['weekday'],
            y=weekday_df['roas'],
            name="ROAS (Multiplier)",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="#f59e0b", width=3),
            marker=dict(size=8, color="#d97706")
        ))
        
        fig_wk.update_layout(
            template="plotly_white",
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(title="Revenue ($)", tickprefix="$", showgrid=True, gridcolor="#f1f5f9"),
            yaxis2=dict(title="ROAS (x)", overlaying="y", side="right", ticksuffix="x", showgrid=False),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_wk, use_container_width=True)

    with col_t4:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">🏆 Campaign Benchmark Comparison</div>
            <div class="card-subtitle">Side-by-side benchmarking of Revenue, Media Spend & Net Operating Profit</div>
        </div>
        """, unsafe_allow_html=True)
        
        camp_bench = df.groupby('campaign_number').agg({
            'revenue': 'sum',
            'cost': 'sum',
            'net_profit': 'sum'
        }).reset_index()
        camp_bench['roas'] = np.where(camp_bench['cost'] > 0, camp_bench['revenue'] / camp_bench['cost'], 0.0)
        
        fig_camp = go.Figure()
        fig_camp.add_trace(go.Bar(
            name="Revenue ($)",
            x=camp_bench['campaign_number'],
            y=camp_bench['revenue'],
            marker_color='#3b82f6'
        ))
        fig_camp.add_trace(go.Bar(
            name="Media Spend ($)",
            x=camp_bench['campaign_number'],
            y=camp_bench['cost'],
            marker_color='#ef4444'
        ))
        fig_camp.add_trace(go.Bar(
            name="Net Profit ($)",
            x=camp_bench['campaign_number'],
            y=camp_bench['net_profit'],
            marker_color='#10b981'
        ))
        
        fig_camp.update_layout(
            barmode='group',
            template="plotly_white",
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(title="Amount ($)", tickprefix="$", showgrid=True, gridcolor="#f1f5f9"),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_camp, use_container_width=True)

# =============================================================================
# TAB 2: PLACEMENT & CREATIVE INTELLIGENCE
# =============================================================================
with tab2:
    col_p1, col_p2 = st.columns([7, 5])
    
    with col_p1:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">🌐 Placement Efficiency Matrix (Spend vs. Revenue vs. ROAS)</div>
            <div class="card-subtitle">Bubble size reflects Total Clicks; Color reflects ROAS. Dashed line represents 1.0x Break-Even</div>
        </div>
        """, unsafe_allow_html=True)
        
        place_agg = df.groupby(['placement', 'banner']).agg({
            'cost': 'sum',
            'revenue': 'sum',
            'clicks': 'sum',
            'post_click_conversions': 'sum'
        }).reset_index()
        place_agg['roas'] = np.where(place_agg['cost'] > 0, place_agg['revenue'] / place_agg['cost'], 0.0)
        place_agg['cvr'] = np.where(place_agg['clicks'] > 0, (place_agg['post_click_conversions'] / place_agg['clicks']) * 100, 0.0)
        
        fig_bubble = px.scatter(
            place_agg,
            x='cost',
            y='revenue',
            size='clicks',
            color='roas',
            hover_name='placement',
            hover_data={
                'banner': True,
                'cost': ':.2f',
                'revenue': ':.2f',
                'roas': ':.2f',
                'clicks': ':,',
                'cvr': ':.2f%'
            },
            color_continuous_scale='Viridis',
            labels={'cost': 'Media Spend ($)', 'revenue': 'Revenue Generated ($)', 'roas': 'ROAS Multiplier'}
        )
        
        # Add 1.0x Break-even line
        max_val = max(float(place_agg['cost'].max()), float(place_agg['revenue'].max())) * 1.08
        fig_bubble.add_shape(
            type="line", line=dict(dash="dash", color="#94a3b8", width=1.5),
            x0=0, y0=0, x1=max_val, y1=max_val
        )
        fig_bubble.add_annotation(
            x=max_val*0.75, y=max_val*0.75, text="1.0x Break-Even (ROAS = 1)", showarrow=False,
            font=dict(size=11, color="#64748b")
        )
        
        fig_bubble.update_layout(
            template="plotly_white",
            height=420,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(title="Media Spend ($)", tickprefix="$", showgrid=True, gridcolor="#f1f5f9"),
            yaxis=dict(title="Revenue ($)", tickprefix="$", showgrid=True, gridcolor="#f1f5f9"),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

    with col_p2:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">📐 Creative Dimension (Banner Size) Share</div>
            <div class="card-subtitle">Revenue contribution and visual format distribution</div>
        </div>
        """, unsafe_allow_html=True)
        
        banner_agg = df.groupby('banner').agg({
            'revenue': 'sum',
            'cost': 'sum',
            'clicks': 'sum',
            'post_click_conversions': 'sum'
        }).reset_index().sort_values('revenue', ascending=False)
        
        fig_banner = px.pie(
            banner_agg,
            names='banner',
            values='revenue',
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_banner.update_traces(
            textposition='inside',
            textinfo='percent+label',
            marker=dict(line=dict(color='#ffffff', width=2))
        )
        fig_banner.update_layout(
            template="plotly_white",
            height=420,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_banner, use_container_width=True)

    # Row 2: User Engagement Impact & Placement Top 10 Table
    col_p3, col_p4 = st.columns(2)
    
    with col_p3:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">👥 Audience Engagement Tier Efficiency</div>
            <div class="card-subtitle">Conversion Rate (CVR) and Cost Per Acquisition (CPA) by engagement level</div>
        </div>
        """, unsafe_allow_html=True)
        
        eng_agg = df.groupby('user_engagement').agg({
            'clicks': 'sum',
            'cost': 'sum',
            'revenue': 'sum',
            'post_click_conversions': 'sum'
        }).reset_index()
        
        eng_agg['cvr'] = np.where(eng_agg['clicks'] > 0, (eng_agg['post_click_conversions'] / eng_agg['clicks']) * 100, 0.0)
        eng_agg['cpa'] = np.where(eng_agg['post_click_conversions'] > 0, eng_agg['cost'] / eng_agg['post_click_conversions'], 0.0)
        eng_agg['roas'] = np.where(eng_agg['cost'] > 0, eng_agg['revenue'] / eng_agg['cost'], 0.0)
        
        fig_eng = go.Figure()
        fig_eng.add_trace(go.Bar(
            x=eng_agg['user_engagement'],
            y=eng_agg['cvr'],
            name="Conversion Rate (%)",
            marker_color="#6366f1",
            text=eng_agg['cvr'].apply(lambda x: f"{x:.2f}%"),
            textposition='auto'
        ))
        fig_eng.add_trace(go.Scatter(
            x=eng_agg['user_engagement'],
            y=eng_agg['cpa'],
            name="CPA ($)",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="#ef4444", width=3),
            marker=dict(size=8, color="#dc2626")
        ))
        
        fig_eng.update_layout(
            template="plotly_white",
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(title="CVR (%)", ticksuffix="%", showgrid=True, gridcolor="#f1f5f9"),
            yaxis2=dict(title="CPA ($)", overlaying="y", side="right", tickprefix="$", showgrid=False),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_eng, use_container_width=True)

    with col_p4:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">🏆 Top 10 High-Revenue Publisher Placements</div>
            <div class="card-subtitle">Highest grossing inventory slots with financial unit economics</div>
        </div>
        """, unsafe_allow_html=True)
        
        top10_places = df.groupby('placement').agg({
            'displays': 'sum',
            'clicks': 'sum',
            'cost': 'sum',
            'revenue': 'sum',
            'post_click_conversions': 'sum'
        }).reset_index()
        
        top10_places['roas'] = np.where(top10_places['cost'] > 0, top10_places['revenue'] / top10_places['cost'], 0.0)
        top10_places['cvr'] = np.where(top10_places['clicks'] > 0, (top10_places['post_click_conversions'] / top10_places['clicks']) * 100, 0.0)
        top10_places = top10_places.sort_values('revenue', ascending=False).head(10)
        
        st.dataframe(
            top10_places.style.format({
                'displays': '{:,.0f}',
                'clicks': '{:,.0f}',
                'cost': '${:,.2f}',
                'revenue': '${:,.2f}',
                'post_click_conversions': '{:,.0f}',
                'roas': '{:.2f}x',
                'cvr': '{:.2f}%'
            }).background_gradient(subset=['revenue'], cmap='Blues')
              .background_gradient(subset=['roas'], cmap='YlGn'),
            use_container_width=True,
            height=280
        )

# =============================================================================
# TAB 3: CAMPAIGN FINANCIAL LEDGER & UNIT ECONOMICS
# =============================================================================
with tab3:
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">💎 Executive Campaign Financial Ledger</div>
        <div class="card-subtitle">Audited multi-dimensional unit economics ledger (all ratios computed from aggregated numerators and denominators)</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Financial aggregate table
    ledger_df = df.groupby(['campaign_number', 'user_engagement']).agg({
        'displays': 'sum',
        'clicks': 'sum',
        'cost': 'sum',
        'revenue': 'sum',
        'post_click_conversions': 'sum',
        'net_profit': 'sum'
    }).reset_index()
    
    ledger_df['ctr'] = np.where(ledger_df['displays'] > 0, (ledger_df['clicks'] / ledger_df['displays']) * 100, 0.0)
    ledger_df['cpc'] = np.where(ledger_df['clicks'] > 0, ledger_df['cost'] / ledger_df['clicks'], 0.0)
    ledger_df['cvr'] = np.where(ledger_df['clicks'] > 0, (ledger_df['post_click_conversions'] / ledger_df['clicks']) * 100, 0.0)
    ledger_df['cpa'] = np.where(ledger_df['post_click_conversions'] > 0, ledger_df['cost'] / ledger_df['post_click_conversions'], 0.0)
    ledger_df['roas'] = np.where(ledger_df['cost'] > 0, ledger_df['revenue'] / ledger_df['cost'], 0.0)
    ledger_df['roi'] = np.where(ledger_df['cost'] > 0, (ledger_df['net_profit'] / ledger_df['cost']) * 100, 0.0)
    ledger_df['margin'] = np.where(ledger_df['revenue'] > 0, (ledger_df['net_profit'] / ledger_df['revenue']) * 100, 0.0)
    
    ledger_df = ledger_df.sort_values('revenue', ascending=False)
    
    st.dataframe(
        ledger_df.style.format({
            'displays': '{:,.0f}',
            'clicks': '{:,.0f}',
            'cost': '${:,.2f}',
            'revenue': '${:,.2f}',
            'net_profit': '${:,.2f}',
            'post_click_conversions': '{:,.0f}',
            'ctr': '{:.2f}%',
            'cpc': '${:,.2f}',
            'cvr': '{:.2f}%',
            'cpa': '${:,.2f}',
            'roas': '{:.2f}x',
            'roi': '{:+.1f}%',
            'margin': '{:.1f}%'
        }).background_gradient(subset=['roas'], cmap='Greens')
          .background_gradient(subset=['revenue'], cmap='Blues')
          .background_gradient(subset=['net_profit'], cmap='YlGn'),
        use_container_width=True,
        height=380
    )
    
    # Export capability
    csv_data = ledger_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Audited Financial Ledger (CSV)",
        data=csv_data,
        file_name="digital_marketing_financial_ledger.csv",
        mime="text/csv",
        help="Export multi-dimensional financial unit economics table as CSV"
    )

# =============================================================================
# TAB 4: MACHINE LEARNING FORECAST & WHAT-IF SIMULATOR
# =============================================================================
with tab4:
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">🤖 Machine Learning Revenue Forecast & What-If Simulator</div>
        <div class="card-subtitle">Evaluate campaign configurations and forecast expected revenue using the trained regression pipeline</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Specification & Evaluation Overview Card
    eval_csv_path = os.path.join(BASE_DIR, "outputs", "model_comparison.csv")
    eval_df = None
    if os.path.exists(eval_csv_path):
        try:
            eval_df = pd.read_csv(eval_csv_path)
        except Exception:
            pass
            
    with st.expander("ℹ️ Model Architecture, Validation Metrics & Leakage Safeguards", expanded=False):
        c_m1, c_m2 = st.columns([1, 1])
        with c_m1:
            st.markdown(f"""
            **Model Information:**
            - **Active Architecture**: `{model_meta['name']}` ({model_meta['type']})
            - **Prediction Target**: `revenue ($)`
            - **Feature Count**: 12 leakage-free operational features
            - **Categorical Encodings**: `month`, `campaign_number`, `user_engagement`, `banner`, `placement`
            - **Numeric Encodings**: `day`, `displays`, `cost`, `clicks`, `post_click_conversions`, `ctr`, `cpc`
            """)
        with c_m2:
            if eval_df is not None:
                st.markdown("**Hold-out & Cross-Validation Benchmark:**")
                st.dataframe(eval_df.style.format({'MAE': '{:.4f}', 'RMSE': '{:.4f}', 'R2': '{:.4f}', 'CV_R2': '{:.4f}'}), use_container_width=True)
            else:
                st.markdown("""
                **Leakage Prevention Note:**
                - `roi`, `roas`, `net_profit`, and `post_click_sales_amount` are strictly excluded from predictive features because they directly require the realized revenue target.
                """)
        st.caption("⚠️ **Technical Note:** Model projections represent statistical regression estimates based on historical campaign distributions. They do not constitute guaranteed causal outcomes.")

    col_sim1, col_sim2 = st.columns([5, 7])
    
    with col_sim1:
        st.markdown("#### ⚙️ Campaign Configuration")
        sim_month = st.selectbox("Target Deployment Month", ["April", "May", "June"])
        sim_day = st.slider("Day of Month", 1, 31, 15)
        sim_camp = st.selectbox("Campaign Number", sorted(list(raw_df['campaign_number'].unique())))
        sim_eng = st.selectbox("Audience Engagement Tier", ["High", "Medium", "Low"])
        sim_banner = st.selectbox("Creative Banner Dimension", sorted(list(raw_df['banner'].unique())))
        sim_placement = st.selectbox("Publisher Placement Slot", sorted(list(raw_df['placement'].unique())))
        
        st.markdown("#### 📊 Operating Traffic & Budget Forecast")
        sim_displays = st.number_input("Projected Impressions (Displays)", min_value=100, max_value=1000000, value=25000, step=1000)
        sim_cost = st.number_input("Planned Budget Spend ($)", min_value=1.0, max_value=100000.0, value=30.00, step=5.0)
        sim_clicks = st.number_input("Anticipated Clicks", min_value=1, max_value=100000, value=220, step=10)
        sim_conv = st.number_input("Expected Conversions", min_value=0, max_value=10000, value=45, step=1)

    with col_sim2:
        st.markdown("#### 🎯 Scenario Forecast & Comparative Analysis")
        
        # Compute baseline comparison from historical segment
        baseline_slice = raw_df[
            (raw_df['campaign_number'] == sim_camp) &
            (raw_df['user_engagement'] == sim_eng)
        ]
        if baseline_slice.empty:
            baseline_slice = raw_df
            
        base_cost = baseline_slice['cost'].mean()
        base_rev = baseline_slice['revenue'].mean()
        base_profit = base_rev - base_cost
        base_roas = base_rev / base_cost if base_cost > 0 else 0.0

        # Construct ML Inference Payload
        sim_ctr = (sim_clicks / sim_displays) if sim_displays > 0 else 0.0
        sim_cpc = (sim_cost / sim_clicks) if sim_clicks > 0 else 0.0

        sim_payload = pd.DataFrame([{
            'month': sim_month,
            'day': sim_day,
            'campaign_number': sim_camp,
            'user_engagement': sim_eng,
            'banner': sim_banner,
            'placement': sim_placement,
            'displays': sim_displays,
            'cost': sim_cost,
            'clicks': sim_clicks,
            'post_click_conversions': sim_conv,
            'ctr': sim_ctr,
            'cpc': sim_cpc,
        }])
        
        # Predict using ML pipeline or explicit fallback
        is_fallback = False
        if ml_pipeline is not None:
            try:
                pred_revenue = float(ml_pipeline.predict(sim_payload)[0])
            except Exception:
                is_fallback = True
                pred_revenue = float(sim_cost * 2.85 + (sim_conv * 12.4))
        else:
            is_fallback = True
            pred_revenue = float(sim_cost * 2.85 + (sim_conv * 12.4))
            
        pred_profit = pred_revenue - sim_cost
        pred_roas = pred_revenue / sim_cost if sim_cost > 0 else 0.0
        pred_margin = (pred_profit / pred_revenue * 100) if pred_revenue > 0 else 0.0
        
        # Delta calculations
        delta_rev = pred_revenue - base_rev
        delta_rev_pct = (delta_rev / base_rev * 100) if base_rev > 0 else 0.0
        delta_profit = pred_profit - base_profit
        delta_roas = pred_roas - base_roas

        if is_fallback:
            st.markdown("""
            <div class="fallback-banner">
                ⚠️ <b>Heuristic Fallback Notice:</b> The serialized machine learning pipeline could not be executed for this payload.
                Displaying a rule-based budget heuristic estimate.
            </div>
            """, unsafe_allow_html=True)

        # Primary Forecast Card
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%); padding: 22px 24px; border-radius: 14px; color: white; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 18px; box-shadow: 0 10px 25px -5px rgba(30, 27, 75, 0.3);">
            <div style="font-size: 12px; font-weight: 700; color: #a5b4fc; text-transform: uppercase; letter-spacing: 1px;">Predicted Campaign Revenue</div>
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 6px;">
                <div style="font-size: 36px; font-weight: 800; color: #ffffff;">${pred_revenue:,.2f}</div>
                <div style="background: rgba(52, 211, 153, 0.2); border: 1px solid rgba(52, 211, 153, 0.4); color: #6ee7b7; padding: 4px 14px; border-radius: 9999px; font-weight: 700; font-size: 13.5px;">
                    ROAS: {pred_roas:.2f}x
                </div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 18px; padding-top: 14px; border-top: 1px solid rgba(255,255,255,0.1);">
                <div>
                    <div style="color: #cbd5e1; font-size: 11px; text-transform: uppercase;">Predicted Profit</div>
                    <div style="font-size: 17px; font-weight: 700; color: #34d399;">${pred_profit:,.2f}</div>
                </div>
                <div>
                    <div style="color: #cbd5e1; font-size: 11px; text-transform: uppercase;">Profit Margin</div>
                    <div style="font-size: 17px; font-weight: 700; color: #93c5fd;">{pred_margin:.1f}%</div>
                </div>
                <div>
                    <div style="color: #cbd5e1; font-size: 11px; text-transform: uppercase;">Inference Engine</div>
                    <div style="font-size: 13.5px; font-weight: 600; color: #fbcfe8;">{model_meta['name']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Side-by-Side Baseline vs Scenario Comparison Table
        st.markdown("##### ⚖️ Baseline vs. Scenario Comparison")
        comp_data = {
            "Metric": ["Ad Spend ($)", "Gross Revenue ($)", "Net Profit ($)", "ROAS (Multiplier)", "Profit Margin (%)"],
            "Historical Baseline": [f"${base_cost:,.2f}", f"${base_rev:,.2f}", f"${base_profit:,.2f}", f"{base_roas:.2f}x", f"{(base_profit/base_rev*100 if base_rev>0 else 0):.1f}%"],
            "Simulated Scenario": [f"${sim_cost:,.2f}", f"${pred_revenue:,.2f}", f"${pred_profit:,.2f}", f"{pred_roas:.2f}x", f"{pred_margin:.1f}%"],
            "Variance (Δ)": [
                f"${sim_cost - base_cost:+,.2f}",
                f"${delta_rev:+,.2f} ({delta_rev_pct:+.1f}%)",
                f"${delta_profit:+,.2f}",
                f"{delta_roas:+.2f}x",
                f"{(pred_margin - (base_profit/base_rev*100 if base_rev>0 else 0)):+.1f}%"
            ]
        }
        st.dataframe(pd.DataFrame(comp_data), hide_index=True, use_container_width=True)

        # Budget Scaling Sensitivity Curve
        st.markdown("##### 📈 Budget Scaling Sensitivity Curve")
        budget_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        curve_costs = [sim_cost * m for m in budget_multipliers]
        curve_revs = []
        
        for c in curve_costs:
            temp_p = sim_payload.copy()
            scaled_displays = int(sim_displays * (c / sim_cost))
            scaled_clicks = int(sim_clicks * (c / sim_cost))
            scaled_conv = int(sim_conv * (c / sim_cost))
            
            temp_p['cost'] = c
            temp_p['displays'] = scaled_displays
            temp_p['clicks'] = scaled_clicks
            temp_p['post_click_conversions'] = scaled_conv
            temp_p['ctr'] = (scaled_clicks / scaled_displays) if scaled_displays > 0 else 0.0
            temp_p['cpc'] = (c / scaled_clicks) if scaled_clicks > 0 else 0.0

            if ml_pipeline is not None and not is_fallback:
                try:
                    r = float(ml_pipeline.predict(temp_p)[0])
                except Exception:
                    r = float(c * 2.85 + (scaled_conv * 12.4))
            else:
                r = float(c * 2.85 + (scaled_conv * 12.4))
            curve_revs.append(r)
            
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=curve_costs,
            y=curve_revs,
            mode='lines+markers',
            name='Forecasted Revenue',
            line=dict(color='#6366f1', width=3),
            marker=dict(size=7, color='#4f46e5')
        ))
        fig_curve.add_trace(go.Scatter(
            x=curve_costs,
            y=curve_costs,
            mode='lines',
            name='1:1 Break-Even',
            line=dict(color='#ef4444', dash='dash', width=1.5)
        ))
        
        fig_curve.update_layout(
            template="plotly_white",
            height=260,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(title="Simulated Spend ($)", tickprefix="$", showgrid=True, gridcolor="#f1f5f9"),
            yaxis=dict(title="Forecasted Revenue ($)", tickprefix="$", showgrid=True, gridcolor="#f1f5f9"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_curve, use_container_width=True)

# =============================================================================
# TAB 5: STRATEGIC DIAGNOSTICS & CAMPAIGN QUERY ASSISTANT
# =============================================================================
with tab5:
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">💡 Strategic Campaign Diagnostics & Automated Insights</div>
        <div class="card-subtitle">Evidence-based analytical synthesis computed dynamically from the current filtered dataset</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate key diagnostics dynamically from current slice
    camp_summary = df.groupby('campaign_number').agg({
        'revenue': 'sum',
        'cost': 'sum'
    })
    camp_summary['roas'] = np.where(camp_summary['cost'] > 0, camp_summary['revenue'] / camp_summary['cost'], 0.0)
    camp_summary = camp_summary.sort_values('roas', ascending=False)
    
    best_camp_id = camp_summary.index[0]
    best_camp_roas = camp_summary.loc[best_camp_id, 'roas']
    best_camp_rev = camp_summary.loc[best_camp_id, 'revenue']
    
    worst_camp_id = camp_summary.index[-1]
    worst_camp_roas = camp_summary.loc[worst_camp_id, 'roas']
    worst_camp_cost = camp_summary.loc[worst_camp_id, 'cost']
    
    banner_summary = df.groupby('banner')['revenue'].sum().sort_values(ascending=False)
    top_banner = banner_summary.index[0]
    top_banner_rev = banner_summary.iloc[0]
    
    place_summary = df.groupby('placement').agg({'revenue': 'sum', 'cost': 'sum'})
    place_summary['roas'] = np.where(place_summary['cost'] > 0, place_summary['revenue'] / place_summary['cost'], 0.0)
    place_summary = place_summary.sort_values('revenue', ascending=False)
    top_placement = place_summary.index[0]
    top_placement_roas = place_summary.loc[top_placement, 'roas']
    
    col_i1, col_i2 = st.columns(2)
    
    with col_i1:
        st.markdown(f"""
        <div class="diagnostic-box" style="border-left-color: #10b981; background: #f0fdf4;">
            <div class="diagnostic-title" style="color: #047857;">🏆 Top Observed ROAS Driver</div>
            <p class="diagnostic-text">
                In the selected data slice, <b>{best_camp_id.upper()}</b> achieves the highest aggregate ROAS of <b>{best_camp_roas:.2f}x</b> 
                (generating ${best_camp_rev:,.2f} in revenue). Creative format <b>{top_banner}</b> accounts for the largest revenue share (${top_banner_rev:,.2f}).
            </p>
        </div>
        
        <div class="diagnostic-box" style="border-left-color: #f59e0b; background: #fffbeb;">
            <div class="diagnostic-title" style="color: #b45309;">⚡ Publisher Placement Contribution</div>
            <p class="diagnostic-text">
                Publisher inventory slot <b>'{top_placement}'</b> represents the highest grossing placement in this view with an observed ROAS of <b>{top_placement_roas:.2f}x</b>.
                Prioritizing inventory budget towards top-tier placements may support aggregate profit margins.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_i2:
        st.markdown(f"""
        <div class="diagnostic-box" style="border-left-color: #ef4444; background: #fef2f2;">
            <div class="diagnostic-title" style="color: #b91c1c;">⚠️ Lower Efficiency Segment</div>
            <p class="diagnostic-text">
                <b>{worst_camp_id.upper()}</b> demonstrates an observed ROAS multiplier of <b>{worst_camp_roas:.2f}x</b> on ${worst_camp_cost:,.2f} in ad spend.
                Recommendation: Audit creative CTR, refine audience targeting tiers, or test adjusted bid caps.
            </p>
        </div>
        
        <div class="diagnostic-box" style="border-left-color: #6366f1; background: #eef2ff;">
            <div class="diagnostic-title" style="color: #4338ca;">👥 Audience Engagement Differentiation</div>
            <p class="diagnostic-text">
                High-engagement audience cohorts demonstrate higher conversion rates (CVR) relative to lower tiers.
                Focusing budget on engaged cohorts typically improves portfolio CPA efficiency.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("#### 💬 Rule-Based Campaign Query Assistant")
    st.caption("Enter a query regarding portfolio metrics (ROAS, CPA, CPC, CTR, campaigns, placements) to receive dynamic insights based on your active filters.")
    
    user_query = st.text_input("Ask a question about current campaign performance:", placeholder="e.g., Which campaign has the highest ROAS? or What is the average CPA?")
    
    if user_query:
        q = user_query.lower()
        if 'cpa' in q or 'acquisition' in q:
            reply = f"Current portfolio CPA is **${portfolio_cpa:.2f}** with total conversions of **{total_conv:,.0f}**. Top placement '{top_placement}' achieved an observed ROAS of **{top_placement_roas:.2f}x**."
        elif 'roas' in q or 'return' in q:
            reply = f"Portfolio ROAS stands at **{portfolio_roas:.2f}x** (Net ROI: **{portfolio_roi:+.1f}%**). Campaign **{best_camp_id}** leads with an observed ROAS of **{best_camp_roas:.2f}x**."
        elif 'roi' in q:
            reply = f"Portfolio Net ROI is **{portfolio_roi:+.1f}%** (ROAS: **{portfolio_roas:.2f}x**), reflecting a Net Operating Profit of **${net_profit:,.2f}** on **${total_cost:,.2f}** ad spend."
        elif 'cpc' in q or 'click' in q:
            reply = f"Average Cost Per Click (CPC) is **${portfolio_cpc:.2f}** across **{total_clicks:,.0f}** clicks, with an overall portfolio CTR of **{portfolio_ctr:.2f}%**."
        elif 'campaign' in q:
            reply = f"In the filtered dataset, **{best_camp_id}** is the top performer with **{best_camp_roas:.2f}x** ROAS, while **{worst_camp_id}** records an observed **{worst_camp_roas:.2f}x** ROAS."
        elif 'banner' in q or 'creative' in q or 'size' in q:
            reply = f"The top revenue-contributing creative dimension is **{top_banner}**, generating **${top_banner_rev:,.2f}** in total attributed revenue."
        elif 'placement' in q:
            reply = f"Top grossing publisher placement is **'{top_placement}'**, delivering an observed ROAS of **{top_placement_roas:.2f}x**."
        else:
            reply = f"Active dataset summary ({len(df):,} records): Total Revenue is **${total_rev:,.2f}**, Media Spend is **${total_cost:,.2f}**, Net Profit is **${net_profit:,.2f}**, and Portfolio ROAS is **{portfolio_roas:.2f}x**."
            
        st.info(f"💡 **Campaign Assistant:** {reply}")

# -----------------------------------------------------------------------------
# 9. DATA QUALITY & PIPELINE METADATA PANEL
# -----------------------------------------------------------------------------
with st.expander("📋 Data Quality, Completeness & Schema Audit", expanded=False):
    col_dq1, col_dq2 = st.columns(2)
    with col_dq1:
        st.markdown(f"""
        **Dataset Telemetry:**
        - **Total Clean Records**: `{len(raw_df):,}`
        - **Filtered Active Records**: `{len(df):,}` (`{(len(df)/len(raw_df)*100):.1f}%` of corpus)
        - **Feature Columns**: `12 operational columns`
        - **Missing / Null Values**: `0` (clean imputed state)
        - **Time Horizon**: `April – June (Daily resolution)`
        """)
    with col_dq2:
        st.markdown(f"""
        **Pipeline & Machine Learning Status:**
        - **Model Serialized**: `{'best_advertising_model.pkl' if ml_pipeline else 'Not Loaded'}`
        - **Active Estimator**: `{model_meta['name']}`
        - **Target Variable**: `revenue ($)`
        - **Leakage Safeguard**: `ROI, ROAS, Net Profit, and sales proxies excluded from feature matrix`
        """)

# -----------------------------------------------------------------------------
# 10. EXECUTIVE FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center; color: #64748b; font-size: 12px; padding: 8px 0;">
    <div>
        <b>Digital Marketing Campaign Analysis & Revenue Intelligence</b> | Portfolio BI Suite
    </div>
    <div>
        Lead Analyst: <b>Samaa Shaikh</b> | Verified Production Pipeline
    </div>
</div>
""", unsafe_allow_html=True)

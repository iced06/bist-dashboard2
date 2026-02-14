"""
Real-Time Technical Analysis Dashboard for Turkish Stock Market (BIST)
Streamlit version with TradingView authentication and original scoring calculations
Includes stock screener for "Chosen Stocks"
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import borsapy as bp
from ta.volatility import BollingerBands
import ta
from datetime import date, datetime, timedelta
import os
from dotenv import load_dotenv

# Page config
st.set_page_config(
    page_title="BIST Technical Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 1rem 0;
        font-weight: bold;
    }
    .status-realtime {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .status-delayed {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffeeba;
    }
    .chosen-stock {
        background-color: #d1ecf1;
        border-left: 4px solid #0c5460;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'chosen_stocks' not in st.session_state:
    st.session_state.chosen_stocks = []

# Full stock list (all BIST stocks)
IMKB = [
    "THYAO", "GARAN", "ASELS", "EREGL", "AKBNK", "SAHOL", "TUPRS", "YKBNK",
    "KCHOL", "PETKM", "VAKBN", "SISE", "TCELL", "ENKAI", "ISCTR", "KOZAL",
    "TTKOM", "DOHOL", "FROTO", "HALKB", "ARCLK", "BIMAS", "EKGYO", "TAVHL",
    "ALARK", "PGSUS", "ODAS", "MGROS", "SOKM", "KRDMD", "ALBRK", "SKBNK",
    "TSKB", "ICBCT", "KLNMA", "AKGRT", "ANHYT", "ANSGR", "AGESA", "TURSG",
    "RAYSG", "CRDFA", "GARFA", "ISFIN", "LIDFA", "SEKFK", "ULUFA", "VAKFN",
    "A1CAP", "GEDIK", "GLBMD", "INFO", "ISMEN", "OSMEN", "OYYAT", "TERA",
    "ALMAD", "CVKMD", "IPEKE", "KOZAA", "PRKME", "ALCAR", "BIENY", "BRSAN",
    "CUSAN", "DNISI", "DOGUB", "EGSER", "ERBOS", "QUAGR", "INTEM", "KLKIM"
]
IMKB = sorted(list(set(IMKB)))

@st.cache_resource
def setup_tradingview_auth():
    """Set up TradingView authentication for real-time data"""
    try:
        username = os.getenv("TRADINGVIEW_USERNAME")
        password = os.getenv("TRADINGVIEW_PASSWORD")
        
        if username and password:
            bp.set_tradingview_credentials(username=username, password=password)
            return True, "✅ TradingView authentication successful! Real-time data enabled."
        else:
            return False, "⚠️ No TradingView credentials found. Using free tier (15-min delay)."
    except Exception as e:
        return False, f"❌ TradingView authentication failed: {str(e)}"

@st.cache_data(ttl=300)
def fetch_stock_data(symbol, start_date="2023-01-01", end_date=None, interval="1d"):
    """Fetch stock data using borsapy"""
    try:
        if end_date is None:
            end_date = date.today().strftime("%Y-%m-%d")
        
        ticker = bp.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)
        
        if df is not None and not df.empty:
            df.columns = [col.title() for col in df.columns]
        
        return df
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_all_indicators(df):
    """
    Calculate ALL technical indicators exactly as in your original code
    This is used for the ORIGINAL scoring calculation (Score 2)
    """
    if df is None or df.empty:
        return None
    
    try:
        # Basic calculations
        df["Return"] = df["Close"].diff()
        df["Return_pct"] = df["Close"].pct_change()
        df["Target_Cls"] = np.where(df.Return > 0, 1, 0)
        df["Vol_diff"] = df["Volume"].diff()
        df["Vol_change"] = df["Volume"].pct_change()
        
        # Bollinger Bands
        indicator_bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
        df["bb_bbm"] = indicator_bb.bollinger_mavg()
        df["bb_bbh"] = indicator_bb.bollinger_hband()
        df["bb_bbl"] = indicator_bb.bollinger_lband()
        
        # MACD
        df["MACD"] = ta.trend.macd(df["Close"], window_slow=26, window_fast=12, fillna=False)
        df["MACDS"] = ta.trend.macd_signal(df["Close"], window_sign=9, fillna=False)
        df["Diff"] = df["MACD"] - df["MACDS"]
        df["Buy_MACD"] = np.where((df["MACD"] > df["MACDS"]), 1, 0)
        df["Buy_MACDS"] = np.where((df["Buy_MACD"] > df["Buy_MACD"].shift(1)), 1, 0)
        df["Buy_MACDS2"] = np.where((df["Diff"] > 0) & (df["Buy_MACDS"] == 1), 2, df["Buy_MACDS"])
        
        # Volume SMA
        df['VSMA15'] = ta.trend.sma_indicator(df['Volume'], window=15)
        
        # OBV
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        
        # RSI
        df["RSI"] = ta.momentum.rsi(df["Close"], window=14, fillna=False)
        df["Buy_RSI"] = np.where((df["RSI"] > 30), 1, 0)
        df["Buy_RSIS"] = np.where((df["Buy_RSI"] > df["Buy_RSI"].shift(1)), 1, 0)
        
        # Awesome Oscillator
        df["AO"] = ta.momentum.awesome_oscillator(df["High"], df["Low"], window1=5, window2=34, fillna=True)
        df["Buy_AO"] = np.where((df["AO"] > 0), 1, 0)
        df["Buy_AOS"] = np.where((df["Buy_AO"] > df["Buy_AO"].shift(1)), 1, 0)
        
        # CCI
        df["CCI"] = ta.trend.cci(df["High"], df["Low"], df["Close"], window=20, fillna=False)
        df["Buy_CCI"] = np.where((df["CCI"] > 0), 1, 0)
        df["Buy_CCIS"] = np.where((df["Buy_CCI"] > df["Buy_CCI"].shift(1)), 1, 0)
        
        # EMA
        df["EMA10"] = ta.trend.ema_indicator(df["Close"], window=10, fillna=False)
        df["EMA30"] = ta.trend.ema_indicator(df["Close"], window=30, fillna=False)
        df["Buy_EMA10"] = np.where((df["Close"] > df["EMA10"]), 1, 0)
        df["Buy_EMA10S"] = np.where((df["Buy_EMA10"] > df["Buy_EMA10"].shift(1)), 1, 0)
        df["Buy_EMA10_EMA30"] = np.where((df["EMA10"] > df["EMA30"]), 1, 0)
        df["Buy_EMA10_EMA30S"] = np.where((df["Buy_EMA10_EMA30"] > df["Buy_EMA10_EMA30"].shift(1)), 1, 0)
        
        # Stochastic
        df["Stochastic"] = ta.momentum.stoch_signal(df["High"], df["Low"], df["Close"], window=3, fillna=False)
        df["Stochastic_Buy"] = np.where((df["Stochastic"] > 20), 1, 0)
        df["Stochastic_BuyS"] = np.where((df["Stochastic_Buy"] > df["Stochastic_Buy"].shift(1)), 1, 0)
        
        # KAMA
        df["KAMA"] = ta.momentum.kama(df["Close"], window=10, pow1=2, pow2=30, fillna=False)
        df["Buy_KAMA"] = np.where((df["Close"] > df["KAMA"]), 1, 0)
        df["Buy_KAMAS"] = np.where((df["Buy_KAMA"] > df["Buy_KAMA"].shift(1)), 1, 0)
        
        # SMA
        df['SMA5'] = ta.trend.sma_indicator(df['Close'], window=5)
        df['SMA22'] = ta.trend.sma_indicator(df['Close'], window=22)
        df['SMA50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df["Buy_SMA5"] = np.where((df["Close"] > df["SMA5"]), 1, 0)
        df["Buy_SMA22"] = np.where((df["Close"] > df["SMA22"]), 1, 0)
        df["Buy_SMA50"] = np.where((df["Close"] > df["SMA50"]), 1, 0)
        df["Buy_SMA5S"] = np.where((df["Buy_SMA5"] > df["Buy_SMA5"].shift(1)), 1, 0)
        df["Buy_SMA22S"] = np.where((df["Buy_SMA22"] > df["Buy_SMA22"].shift(1)), 1, 0)
        df["Buy_SMA50S"] = np.where((df["Buy_SMA50"] > df["Buy_SMA50"].shift(1)), 1, 0)
        
        # CMF
        df["CMF"] = ta.volume.chaikin_money_flow(df["High"], df["Low"], df["Close"], df["Volume"], window=20, fillna=False)
        df["Buy_CMF"] = np.where((df["CMF"] > 0), 1, 0)
        df["Buy_CMFS"] = np.where((df["Buy_CMF"] > df["Buy_CMF"].shift(1)), 1, 0)
        
        return df
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
        return df

def calculate_original_scores(df):
    """
    Calculate scores EXACTLY as in your original code
    Score1 (Indicator Score 2): Sum of all buy signals
    Score2 (Volume Score 2): Volume / VSMA15
    """
    if df is None or df.empty:
        return 0, 0
    
    try:
        latest = df.iloc[-1]
        
        # EXACTLY your original formula
        indicator_score_2 = (
            latest["Buy_MACDS2"] + 
            latest["Buy_AOS"] + 
            latest["Buy_EMA10_EMA30S"] + 
            latest["Buy_SMA5S"] + 
            latest["Buy_SMA22S"] + 
            latest["Buy_RSIS"] + 
            latest["Stochastic_BuyS"] + 
            latest["Buy_CCIS"] + 
            latest["Buy_KAMAS"] + 
            latest["Buy_CMFS"]
        )
        
        # EXACTLY your original volume score formula
        volume_score_2 = latest["Volume"] / latest["VSMA15"]
        
        return float(indicator_score_2), float(volume_score_2)
    except Exception as e:
        st.error(f"Error calculating original scores: {e}")
        return 0, 0

def calculate_simplified_scores(df):
    """
    Calculate simplified scores (Score 1) - the ones I created
    These are simpler and more intuitive
    """
    if df is None or df.empty:
        return 0, 0
    
    latest = df.iloc[-1]
    indicator_score = 0
    volume_score = 0
    
    # Simplified Indicator Score
    if latest['RSI'] < 30:
        indicator_score += 1
    elif latest['RSI'] > 70:
        indicator_score -= 1
    
    if latest['MACD'] > latest['MACDS']:
        indicator_score += 1
    else:
        indicator_score -= 1
    
    if latest['Close'] > latest['SMA5'] > latest['SMA22']:
        indicator_score += 1
    elif latest['Close'] < latest['SMA5'] < latest['SMA22']:
        indicator_score -= 1
    
    if latest['Close'] < latest['bb_bbl']:
        indicator_score += 1
    elif latest['Close'] > latest['bb_bbh']:
        indicator_score -= 1
    
    if len(df) >= 5:
        price_change_5d = (latest['Close'] - df.iloc[-5]['Close']) / df.iloc[-5]['Close']
        if price_change_5d > 0.05:
            indicator_score += 1
        elif price_change_5d < -0.05:
            indicator_score -= 1
    
    indicator_score = max(0, min(5, indicator_score + 2.5))
    
    # Simplified Volume Score (normalized to 0-5 scale)
    volume_ratio = latest["Volume"] / latest["VSMA15"]
    if volume_ratio > 2:
        volume_score = 5
    elif volume_ratio > 1.5:
        volume_score = 4
    elif volume_ratio > 1.2:
        volume_score = 3
    elif volume_ratio > 0.8:
        volume_score = 2
    else:
        volume_score = 1
    
    return round(indicator_score, 1), round(volume_score, 1)

def screen_chosen_stocks(stock_list):
    """
    Screen stocks based on criteria:
    - Indicator Score 2 >= 3
    - Volume Score 2 > 0.7
    """
    chosen_stocks = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, symbol in enumerate(stock_list):
        status_text.text(f"Screening {symbol}... ({idx+1}/{len(stock_list)})")
        progress_bar.progress((idx + 1) / len(stock_list))
        
        try:
            df = fetch_stock_data(symbol, start_date="2023-01-01")
            if df is not None and not df.empty:
                df = calculate_all_indicators(df)
                indicator_score_2, volume_score_2 = calculate_original_scores(df)
                
                # Apply your criteria
                if indicator_score_2 >= 3 and volume_score_2 > 0.7:
                    chosen_stocks.append({
                        'symbol': symbol,
                        'indicator_score_2': round(indicator_score_2, 2),
                        'volume_score_2': round(volume_score_2, 2)
                    })
        except Exception as e:
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    return chosen_stocks

def create_gauge_chart(value, title, max_value=5):
    """Create a gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 18}},
        gauge={
            'axis': {'range': [None, max_value]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 2], 'color': "#ffcccc"},
                {'range': [2, 3.5], 'color': "#ffffcc"},
                {'range': [3.5, max_value], 'color': "#ccffcc"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value / 2
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_candlestick_chart(df, symbol):
    """Create candlestick chart with indicators"""
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='Price'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['bb_bbh'], mode='lines',
        line=dict(color='rgba(255, 0, 0, 0.5)', width=1), name='BB Upper'
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['bb_bbl'], mode='lines',
        line=dict(color='rgba(0, 0, 255, 0.5)', width=1),
        name='BB Lower', fill='tonexty', fillcolor='rgba(200, 200, 200, 0.2)'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA5'], mode='lines',
        line=dict(color='orange', width=2), name='SMA 5'
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA22'], mode='lines',
        line=dict(color='green', width=2), name='SMA 22'
    ))
    
    fig.update_layout(
        title=f"{symbol} - Price Chart",
        xaxis_title="Date", yaxis_title="Price (TRY)",
        hovermode='x unified', height=500,
        xaxis_rangeslider_visible=False
    )
    return fig

def create_volume_chart(df):
    """Create volume chart"""
    fig = go.Figure()
    colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' 
              for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['VSMA15'], mode='lines',
        line=dict(color='blue', width=2), name='Volume SMA 15'
    ))
    fig.update_layout(title="Volume Analysis", xaxis_title="Date", yaxis_title="Volume", height=300)
    return fig

def create_macd_chart(df):
    """Create MACD chart"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines',
                             line=dict(color='blue', width=2), name='MACD'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MACDS'], mode='lines',
                             line=dict(color='red', width=2), name='Signal'))
    colors = ['green' if val > 0 else 'red' for val in df['Diff']]
    fig.add_trace(go.Bar(x=df.index, y=df['Diff'], name='Histogram', marker_color=colors))
    fig.update_layout(title="MACD Indicator", xaxis_title="Date", yaxis_title="Value", height=300)
    return fig

def create_rsi_chart(df):
    """Create RSI chart"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines',
                             line=dict(color='purple', width=2), name='RSI'))
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    fig.update_layout(title="RSI Indicator", xaxis_title="Date", 
                     yaxis_title="RSI", yaxis_range=[0, 100], height=300)
    return fig

# Main App
def main():
    st.markdown('<h1 class="main-header">📊 Turkish Stock Market Technical Analysis</h1>', 
                unsafe_allow_html=True)
    
    authenticated, auth_message = setup_tradingview_auth()
    st.session_state.authenticated = authenticated
    
    status_class = "status-realtime" if authenticated else "status-delayed"
    status_icon = "🎉" if authenticated else "⚠️"
    st.markdown(f'<div class="status-box {status_class}">{status_icon} {auth_message}</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Mode selection
        mode = st.radio(
            "Select Mode",
            options=["📊 Single Stock Analysis", "🔍 Stock Screener (Chosen Stocks)"],
            index=0
        )
        
        if mode == "📊 Single Stock Analysis":
            selected_stock = st.selectbox(
                "Select Stock Symbol",
                options=IMKB,
                index=IMKB.index("THYAO") if "THYAO" in IMKB else 0
            )
            
            st.subheader("📅 Date Range")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
            with col2:
                end_date = st.date_input("End Date", value=datetime.now())
            
            if authenticated:
                st.subheader("⏱️ Time Interval")
                interval = st.selectbox("Select Interval", options=["1d", "1h", "4h", "15m", "5m"])
            else:
                interval = "1d"
                st.info("📝 Intraday intervals available with TradingView authentication")
        else:
            st.info("🔍 Screen all stocks for: Indicator Score 2 >= 3 AND Volume Score 2 > 0.7")
            if st.button("🚀 Run Stock Screener", use_container_width=True):
                with st.spinner("Screening all stocks... This may take a few minutes..."):
                    st.session_state.chosen_stocks = screen_chosen_stocks(IMKB)
                st.success(f"✅ Found {len(st.session_state.chosen_stocks)} chosen stocks!")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📖 Scoring System")
        st.markdown("""
        **Score 1 (Simplified):**
        - 0-5 scale, easier to interpret
        
        **Score 2 (Original):**
        - Your exact calculation
        - 10 buy signals summed
        - Volume ratio to 15-day SMA
        """)
    
    # Main content
    if mode == "📊 Single Stock Analysis":
        if selected_stock:
            with st.spinner(f"Loading data for {selected_stock}..."):
                df = fetch_stock_data(selected_stock, start_date=start_date.strftime("%Y-%m-%d"),
                                     end_date=end_date.strftime("%Y-%m-%d"), interval=interval)
            
            if df is not None and not df.empty:
                df = calculate_all_indicators(df)
                
                # Calculate BOTH sets of scores
                ind_score_1, vol_score_1 = calculate_simplified_scores(df)
                ind_score_2, vol_score_2 = calculate_original_scores(df)
                
                latest = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else latest
                
                # Key metrics
                st.subheader(f"📈 {selected_stock} - Latest Data")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    price_change = latest['Close'] - prev['Close']
                    price_change_pct = (price_change / prev['Close']) * 100
                    st.metric("Close Price", f"₺{latest['Close']:.2f}",
                             f"{price_change:.2f} ({price_change_pct:.2f}%)")
                with col2:
                    st.metric("Volume", f"{latest['Volume']:,.0f}")
                with col3:
                    st.metric("RSI", f"{latest['RSI']:.2f}")
                with col4:
                    st.metric("MACD", f"{latest['MACD']:.4f}")
                
                # Score comparison
                st.subheader("🎯 Score Comparison")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Simplified Scores (Score 1)")
                    subcol1, subcol2 = st.columns(2)
                    with subcol1:
                        st.plotly_chart(create_gauge_chart(ind_score_1, "Indicator Score 1"), 
                                       use_container_width=True)
                    with subcol2:
                        st.plotly_chart(create_gauge_chart(vol_score_1, "Volume Score 1"), 
                                       use_container_width=True)
                
                with col2:
                    st.markdown("#### Original Scores (Score 2) ⭐")
                    subcol1, subcol2 = st.columns(2)
                    with subcol1:
                        st.plotly_chart(create_gauge_chart(ind_score_2, "Indicator Score 2", max_value=10), 
                                       use_container_width=True)
                    with subcol2:
                        st.metric("Volume Score 2", f"{vol_score_2:.2f}")
                        if vol_score_2 > 0.7:
                            st.success("✅ Above 0.7 threshold")
                        else:
                            st.warning("⚠️ Below 0.7 threshold")
                
                # Chosen Stock Status
                if ind_score_2 >= 3 and vol_score_2 > 0.7:
                    st.markdown("""
                    <div class="chosen-stock">
                        <h3>⭐ This is a CHOSEN STOCK! ⭐</h3>
                        <p>Indicator Score 2 >= 3 AND Volume Score 2 > 0.7</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Charts
                st.subheader("📊 Price Chart")
                st.plotly_chart(create_candlestick_chart(df, selected_stock), use_container_width=True)
                
                tab1, tab2, tab3 = st.tabs(["📊 Volume", "📈 MACD", "📉 RSI"])
                with tab1:
                    st.plotly_chart(create_volume_chart(df), use_container_width=True)
                with tab2:
                    st.plotly_chart(create_macd_chart(df), use_container_width=True)
                with tab3:
                    st.plotly_chart(create_rsi_chart(df), use_container_width=True)
                
                with st.expander("📋 View Raw Data"):
                    st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD']].tail(50))
            else:
                st.error(f"❌ No data available for {selected_stock}")
    
    else:  # Stock Screener Mode
        st.subheader("🔍 Chosen Stocks - Stock Screener Results")
        
        if st.session_state.chosen_stocks:
            st.success(f"Found {len(st.session_state.chosen_stocks)} stocks meeting criteria:")
            st.info("**Criteria:** Indicator Score 2 >= 3 AND Volume Score 2 > 0.7")
            
            # Display as table
            df_chosen = pd.DataFrame(st.session_state.chosen_stocks)
            df_chosen = df_chosen.sort_values('indicator_score_2', ascending=False)
            
            st.dataframe(
                df_chosen.style.background_gradient(subset=['indicator_score_2', 'volume_score_2'], 
                                                   cmap='RdYlGn'),
                use_container_width=True
            )
            
            # Display individual cards
            cols = st.columns(3)
            for idx, stock in enumerate(df_chosen.to_dict('records')):
                with cols[idx % 3]:
                    st.markdown(f"""
                    <div class="chosen-stock">
                        <h4>{stock['symbol']}</h4>
                        <p>Indicator Score 2: <b>{stock['indicator_score_2']}</b></p>
                        <p>Volume Score 2: <b>{stock['volume_score_2']}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("👆 Click 'Run Stock Screener' in the sidebar to find chosen stocks")
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: #7f8c8d;'>
            Data powered by borsapy (TradingView) | 
            {'✅ Real-time' if authenticated else '⏱️ 15-min delayed'} | 
            Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

"""
Real-Time Technical Analysis Dashboard for Turkish Stock Market (BIST)
Streamlit version with TradingView authentication and original scoring calculations
Includes stock screener for "Chosen Stocks" with FILTER option
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
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
    .filter-info {
        background-color: #fff3cd;
        border-left: 4px solid #856404;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'chosen_stocks' not in st.session_state:
    st.session_state.chosen_stocks = []
if 'filter_chosen_only' not in st.session_state:
    st.session_state.filter_chosen_only = False

# Full stock list
IMKB = [
    "ALBRK", "GARAN", "HALKB", "ISCTR", "SKBNK", "TSKB", "ICBCT", "KLNMA",
    "VAKBN", "YKBNK", "AKGRT", "ANHYT", "ANSGR", "AGESA", "TURSG", "RAYSG", 
    "CRDFA", "GARFA", "ISFIN", "LIDFA", "SEKFK", "ULUFA", "VAKFN", "A1CAP", 
    "GEDIK", "GLBMD", "INFO", "ISMEN", "OSMEN", "OYYAT", "TERA", "ALMAD", 
    "CVKMD", "PRKME", "ALCAR", "BIENY", "BRSAN", 
    "CUSAN", "DNISI", "DOGUB", "EGSER", "ERBOS", "QUAGR", "INTEM", "KLKIM", 
    "KLSER", "KLMSN", "KUTPO", "PNLSN", "SAFKR", "ERCB", "SISE", "USAK", 
    "YYAPI", "AFYON", "AKCNS", "BTCIM", "BSOKE", "BOBET", "BUCIM", "CMBTN", 
    "CMENT", "CIMSA", "GOLTS", "KONYA", "OYAKC", "NIBAS", "NUHCM", "ARCLK", 
    "ARZUM", "SILVR", "VESBE", "VESTL", "BMSCH", "BMSTL", "EREGL", "IZMDC", 
    "KCAER", "KRDMA", "KRDMB", "KRDMD", "TUCLK", "YKSLN", "AHGAZ", "AKENR", 
    "AKFYE", "AKSEN", "AKSUE", "ALFAS", "ASTOR", "ARASE", "AYDEM", "AYEN",
    "BASGZ", "BIOEN", "CONSE", "CWENE", "CANTE", "EMKEL", "ENJSA", "ENERY", 
    "ESEN", "GWIND", "GEREL", "HUNER", "IZENR", "KARYE", "NATEN", "NTGAZ", 
    "MAGEN", "ODAS", "SMRTG", "TATEN", "ZEDUR", "ZOREN", "ATAKP", "AVOD", 
    "AEFES", "BANVT", "BYDNR", "BIGCH", "CCOLA", "DARDL", "EKIZ", "EKSUN", 
    "ELITE", "ERSU", "FADE", "FRIGO", "GOKNR", "KAYSE", "KENT", "KERVT", 
    "KNFRT", "KRSTL", "KRVGD", "KTSKR", "MERKO", "OFSYM", "ORCAY", "OYLUM", 
    "PENGD", "PETUN", "PINSU", "PNSUT", "SELGD", "SELVA", "SOKE", "TBORG", 
    "TATGD", "TUKAS", "ULKER", "ULUUN", "YYLGD", "BIMAS", "KIMMR", "GMTAS", 
    "SOKM", "BIZIM", "CRFSA", "MGROS", "AKYHO", "ALARK", "MARKA", "ATSYH", 
    "BRYAT", "COSMO", "DAGHL", "DOHOL", "DERHL", "ECZYT", "ENKAI", "EUHOL", 
    "GLYHO", "GLRYH", "GSDHO", "HEDEF", "IEYHO", "IHLAS", "INVES", "KERVN", 
    "KLRHO", "KCHOL", "BERA", "MZHLD", "MMCAS", "METRO", "NTHOL", "OSTIM",
    "POLHO", "RALYH", "SAHOL", "TAVHL", "TKFEN", "UFUK", "VERUS", "AGHOL", 
    "YESIL", "UNLU", "ADESE", "AKFGY", "AKMGY", "AKSGY", "ALGYO", "ASGYO", 
    "ATAGY", "AGYO", "AVGYO", "DAPGM", "DZGYO", "DGGYO", "EDIP", "EYGYO", 
    "EKGYO", "FZLGY", "IDGYO", "IHLGM", "ISGYO", "KZBGY", "KLGYO", "KRGYO", 
    "KUYAS", "MSGYO", "NUGYO", "OZKGY", "OZGYO", "PAGYO", "PSGYO", "PEKGY", 
    "RYGYO", "SEGYO", "SRVGY", "SNGYO", "TRGYO", "TDGYO", "TSGYO", "TURGG",
    "VKGYO", "YGGYO", "YGYO", "ZRGYO", "ALCTL", "ARDYZ", "ARENA", "INGRM", 
    "ASELS", "ATATP", "AZTEK", "DGATE", "DESPC", "EDATA", "FORTE", "HTTBT", 
    "KFEIN", "SDTTR", "SMART", "ESCOM", "FONET", "INDES", "KAREL", "KRONT",
    "LINK", "LOGO", "MANAS", "MTRKS", "MIATK", "MOBTL", "NETAS", "OBASE", 
    "PENTA", "TKNSA", "VBTYZ", "ARSAN", "BLCYT", "BRKO", "BRMEN", "BOSSA", 
    "DAGI", "DERIM", "DESA", "DIRIT", "EBEBK", "ENSRI", "HATEK", "ISSEN", 
    "KRTEK", "LUKSK", "MNDRS", "RUBNS", "SKTAS", "SNPAM", "SUNTK", "YATAS", 
    "YUNSA", "ADEL", "ANGEN", "ANELE", "BNTAS", "BRKVY", "BRLSM", "BURCE", 
    "BURVA", "BVSAN", "CEOEM", "DGNMO", "EMNIS", "EUPWR", "ESCAR", "FORMT", 
    "FLAP", "GESAN", "GLCVY", "GENTS", "HKTM", "IHEVA", "IHAAS", "IMASM", 
    "KTLEV", "KLSYN", "KONTR", "MACKO", "MAVI", "MAKIM", "MAKTK", "MEPET", 
    "ORGE", "PARSN", "TGSAS", "PRKAB", "PAPIL", "PCILT", "PKART", "PSDTC", 
    "SANEL", "SNICA", "SANKO", "SARKY", "SNKRN", "KUVVA", "OZSUB", "SONME", 
    "SUMAS", "SUWEN", "TLMAN", "ULUSE", "VAKKO", "YAPRK", "YAYLA", "YEOTK", 
    "AVHOL", "BEYAZ", "DENGE", "IZFAS", "IZINV", "MEGAP", "OZRDN", "PASEU", 
    "PAMEL", "POLTK", "RODRG", "ASUZU", "DOAS", "FROTO", "KARSN", "OTKAR", 
    "TOASO", "TMSN", "TTRAK", "BFREN", "BRISA", "CELHA", "CEMAS", "CEMTS", 
    "DOKTA", "DMSAS", "DITAS", "EGEEN", "FMIZP", "GOODY", "JANTS", "KATMR", 
    "AYGAZ", "CASA", "TUPRS", "TRCAS", "ACSEL", "AKSA", "ALKIM", "BAGFS", 
    "BAYRK", "BRKSN", "DYOBY", "EGGUB", "EGPRO", "EPLAS", "EUREN", "GUBRF", 
    "ISKPL", "KMPUR", "KOPOL", "KORDS", "KRPLS", "MRSHL", "MERCN", "PETKM", 
    "RNPOL", "SANFM", "SASA", "TARKM", "ALKA", "BAKAB", "BARMA", "DURDO", 
    "GEDZA", "GIPTA", "KAPLM", "KARTN", "KONKA", "MNDTR", "PRZMA", "SAMAT", 
    "TEZOL", "VKING", "HUBVC", "GOZDE", "HDFGS", "ISGSY", "PRDGS", "VERTU", 
    "DOBUR", "HURGZ", "IHGZT", "IHYAY", "AYCES", "AVTUR", "ETILR", "MAALT", 
    "METUR", "PKENT", "TEKTU", "ULAS", "CLEBI", "GSDDE", "GRSEL", "GZNMI", 
    "PGSUS", "PLTUR", "RYSAS", "LIDER", "TUREX", "THYAO", "TCELL", "TTKOM", 
    "DEVA", "ECILC", "GENIL", "MEDTR", "MPARK", "EGEPO", "ONCSM", "RTALB", 
    "SELEC", "TNZTP", "TRILC", "ATLAS"
]
IMKB = sorted(list(set(IMKB)))

@st.cache_resource
def setup_tradingview_auth():
    try:
        username = os.getenv("TRADINGVIEW_USERNAME")
        password = os.getenv("TRADINGVIEW_PASSWORD")
        if username and password:
            bp.set_tradingview_credentials(username=username, password=password)
            return True, "✅ Real-time data enabled"
        return False, "⚠️ Using free tier (15-min delay)"
    except:
        return False, "❌ Authentication failed"

@st.cache_data(ttl=300)
def fetch_stock_data(symbol, start_date="2023-01-01", end_date=None, interval="1d"):
    try:
        if end_date is None:
            end_date = date.today().strftime("%Y-%m-%d")
        ticker = bp.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)
        if df is not None and not df.empty:
            df.columns = [col.title() for col in df.columns]
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def calculate_all_indicators(df):
    """Calculate ALL indicators - EXACTLY as in your original code"""
    if df is None or df.empty:
        return None
    try:
        df["Return"] = df["Close"].diff()
        df["Return_pct"] = df["Close"].pct_change()
        df["Target_Cls"] = np.where(df.Return > 0, 1, 0)
        df["Vol_diff"] = df["Volume"].diff()
        df["Vol_change"] = df["Volume"].pct_change()
        
        indicator_bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
        df["bb_bbm"] = indicator_bb.bollinger_mavg()
        df["bb_bbh"] = indicator_bb.bollinger_hband()
        df["bb_bbl"] = indicator_bb.bollinger_lband()
        
        df["MACD"] = ta.trend.macd(df["Close"], window_slow=26, window_fast=12, fillna=False)
        df["MACDS"] = ta.trend.macd_signal(df["Close"], window_sign=9, fillna=False)
        df["Diff"] = df["MACD"] - df["MACDS"]
        df["Buy_MACD"] = np.where((df["MACD"] > df["MACDS"]), 1, 0)
        df["Buy_MACDS"] = np.where((df["Buy_MACD"] > df["Buy_MACD"].shift(1)), 1, 0)
        df["Buy_MACDS2"] = np.where((df["Diff"] > 0) & (df["Buy_MACDS"] == 1), 2, df["Buy_MACDS"])
        
        df['VSMA15'] = ta.trend.sma_indicator(df['Volume'], window=15)
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        
        df["RSI"] = ta.momentum.rsi(df["Close"], window=14, fillna=False)
        df["Buy_RSI"] = np.where((df["RSI"] > 30), 1, 0)
        df["Buy_RSIS"] = np.where((df["Buy_RSI"] > df["Buy_RSI"].shift(1)), 1, 0)
        
        df["AO"] = ta.momentum.awesome_oscillator(df["High"], df["Low"], window1=5, window2=34, fillna=True)
        df["Buy_AO"] = np.where((df["AO"] > 0), 1, 0)
        df["Buy_AOS"] = np.where((df["Buy_AO"] > df["Buy_AO"].shift(1)), 1, 0)
        
        df["CCI"] = ta.trend.cci(df["High"], df["Low"], df["Close"], window=20, fillna=False)
        df["Buy_CCI"] = np.where((df["CCI"] > 0), 1, 0)
        df["Buy_CCIS"] = np.where((df["Buy_CCI"] > df["Buy_CCI"].shift(1)), 1, 0)
        
        df["EMA10"] = ta.trend.ema_indicator(df["Close"], window=10, fillna=False)
        df["EMA30"] = ta.trend.ema_indicator(df["Close"], window=30, fillna=False)
        df["Buy_EMA10"] = np.where((df["Close"] > df["EMA10"]), 1, 0)
        df["Buy_EMA10S"] = np.where((df["Buy_EMA10"] > df["Buy_EMA10"].shift(1)), 1, 0)
        df["Buy_EMA10_EMA30"] = np.where((df["EMA10"] > df["EMA30"]), 1, 0)
        df["Buy_EMA10_EMA30S"] = np.where((df["Buy_EMA10_EMA30"] > df["Buy_EMA10_EMA30"].shift(1)), 1, 0)
        
        df["Stochastic"] = ta.momentum.stoch_signal(df["High"], df["Low"], df["Close"], window=3, fillna=False)
        df["Stochastic_Buy"] = np.where((df["Stochastic"] > 20), 1, 0)
        df["Stochastic_BuyS"] = np.where((df["Stochastic_Buy"] > df["Stochastic_Buy"].shift(1)), 1, 0)
        
        df["KAMA"] = ta.momentum.kama(df["Close"], window=10, pow1=2, pow2=30, fillna=False)
        df["Buy_KAMA"] = np.where((df["Close"] > df["KAMA"]), 1, 0)
        df["Buy_KAMAS"] = np.where((df["Buy_KAMA"] > df["Buy_KAMA"].shift(1)), 1, 0)
        
        df['SMA5'] = ta.trend.sma_indicator(df['Close'], window=5)
        df['SMA22'] = ta.trend.sma_indicator(df['Close'], window=22)
        df['SMA50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df["Buy_SMA5"] = np.where((df["Close"] > df["SMA5"]), 1, 0)
        df["Buy_SMA22"] = np.where((df["Close"] > df["SMA22"]), 1, 0)
        df["Buy_SMA50"] = np.where((df["Close"] > df["SMA50"]), 1, 0)
        df["Buy_SMA5S"] = np.where((df["Buy_SMA5"] > df["Buy_SMA5"].shift(1)), 1, 0)
        df["Buy_SMA22S"] = np.where((df["Buy_SMA22"] > df["Buy_SMA22"].shift(1)), 1, 0)
        df["Buy_SMA50S"] = np.where((df["Buy_SMA50"] > df["Buy_SMA50"].shift(1)), 1, 0)
        
        df["CMF"] = ta.volume.chaikin_money_flow(df["High"], df["Low"], df["Close"], df["Volume"], window=20, fillna=False)
        df["Buy_CMF"] = np.where((df["CMF"] > 0), 1, 0)
        df["Buy_CMFS"] = np.where((df["Buy_CMF"] > df["Buy_CMF"].shift(1)), 1, 0)
        
        return df
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
        return df

def calculate_original_scores(df):
    """YOUR EXACT ORIGINAL FORMULAS"""
    if df is None or df.empty:
        return 0, 0
    try:
        latest = df.iloc[-1]
        indicator_score_2 = (
            latest["Buy_MACDS2"] + latest["Buy_AOS"] + latest["Buy_EMA10_EMA30S"] + 
            latest["Buy_SMA5S"] + latest["Buy_SMA22S"] + latest["Buy_RSIS"] + 
            latest["Stochastic_BuyS"] + latest["Buy_CCIS"] + latest["Buy_KAMAS"] + latest["Buy_CMFS"]
        )
        volume_score_2 = latest["Volume"] / latest["VSMA15"]
        return float(indicator_score_2), float(volume_score_2)
    except:
        return 0, 0

def calculate_simplified_scores(df):
    """Simplified scoring for comparison"""
    if df is None or df.empty:
        return 0, 0
    latest = df.iloc[-1]
    score = 0
    if latest['RSI'] < 30: score += 1
    elif latest['RSI'] > 70: score -= 1
    if latest['MACD'] > latest['MACDS']: score += 1
    else: score -= 1
    if latest['Close'] > latest['SMA5'] > latest['SMA22']: score += 1
    elif latest['Close'] < latest['SMA5'] < latest['SMA22']: score -= 1
    if latest['Close'] < latest['bb_bbl']: score += 1
    elif latest['Close'] > latest['bb_bbh']: score -= 1
    if len(df) >= 5:
        price_change = (latest['Close'] - df.iloc[-5]['Close']) / df.iloc[-5]['Close']
        if price_change > 0.05: score += 1
        elif price_change < -0.05: score -= 1
    score = max(0, min(5, score + 2.5))
    vol_ratio = latest["Volume"] / latest["VSMA15"]
    vol_score = 5 if vol_ratio > 2 else 4 if vol_ratio > 1.5 else 3 if vol_ratio > 1.2 else 2 if vol_ratio > 0.8 else 1
    return round(score, 1), round(vol_score, 1)

def screen_chosen_stocks(stock_list):
    """Screen for Indicator Score 2 >= 3 AND Volume Score 2 > 0.7"""
    chosen = []
    progress = st.progress(0)
    status = st.empty()
    for idx, symbol in enumerate(stock_list):
        status.text(f"Screening {symbol}... ({idx+1}/{len(stock_list)})")
        progress.progress((idx + 1) / len(stock_list))
        try:
            df = fetch_stock_data(symbol, start_date="2023-01-01")
            if df is not None and not df.empty:
                df = calculate_all_indicators(df)
                ind_score, vol_score = calculate_original_scores(df)
                if ind_score >= 3 and vol_score > 0.7:
                    chosen.append({
                        'symbol': symbol, 
                        'indicator_score_2': round(ind_score, 2), 
                        'volume_score_2': round(vol_score, 2)
                    })
        except:
            continue
    progress.empty()
    status.empty()
    return chosen

def create_gauge(value, title, max_val=5):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, title={'text': title, 'font': {'size': 18}},
        gauge={'axis': {'range': [None, max_val]}, 'bar': {'color': "#1f77b4"},
               'steps': [{'range': [0, 2], 'color': "#ffcccc"}, 
                        {'range': [2, 3.5], 'color': "#ffffcc"}, 
                        {'range': [3.5, max_val], 'color': "#ccffcc"}],
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': max_val/2}}
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_candlestick(df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_bbh'], mode='lines', line=dict(color='rgba(255,0,0,0.5)', width=1), name='BB Upper'))
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_bbl'], mode='lines', line=dict(color='rgba(0,0,255,0.5)', width=1), name='BB Lower', fill='tonexty', fillcolor='rgba(200,200,200,0.2)'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA5'], mode='lines', line=dict(color='orange', width=2), name='SMA 5'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA22'], mode='lines', line=dict(color='green', width=2), name='SMA 22'))
    fig.update_layout(title=f"{symbol} - Price Chart", xaxis_title="Date", yaxis_title="Price (TRY)", height=500, xaxis_rangeslider_visible=False, hovermode='x unified')
    return fig

def create_volume_chart(df):
    fig = go.Figure()
    colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors))
    fig.add_trace(go.Scatter(x=df.index, y=df['VSMA15'], mode='lines', line=dict(color='blue', width=2), name='VSMA 15'))
    fig.update_layout(title="Volume Analysis", xaxis_title="Date", yaxis_title="Volume", height=300)
    return fig

def create_macd_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', line=dict(color='blue', width=2), name='MACD'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MACDS'], mode='lines', line=dict(color='red', width=2), name='Signal'))
    colors = ['green' if val > 0 else 'red' for val in df['Diff']]
    fig.add_trace(go.Bar(x=df.index, y=df['Diff'], name='Histogram', marker_color=colors))
    fig.update_layout(title="MACD Indicator", xaxis_title="Date", yaxis_title="Value", height=300)
    return fig

def create_rsi_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', line=dict(color='purple', width=2), name='RSI'))
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    fig.update_layout(title="RSI Indicator", xaxis_title="Date", yaxis_title="RSI", yaxis_range=[0, 100], height=300)
    return fig

def main():
    st.markdown('<h1 class="main-header">📊 Turkish Stock Market Analysis</h1>', unsafe_allow_html=True)
    
    auth, msg = setup_tradingview_auth()
    st.session_state.authenticated = auth
    css = "status-realtime" if auth else "status-delayed"
    st.markdown(f'<div class="status-box {css}">{"🎉" if auth else "⚠️"} {msg}</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ Settings")
        mode = st.radio("Mode", ["📊 Single Stock", "🔍 Stock Screener"])
        
        if mode == "📊 Single Stock":
            # NEW: Add filter toggle
            st.markdown("---")
            filter_chosen = st.checkbox(
                "🔍 Show Only Chosen Stocks",
                value=st.session_state.filter_chosen_only,
                help="Filter dropdown to show only stocks that meet chosen criteria (Indicator Score 2 >= 3 AND Volume Score 2 > 0.7)"
            )
            st.session_state.filter_chosen_only = filter_chosen
            
            # Get list of stocks to show
            if filter_chosen:
                if st.session_state.chosen_stocks:
                    available_stocks = [s['symbol'] for s in st.session_state.chosen_stocks]
                    st.markdown(f'<div class="filter-info">📌 Showing {len(available_stocks)} chosen stocks</div>', unsafe_allow_html=True)
                else:
                    st.warning("⚠️ No chosen stocks found. Run Stock Screener first or uncheck the filter.")
                    available_stocks = IMKB
            else:
                available_stocks = IMKB
            
            # Stock selection
            stock = st.selectbox(
                "Select Stock Symbol",
                options=available_stocks,
                index=0 if available_stocks else 0
            )
            
            # Date range
            st.subheader("📅 Date Range")
            col1, col2 = st.columns(2)
            with col1:
                start = st.date_input("Start", value=datetime.now() - timedelta(days=365))
            with col2:
                end = st.date_input("End", value=datetime.now())
            
            # Interval
            if auth:
                st.subheader("⏱️ Interval")
                interval = st.selectbox("Select", ["1d", "1h", "4h", "15m", "5m"])
            else:
                interval = "1d"
                st.info("📝 Intraday intervals available with TradingView auth")
        else:
            st.info("🔍 Criteria: Indicator Score 2 >= 3 AND Volume Score 2 > 0.7")
            if st.button("🚀 Run Screener", use_container_width=True):
                with st.spinner("Screening all stocks..."):
                    st.session_state.chosen_stocks = screen_chosen_stocks(IMKB)
                st.success(f"✅ Found {len(st.session_state.chosen_stocks)} chosen stocks!")
                st.info("💡 Tip: Switch to Single Stock mode and enable 'Show Only Chosen Stocks' filter to analyze them!")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown("""
        **Score 1**: Simplified (0-5)
        **Score 2**: Your Original ⭐
        
        **Chosen Stocks**: 
        - Indicator Score 2 >= 3
        - Volume Score 2 > 0.7
        """)
    
    # Main content
    if mode == "📊 Single Stock":
        if not available_stocks:
            st.warning("⚠️ No stocks available. Please run Stock Screener first or uncheck the filter.")
            return
            
        with st.spinner(f"Loading {stock}..."):
            df = fetch_stock_data(stock, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), interval)
        
        if df is not None and not df.empty:
            df = calculate_all_indicators(df)
            ind1, vol1 = calculate_simplified_scores(df)
            ind2, vol2 = calculate_original_scores(df)
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            st.subheader(f"📈 {stock}")
            
            # Display if filtered
            if st.session_state.filter_chosen_only:
                st.markdown('<div class="filter-info">🔍 <b>Filtered View:</b> Showing only chosen stocks</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            change = latest['Close'] - prev['Close']
            pct = (change / prev['Close']) * 100
            with col1:
                st.metric("Price", f"₺{latest['Close']:.2f}", f"{change:.2f} ({pct:.2f}%)")
            with col2:
                st.metric("Volume", f"{latest['Volume']:,.0f}")
            with col3:
                st.metric("RSI", f"{latest['RSI']:.2f}")
            with col4:
                st.metric("MACD", f"{latest['MACD']:.4f}")
            
            st.subheader("🎯 Score Comparison")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Score 1 (Simplified)")
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(create_gauge(ind1, "Indicator 1"), use_container_width=True)
                with c2:
                    st.plotly_chart(create_gauge(vol1, "Volume 1"), use_container_width=True)
            with col2:
                st.markdown("#### Score 2 (Original) ⭐")
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(create_gauge(ind2, "Indicator 2", 10), use_container_width=True)
                with c2:
                    st.metric("Volume Score 2", f"{vol2:.2f}")
                    if vol2 > 0.7:
                        st.success("✅ Above 0.7")
                    else:
                        st.warning("⚠️ Below 0.7")
            
            if ind2 >= 3 and vol2 > 0.7:
                st.markdown('<div class="chosen-stock"><h3>⭐ THIS IS A CHOSEN STOCK! ⭐</h3><p>Meets criteria: Indicator Score 2 >= 3 AND Volume Score 2 > 0.7</p></div>', unsafe_allow_html=True)
            
            st.subheader("📊 Charts")
            st.plotly_chart(create_candlestick(df, stock), use_container_width=True)
            
            tab1, tab2, tab3 = st.tabs(["📊 Volume", "📈 MACD", "📉 RSI"])
            with tab1:
                st.plotly_chart(create_volume_chart(df), use_container_width=True)
            with tab2:
                st.plotly_chart(create_macd_chart(df), use_container_width=True)
            with tab3:
                st.plotly_chart(create_rsi_chart(df), use_container_width=True)
            
            with st.expander("📋 Raw Data"):
                st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD']].tail(50), use_container_width=True)
        else:
            st.error(f"❌ No data available for {stock}")
    else:
        st.subheader("🔍 Chosen Stocks - Stock Screener Results")
        if st.session_state.chosen_stocks:
            st.success(f"✅ Found {len(st.session_state.chosen_stocks)} chosen stocks")
            st.info("💡 **Tip:** Switch to Single Stock mode and enable 'Show Only Chosen Stocks' filter to analyze these stocks individually!")
            
            df_chosen = pd.DataFrame(st.session_state.chosen_stocks).sort_values('indicator_score_2', ascending=False)
            st.dataframe(
                df_chosen.style.background_gradient(subset=['indicator_score_2', 'volume_score_2'], cmap='RdYlGn'),
                use_container_width=True
            )
            
            cols = st.columns(3)
            for idx, s in enumerate(df_chosen.to_dict('records')):
                with cols[idx % 3]:
                    st.markdown(f"""
                    <div class="chosen-stock">
                        <h4>{s['symbol']}</h4>
                        <p><b>Indicator Score 2:</b> {s['indicator_score_2']}</p>
                        <p><b>Volume Score 2:</b> {s['volume_score_2']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("👆 Click 'Run Stock Screener' in the sidebar to find chosen stocks")
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: #7f8c8d;'>
            Data: borsapy (TradingView) | {'✅ Real-time' if auth else '⏱️ 15-min delayed'} | 
            Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

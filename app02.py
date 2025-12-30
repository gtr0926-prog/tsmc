import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 頁面基本設定 ---
st.set_page_config(page_title="2330 AI 戰略指揮所", layout="wide")

# 深色模式 UI 優化
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    [data-testid="stSidebar"] { background-color: #262730; }
    h1, h2, h3, h4, h5, p, span { color: #fafafa !important; }
    .stMetric { background-color: #1e201f; padding: 15px; border-radius: 12px; border: 1px solid #333; }
    div.stButton > button {
        width: 100%; border-radius: 8px; height: 3.5em;
        background-color: #ff4b4b; color: white; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 台積電 (2330) AI 戰略指揮所")
st.caption("數據來源：Yahoo Finance | 功能：十字準線追蹤 + 5% 獲利調節")

# --- 2. 側邊欄：資金與參數設定 ---
st.sidebar.header("💰 我的資金部位")
total_capital = st.sidebar.number_input("可用操作資金 (TWD)", min_value=0, value=1000000, step=10000)
current_shares = st.sidebar.number_input("持有股數 (1張=1000股)", min_value=0, value=0, step=1000)
avg_cost = st.sidebar.number_input("買進平均成本 (TWD)", min_value=1.0, value=1000.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("🛡️ 調節策略")
take_profit_threshold = st.sidebar.slider("獲利調節門檻 (%)", 1.0, 15.0, 5.0, help="當獲利超過此趴數且AI看空時，建議賣出")
adjust_ratio = st.sidebar.slider("調節賣出比例 (%)", 10, 100, 50)

if st.sidebar.button("🔄 重新載入數據"):
    st.cache_data.clear()

# --- 3. 數據抓取與特徵工程 ---
@st.cache_data(ttl=600) # 每 10 分鐘更新
def fetch_data():
    tickers = ["2330.TW", "TSM", "^SOX"]
    # progress=False 隱藏下載進度條
    df = yf.download(tickers, period="2y", interval="1d", progress=False)
    
    # 處理 Multi-Index 欄位 (yfinance 新版格式)
    try:
        # 嘗試直接存取，如果失敗則用 xs 或調整層級
        close = df['Close']['2330.TW']
        tsm = df['Close']['TSM']
        sox = df['Close']['^SOX']
        vol = df['Volume']['2330.TW']
    except:
        # 兼容性處理
        close = df.iloc[:, df.columns.get_level_values(1)=='2330.TW']['Close']
        tsm = df.iloc[:, df.columns.get_level_values(1)=='TSM']['Close']
        sox = df.iloc[:, df.columns.get_level_values(1)=='^SOX']['Close']
        vol = df.iloc[:, df.columns.get_level_values(1)=='2330.TW']['Volume']

    data = pd.DataFrame({
        'Close': close,
        'Volume': vol,
        'TSM_Close': tsm,
        'SOX_Close': sox
    }).dropna()
    
    # 技術指標
    data['MA5'] = data['Close'].rolling(5).mean()
    data['MA20'] = data['Close'].rolling(20).mean()
    data['ADR_Premium'] = (data['TSM_Close'] * 31 / 5) - data['Close']
    
    # 標籤
    data['Target_Class'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data['Target_Price'] = data['Close'].shift(-1)
    
    return data.dropna()

data = fetch_data()

# --- 4. 模型訓練 ---
features = ['Close', 'MA5', 'MA20', 'ADR_Premium', 'SOX_Close']
X = data[features]
y_cls = data['Target_Class']
y_reg = data['Target_Price']

clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y_cls)
reg = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_reg)

# --- 5. 決策計算 ---
latest_data = X.iloc[[-1]]
latest_price = data['Close'].iloc[-1]
pred_up = clf.predict(latest_data)[0]
pred_price = reg.predict(latest_data)[0]

current_profit_pct = ((latest_price - avg_cost) / avg_cost) * 100
target_exit_price = avg_cost * (1 + take_profit_threshold / 100)

decision = "觀望 / 續抱"
suggested_shares = 0
color = "#2962ff"
note = "AI 訊號中性，建議等待。"

if pred_up == 1:
    decision = "建議買進"
    color = "#00c853"
    suggested_shares = int((total_capital * 0.3) // latest_price)
    note = "AI 看好明日走勢，建議適量佈局。"
else:
    if current_shares > 0:
        if current_profit_pct >= take_profit_threshold:
            decision = f"獲利調節 (>{take_profit_threshold}%)"
            color = "#ff9100"
            suggested_shares = int(current_shares * (adjust_ratio / 100))
            note = f"獲利達 {current_profit_pct:.1f}% 且 AI 看空，建議入袋為安。"
        else:
            decision = "續抱 (未達門檻)"
            color = "#2962ff"
            note = f"獲利僅 {current_profit_pct:.1f}% 未達 {take_profit_threshold}%，避免頻繁交易。"

# --- 6. 儀表板 ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("當前股價", f"{latest_price:.1f}")
c2.metric("我的成本", f"{avg_cost:.1f}")
c3.metric("目前損益", f"{current_profit_pct:.2f}%", delta=f"{take_profit_threshold}% 目標")
c4.metric("AI 目標價", f"{pred_price:.1f}")

st.markdown(f"""
<div style="background-color: {color}; padding: 25px; border-radius: 15px; text-align: center; margin: 20px 0;">
    <h1 style="margin:0; font-size: 2.5em; color: white !important;">{decision}</h1>
    <h2 style="margin:10px 0; color: white !important;">建議操作股數：{suggested_shares:,} 股</h2>
    <p style="font-size: 1.1em; opacity: 0.9; color: white !important;">{note}</p>
</div>
""", unsafe_allow_html=True)

# --- 7. 走勢圖 (含十字線) ---
st.subheader("📊 戰略走勢圖 (含十字準線)")
plot_df = data.iloc[-100:]

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

# 主圖
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'], name='收盤價', line=dict(color='#2962ff', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=[avg_cost]*len(plot_df), name='成本線', line=dict(color='white', dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=[target_exit_price]*len(plot_df), name='調節門檻', line=dict(color='#ff9100', dash='dot')), row=1, col=1)

# 副圖 (成交量)
fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], name='成交量', marker_color='#555'), row=2, col=1)

# 【關鍵修改】開啟十字線與釘選模式
fig.update_layout(
    height=600, 
    template="plotly_dark", 
    margin=dict(l=10, r=10, t=20, b=10), 
    legend=dict(orientation="h", y=1.05),
    hovermode="x unified" # 開啟 X 軸統一顯示 (垂直線)
)

# 進階十字線設定 (讓線更明顯)
fig.update_xaxes(
    showspikes=True, # 顯示釘選線
    spikemode='across', # 線延伸到底
    spikesnap='cursor', # 對齊游標
    showline=True, 
    showgrid=True,
    spikecolor="white", # 十字線顏色
    spikethickness=1
)
fig.update_yaxes(
    showspikes=True, # Y軸也顯示
    spikemode='across',
    spikecolor="white",
    spikethickness=1
)

st.plotly_chart(fig, use_container_width=True)

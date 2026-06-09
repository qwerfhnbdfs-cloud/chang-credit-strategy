# -*- coding: utf-8 -*-
"""
长信用策略 - Streamlit 网页版
运行: streamlit run app.py
部署: Streamlit Cloud (streamlit.io/cloud)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from io import BytesIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="长信用策略可视化",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============ 核心计算模块（完全保留 server.py 逻辑）============

class StrategyCalculator:
    """核心策略计算类"""

    def __init__(self, excel_bytes):
        self.excel_bytes = excel_bytes
        self.wealth_df = None
        self.curve_df = None
        self.rrg_df = None
        self.price_df = None
        self.signal_df = None
        self.load_data()

    def load_data(self):
        """从Excel加载原始数据"""
        w_raw = pd.read_excel(BytesIO(self.excel_bytes), sheet_name='财富指数—趋势', header=None)
        self.wealth_df = pd.DataFrame({
            '日期': pd.to_datetime(w_raw.iloc[2:, 0].values),
            '财富1年': w_raw.iloc[2:, 1].astype(float).values,
            '财富3年': w_raw.iloc[2:, 2].astype(float).values,
            '财富5年': w_raw.iloc[2:, 3].astype(float).values,
            '财富7年': w_raw.iloc[2:, 4].astype(float).values,
            '财富10年': w_raw.iloc[2:, 5].astype(float).values,
            '财富10年+': w_raw.iloc[2:, 6].astype(float).values,
            '短端平均': w_raw.iloc[2:, 7].astype(float).values,
            '长端平均': w_raw.iloc[2:, 8].astype(float).values,
        })

        c_raw = pd.read_excel(BytesIO(self.excel_bytes), sheet_name='信用曲线—价格', header=None)
        self.curve_df = pd.DataFrame({
            '日期': pd.to_datetime(c_raw.iloc[2:, 0].values),
            '收益率3Y': c_raw.iloc[2:, 1].astype(float).values,
            '收益率5Y': c_raw.iloc[2:, 2].astype(float).values,
            '收益率7Y': c_raw.iloc[2:, 3].astype(float).values,
            '收益率10Y': c_raw.iloc[2:, 4].astype(float).values,
            '国开7Y': c_raw.iloc[2:, 5].astype(float).values,
            '国开10Y': c_raw.iloc[2:, 6].astype(float).values,
            '信用利差7Y': c_raw.iloc[2:, 7].astype(float).values,
            '信用利差10Y': c_raw.iloc[2:, 8].astype(float).values,
        })

    def calculate_rrg(self, ratio_ma=42, x_ma=21, momentum_days=14):
        df = self.wealth_df.copy()
        df['比值'] = df['长端平均'] / df['短端平均']
        df['比值MA'] = df['比值'].rolling(window=ratio_ma, min_periods=ratio_ma).mean()
        df['X轴'] = (df['比值'] - df['比值MA']) / df['比值MA'] * 100
        df[f'X轴{x_ma}日MA'] = df['X轴'].rolling(window=x_ma, min_periods=x_ma).mean()
        df['趋势方向'] = '下降'
        df.loc[df[f'X轴{x_ma}日MA'] > 0, '趋势方向'] = '上升'
        df.loc[df[f'X轴{x_ma}日MA'].isna(), '趋势方向'] = np.nan
        shift_days = max(1, momentum_days - 1)
        df['Y轴'] = df['X轴'] - df['X轴'].shift(shift_days)

        conditions = [
            (df['X轴'] > 0) & (df['Y轴'] > 0),
            (df['X轴'] > 0) & (df['Y轴'] <= 0),
            (df['X轴'] <= 0) & (df['Y轴'] > 0),
            (df['X轴'] <= 0) & (df['Y轴'] <= 0),
        ]
        df['四象限'] = '待定'
        df.loc[conditions[0], '四象限'] = '领先'
        df.loc[conditions[1], '四象限'] = '减弱'
        df.loc[conditions[2], '四象限'] = '改善'
        df.loc[conditions[3], '四象限'] = '滞后'
        df.loc[df[f'X轴{x_ma}日MA'].isna(), '四象限'] = np.nan
        self.rrg_df = df
        return df

    def calculate_price(self, bollinger_ma=60, bollinger_std=2.0,
                        percentile_window=252, cheap_threshold=0.55, expensive_threshold=0.15):
        df = self.curve_df.copy()
        df['收益率MA'] = df['收益率7Y'].rolling(window=bollinger_ma, min_periods=bollinger_ma).mean()
        df['收益率STD'] = df['收益率7Y'].rolling(window=bollinger_ma, min_periods=bollinger_ma).std()
        df['收益率上轨'] = df['收益率MA'] + bollinger_std * df['收益率STD']
        df['收益率下轨'] = df['收益率MA'] - bollinger_std * df['收益率STD']
        bw = df['收益率上轨'] - df['收益率下轨']
        df['收益率布林带位置'] = np.where(bw != 0, (df['收益率7Y'] - df['收益率下轨']) / bw * 100, np.nan)

        df['利差MA'] = df['信用利差7Y'].rolling(window=bollinger_ma, min_periods=bollinger_ma).mean()
        df['利差STD'] = df['信用利差7Y'].rolling(window=bollinger_ma, min_periods=bollinger_ma).std()
        df['利差上轨'] = df['利差MA'] + bollinger_std * df['利差STD']
        df['利差下轨'] = df['利差MA'] - bollinger_std * df['利差STD']
        bw2 = df['利差上轨'] - df['利差下轨']
        df['利差布林带位置'] = np.where(bw2 != 0, (df['信用利差7Y'] - df['利差下轨']) / bw2 * 100, np.nan)
        df['布林带平均位置'] = (df['收益率布林带位置'] + df['利差布林带位置']) / 2

        def percentrank_exc(series, window):
            result = np.full(len(series), np.nan)
            for i in range(window - 1, len(series)):
                hist = series.iloc[i - window + 1:i].values
                if len(hist) > 0 and not np.all(np.isnan(hist)):
                    count_less = np.sum(hist < series.iloc[i])
                    count_equal = np.sum(hist == series.iloc[i])
                    n = len(hist)
                    result[i] = (count_less + count_equal * 0.5) / (n + 1)
            return result

        df['收益率分位数'] = percentrank_exc(df['收益率7Y'], percentile_window)
        df['利差分位数'] = percentrank_exc(df['信用利差7Y'], percentile_window)
        df['综合分位数'] = (df['收益率分位数'] + df['利差分位数']) / 2

        conds = [
            df['综合分位数'] >= cheap_threshold,
            (df['综合分位数'] >= expensive_threshold) & (df['综合分位数'] < cheap_threshold),
            df['综合分位数'] < expensive_threshold,
        ]
        df['价格区间'] = np.select(conds, ['便宜', '中性', '贵'], default='待定')
        df.loc[df['综合分位数'].isna(), '价格区间'] = np.nan
        self.price_df = df
        return df

    def calculate_signals(self, confirm_days=3):
        df = pd.merge(
            self.rrg_df[['日期', '四象限', '趋势方向', 'Y轴', 'X轴']],
            self.price_df[['日期', '价格区间', '综合分位数', '收益率7Y']],
            on='日期', how='outer'
        )
        df = df.sort_values('日期').reset_index(drop=True)
        df['价格区间'] = df['价格区间'].fillna('待定')

        position = '空仓'
        df['确认后持仓'] = '空仓'
        df['交易动作'] = ''
        df['信号原因'] = ''

        for i in range(1, len(df)):
            prev = df.iloc[i - 1]
            curr = df.iloc[i]

            if (prev['四象限'] == '改善' and curr['四象限'] == '领先' and
                curr['价格区间'] in ['便宜', '中性', '待定'] and
                position == '空仓'):
                position = '持仓'
                df.loc[i, '交易动作'] = '买入'
                df.loc[i, '信号原因'] = '改善→领先,买入'

            elif (prev['四象限'] == '减弱' and curr['四象限'] == '滞后' and
                  position == '持仓'):
                position = '空仓'
                df.loc[i, '交易动作'] = '卖出'
                df.loc[i, '信号原因'] = '减弱→滞后,卖出'

            elif (prev['趋势方向'] == '上升' and curr['趋势方向'] == '下降' and
                  position == '持仓'):
                position = '空仓'
                df.loc[i, '交易动作'] = '卖出'
                df.loc[i, '信号原因'] = '趋势转降,卖出'

            df.loc[i, '确认后持仓'] = position

        self.signal_df = df
        return df

    def get_backtest_stats(self):
        df = self.signal_df.copy()
        trades = df[df['交易动作'].isin(['买入', '卖出'])]
        hold = len(df[df['确认后持仓'] == '持仓'])
        empty = len(df[df['确认后持仓'] == '空仓'])
        coverage = hold / len(df) * 100

        df['日收益'] = -df['收益率7Y'].diff() / 100
        df['策略日收益'] = np.where(df['确认后持仓'] == '持仓', df['日收益'].fillna(0), 0)
        df['策略累计'] = (1 + df['策略日收益']).cumprod() - 1
        df['买入持有累计'] = (1 + df['日收益'].fillna(0)).cumprod() - 1

        latest = df.iloc[-1]
        return {
            '总天数': len(df),
            '交易次数': len(trades),
            '持仓天数': hold,
            '空仓天数': empty,
            '覆盖率': round(coverage, 1),
            '策略收益': round(df['策略累计'].iloc[-1] * 100, 2),
            '买入持有收益': round(df['买入持有累计'].iloc[-1] * 100, 2),
            '超额收益': round((df['策略累计'].iloc[-1] - df['买入持有累计'].iloc[-1]) * 100, 2),
            '最新信号': latest['确认后持仓'],
            '最新四象限': latest['四象限'],
            '最新价格区间': latest['价格区间'],
            '最新趋势': latest['趋势方向'],
            '最新综合分位数': round(latest['综合分位数'] * 100, 2) if pd.notna(latest['综合分位数']) else 'N/A',
            '最新收益率': round(latest['收益率7Y'], 4),
        }

    def run_full(self, params):
        """执行完整计算并返回结果"""
        rrg = self.calculate_rrg(params['ratio_ma'], params['x_ma'], params['momentum_days'])
        price = self.calculate_price(params['bollinger_ma'], params['bollinger_std'],
                                     params['percentile_window'], params['cheap_threshold'] / 100,
                                     params['expensive_threshold'] / 100)
        signal = self.calculate_signals(params.get('confirm_days', 3))
        stats = self.get_backtest_stats()
        return rrg, price, signal, stats


# ============ Streamlit UI ============

st.title("📊 长信用择时策略—基于趋势和价格双维信号")

# ============ 数据加载配置 ============
# GitHub Raw URL（部署后改成你的实际链接）
# 格式: https://raw.githubusercontent.com/用户名/仓库名/分支名/文件名.xlsx
DATA_URL = "https://raw.githubusercontent.com/qwerfhnbdfs-cloud/chang-credit-strategy/main/中债财富指数对应的期限轮动策略V1.xlsx"

# --- 侧边栏参数 ---
st.sidebar.header("⚙️ 参数设置")

st.sidebar.subheader("📈 RRG趋势参数")
ratio_ma = st.sidebar.slider("比值MA周期", 5, 120, 42)
x_ma = st.sidebar.slider("X轴MA周期", 5, 60, 21)
momentum_days = st.sidebar.slider("动量周期", 5, 60, 14)

st.sidebar.subheader("💰 价格指标参数")
bollinger_ma = st.sidebar.slider("布林带MA", 10, 120, 60)
bollinger_std = st.sidebar.slider("布林带倍数", 0.5, 4.0, 2.0, 0.1)
percentile_window = st.sidebar.slider("PERCENTRANK窗口", 60, 500, 252)
cheap_threshold = st.sidebar.slider("便宜阈值(%)", 30, 80, 55)
expensive_threshold = st.sidebar.slider("贵阈值(%)", 5, 40, 15)

params = {
    'ratio_ma': ratio_ma,
    'x_ma': x_ma,
    'momentum_days': momentum_days,
    'bollinger_ma': bollinger_ma,
    'bollinger_std': bollinger_std,
    'percentile_window': percentile_window,
    'cheap_threshold': cheap_threshold,
    'expensive_threshold': expensive_threshold,
    'confirm_days': 3,
}

# ============ 数据加载 ============
@st.cache_data(ttl=300, show_spinner=False)
def load_data_from_url(url):
    """从URL下载Excel数据（缓存5分钟）"""
    # 添加时间戳参数绕过CDN缓存
    url_with_ts = f"{url}?t={int(time.time() // 300)}"
    resp = requests.get(url_with_ts, timeout=30)
    resp.raise_for_status()
    return StrategyCalculator(resp.content)

def load_data_from_upload(file):
    """从上传文件加载数据"""
    return StrategyCalculator(file.read())

st.sidebar.header("📁 数据文件")

# 手动上传（覆盖模式）
uploaded_file = st.sidebar.file_uploader(
    "📤 临时上传覆盖（可选）",
    type=["xlsx", "xls"],
    help="留空则自动从GitHub读取最新数据；上传则临时用上传的文件"
)

calc = None
data_source = ""

# 优先使用手动上传的文件
if uploaded_file is not None:
    try:
        calc = load_data_from_upload(uploaded_file)
        data_source = f"手动上传: {uploaded_file.name}"
        st.sidebar.success(f"✅ 已加载上传文件: {uploaded_file.name}")
    except Exception as e:
        st.sidebar.error(f"❌ 上传文件加载失败: {e}")

# 如果没有手动上传，尝试从URL自动加载
if calc is None:
    if "你的用户名" not in DATA_URL:
        try:
            with st.spinner("🔄 正在从GitHub加载最新数据..."):
                calc = load_data_from_url(DATA_URL)
            data_source = "GitHub 自动同步"
            st.sidebar.success("✅ 已从GitHub加载最新数据")
        except Exception as e:
            st.sidebar.error(f"❌ GitHub数据加载失败: {e}")
            st.sidebar.info("💡 请检查 DATA_URL 配置是否正确，或临时上传文件")
    else:
        st.sidebar.warning("⚠️ 请先在代码中设置 DATA_URL")

if calc is None:
    st.info("👆 请在左侧上传Excel数据文件，或配置 GitHub 数据源")
    st.stop()

st.sidebar.markdown(f"<small>📡 数据来源: {data_source}</small>", unsafe_allow_html=True)

# --- 执行计算 ---
with st.spinner("🔄 正在计算..."):
    rrg, price, signal, stats = calc.run_full(params)

# --- 更新时间 ---
latest_date_str = signal.iloc[-1]['日期'].strftime('%Y/%m/%d') if pd.notna(signal.iloc[-1]['日期']) else 'N/A'
st.markdown(f"""
<div style="text-align:right;font-size:13px;color:#888;margin-bottom:4px;">
    更新时间：{latest_date_str}
</div>
""", unsafe_allow_html=True)

# --- 顶部统计栏 ---
s = stats
st.markdown("---")
cols = st.columns(6)
metrics = [
    ("最新信号", f"{'🟢' if s['最新信号'] == '持仓' else '🔴'} {s['最新信号']}"),
    ("四象限", s['最新四象限']),
    ("趋势", s['最新趋势']),
    ("价格区间", s['最新价格区间']),
    ("综合分位", f"{s['最新综合分位数']}%"),
    ("策略收益", f"{s['策略收益']}%"),
]
for col, (label, value) in zip(cols, metrics):
    col.markdown(f"""
    <div style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:12px;border-radius:10px;text-align:center;">
        <div style="font-size:12px;opacity:0.85;">{label}</div>
        <div style="font-size:18px;font-weight:700;margin-top:4px;">{value}</div>
    </div>
    """, unsafe_allow_html=True)
st.markdown("---")

# --- Tab页面 ---
tab_rrg, tab_price, tab_signal, tab_backtest, tab_rules = st.tabs([
    "📈 RRG趋势", "💰 价格指标", "📡 信号判断", "📊 回测统计", "📋 信号规则"
])

# ==================== Tab 1: RRG趋势 ====================
with tab_rrg:
    rrg_plot = rrg.copy()
    rrg_plot['日期str'] = rrg_plot['日期'].dt.strftime('%Y-%m-%d')
    dates = rrg_plot['日期str'].tolist()
    xma_col = f'X轴{x_ma}日MA'

    c1, c2 = st.columns(2)

    with c1:
        # 比值走势
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=dates, y=rrg_plot['比值'], mode='lines', name='比值',
                                   line=dict(width=1, color='#3498db')))
        fig1.add_trace(go.Scatter(x=dates, y=rrg_plot['比值MA'], mode='lines', name='比值MA',
                                   line=dict(width=1.5, dash='dash', color='#e74c3c')))
        fig1.update_layout(title="长端/短端财富指数比值走势", xaxis_tickangle=-45,
                           height=340, margin=dict(l=40, r=20, t=40, b=60),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig1, use_container_width=True, key="rrg_ratio")

        # X轴MA趋势判断
        fig3 = go.Figure()
        xma_data = rrg_plot[xma_col].values
        fig3.add_trace(go.Scatter(x=dates, y=xma_data, mode='lines', name=xma_col,
                                   line=dict(width=1.5, color='#667eea')))
        fig3.add_hline(y=0, line_dash="dash", line_color="#e74c3c", line_width=1.5)
        fig3.add_trace(go.Scatter(
            x=dates, y=np.where(xma_data > 0, xma_data, 0),
            fill='tozeroy', fillcolor='rgba(46,204,113,0.2)', mode='none', name='上升趋势区'
        ))
        fig3.add_trace(go.Scatter(
            x=dates, y=np.where(xma_data <= 0, xma_data, 0),
            fill='tozeroy', fillcolor='rgba(231,76,60,0.2)', mode='none', name='下降趋势区'
        ))
        fig3.update_layout(title=f"{xma_col}趋势判断（>0上升，≤0下降）", xaxis_tickangle=-45,
                           height=340, margin=dict(l=40, r=20, t=40, b=60),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig3, use_container_width=True, key="rrg_xma")

    with c2:
        # X轴与Y轴
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=dates, y=rrg_plot['X轴'], mode='lines', name='X轴(相对强度)',
                                   line=dict(width=1.2, color='#3498db')))
        fig2.add_trace(go.Scatter(x=dates, y=rrg_plot['Y轴'], mode='lines', name='Y轴(动量)',
                                   line=dict(width=1.2, color='#e74c3c')))
        fig2.add_hline(y=0, line_dash="dash", line_color="#999", line_width=1)
        fig2.update_layout(title="X轴相对强度与Y轴动量", xaxis_tickangle=-45,
                           height=340, margin=dict(l=40, r=20, t=40, b=60),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig2, use_container_width=True, key="rrg_xy")

        # RRG旋转图
        valid = rrg_plot.dropna(subset=['X轴', 'Y轴']).copy()
        quad_colors = {'领先': '#2ecc71', '减弱': '#f39c12', '改善': '#3498db', '滞后': '#e74c3c'}
        fig4 = go.Figure()

        for q in ['领先', '减弱', '改善', '滞后']:
            qd = valid[valid['四象限'] == q]
            if len(qd) > 0:
                fig4.add_trace(go.Scatter(
                    x=qd['X轴'], y=qd['Y轴'], mode='markers',
                    marker=dict(size=5, color=quad_colors.get(q, '#999')),
                    name=q, opacity=0.7
                ))

        if len(valid) > 0:
            last = valid.iloc[-1]
            fig4.add_trace(go.Scatter(
                x=[last['X轴']], y=[last['Y轴']], mode='markers+text',
                marker=dict(size=18, color='#f1c40f', line=dict(color='#e74c3c', width=2)),
                text=[last['日期str']], textposition="top center",
                textfont=dict(size=11, color='#e74c3c'),
                name='当前位置'
            ))
            fig4.add_trace(go.Scatter(
                x=valid['X轴'], y=valid['Y轴'], mode='lines',
                line=dict(width=0.5, color='#ccc', dash='dot'),
                name='轨迹', showlegend=False
            ))

        fig4.add_hline(y=0, line_dash="solid", line_color="#999")
        fig4.add_vline(x=0, line_dash="solid", line_color="#999")
        fig4.update_layout(title="RRG旋转图 (X轴 vs Y轴)", height=340,
                           xaxis_title="X轴(相对强度)", yaxis_title="Y轴(动量)",
                           margin=dict(l=40, r=20, t=40, b=40),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig4, use_container_width=True, key="rrg_scatter")


# ==================== Tab 2: 价格指标 ====================
with tab_price:
    price_plot = price.copy()
    price_plot['日期str'] = price_plot['日期'].dt.strftime('%Y-%m-%d')
    dates_p = price_plot['日期str'].tolist()

    c1, c2 = st.columns(2)

    with c1:
        # 收益率布林带
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=dates_p, y=price_plot['收益率7Y'], mode='lines', name='7Y收益率',
                                   line=dict(width=1.2, color='#3498db')))
        fig5.add_trace(go.Scatter(x=dates_p, y=price_plot['收益率上轨'], mode='lines', name='上轨',
                                   line=dict(width=0.8, dash='dash', color='#95a5a6')))
        fig5.add_trace(go.Scatter(x=dates_p, y=price_plot['收益率下轨'], mode='lines', name='下轨',
                                   line=dict(width=0.8, dash='dash', color='#95a5a6')))
        fig5.update_layout(title="7Y AA+收益率与布林带", xaxis_tickangle=-45,
                           height=340, margin=dict(l=40, r=20, t=40, b=60),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig5, use_container_width=True, key="price_yield")

        # 综合分位数
        fig7 = go.Figure()
        pct_vals = price_plot['综合分位数'] * 100
        fig7.add_trace(go.Scatter(x=dates_p, y=pct_vals, mode='lines', name='综合分位数%',
                                   line=dict(width=1.5, color='#667eea')))
        fig7.add_hline(y=cheap_threshold, line_dash="dash", line_color="#27ae60",
                        annotation_text=f"便宜线({cheap_threshold}%)", annotation_position="right")
        fig7.add_hline(y=expensive_threshold, line_dash="dash", line_color="#e74c3c",
                        annotation_text=f"贵线({expensive_threshold}%)", annotation_position="right")
        fig7.update_layout(title="综合估值分位数", xaxis_tickangle=-45, yaxis_range=[0, 100],
                           height=340, margin=dict(l=40, r=20, t=40, b=60),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig7, use_container_width=True, key="price_pct")

    with c2:
        # 信用利差布林带
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=dates_p, y=price_plot['信用利差7Y'], mode='lines', name='信用利差',
                                   line=dict(width=1.2, color='#9b59b6')))
        fig6.add_trace(go.Scatter(x=dates_p, y=price_plot['利差上轨'], mode='lines', name='上轨',
                                   line=dict(width=0.8, dash='dash', color='#95a5a6')))
        fig6.add_trace(go.Scatter(x=dates_p, y=price_plot['利差下轨'], mode='lines', name='下轨',
                                   line=dict(width=0.8, dash='dash', color='#95a5a6')))
        fig6.update_layout(title="7Y AA+信用利差与布林带", xaxis_tickangle=-45,
                           height=340, margin=dict(l=40, r=20, t=40, b=60),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig6, use_container_width=True, key="price_spread")


# ==================== Tab 3: 信号判断 ====================
with tab_signal:
    sig_plot = signal.copy()
    sig_plot['日期str'] = sig_plot['日期'].dt.strftime('%Y-%m-%d')
    dates_s = sig_plot['日期str'].tolist()

    c1, c2 = st.columns(2)

    with c1:
        # 持仓状态时间线
        fig8 = go.Figure()
        hold_mask = sig_plot['确认后持仓'] == '持仓'
        empty_mask = sig_plot['确认后持仓'] == '空仓'
        fig8.add_trace(go.Scatter(
            x=dates_s, y=hold_mask.astype(int),
            fill='tozeroy', fillcolor='rgba(46,204,113,0.5)', mode='none', name='持仓'
        ))
        fig8.add_trace(go.Scatter(
            x=dates_s, y=empty_mask.astype(int),
            fill='tozeroy', fillcolor='rgba(231,76,60,0.5)', mode='none', name='空仓'
        ))
        fig8.update_layout(title="持仓状态时间线", xaxis_tickangle=-45,
                           height=340, yaxis=dict(tickvals=[0, 1], ticktext=['空仓', '持仓']),
                           margin=dict(l=50, r=20, t=40, b=60),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig8, use_container_width=True, key="sig_position")

        # X轴 vs Y轴联合走势
        fig10 = go.Figure()
        fig10.add_trace(go.Scatter(x=dates_s, y=sig_plot['X轴'], mode='lines', name='X轴',
                                    line=dict(width=1, color='#3498db')))
        fig10.add_trace(go.Scatter(x=dates_s, y=sig_plot['Y轴'], mode='lines', name='Y轴',
                                    line=dict(width=1, color='#e74c3c')))
        fig10.add_hline(y=0, line_dash="dash", line_color="#999")
        fig10.update_layout(title="X轴 vs Y轴联合走势", xaxis_tickangle=-45,
                           height=340, margin=dict(l=40, r=20, t=40, b=60),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig10, use_container_width=True, key="sig_xy")

    with c2:
        # 交易点位
        fig9 = go.Figure()
        fig9.add_trace(go.Scatter(x=dates_s, y=sig_plot['收益率7Y'], mode='lines', name='7Y收益率',
                                   line=dict(width=1, color='#3498db')))
        buy_pts = sig_plot[sig_plot['交易动作'] == '买入']
        sell_pts = sig_plot[sig_plot['交易动作'] == '卖出']
        if len(buy_pts) > 0:
            fig9.add_trace(go.Scatter(
                x=buy_pts['日期str'], y=buy_pts['收益率7Y'], mode='markers+text',
                marker=dict(size=14, color='#27ae60', symbol='triangle-up'),
                text=['买'] * len(buy_pts), textposition="top center",
                textfont=dict(size=10, color='#27ae60'),
                name='买入'
            ))
        if len(sell_pts) > 0:
            fig9.add_trace(go.Scatter(
                x=sell_pts['日期str'], y=sell_pts['收益率7Y'], mode='markers+text',
                marker=dict(size=14, color='#e74c3c', symbol='triangle-down'),
                text=['卖'] * len(sell_pts), textposition="bottom center",
                textfont=dict(size=10, color='#e74c3c'),
                name='卖出'
            ))
        fig9.update_layout(title="7Y收益率与交易点位", xaxis_tickangle=-45,
                           height=340, margin=dict(l=40, r=20, t=40, b=60),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig9, use_container_width=True, key="sig_trades")

        # 四象限时间分布
        q_map = {'领先': 3, '改善': 2, '减弱': 1, '滞后': 0}
        q_colors_map = {'领先': '#2ecc71', '改善': '#3498db', '减弱': '#f39c12', '滞后': '#e74c3c'}
        fig11 = go.Figure()
        for q, val in q_map.items():
            qd = sig_plot[sig_plot['四象限'] == q]
            if len(qd) > 0:
                fig11.add_trace(go.Scatter(
                    x=qd['日期str'], y=[val] * len(qd), mode='markers',
                    marker=dict(size=4, color=q_colors_map[q]), name=q
                ))
        # 买入/卖出触发标记
        buy_trig = []
        sell_trig = []
        for i in range(1, len(sig_plot)):
            prev = sig_plot.iloc[i - 1]
            curr = sig_plot.iloc[i]
            if prev['四象限'] == '改善' and curr['四象限'] == '领先':
                buy_trig.append(i)
            if prev['四象限'] == '减弱' and curr['四象限'] == '滞后':
                sell_trig.append(i)
            if prev['趋势方向'] == '上升' and curr['趋势方向'] == '下降':
                sell_trig.append(i)
        if buy_trig:
            fig11.add_trace(go.Scatter(
                x=[sig_plot.iloc[i]['日期str'] for i in buy_trig],
                y=[3.5] * len(buy_trig), mode='markers',
                marker=dict(size=10, color='#27ae60', symbol='triangle-up'),
                name='买入触发'
            ))
        if sell_trig:
            fig11.add_trace(go.Scatter(
                x=[sig_plot.iloc[i]['日期str'] for i in sell_trig],
                y=[-0.5] * len(sell_trig), mode='markers',
                marker=dict(size=10, color='#e74c3c', symbol='triangle-down'),
                name='卖出触发'
            ))
        fig11.update_layout(title="四象限时间分布", xaxis_tickangle=-45,
                           yaxis=dict(tickvals=[0, 1, 2, 3], ticktext=['滞后', '减弱', '改善', '领先'],
                                      range=[-0.8, 3.8]),
                           height=340, margin=dict(l=50, r=20, t=40, b=60),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig11, use_container_width=True, key="sig_quad")


# ==================== Tab 4: 回测统计 ====================
with tab_backtest:
    sig_bt = signal.copy()
    sig_bt['日收益'] = -sig_bt['收益率7Y'].diff() / 100
    sig_bt['策略日收益'] = np.where(sig_bt['确认后持仓'] == '持仓', sig_bt['日收益'].fillna(0), 0)
    sig_bt['策略累计'] = (1 + sig_bt['策略日收益']).cumprod() - 1
    sig_bt['买入持有累计'] = (1 + sig_bt['日收益'].fillna(0)).cumprod() - 1
    sig_bt['日期str'] = sig_bt['日期'].dt.strftime('%Y-%m-%d')
    dates_bt = sig_bt['日期str'].tolist()

    c1, c2 = st.columns([2, 1])

    with c1:
        # 累计收益对比
        fig12 = go.Figure()
        fig12.add_trace(go.Scatter(
            x=dates_bt, y=sig_bt['策略累计'] * 100,
            mode='lines', name='策略累计收益%',
            line=dict(width=1.5, color='#3498db')
        ))
        fig12.add_trace(go.Scatter(
            x=dates_bt, y=sig_bt['买入持有累计'] * 100,
            mode='lines', name='买入持有累计收益%',
            line=dict(width=1, color='#95a5a6')
        ))
        fig12.update_layout(title="累计收益对比", xaxis_tickangle=-45,
                           yaxis_title="收益率%", height=380,
                           margin=dict(l=50, r=20, t=40, b=60),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig12, use_container_width=True, key="bt_cum")

    with c2:
        # 持仓分布饼图
        hold_cnt = len(sig_bt[sig_bt['确认后持仓'] == '持仓'])
        empty_cnt = len(sig_bt[sig_bt['确认后持仓'] == '空仓'])
        fig13 = go.Figure(data=[go.Pie(
            labels=['持仓', '空仓'], values=[hold_cnt, empty_cnt],
            hole=0.4, marker_colors=['#2ecc71', '#e74c3c'],
            textinfo='label+value+percent'
        )])
        fig13.update_layout(title="持仓分布", height=380,
                           margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig13, use_container_width=True, key="bt_pie")

    # 年度收益
    st.subheader("年度收益对比")
    sig_bt['年份'] = sig_bt['日期str'].str[:4]
    yearly = []
    for y, g in sig_bt.groupby('年份'):
        if len(g) <= 10:
            continue
        g = g.sort_values('日期').reset_index(drop=True)
        strat_ret = 1
        bh_ret = 1
        for i in range(1, len(g)):
            daily = -(g.iloc[i]['收益率7Y'] - g.iloc[i - 1]['收益率7Y']) / 100
            if g.iloc[i]['确认后持仓'] == '持仓':
                strat_ret *= (1 + daily)
            bh_ret *= (1 + daily)
        yearly.append({'年份': y, '策略': round((strat_ret - 1) * 100, 2),
                       '买入持有': round((bh_ret - 1) * 100, 2)})
    if yearly:
        df_yearly = pd.DataFrame(yearly)
        fig14 = go.Figure()
        fig14.add_trace(go.Bar(x=df_yearly['年份'], y=df_yearly['策略'], name='策略',
                                marker_color='#3498db'))
        fig14.add_trace(go.Bar(x=df_yearly['年份'], y=df_yearly['买入持有'], name='买入持有',
                                marker_color='#95a5a6'))
        fig14.update_layout(barmode='group', height=300,
                           margin=dict(l=50, r=20, t=30, b=40),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig14, use_container_width=True, key="bt_yearly")

    # 交易记录
    st.subheader("交易记录")
    trades = signal[signal['交易动作'] != ''][['日期', '交易动作', '信号原因', '收益率7Y', '四象限', '价格区间']].copy()
    trades['日期'] = trades['日期'].dt.strftime('%Y-%m-%d')
    trades['收益率7Y'] = trades['收益率7Y'].round(4)
    trades_display = trades.rename(columns={
        '日期': '日期', '交易动作': '动作', '信号原因': '原因',
        '收益率7Y': '7Y收益率', '四象限': '四象限', '价格区间': '价格区间'
    })
    st.dataframe(trades_display, use_container_width=True, height=300)

    # 回测统计卡片
    st.subheader("回测统计")
    sc = stats
    scols = st.columns(5)
    sc_items = [
        ("总天数", sc['总天数']),
        ("交易次数", sc['交易次数']),
        ("持仓天数", sc['持仓天数']),
        ("空仓天数", sc['空仓天数']),
        ("覆盖率", f"{sc['覆盖率']}%"),
    ]
    for scol, (sl, sv) in zip(scols, sc_items):
        scol.metric(sl, sv)
    scols2 = st.columns(3)
    scols2[0].metric("策略收益", f"{sc['策略收益']}%")
    scols2[1].metric("买入持有收益", f"{sc['买入持有收益']}%")
    scols2[2].metric("超额收益", f"{sc['超额收益']}%")


# ==================== Tab 5: 信号规则 ====================
with tab_rules:
    latest = signal.iloc[-1]
    latest_date = latest['日期'].strftime('%Y-%m-%d') if pd.notna(latest['日期']) else 'N/A'

    # ===== 策略原理说明（新增）=====
    st.subheader("📖 策略原理")

    with st.expander("点击展开：了解策略背后的逻辑", expanded=True):
        st.markdown("""
        <div style="font-size:14px;line-height:1.8;color:#333;">
        <p>本策略采用<b>双维度信号系统</b>，通过<b>趋势维度（RRG旋转图）</b>和<b>价格维度（估值分位数）</b>共同判断买卖时机。</p>
        </div>
        """, unsafe_allow_html=True)

        # X轴和Y轴计算方式
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div style="background:#f0f7ff;border-radius:10px;padding:16px;border-left:4px solid #3498db;">
                <h4 style="color:#3498db;margin-top:0;">📐 X轴 — 相对强度</h4>
                <div style="font-size:13px;line-height:1.8;">
                    <p><b>计算方式：</b></p>
                    <ol>
                        <li>比值 = 长端平均财富指数 ÷ 短端平均财富指数</li>
                        <li>比值MA = 比值的移动平均值</li>
                        <li><b>X轴 = (比值 - 比值MA) ÷ 比值MA × 100</b></li>
                    </ol>
                    <p><b>含义：</b></p>
                    <ul>
                        <li>X轴 > 0：长端表现<strong style="color:#27ae60;">强于</strong>短端（相对强度为正）</li>
                        <li>X轴 ≤ 0：长端表现<strong style="color:#e74c3c;">弱于</strong>短端（相对强度为负）</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div style="background:#fff5f0;border-radius:10px;padding:16px;border-left:4px solid #e74c3c;">
                <h4 style="color:#e74c3c;margin-top:0;">📐 Y轴 — 动量</h4>
                <div style="font-size:13px;line-height:1.8;">
                    <p><b>计算方式：</b></p>
                    <ol>
                        <li>取X轴当前值</li>
                        <li>取X轴N日前值（动量周期，默认14日）</li>
                        <li><b>Y轴 = X轴当前值 - X轴N日前值</b></li>
                    </ol>
                    <p><b>含义：</b></p>
                    <ul>
                        <li>Y轴 > 0：相对强度在<strong style="color:#27ae60;">增加</strong>（动量向上）</li>
                        <li>Y轴 ≤ 0：相对强度在<strong style="color:#e74c3c;">减弱</strong>（动量向下）</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 四象限矩阵图
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#667eea;'>🎯 四象限含义矩阵</h4>", unsafe_allow_html=True)

        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;margin:16px 0;">
            <!-- X轴列标题 -->
            <div style="display:flex;width:100%;max-width:600px;margin-left:80px;">
                <div style="flex:1;text-align:center;padding:6px;font-size:13px;font-weight:bold;color:#e74c3c;background:#ffebee;border-radius:6px 6px 0 0;margin-right:4px;">X轴 ≤ 0<br><small>相对强度为负</small></div>
                <div style="flex:1;text-align:center;padding:6px;font-size:13px;font-weight:bold;color:#27ae60;background:#e8f5e9;border-radius:6px 6px 0 0;">X轴 > 0<br><small>相对强度为正</small></div>
            </div>
            <!-- 第一行：Y轴 > 0 -->
            <div style="display:flex;width:100%;max-width:600px;">
                <div style="width:76px;display:flex;align-items:center;justify-content:center;padding:8px;background:#e8f5e9;font-weight:bold;font-size:12px;color:#2e7d32;border-radius:6px 0 0 0;margin-right:4px;">Y轴 > 0<br>动量向上</div>
                <div style="flex:1;padding:14px;background:#d6eaf8;text-align:center;border:2px solid #3498db;border-right:1px solid #3498db;border-bottom:1px solid #3498db;">
                    <div style="font-size:18px;font-weight:bold;color:#3498db;">🔵 改善</div>
                    <div style="font-size:12px;color:#333;margin-top:6px;">相对强度为负<br>但动量向上<br><b style="color:#3498db;">→ 关注区域</b></div>
                </div>
                <div style="flex:1;padding:14px;background:#d5f5e3;text-align:center;border:2px solid #27ae60;border-bottom:1px solid #27ae60;border-radius:0 6px 0 0;">
                    <div style="font-size:18px;font-weight:bold;color:#27ae60;">🟢 领先</div>
                    <div style="font-size:12px;color:#333;margin-top:6px;">相对强度为正<br>且动量向上<br><b style="color:#27ae60;">→ 最佳买入区域</b></div>
                </div>
            </div>
            <!-- 第二行：Y轴 ≤ 0 -->
            <div style="display:flex;width:100%;max-width:600px;">
                <div style="width:76px;display:flex;align-items:center;justify-content:center;padding:8px;background:#ffebee;font-weight:bold;font-size:12px;color:#c62828;border-radius:0 0 0 6px;margin-right:4px;">Y轴 ≤ 0<br>动量向下</div>
                <div style="flex:1;padding:14px;background:#fadbd8;text-align:center;border:2px solid #e74c3c;border-right:1px solid #e74c3c;border-top:1px solid #e74c3c;border-radius:0 0 0 6px;">
                    <div style="font-size:18px;font-weight:bold;color:#e74c3c;">🔴 滞后</div>
                    <div style="font-size:12px;color:#333;margin-top:6px;">相对强度为负<br>且动量向下<br><b style="color:#e74c3c;">→ 卖出/避险区域</b></div>
                </div>
                <div style="flex:1;padding:14px;background:#fdebd0;text-align:center;border:2px solid #f39c12;border-top:1px solid #f39c12;">
                    <div style="font-size:18px;font-weight:bold;color:#f39c12;">🟡 减弱</div>
                    <div style="font-size:12px;color:#333;margin-top:6px;">相对强度为正<br>但动量向下<br><b style="color:#f39c12;">→ 警惕区域</b></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size:13px;color:#666;margin-top:4px;text-align:center;">
            <b>趋势判断：</b>X轴MA > 0 为上升趋势，X轴MA ≤ 0 为下降趋势（与四象限独立判断）
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ===== 原有内容保留 =====
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style="background:#d5f5e3;border-radius:10px;padding:20px;border-left:5px solid #27ae60;">
            <h3 style="color:#27ae60;">📗 买入信号</h3>
            <div style="font-size:14px;line-height:1.8;">
                <p><b>触发条件：</b></p>
                <ul>
                    <li>四象限从 <b>改善</b> → <b>领先</b></li>
                    <li>且价格区间为 <b>便宜</b> 或 <b>中性</b></li>
                </ul>
                <p><b>执行：</b>状态转换当日立即执行买入</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style="background:#fadbd8;border-radius:10px;padding:20px;border-left:5px solid #e74c3c;">
            <h3 style="color:#e74c3c;">📕 卖出信号（满足任一）</h3>
            <div style="font-size:14px;line-height:1.8;">
                <p><b>条件1：四象限转换</b></p>
                <ul>
                    <li>四象限从 <b>减弱</b> → <b>滞后</b></li>
                </ul>
                <p><b>条件2：趋势反转</b></p>
                <ul>
                    <li>趋势从 <b>上升</b> → <b>下降</b></li>
                </ul>
                <p><b>执行：</b>状态转换当日立即执行卖出</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    sig_color = "#2ecc71" if latest['确认后持仓'] == '持仓' else "#fadbd8"
    st.markdown(f"""
    <div style="padding:20px;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);border-radius:10px;text-align:center;color:white;">
        <div style="font-size:13px;opacity:0.9;margin-bottom:8px;">当前状态</div>
        <div style="font-size:18px;font-weight:600;">
            {latest_date} | 四象限: <b>{latest['四象限']}</b> | 价格: <b>{latest['价格区间']}</b> | 趋势: <b>{latest['趋势方向']}</b>
        </div>
        <div style="font-size:26px;font-weight:700;margin-top:10px;">
            信号: <span style="color:{sig_color};">{latest['确认后持仓']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="padding:20px;background:#fff;border-radius:10px;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
        <h3 style="font-size:14px;color:#667eea;">📊 信号流程图</h3>
        <div style="display:flex;align-items:center;justify-content:center;gap:8px;flex-wrap:wrap;font-size:13px;margin-top:12px;">
            <div style="padding:10px 16px;background:#3498db;color:#fff;border-radius:8px;">观察状态</div>
            <div style="color:#999;">→</div>
            <div style="padding:10px 16px;background:#f39c12;color:#fff;border-radius:8px;">检测转换</div>
            <div style="color:#999;">→</div>
            <div style="padding:10px 16px;background:#9b59b6;color:#fff;border-radius:8px;">当日立即执行</div>
            <div style="color:#999;">→</div>
            <div style="padding:10px 16px;background:#27ae60;color:#fff;border-radius:8px;">执行交易</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

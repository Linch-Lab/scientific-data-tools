import streamlit as st
import plotly.graph_objects as go
import numpy as np

# --- 頁面設定 ---
st.set_page_config(page_title="電化學電位換算器", layout="wide")

st.title("🧪 電化學電位換算 App")
st.markdown("本工具用於計算不同參比電極之間的電位換算，並提供直觀的相對位置示意。")

# --- 常數定義 (25°C 下相對於 SHE 的電位) ---
# 參考來源: Bard, A. J., & Faulkner, L. R. (2001). Electrochemical Methods.
AG_AGCL_POTENTIALS = {
    "Saturated KCl (0.197V)": 0.197,
    "3.5M KCl (0.205V)": 0.205,
    "3.0M KCl (0.210V)": 0.210,
    "1.0M KCl (0.235V)": 0.235
}

# --- 側邊欄輸入 ---
st.sidebar.header("參數設定")
ph = st.sidebar.number_input("pH 值", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
temp_c = st.sidebar.number_input("溫度 (°C)", value=25.0)
input_val = st.sidebar.number_input("輸入電位值 (V)", value=0.0, format="%.3f")

# 換算模式選擇
modes = ["SHE", "RHE", "Ag/AgCl"]
from_mode = st.sidebar.selectbox("來源電極 (From)", modes, index=2)
to_mode = st.sidebar.selectbox("目標電極 (To)", modes, index=1)

# 若選擇 Ag/AgCl，顯示濃度選項
ag_conc = None
if from_mode == "Ag/AgCl" or to_mode == "Ag/AgCl":
    ag_conc = st.sidebar.selectbox("Ag/AgCl KCl 濃度", list(AG_AGCL_POTENTIALS.keys()))

# --- 核心邏輯計算 ---
# R = 8.314, F = 96485
nernst_slope = (2.303 * 8.314 * (temp_c + 273.15)) / 96485

def get_offset_to_she(mode, ag_key=None):
    """回傳該電極相對於 SHE 的電位偏壓 (E_electrode vs SHE)"""
    if mode == "SHE":
        return 0.0
    elif mode == "RHE":
        # E_RHE = E_SHE - slope * pH
        return -nernst_slope * ph
    elif mode == "Ag/AgCl":
        return AG_AGCL_POTENTIALS[ag_key]
    return 0.0

# 計算換算結果
offset_from = get_offset_to_she(from_mode, ag_conc)
offset_to = get_offset_to_she(to_mode, ag_conc)

# 公式：V_to = V_from + (Offset_from - Offset_to)
result_val = input_val + (offset_from - offset_to)

# --- 顯示結果與公式 ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("計算結果")
    st.metric(label=f"電位 (vs {to_mode})", value=f"{result_val:.4f} V")
    
    # 顯示通用公式
    st.info("💡 **換算公式說明**")
    if to_mode == "RHE" and from_mode == "Ag/AgCl":
        formula_raw = r"E_{RHE} = E_{Ag/AgCl} + E_{Ag/AgCl(ref)} + 0.0591 \times pH"
        formula_val = rf"E_{{RHE}} = {input_val} + {offset_from} + {nernst_slope:.4f} \times {ph}"
    elif to_mode == "Ag/AgCl" and from_mode == "RHE":
        formula_raw = r"E_{Ag/AgCl} = E_{RHE} - E_{Ag/AgCl(ref)} - 0.0591 \times pH"
        formula_val = rf"E_{{Ag/AgCl}} = {input_val} - {AG_AGCL_POTENTIALS[ag_conc]} - {nernst_slope:.4f} \times {ph}"
    else:
        formula_raw = r"E_{target} = E_{input} + (Offset_{input} - Offset_{target})"
        formula_val = rf"E_{{target}} = {input_val} + ({offset_from:.3f} - {offset_to:.3f})"

    st.write("通用公式：")
    st.latex(formula_raw)
    st.write("帶入參數：")
    st.latex(formula_val)

with col2:
    st.subheader("電位尺標示意圖 (vs SHE)")
    
    # 準備尺標數據
    labels = ["SHE", "RHE", "Ag/AgCl"]
    positions = [0, get_offset_to_she("RHE"), AG_AGCL_POTENTIALS[ag_conc if ag_conc else "Saturated KCl (0.197V)"]]
    
    fig = go.Figure()
    # 繪製尺標線
    fig.add_trace(go.Scatter(x=positions, y=[0,0,0], mode="markers+text",
                             text=labels, textposition="top center",
                             marker=dict(size=12, color="royalblue"),
                             name="電極基準點"))
    
    # 繪製用戶當前輸入點
    user_pos_vs_she = input_val + offset_from
    fig.add_trace(go.Scatter(x=[user_pos_vs_she], y=[0], mode="markers",
                             marker=dict(size=15, color="red", symbol="x"),
                             name="你的輸入位置"))

    fig.update_layout(xaxis_title="電位 (V vs SHE)", yaxis_showticklabels=False, 
                      height=300, margin=dict(l=20, r=20, t=40, b=20))
    fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
    
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.markdown("""
### ## Dr. Bill 的專業筆記：
1. **溫度補償**：本程式已將 $2.303 RT/F$ 納入計算，這會隨溫度改變 RHE 的斜率（25°C 時約為 0.0591 V/pH）。
2. **過電位計算**：請記得「過電位 ($\eta$)」通常定義為 $\eta = E_{measured} - E_{reversible}$。此 App 協助你完成第一步：將所有電位換算至同一基準面。
""")
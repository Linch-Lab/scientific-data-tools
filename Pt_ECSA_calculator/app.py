import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import io

# --- 1. 常數與設定 ---
Q_REF_PT = 210.0  # uC/cm2
F = 96485.0       # C/mol
R_GAS = 8.314     # J/(mol K)

E0_REF_DICT = {
    "sat.": 0.1976, "3.5M": 0.205, "3M": 0.210, "1M": 0.235
}

st.set_page_config(page_title="Pt ECSA Analyzer", layout="wide")

# --- 2. 核心邏輯函數 (從 Tkinter 移植) ---
def identify_cycles(potential_array):
    """識別 CV 圈數"""
    v_range = np.max(potential_array) - np.min(potential_array)
    # 尋找波峰
    peaks, _ = find_peaks(-potential_array, prominence=v_range*0.1)
    
    cycles_indices = []
    if len(peaks) < 2:
        cycles_indices.append((0, len(potential_array)-1))
    else:
        for i in range(len(peaks)-1):
            cycles_indices.append((peaks[i], peaks[i+1]))
    return cycles_indices

def get_ref_shift(mode, ph, temp_c, kcl_type):
    """計算電位校正值"""
    if mode == "None (Raw)": return 0.0
    
    temp_k = temp_c + 273.15
    e0_ag_agcl = E0_REF_DICT.get(kcl_type, 0.1976)
    nernst_slope = 2.303 * R_GAS * temp_k / F
    shift_val = e0_ag_agcl + (nernst_slope * ph)
    
    if mode == "Ag/AgCl -> RHE": return shift_val
    elif mode == "RHE -> Ag/AgCl": return -shift_val
    return 0.0

# --- 3. 側邊欄：參數設定 ---
st.sidebar.header("⚙️ Parameters")

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload CV Data (.txt/.csv)", type=["txt", "csv"])

st.sidebar.subheader("Experimental Setup")
scan_rate = st.sidebar.number_input("Scan Rate (mV/s)", value=100.0, step=10.0)
area = st.sidebar.number_input("Active Area (cm²)", value=0.196, format="%.4f")
loading = st.sidebar.number_input("Pt Loading (mg/cm²)", value=0.1, step=0.01)

st.sidebar.subheader("Potential Calibration")
ref_mode = st.sidebar.selectbox("Ref Mode", ["Ag/AgCl -> RHE", "RHE -> Ag/AgCl", "None (Raw)"])
if "Raw" not in ref_mode:
    ph = st.sidebar.number_input("pH", value=1.0, step=0.1)
    temp_c = st.sidebar.number_input("Temp (°C)", value=25.0)
    kcl = st.sidebar.selectbox("KCl Conc.", ["sat.", "3.5M", "3M", "1M"])
else:
    ph, temp_c, kcl = 1.0, 25.0, "sat."

# --- 4. 主畫面：數據處理與互動 ---
st.title("🧪 Pt-ECSA Analyzer (Web Version)")

if uploaded_file is not None:
    # 讀取數據
    try:
        # 嘗試讀取，自動偵測分隔符
        raw_df = pd.read_csv(uploaded_file, sep=None, engine='python')
        # 假設前兩欄是 V 和 I
        V_full = pd.to_numeric(raw_df.iloc[:, 0], errors='coerce').values
        I_full = pd.to_numeric(raw_df.iloc[:, 1], errors='coerce').values
        
        # 移除 NaN
        mask = ~np.isnan(V_full) & ~np.isnan(I_full)
        V_full = V_full[mask]
        I_full = I_full[mask]

        # 識別圈數
        cycles = identify_cycles(V_full)
        cycle_options = [f"Cycle {i+1}" for i in range(len(cycles))]
        
        # 佈局：上方選擇圈數與單位
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_cycle_label = st.selectbox("Select Cycle", cycle_options, index=len(cycle_options)-1)
        with col2:
            pot_unit = st.selectbox("Potential Unit", ["V", "mV"])
        with col3:
            curr_unit = st.selectbox("Current Unit", ["mA/cm2", "mA", "A"])

        # 取得當前圈數據
        c_idx = cycle_options.index(selected_cycle_label)
        start, end = cycles[c_idx]
        V_raw = V_full[start:end]
        I_raw = I_full[start:end]

        # --- 計算部分 ---
        
        # 1. 校正電位
        shift = get_ref_shift(ref_mode, ph, temp_c, kcl)
        V_calib = V_raw + shift
        
        # 2. 篩選正掃 (Anodic)
        diff_V = np.diff(V_calib, append=V_calib[-1])
        mask_anodic = diff_V > 0
        V_anodic = V_calib[mask_anodic]
        I_anodic = I_raw[mask_anodic]

        if len(V_anodic) > 0:
            # --- 互動式範圍調整 (使用 Columns) ---
            st.subheader("Analysis Ranges")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**DL Fit Range (Red)**")
                dl_start = st.number_input("DL Start (V)", value=0.4, step=0.05)
                dl_end = st.number_input("DL End (V)", value=0.6, step=0.05)
            with c2:
                st.markdown("**Integration Range (Cyan)**")
                h_start = st.number_input("Int Start (V)", value=0.05, step=0.05)
                h_end = st.number_input("Int End (V)", value=0.4, step=0.05)
            with c3:
                st.markdown("**Baseline Offset**")
                offset_val = st.number_input("Offset (mA/cm2)", value=0.0, step=0.01)

            # --- ECSA 核心運算 ---
            # Double Layer Fit
            mask_dl = (V_anodic >= dl_start) & (V_anodic <= dl_end)
            V_dl = V_anodic[mask_dl]
            I_dl = I_anodic[mask_dl]

            slope, intercept = 0, 0
            if len(V_dl) > 1:
                slope, intercept = np.polyfit(V_dl, I_dl, 1)
            elif len(V_dl) == 1:
                intercept = I_dl[0] - slope * V_dl[0]

            # Integration
            mask_integ = (V_anodic >= h_start) & (V_anodic <= h_end)
            V_integ = V_anodic[mask_integ]
            I_integ = I_anodic[mask_integ]

            # Baseline calculation
            I_baseline = slope * V_integ + intercept
            I_net = I_integ - I_baseline
            I_net = np.maximum(I_net, 0) # 只取大於基線的部分

            # Trapz Integration
            area_AV = np.trapz(I_net, V_integ) if len(V_integ) > 1 else 0
            
            # Unit Conversion Logic
            scan_rate_v_s = scan_rate / 1000.0
            charge_uC = (area_AV / scan_rate_v_s) * 1e6
            ecsa_cm2 = charge_uC / Q_REF_PT
            
            total_mass_g = (loading * area) / 1000.0
            ms_ecsa = (ecsa_cm2 / 10000.0) / total_mass_g if total_mass_g > 0 else 0

            # --- 結果顯示 ---
            st.divider()
            res1, res2, res3 = st.columns(3)
            res1.metric("Charge (uC)", f"{charge_uC:.2f}")
            res2.metric("ECSA (cm²)", f"{ecsa_cm2:.2f}")
            res3.metric("Mass-Specific ECSA (m²/g)", f"{ms_ecsa:.2f}")
            
            # --- 繪圖 (Matplotlib) ---
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 單位轉換因子
            v_fac = 1000.0 if pot_unit == "mV" else 1.0
            v_label = f"Potential ({pot_unit})"
            
            if curr_unit == "mA":
                i_fac = 1000.0
                i_label = "Current (mA)"
            elif curr_unit == "mA/cm2":
                i_fac = 1000.0 / area
                i_label = "Current Density (mA/cm²)"
            else:
                i_fac = 1.0
                i_label = "Current (A)"

            # 1. 原始 CV
            ax.plot(V_calib * v_fac, I_raw * i_fac, 'k-', alpha=0.6, label="Cycle Data")
            
            # 2. 輔助線 (DL Fit - Red)
            # 延伸整條線以便觀察
            V_line = np.linspace(min(V_anodic), max(V_anodic), 200)
            I_line_red = slope * V_line + intercept
            ax.plot(V_line * v_fac, I_line_red * i_fac, 'r--', label="DL Fit (Base)")

            # 3. 偏移線 (Parallel Ref - Blue)
            # Offset is defined in Display Units (usually mA/cm2)
            # 所以直接在顯示層級減去 offset
            I_line_blue_disp = (I_line_red * i_fac) - offset_val
            ax.plot(V_line * v_fac, I_line_blue_disp, 'b:', label="Parallel Ref")

            # 4. 填色區域 (Cyan)
            if len(V_integ) > 0:
                # 這裡要很小心單位，基線也要轉成顯示單位
                y1 = I_integ * i_fac
                y2 = (slope * V_integ + intercept) * i_fac
                
                # 只填色實際電流 > 基線的部分
                ax.fill_between(V_integ * v_fac, y1, y2, where=(y1 > y2), 
                                color='cyan', alpha=0.5, label="Integrated Area")

            # 標記範圍線
            ax.axvline(dl_start * v_fac, color='green', linestyle=':', alpha=0.3)
            ax.axvline(dl_end * v_fac, color='green', linestyle=':', alpha=0.3)

            ax.set_xlabel(v_label)
            ax.set_ylabel(i_label)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)

            # --- 下載數據 ---
            csv_buffer = io.StringIO()
            # 建立下載用的 DataFrame
            df_export = pd.DataFrame({
                v_label: V_calib * v_fac,
                i_label: I_raw * i_fac
            })
            df_export.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Processed Data as CSV",
                data=csv_buffer.getvalue(),
                file_name=f"ECSA_Result_{selected_cycle_label}.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Please upload a CV data file from the sidebar to begin.")
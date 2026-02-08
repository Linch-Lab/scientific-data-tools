import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import io

# ==========================================
# 1. Constants & Configuration
# ==========================================
Q_REF_PT = 210.0  # uC/cm2
F = 96485.0       # C/mol
R_GAS = 8.314     # J/(mol K)

E0_REF_DICT = {
    "sat.": 0.1976, 
    "3.5M": 0.205,
    "3M": 0.210,
    "1M": 0.235
}

st.set_page_config(page_title="Pt ECSA Analyzer Pro", layout="wide")

# ==========================================
# 2. Math Helpers
# ==========================================

def get_linear_intersection(x1, y1, x2, y2, base_y1, base_y2):
    """Finds intersection V where Data Line crosses the Baseline (Red Line)."""
    if x2 == x1: return x1, y1
    m1 = (y2 - y1) / (x2 - x1)
    m2 = (base_y2 - base_y1) / (x2 - x1)
    if m1 == m2: return x1, y1
    x_cross = (base_y1 - y1 + x1 * (m1 - m2)) / (m1 - m2)
    y_cross = m1 * (x_cross - x1) + y1
    return x_cross, y_cross

def calculate_precise_area(V_arr, I_curve, I_base):
    """Calculates area between Curve and Red Baseline (DL Fit)."""
    total_area = 0.0
    V_fill, I_fill_top, I_fill_bot = [], [], []

    for i in range(len(V_arr) - 1):
        x1, x2 = V_arr[i], V_arr[i+1]
        y1, y2 = I_curve[i], I_curve[i+1]
        b1, b2 = I_base[i], I_base[i+1]
        
        diff1 = y1 - b1
        diff2 = y2 - b2
        
        # Case A: Entirely Above Baseline
        if diff1 >= 0 and diff2 >= 0:
            width = x2 - x1
            avg_height = (diff1 + diff2) / 2.0
            total_area += width * avg_height
            V_fill.extend([x1, x2])
            I_fill_top.extend([y1, y2])
            I_fill_bot.extend([b1, b2])

        # Case B: Crossing Up
        elif diff1 < 0 and diff2 > 0:
            x_cross, y_cross = get_linear_intersection(x1, y1, x2, y2, b1, b2)
            total_area += 0.5 * (x2 - x_cross) * diff2
            V_fill.extend([x_cross, x2])
            I_fill_top.extend([y_cross, y2])
            I_fill_bot.extend([y_cross, b2])

        # Case C: Crossing Down
        elif diff1 > 0 and diff2 < 0:
            x_cross, y_cross = get_linear_intersection(x1, y1, x2, y2, b1, b2)
            total_area += 0.5 * (x_cross - x1) * diff1
            V_fill.extend([x1, x_cross])
            I_fill_top.extend([y1, y_cross])
            I_fill_bot.extend([b1, y_cross])
            
    return total_area, V_fill, I_fill_top, I_fill_bot

# ==========================================
# 3. Core Logic
# ==========================================

def identify_cycles(potential_array):
    v_range = np.max(potential_array) - np.min(potential_array)
    peaks, _ = find_peaks(-potential_array, prominence=v_range*0.1)
    cycles_indices = []
    if len(peaks) < 2:
        cycles_indices.append((0, len(potential_array)-1))
    else:
        for i in range(len(peaks)-1):
            cycles_indices.append((peaks[i], peaks[i+1]))
    return cycles_indices

def get_ref_shift(mode, ph, temp_c, kcl_type):
    if mode == "None (Raw)": return 0.0
    temp_k = temp_c + 273.15
    e0_ag_agcl = E0_REF_DICT.get(kcl_type, 0.1976)
    nernst_slope = 2.303 * R_GAS * temp_k / F 
    shift_val = e0_ag_agcl + (nernst_slope * ph)
    return shift_val if mode == "Ag/AgCl -> RHE" else -shift_val

# ==========================================
# 4. Sidebar UI
# ==========================================
st.sidebar.header("⚙️ Parameters")
uploaded_file = st.sidebar.file_uploader("Upload CV Data (.txt/.csv)", type=["txt", "csv"])

st.sidebar.subheader("Experimental Setup")
scan_rate = st.sidebar.number_input("Scan Rate (mV/s)", value=50.0, step=10.0)
area = st.sidebar.number_input("Active Area (cm²)", value=0.196, format="%.4f")
loading = st.sidebar.number_input("Pt Loading (mg/cm²)", value=0.1, step=0.01)

st.sidebar.subheader("Calibration")
ref_mode = st.sidebar.selectbox("Ref Mode", ["Ag/AgCl -> RHE", "RHE -> Ag/AgCl", "None (Raw)"])
if "Raw" not in ref_mode:
    ph = st.sidebar.number_input("pH", value=1.0, step=0.1)
    temp_c = st.sidebar.number_input("Temp (°C)", value=25.0)
    kcl = st.sidebar.selectbox("KCl Conc.", ["sat.", "3.5M", "3M", "1M"])
else:
    ph, temp_c, kcl = 1.0, 25.0, "sat."

# ==========================================
# 5. Main Logic
# ==========================================
st.title("🧪 Pt-ECSA Analyzer Pro")

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file, sep=None, engine='python')
        V_full = pd.to_numeric(raw_df.iloc[:, 0], errors='coerce').values
        I_full = pd.to_numeric(raw_df.iloc[:, 1], errors='coerce').values
        mask = ~np.isnan(V_full) & ~np.isnan(I_full)
        V_full, I_full = V_full[mask], I_full[mask]

        cycles = identify_cycles(V_full)
        cycle_options = [f"Cycle {i+1}" for i in range(len(cycles))]
        
        col1, col2, col3 = st.columns(3)
        with col1: selected_cycle_label = st.selectbox("Select Cycle", cycle_options, index=len(cycle_options)-1)
        with col2: pot_unit = st.selectbox("Potential Unit", ["V", "mV"])
        with col3: curr_unit = st.selectbox("Current Unit", ["mA/cm2", "mA", "A"])

        c_idx = cycle_options.index(selected_cycle_label)
        start, end = cycles[c_idx]
        V_raw = V_full[start:end]
        I_raw = I_full[start:end]

        shift = get_ref_shift(ref_mode, ph, temp_c, kcl)
        V_calib = V_raw + shift
        diff_V = np.diff(V_calib, append=V_calib[-1])
        
        # --- Separate Anodic (Upper) and Cathodic (Lower) ---
        mask_anodic = diff_V > 0
        V_anodic = V_calib[mask_anodic]
        I_anodic = I_raw[mask_anodic]
        
        mask_cathodic = diff_V <= 0
        V_cathodic = V_calib[mask_cathodic]
        I_cathodic = I_raw[mask_cathodic]

        if len(V_anodic) > 0:
            st.subheader("Analysis Ranges")
            
            # --- Auto-Find & UI Logic ---
            default_dl_start = 0.4
            default_dl_end = 0.6
            default_h_start = 0.05
            default_h_end = 0.4

            auto_find_start = st.checkbox("✅ Auto-Find Integration Start (Valley Detection)", value=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**1. DL Fit Range (Red)**")
                dl_start = st.number_input("DL Start (V)", value=default_dl_start, step=0.05)
                dl_end = st.number_input("DL End (V)", value=default_dl_end, step=0.05)

            # --- Auto-Find Logic ---
            if auto_find_start:
                mask_search = (V_anodic >= 0.02) & (V_anodic <= dl_start)
                if np.any(mask_search):
                    V_search = V_anodic[mask_search]
                    I_search = I_anodic[mask_search]
                    min_idx = np.argmin(I_search)
                    default_h_start = float(V_search[min_idx])

            with c2:
                st.markdown("**2. Integration Range (Cyan)**")
                h_start = st.number_input("Int Start (V)", value=default_h_start, step=0.01, format="%.3f", disabled=auto_find_start)
                h_end = st.number_input("Int End (V)", value=default_h_end, step=0.05)

            # --- CORE ALGORITHM ---

            # 1. Double Layer Fit (Red Line) - ANODIC
            mask_dl = (V_anodic >= dl_start) & (V_anodic <= dl_end)
            V_dl, I_dl = V_anodic[mask_dl], I_anodic[mask_dl]
            slope, intercept = 0, 0
            if len(V_dl) > 1:
                slope, intercept = np.polyfit(V_dl, I_dl, 1)

            # 2. Blue Line Logic (Parallel to Red, touching Lower Half Highest Point)
            blue_intercept = intercept # Fallback
            
            if len(V_cathodic) > 0:
                mask_cat_dl = (V_cathodic >= dl_start) & (V_cathodic <= dl_end)
                V_cat_dl = V_cathodic[mask_cat_dl]
                I_cat_dl = I_cathodic[mask_cat_dl]
                
                if len(V_cat_dl) > 0:
                    max_idx = np.argmax(I_cat_dl)
                    V_max_cat = V_cat_dl[max_idx]
                    I_max_cat = I_cat_dl[max_idx]
                    # Calculate new intercept for Blue Line
                    blue_intercept = I_max_cat - slope * V_max_cat
            
            # 3. Integration (Anodic Curve vs Red Line)
            mask_integ = (V_anodic >= h_start) & (V_anodic <= h_end)
            V_integ = V_anodic[mask_integ]
            I_integ_curve = I_anodic[mask_integ]
            
            # Baseline is strictly the Red Line (DL Fit)
            I_integ_base = slope * V_integ + intercept
            
            area_AV, V_fill, I_fill_top, I_fill_bot = calculate_precise_area(
                V_integ, I_integ_curve, I_integ_base
            )

            # 4. Physics Calculation
            scan_rate_v_s = scan_rate / 1000.0
            charge_uC = (area_AV / scan_rate_v_s) * 1e6
            ecsa_cm2 = charge_uC / Q_REF_PT
            total_mass_g = (loading * area) / 1000.0
            ms_ecsa = (ecsa_cm2 / 10000.0) / total_mass_g if total_mass_g > 0 else 0

            # --- DISPLAY ---
            st.divider()
            r1, r2, r3 = st.columns(3)
            r1.metric("Charge (uC)", f"{charge_uC:.2f}")
            r2.metric("ECSA (cm²)", f"{ecsa_cm2:.2f}")
            r3.metric("Mass-Specific (m²/g)", f"{ms_ecsa:.2f}")

            # --- PLOTTING ---
            fig, ax = plt.subplots(figsize=(10, 6))
            
            v_fac = 1000.0 if pot_unit == "mV" else 1.0
            if curr_unit == "mA": i_fac = 1000.0
            elif curr_unit == "mA/cm2": i_fac = 1000.0 / area
            else: i_fac = 1.0
            
            # 1. Raw Data
            ax.plot(V_calib * v_fac, I_raw * i_fac, 'k-', alpha=0.6, label="Cycle Data")
            
            # 2. Red Line (DL Fit)
            V_line = np.linspace(min(V_anodic), max(V_anodic), 200)
            I_red = slope * V_line + intercept
            ax.plot(V_line * v_fac, I_red * i_fac, 'r--', label="DL Fit (Red)")
            
            # 3. Blue Line (Lower Half Tangent)
            I_blue = slope * V_line + blue_intercept
            ax.plot(V_line * v_fac, I_blue * i_fac, 'b:', alpha=0.8, label="Cathodic Tangent (Blue)")
            
            # 4. Fill Area
            if len(V_fill) > 0:
                ax.fill_between(np.array(V_fill) * v_fac, 
                                np.array(I_fill_top) * i_fac, 
                                np.array(I_fill_bot) * i_fac, 
                                color='cyan', alpha=0.5, label="ECSA Area")

            # Markers
            ax.axvline(dl_start * v_fac, color='g', ls=':', alpha=0.4)
            ax.axvline(dl_end * v_fac, color='g', ls=':', alpha=0.4, label="DL Fit Range")
            
            ax.axvline(h_start * v_fac, color='m', ls='--', alpha=0.4, label="Int Start")
            ax.axvline(h_end * v_fac, color='m', ls=':', alpha=0.4)
            
            ax.set_xlabel(f"Potential ({pot_unit})")
            ax.set_ylabel(f"Current ({curr_unit})")
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # --- FIXED EXPORT LOGIC ---
            csv_buf = io.StringIO()
            # 1. Create Dataframe with ONLY Cycle Data (Length guaranteed to match)
            # 2. Use user-selected Units
            df_export = pd.DataFrame({
                f"Potential ({pot_unit})": V_calib * v_fac,
                f"Current ({curr_unit})": I_raw * i_fac
            })
            
            # 3. Download Button
            df_export.to_csv(csv_buf, index=False)
            st.download_button(
                label="📥 Download Cycle Data (CSV)",
                data=csv_buf.getvalue(),
                file_name=f"Cycle_{selected_cycle_label}_Export.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error: {e}")
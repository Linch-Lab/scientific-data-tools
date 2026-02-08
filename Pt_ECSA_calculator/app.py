import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
import io

# ==========================================
# 1. Constants & Configuration
# ==========================================
Q_REF_PT = 210.0  # Charge constant for Pt (uC/cm2)
F = 96485.0       # Faraday's constant (C/mol)
R_GAS = 8.314     # Gas constant (J/(mol K))

# Reference electrode potentials vs SHE (Standard Hydrogen Electrode)
E0_REF_DICT = {
    "sat.": 0.1976, # Ag/AgCl (sat. KCl)
    "3.5M": 0.205,
    "3M": 0.210,
    "1M": 0.235
}

st.set_page_config(page_title="Pt ECSA Analyzer", layout="wide")

# ==========================================
# 2. Core Logic Functions
# ==========================================

def identify_cycles(potential_array):
    """
    Identifies individual CV cycles by finding potential peaks.
    """
    v_range = np.max(potential_array) - np.min(potential_array)
    # Find peaks (top of the CV cycle) to split data
    peaks, _ = find_peaks(-potential_array, prominence=v_range*0.1)
    
    cycles_indices = []
    if len(peaks) < 2:
        # If no peaks found, treat the whole file as one cycle
        cycles_indices.append((0, len(potential_array)-1))
    else:
        # Pair up indices to define start and end of each cycle
        for i in range(len(peaks)-1):
            cycles_indices.append((peaks[i], peaks[i+1]))
    return cycles_indices

def get_ref_shift(mode, ph, temp_c, kcl_type):
    """
    Calculates the potential shift needed to convert to RHE.
    Formula: E_RHE = E_AgAgCl + E0 + 0.0591 * pH
    """
    if mode == "None (Raw)": return 0.0
    
    temp_k = temp_c + 273.15
    e0_ag_agcl = E0_REF_DICT.get(kcl_type, 0.1976)
    # Nernst slope depends on temperature
    nernst_slope = 2.303 * R_GAS * temp_k / F 
    
    shift_val = e0_ag_agcl + (nernst_slope * ph)
    
    if mode == "Ag/AgCl -> RHE": return shift_val
    elif mode == "RHE -> Ag/AgCl": return -shift_val
    return 0.0

# ==========================================
# 3. Sidebar: Parameters
# ==========================================
st.sidebar.header("⚙️ Parameters")

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload CV Data (.txt/.csv)", type=["txt", "csv"])

st.sidebar.subheader("Experimental Setup")
scan_rate = st.sidebar.number_input("Scan Rate (mV/s)", value=100.0, step=10.0)
area = st.sidebar.number_input("Active Area (cm²)", value=0.196, format="%.4f")
loading = st.sidebar.number_input("Pt Loading (mg/cm²)", value=0.1, step=0.01)

st.sidebar.subheader("Potential Calibration")
ref_mode = st.sidebar.selectbox("Ref Mode", ["Ag/AgCl -> RHE", "RHE -> Ag/AgCl", "None (Raw)"])

# Conditional display for pH and Temp
if "Raw" not in ref_mode:
    ph = st.sidebar.number_input("pH", value=1.0, step=0.1)
    temp_c = st.sidebar.number_input("Temp (°C)", value=25.0)
    kcl = st.sidebar.selectbox("KCl Conc.", ["sat.", "3.5M", "3M", "1M"])
else:
    ph, temp_c, kcl = 1.0, 25.0, "sat."

# ==========================================
# 4. Main App Logic
# ==========================================
st.title("🧪 Pt-ECSA Analyzer")

if uploaded_file is not None:
    try:
        # --- A. Data Loading ---
        # Read file, assume structure is [Potential, Current]
        raw_df = pd.read_csv(uploaded_file, sep=None, engine='python')
        V_full = pd.to_numeric(raw_df.iloc[:, 0], errors='coerce').values
        I_full = pd.to_numeric(raw_df.iloc[:, 1], errors='coerce').values
        
        # Clean NaNs
        mask = ~np.isnan(V_full) & ~np.isnan(I_full)
        V_full = V_full[mask]
        I_full = I_full[mask]

        # --- B. Cycle Selection ---
        cycles = identify_cycles(V_full)
        cycle_options = [f"Cycle {i+1}" for i in range(len(cycles))]
        
        # UI Layout for selection
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_cycle_label = st.selectbox("Select Cycle", cycle_options, index=len(cycle_options)-1)
        with col2:
            pot_unit = st.selectbox("Potential Unit", ["V", "mV"])
        with col3:
            curr_unit = st.selectbox("Current Unit", ["mA/cm2", "mA", "A"])

        # Extract specific cycle data
        c_idx = cycle_options.index(selected_cycle_label)
        start, end = cycles[c_idx]
        V_raw = V_full[start:end]
        I_raw = I_full[start:end]

        # --- C. Pre-Processing ---
        # 1. Apply Reference Shift (Calibration)
        shift = get_ref_shift(ref_mode, ph, temp_c, kcl)
        V_calib = V_raw + shift
        
        # 2. Filter Anodic Scan (Forward scan, where V is increasing)
        diff_V = np.diff(V_calib, append=V_calib[-1])
        mask_anodic = diff_V > 0
        V_anodic = V_calib[mask_anodic]
        I_anodic = I_raw[mask_anodic]

        if len(V_anodic) > 0:
            # --- D. Analysis Range UI ---
            st.subheader("Analysis Ranges")
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("**1. DL Fit Range (Red Line)**")
                st.caption("Double Layer Region (flat part)")
                dl_start = st.number_input("DL Start (V)", value=0.4, step=0.05)
                dl_end = st.number_input("DL End (V)", value=0.6, step=0.05)
            
            with c2:
                st.markdown("**2. Integration Range (Cyan Area)**")
                st.caption("H-Desorption Region (peaks)")
                h_start = st.number_input("Int Start (V)", value=0.05, step=0.05)
                h_end = st.number_input("Int End (V)", value=0.4, step=0.05)

            # --- E. Mathematical Calculation (All in Amps & Volts) ---
            
            # Step 1: Linear Fit for Double Layer (The Red Line)
            mask_dl = (V_anodic >= dl_start) & (V_anodic <= dl_end)
            V_dl = V_anodic[mask_dl]
            I_dl = I_anodic[mask_dl]

            slope, intercept = 0, 0
            if len(V_dl) > 1:
                # polyfit returns [slope, intercept]
                slope, intercept = np.polyfit(V_dl, I_dl, 1) 
            elif len(V_dl) == 1:
                intercept = I_dl[0] - slope * V_dl[0]

            # Step 2: Prepare Integration Data
            mask_integ = (V_anodic >= h_start) & (V_anodic <= h_end)
            V_integ = V_anodic[mask_integ]
            I_integ = I_anodic[mask_integ]

            # Step 3: Auto-Calculate Offset (The Blue Line Logic)
            # We want to shift the Red Line down so it touches the bottom of the curve.
            offset_amps = 0.0
            if len(V_integ) > 0:
                # Predict what the Red Line current would be in the integration range
                I_red_pred = slope * V_integ + intercept
                
                # Calculate difference: (Red Line) - (Actual Curve)
                diff_amps = I_red_pred - I_integ
                
                # Find the maximum positive difference.
                # If diff > 0, it means the Red Line is ABOVE the curve.
                # We need to lower the line by this max amount.
                max_diff = np.max(diff_amps)
                
                # If max_diff is negative, the line is already entirely below the curve, so offset is 0.
                offset_amps = max_diff if max_diff > 0 else 0.0

            # Step 4: Define the Final Baseline (Blue Line)
            # Baseline = (Slope*V + Intercept) - Offset
            I_baseline_amps = (slope * V_integ + intercept) - offset_amps
            
            # Step 5: Net Current (Curve - Baseline)
            I_net = I_integ - I_baseline_amps
            I_net = np.maximum(I_net, 0) # Remove negative values (where baseline > curve)

            # Step 6: Integration (Area under curve)
            area_AV = trapezoid(I_net, V_integ) if len(V_integ) > 1 else 0
            
            # Step 7: Physics Unit Conversion (Coulombs -> ECSA)
            scan_rate_v_s = scan_rate / 1000.0  # mV/s -> V/s
            charge_uC = (area_AV / scan_rate_v_s) * 1e6 # Convert C to uC
            ecsa_cm2 = charge_uC / Q_REF_PT
            
            # Mass Activity Calculation
            total_mass_g = (loading * area) / 1000.0 # mg -> g
            ms_ecsa = (ecsa_cm2 / 10000.0) / total_mass_g if total_mass_g > 0 else 0

            # --- F. Display Results ---
            st.divider()
            res1, res2, res3 = st.columns(3)
            res1.metric("Charge (uC)", f"{charge_uC:.2f}")
            res2.metric("ECSA (cm²)", f"{ecsa_cm2:.2f}")
            res3.metric("Mass-Specific ECSA (m²/g)", f"{ms_ecsa:.2f}")
            
            # --- G. Plotting (Matplotlib) ---
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 1. Define Display Factors (Scaling for Plot Only)
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

            # 2. Plot Raw Data (Black)
            # Important: Multiply Amps by i_fac to get mA or mA/cm2
            ax.plot(V_calib * v_fac, I_raw * i_fac, 'k-', alpha=0.6, label="Cycle Data")
            
            # 3. Plot Red Line (DL Fit)
            V_line = np.linspace(min(V_anodic), max(V_anodic), 200)
            I_line_red_amps = slope * V_line + intercept
            ax.plot(V_line * v_fac, I_line_red_amps * i_fac, 'r--', label="DL Fit (Base)")

            # 4. Plot Blue Line (Parallel Baseline)
            # Correction: Subtract offset_amps first, THEN multiply by i_fac
            I_line_blue_amps = I_line_red_amps - offset_amps
            ax.plot(V_line * v_fac, I_line_blue_amps * i_fac, 'b:', label="Integ. Baseline")

            # 5. Fill Cyan Area
            if len(V_integ) > 0:
                # Calculate Y coordinates in Display Units
                y_curve_disp = I_integ * i_fac
                
                # Calculate Baseline in Display Units
                # (Slope*V + Int - Offset) * Factor
                y_base_disp = ((slope * V_integ + intercept) - offset_amps) * i_fac
                
                # Fill between the Curve and the Blue Baseline
                ax.fill_between(V_integ * v_fac, 
                                y_curve_disp, 
                                y_base_disp, 
                                where=(y_curve_disp >= y_base_disp), 
                                color='cyan', alpha=0.5, label="ECSA Area")

            # 6. Plot Markers (Green Lines)
            ax.axvline(dl_start * v_fac, color='green', linestyle=':', alpha=0.3)
            ax.axvline(dl_end * v_fac, color='green', linestyle=':', alpha=0.3)

            ax.set_xlabel(v_label)
            ax.set_ylabel(i_label)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)

            # --- H. Download CSV ---
            csv_buffer = io.StringIO()
            df_export = pd.DataFrame({
                v_label: V_calib * v_fac,
                i_label: I_raw * i_fac,
                "Baseline_Subtracted_Current": (I_raw - ((slope * V_calib + intercept) - offset_amps)) * i_fac
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
        # Debug helper (optional)
        st.write("Detailed Error:", str(e))

else:
    st.info("Please upload a CV data file from the sidebar to begin.")
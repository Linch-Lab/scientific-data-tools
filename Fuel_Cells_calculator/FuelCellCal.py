import streamlit as st
import pandas as pd

# ==========================================
# 物理常數定義
# ==========================================
F = 96485.3329  # Faraday constant (s A / mol)
MOLAR_VOL_STP = 22.414 # L/mol (STP)

# 標準熱力學常數 (298.15 K)
DH_LHV = 241.82 # kJ/mol
DH_HHV = 285.80 # kJ/mol

# 熱中性電壓
V_TN_LHV = 1.254 # V
V_TN_HHV = 1.482 # V

# 環境溫度 (for Carnot)
TEMP_AMB_C = 25.0
TEMP_AMB_K = TEMP_AMB_C + 273.15

# ==========================================
# 單位轉換工具
# ==========================================
def convert_current_to_amps(value, unit, area_cm2):
    if unit == "A": return value
    elif unit == "mA": return value / 1000.0
    elif unit == "A/cm²": return value * area_cm2
    elif unit == "mA/cm²": return (value / 1000.0) * area_cm2
    return 0.0

def convert_power_output(kw_value, target_unit, area_cm2):
    if target_unit == "kW": return kw_value
    elif target_unit == "W": return kw_value * 1000.0
    elif target_unit == "kW/cm²": return kw_value / area_cm2
    elif target_unit == "W/cm²": return (kw_value * 1000.0) / area_cm2
    return 0.0

# ==========================================
# 核心物理引擎
# ==========================================
def calculate_physics_engine(
    current_total_amps, cell_voltage, n_cells,
    lambda_h2, lambda_oxidant,
    o2_concentration_pct,
    bop_power_kw, heat_rec_pct,
    temp_c
):
    temp_k = temp_c + 273.15
    
    # 1. 狀態判斷
    if temp_c < 100.0:
        v_rev_t = 1.229 - 0.85e-3 * (temp_k - 298.15)
        state_note = "產物：液態水相 (Liquid Phase H₂O)"
    else:
        v_rev_t = 1.18 - 0.23e-3 * (temp_k - 298.15)
        state_note = "產物：氣態水相 (Gas Phase H₂O)"

    dg_t = (2 * F * v_rev_t) / 1000.0

    # 2. 基礎功率
    stack_power_gross_kw = current_total_amps * cell_voltage * n_cells / 1000.0
    stack_power_net_kw = stack_power_gross_kw - bop_power_kw

    # 3. 氣體供應與能量流
    mol_h2_cons = (current_total_amps * n_cells) / (2 * F)
    mol_h2_supply = mol_h2_cons * lambda_h2
    
    energy_input_lhv = mol_h2_supply * DH_LHV

    # 4. 熱產生與回收
    heat_gen_total_kw = energy_input_lhv - stack_power_gross_kw
    
    heat_recovered_kw = heat_gen_total_kw * (heat_rec_pct / 100.0)
    heat_waste_kw = heat_gen_total_kw - heat_recovered_kw
    
    # 5. 效率計算
    eff_carnot = (1 - (TEMP_AMB_K / temp_k)) * 100
    eff_theo_lhv = (dg_t / DH_LHV) * 100
    eff_stack_lhv = (cell_voltage / V_TN_LHV) * 100
    eff_voltage_rev = (cell_voltage / v_rev_t) * 100 if v_rev_t > 0 else 0
    
    eff_sys_elec_lhv = (stack_power_net_kw / energy_input_lhv * 100) if energy_input_lhv > 0 else 0
    eff_chp_lhv = ((stack_power_net_kw + heat_recovered_kw) / energy_input_lhv * 100) if energy_input_lhv > 0 else 0

    # 6. 流量計算
    slpm_factor = 60 * MOLAR_VOL_STP
    
    # H2
    flow_h2_cons = mol_h2_cons * slpm_factor
    flow_h2_set = mol_h2_supply * slpm_factor
    
    # Oxidant Logic
    mol_ox_pure_cons = (current_total_amps * n_cells) / (4 * F) # 純氧莫耳數
    flow_o2_pure_slpm = mol_ox_pure_cons * slpm_factor         # 純氧 SLPM
    
    y_o2 = o2_concentration_pct / 100.0 if o2_concentration_pct > 0 else 0.21
    
    # 實際混合氣體流量
    flow_ox_mix_cons = flow_o2_pure_slpm / y_o2
    flow_ox_mix_set = flow_ox_mix_cons * lambda_oxidant

    return {
        "Temp_K": temp_k,
        "V_Rev_T": v_rev_t,
        "State_Note": state_note,
        "Eff_Carnot": eff_carnot,
        "Power_Gross_kW": stack_power_gross_kw,
        "Power_Net_kW": stack_power_net_kw,
        "Energy_Input_LHV": energy_input_lhv,
        "Heat_Total_kW": heat_gen_total_kw,
        "Heat_Recovered_kW": heat_recovered_kw,
        "Heat_Waste_kW": heat_waste_kw,
        "Eff_Theo_LHV": eff_theo_lhv,
        "Eff_Stack_LHV": eff_stack_lhv,
        "Eff_Voltage_Rev": eff_voltage_rev,
        "Eff_Sys_Elec_LHV": eff_sys_elec_lhv,
        "Eff_CHP_LHV": eff_chp_lhv,
        "Flow_H2_Cons": flow_h2_cons, "Flow_H2_Set": flow_h2_set,
        "Flow_O2_Pure_Cons": flow_o2_pure_slpm,
        "Flow_Ox_Mix_Cons": flow_ox_mix_cons,
        "Flow_Ox_Mix_Set": flow_ox_mix_set,
    }

# ==========================================
# UI 介面
# ==========================================
def main():
    st.set_page_config(page_title="FC Calculator - Dr. Ching-Hsien Lin", layout="wide", page_icon="⚡")
    st.title("⚡ 燃料電池計算機")
    
    with st.sidebar:
        st.header("⚙️ 單位設定")
        flow_unit_opt = st.radio("流量單位", ["SLPM", "SCCM"], horizontal=True)
        flow_factor = 1000.0 if flow_unit_opt == "SCCM" else 1.0
        st.divider()
        power_unit_opt = st.selectbox("功率顯示單位", ["kW", "W", "W/cm²", "kW/cm²"])
        st.info(f"模式：{flow_unit_opt} | {power_unit_opt}")

    col_input, col_gap, col_output = st.columns([12, 1, 14])

    # ==========================
    # 1. 左側：輸入
    # ==========================
    with col_input:
        st.subheader("1. 工況設定 (Operating Conditions)")
        
        c1, c2, c3 = st.columns([2, 2, 2])
        n_cells = c1.number_input("電池片數", value=100, step=1)
        area = c2.number_input("反應面積 [cm²]", value=300.0, step=10.0)
        temp_c = c3.number_input("工作溫度 [°C]", value=65.0, step=5.0, min_value=0.0, max_value=1000.0)
        
        c4, c5, c6 = st.columns([2, 1, 3])
        curr_val = c4.number_input("電流設定", value=1.0, step=0.1)
        curr_unit = c5.selectbox("單位", ["A/cm²", "mA/cm²", "A", "mA"])
        v_cell = c6.number_input("單電池電壓 [V]", value=0.65, step=0.01)
        
        current_total_amps = convert_current_to_amps(curr_val, curr_unit, area)
        gross_power_kw = current_total_amps * v_cell * n_cells / 1000.0
        disp_power = convert_power_output(gross_power_kw, power_unit_opt, area)
        st.success(f"👉 電堆總功率 (Gross): **{disp_power:.3f} {power_unit_opt}**")
        
        st.divider()
        
        st.subheader(f"2. 氣體流量 ({flow_unit_opt})")
        
        mol_h2_ref = (current_total_amps * n_cells) / (2 * F)
        f_h2_ref = (mol_h2_ref * 60 * MOLAR_VOL_STP) * flow_factor
        
        # H2
        st.markdown("##### 🟢 Anode (H2)")
        h1, h2, h3 = st.columns([3, 3, 4])
        h1.metric("理論消耗", f"{f_h2_ref:.1f}")
        stoic_h2 = h2.number_input("H2 Stoich", value=1.5, step=0.1)
        h3.metric("👉 設定流量", f"{f_h2_ref * stoic_h2:.1f}", delta="Input")
        
        # Oxidant
        st.markdown("##### 🔵 Cathode (Oxidant)")
        o2_conc = st.number_input("氧氣濃度 [%]", value=21.0, step=0.5)
        
        # Calc
        y_o2 = o2_conc / 100.0
        mol_ox_pure = (current_total_amps * n_cells) / (4 * F)
        f_o2_pure = (mol_ox_pure * 60 * MOLAR_VOL_STP) * flow_factor # Pure O2
        f_ox_mix = (f_o2_pure / y_o2) # Mix Flow
        
        o1, o2, o3 = st.columns([3, 3, 4])
        
        with o1:
            st.metric("理論消耗 (Mix)", f"{f_ox_mix:.1f}")
            
        stoic_ox = o2.number_input("Ox Stoich", value=2.0, step=0.1)
        
        # [Corrected Logic] 流量當量 = 設定總流量 / 純氧理論消耗
        f_ox_mix_set = f_ox_mix * stoic_ox
        
        # Dr. Max Check: Avoid division by zero
        if f_o2_pure > 0:
            flow_equivalent = f_ox_mix_set / f_o2_pure
        else:
            flow_equivalent = 0.0
        
        # 顯示 Pure O2 與 流量當量
        with o1:
            st.caption(f"純氧當量: {f_o2_pure:.1f} | 流量當量: {flow_equivalent:.2f}")

        o3.metric("👉 設定流量 (Mix)", f"{f_ox_mix_set:.1f}", delta="Input")

    # ==========================
    # 2. 右側：分析
    # ==========================
    with col_output:
        st.subheader("3. 效率與能量分析")
        
        with st.expander("🔧 系統參數 (BOP & Heat Recovery)", expanded=True):
            col_bop, col_heat = st.columns(2)
            with col_bop:
                enable_bop = st.checkbox("啟用 BOP 功耗", value=False)
                bop_val = st.number_input("BOP 功耗 [kW]", value=0.5, disabled=not enable_bop)
                bop_calc = bop_val if enable_bop else 0.0
            with col_heat:
                enable_heat_rec = st.checkbox("啟用熱能回收 (CHP)", value=True)
                heat_rec_pct = st.slider("熱回收效率 [%]", 0.0, 100.0, 60.0, disabled=not enable_heat_rec)
                heat_rec_calc = heat_rec_pct if enable_heat_rec else 0.0

        # 計算
        res = calculate_physics_engine(
            current_total_amps, v_cell, n_cells,
            stoic_h2, stoic_ox, o2_conc,
            bop_calc, heat_rec_calc, temp_c
        )

        def show_p(kw): return f"{convert_power_output(kw, power_unit_opt, area):.3f}"
        unit_str = power_unit_opt

        # 功率面板
        p1, p2, p3 = st.columns(3)
        p1.metric("電堆總功率", f"{show_p(res['Power_Gross_kW'])} {unit_str}")
        p2.metric("系統淨功率", f"{show_p(res['Power_Net_kW'])} {unit_str}")
        p3.metric("回收熱功率", f"{show_p(res['Heat_Recovered_kW'])} {unit_str}")

        # 系統能量流總表
        st.markdown(f"#### ⚖️ 系統能量流總表 (Energy Balance Sheet) [{unit_str}]")
        st.caption("正號 (+)：能量輸入 | 負號 (-)：能量輸出或損耗")
        
        input_fuel = res['Energy_Input_LHV']
        out_net_elec = -1 * res['Power_Net_kW']
        out_rec_heat = -1 * res['Heat_Recovered_kW']
        loss_waste_heat = -1 * res['Heat_Waste_kW']
        loss_bop = -1 * bop_calc
        
        balance_check = input_fuel + out_net_elec + out_rec_heat + loss_waste_heat + loss_bop
        
        balance_data = [
            ("1. 燃料總能量輸入 (Input: Fuel LHV)", f"+ {show_p(input_fuel)}"),
            ("2. 淨電力輸出 (Output: Net Power)", f"{show_p(out_net_elec)}"),
            ("3. 有效回收熱 (Output: Recovered Heat)", f"{show_p(out_rec_heat)}"),
            ("4. BOP 寄生功耗 (Loss: BOP)", f"{show_p(loss_bop)}"),
            ("5. 廢熱排放 (Loss: Waste Heat)", f"{show_p(loss_waste_heat)}"),
            ("---", "---"),
            ("∑ 能量平衡檢查 (Balance Check)", f"{show_p(balance_check)}")
        ]
        
        st.table(pd.DataFrame(balance_data, columns=["能量流項目 (Energy Flow Item)", f"數值 ({unit_str})"]))

        # 效率報告
        st.markdown("#### 📊 效率指標比較")
        st.info(f"當前溫度 {temp_c}°C ({res['Temp_K']:.1f} K) | 模型狀態：**{res['State_Note']}**")
        
        eff_display = [
            ("理想卡諾效率 (Carnot Limit)", f"{res['Eff_Carnot']:.1f} %", "1 - T_amb / T_stack"),
            ("FC 熱力學極限 (Delta G / H)", f"{res['Eff_Theo_LHV']:.1f} %", f"V_rev = {res['V_Rev_T']:.3f} V"),
            ("系統純電效率 (System Electric)", f"{res['Eff_Sys_Elec_LHV']:.1f} %", "淨電功率 / 燃料輸入"),
            ("系統總熱電效率 (Total CHP)", f"{res['Eff_CHP_LHV']:.1f} %", "有用功 / 燃料輸入")
        ]
        st.table(pd.DataFrame(eff_display, columns=["指標", "數值 (LHV)", "備註"]))

if __name__ == "__main__":
    main()
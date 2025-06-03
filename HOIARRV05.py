# ==============================================================================
# IMPORT LIBRARIES (การนำเข้าไลบรารีที่จำเป็น)
# ==============================================================================
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# PAGE CONFIGURATION (การตั้งค่าหน้าเว็บเบื้องต้น)
# ==============================================================================
st.set_page_config(page_title="HOIA-RR Tool", page_icon=":tada:", layout="centered")

# --- START: CSS Styles ---
st.markdown("""
<style>
/* CSS ทั่วไปสำหรับ Header */
.custom-header { font-size: 20px; font-weight: bold; margin-top: 0px !important; padding-top: 0px !important; }
div[data-testid="stHorizontalBlock"] > div div[data-testid="stMetric"] {
    border: 1px solid #ddd; padding: 0.75rem; border-radius: 0.5rem; height: 100%;
    display: flex; flex-direction: column; justify-content: center;
}
div[data-testid="stHorizontalBlock"] > div:nth-child(1) div[data-testid="stMetric"] { background-color: #e6fffa; border-color: #b2f5ea; }
div[data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stMetric"] { background-color: #fff3e0; border-color: #ffe0b2; }
div[data-testid="stHorizontalBlock"] > div:nth-child(3) div[data-testid="stMetric"] { background-color: #fce4ec; border-color: #f8bbd0; }
div[data-testid="stHorizontalBlock"] > div:nth-child(4) div[data-testid="stMetric"] { background-color: #e3f2fd; border-color: #bbdefb; }
div[data-testid="stHorizontalBlock"] > div div[data-testid="stMetric"] [data-testid="stMetricLabel"] > div,
div[data-testid="stHorizontalBlock"] > div div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stHorizontalBlock"] > div div[data-testid="stMetric"] [data-testid="stMetricDelta"]
{ color: #262730 !important; }
div[data-testid="stMetric"] [data-testid="stMetricLabel"] > div { /* Label text */
    font-size: 0.8rem !important; line-height: 1.2 !important; white-space: normal !important;
    overflow-wrap: break-word !important; word-break: break-word; display: block !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.3rem !important; }
div[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }
div[data-testid="stHorizontalBlock"] > div .stExpander {
    border: none !important; box-shadow: none !important; padding: 0 !important; margin-top: 0.5rem;
}
div[data-testid="stHorizontalBlock"] > div .stExpander header {
    padding: 0.25rem 0.5rem !important; font-size: 0.75rem !important; border-radius: 0.25rem;
}
div[data-testid="stHorizontalBlock"] > div .stExpander div[data-testid="stExpanderDetails"] {
    max-height: 200px; overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)
# --- END: CSS Styles ---

# ==============================================================================
# HEADER SECTION (ส่วนหัวของแอปพลิเคชัน)
# ==============================================================================
with st.container():
    st.markdown('<p class="custom-header">📚 เครื่องมือวิเคราะห์ผลการรายงานอุบัติการณ์ของโรงพยาบาล (HOIA-RR Tool)</p>',
                unsafe_allow_html=True)

# ==============================================================================
# STATIC DATA LOADING & DEFINITIONS (การโหลดข้อมูลคงที่และคำจำกัดความ)
# ==============================================================================
uploaded_file = st.sidebar.file_uploader("👨‍💻 กรุณาอัปโหลดไฟล์รายงานอุบัติการณ์ (.xlsx)", type=".xlsx")

PSG9_FILE_PATH = "PSG9code.xlsx"
SENTINEL_FILE_PATH = "Sentinel2024.xlsx"
ALLCODE_FILE_PATH = "Code2024.xlsx"
psg9_r_codes_for_counting = set()
sentinel_composite_keys = set()
df2 = pd.DataFrame()
PSG9code_df_master = pd.DataFrame()

try:
    if PSG9_FILE_PATH:
        PSG9code_df_master = pd.read_excel(PSG9_FILE_PATH)
        if 'รหัส' in PSG9code_df_master.columns:
            psg9_r_codes_for_counting = set(PSG9code_df_master['รหัส'].astype(str).str.strip().unique())
            if 'หมวดหมู่PSG' not in PSG9code_df_master.columns:
                st.sidebar.warning(f"ไฟล์ PSG9 ({PSG9_FILE_PATH}) ไม่มีคอลัมน์ 'หมวดหมู่PSG'")
        else:
            st.sidebar.error(f"ไฟล์ PSG9 ({PSG9_FILE_PATH}) ไม่มีคอลัมน์ 'รหัส'.")
            PSG9code_df_master = pd.DataFrame()
    else:
        st.sidebar.warning("ไม่ได้กำหนด Path สำหรับ PSG9code.xlsx")
        PSG9code_df_master = pd.DataFrame()

    Sentinel2024_df = pd.read_excel(SENTINEL_FILE_PATH)
    if 'รหัส' in Sentinel2024_df.columns and 'Impact' in Sentinel2024_df.columns:
        Sentinel2024_df['รหัส'] = Sentinel2024_df['รหัส'].astype(str).str.strip()
        Sentinel2024_df['Impact'] = Sentinel2024_df['Impact'].astype(str).str.strip()
        Sentinel2024_df['Sentinel code'] = Sentinel2024_df[['รหัส', 'Impact']].agg('-'.join, axis=1)
        sentinel_composite_keys = set(Sentinel2024_df['Sentinel code'].drop_duplicates())
    else:
        st.sidebar.error(f"Sentinel file missing 'รหัส' or 'Impact'.")

    allcode2024_df = pd.read_excel(ALLCODE_FILE_PATH)
    if 'รหัส' in allcode2024_df.columns:
        required_df2_cols = ["รหัส", "ชื่ออุบัติการณ์ความเสี่ยง", "กลุ่ม", "หมวด"]
        if not any(col not in allcode2024_df.columns for col in required_df2_cols):
            df2 = allcode2024_df[required_df2_cols].drop_duplicates().copy()
        else:
            st.sidebar.error(f"AllCode file missing columns for df2.")
    else:
        st.sidebar.error(f"AllCode file missing 'รหัส'.")
except FileNotFoundError as e:
    st.sidebar.error(f"ไม่พบไฟล์นิยาม: {e}.")
    psg9_r_codes_for_counting = set();
    sentinel_composite_keys = set();
    df2 = pd.DataFrame();
    PSG9code_df_master = pd.DataFrame()
except Exception as e:
    st.sidebar.error(f"Error loading definition files: {e}")
    psg9_r_codes_for_counting = set();
    sentinel_composite_keys = set();
    df2 = pd.DataFrame();
    PSG9code_df_master = pd.DataFrame()

color_discrete_map = {'Critical': 'red', 'High': 'orange', 'Medium': 'yellow', 'Low': 'green'}
month_label = {1: '01 มกราคม', 2: '02 กุมภาพันธ์', 3: '03 มีนาคม', 4: '04 เมษายน', 5: '05 พฤษภาคม', 6: '06 มิถุนายน',
               7: '07 กรกฎาคม', 8: '08 สิงหาคม', 9: '09 กันยายน', 10: '10 ตุลาคม', 11: '11 พฤศจิกายน', 12: '12 ธันวาคม'}
PSG9_label_dict = {1: '01 การผ่าตัดผิดคน ผิดข้าง ผิดตำแหน่ง ผิดหัตถการ',
                   2: '02 การติดเชื้อที่สำคัญในสถานยาบาลตามบริบทขององค์กรในกลุ่ม SSI, VAP, CAUTI, CABSI',
                   3: '03 บุคลากรติดเชื้อจากการปฏิบัติหน้าที่', 4: '04 การเกิด Medication Error และ Adverse Drug Event',
                   5: '05 การให้เลือดผิดคน ผิดหมู่ ผิดชนิด', 6: '06 การระบุตัวผู้ป่วยผิดพลาด',
                   7: '07 ความคลาดเคลื่อนในการวินิจฉัยโรค',
                   8: '08 การรายงานผลการตรวจทางห้องปฏิบัติการ/พยาธิวิทยา คลาดเคลื่อน',
                   9: '09 การคัดกรองที่ห้องฉุกเฉินคลาดเคลื่อน', 0: '00 Non PSG9'}
type_name = {'CPS': 'Safe Surgery', 'CPI': 'Infection Prevention and Control', 'CPM': 'Medication & Blood Safety',
             'CPP': 'Patient Care Process', 'CPL': 'Line, Tube & Catheter and Laboratory', 'CPE': 'Emergency Response',
             'CSG': 'Gynecology & Obstetrics diseases and procedure', 'CSS': 'Surgical diseases and procedure',
             'CSM': 'Medical diseases and procedure', 'CSP': 'Pediatric diseases and procedure',
             'CSO': 'Orthopedic diseases and procedure', 'CSD': 'Dental diseases and procedure',
             'GPS': 'Social Media and Communication', 'GPI': 'Infection and Exposure',
             'GPM': 'Mental Health and Mediation', 'GPP': 'Process of work', 'GPL': 'Lane (Traffic) and Legal Issues',
             'GPE': 'Environment and Working Conditions', 'GOS': 'Strategy, Structure, Security',
             'GOI': 'Information Technology & Communication, Internal control & Inventory',
             'GOM': 'Manpower, Management', 'GOP': 'Policy, Process of work & Operation',
             'GOL': 'Licensed & Professional certificate', 'GOE': 'Economy'}
colors2 = np.array([["#e1f5fe", "#f6c8b6", "#dd191d", "#dd191d", "#dd191d", "#dd191d", "#dd191d"],
                    ["#e1f5fe", "#f6c8b6", "#ff8f00", "#ff8f00", "#dd191d", "#dd191d", "#dd191d"],
                    ["#e1f5fe", "#f6c8b6", "#ffee58", "#ffee58", "#ff8f00", "#dd191d", "#dd191d"],
                    ["#e1f5fe", "#f6c8b6", "#42db41", "#ffee58", "#ffee58", "#ff8f00", "#ff8f00"],
                    ["#e1f5fe", "#f6c8b6", "#42db41", "#42db41", "#42db41", "#ffee58", "#ffee58"],
                    ["#e1f5fe", "#f6c8b6", "#f6c8b6", "#f6c8b6", "#f6c8b6", "#f6c8b6", "#f6c8b6"],
                    ["#e1f5fe", "#e1f5fe", "#e1f5fe", "#e1f5fe", "#e1f5fe", "#e1f5fe", "#e1f5fe"]])
risk_color_data = {
    'Category Color': ["Critical", "Critical", "Critical", "Critical", "Critical", "High", "High", "Critical",
                       "Critical", "Critical", "Medium", "Medium", "High", "Critical", "Critical", "Low", "Medium",
                       "Medium", "High", "High", "Low", "Low", "Low", "Medium", "Medium"],
    'Risk Level': ["51", "52", "53", "54", "55", "41", "42", "43", "44", "45", "31", "32", "33", "34", "35", "21", "22",
                   "23", "24", "25", "11", "12", "13", "14", "15"]}
risk_color_df = pd.DataFrame(risk_color_data)
incident_codes_list = ['CPS101', 'CPS102', 'CPS103', 'CPI201', 'CPI202', 'CPI203', 'CPS111', 'GPI201', 'GPI202',
                       'GPI203', 'GPI204', 'CPM101', 'CPM201', 'CPM202', 'CPM203', 'CPM204', 'CPM205', 'CPM501',
                       'CPP101', 'CPP301', 'CPL201', 'CPL203', 'CPE402', 'CPE403', 'CPE405', 'CPE407', 'CPE408',
                       'CPM105', 'CPP205', 'CPS108', 'CPS105', 'CPS308', 'CPP405', 'CPS307', 'GOS301', 'GPE206',
                       'GPS101', 'GPS201']


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
@st.cache_data
def load_data(uploaded_file_obj): return pd.read_excel(uploaded_file_obj)


def create_goal_summary_table(data_df_goal, goal_category_name_param, e_up_non_numeric_levels_param,
                              e_up_numeric_levels_param=None):
    # (เนื้อหาฟังก์ชันเหมือนเดิม)
    df_filtered_by_goal_cat = data_df_goal[data_df_goal['หมวด'] == goal_category_name_param].copy();
    if df_filtered_by_goal_cat.empty: return pd.DataFrame()
    if 'Incident Type' not in df_filtered_by_goal_cat.columns or 'Impact' not in df_filtered_by_goal_cat.columns: return pd.DataFrame()
    try:
        pvt_table_goal = pd.crosstab(df_filtered_by_goal_cat['Incident Type'], df_filtered_by_goal_cat['Impact'],
                                     margins=True, margins_name='รวมทั้งหมด')
    except Exception:
        return pd.DataFrame()
    if 'รวมทั้งหมด' in pvt_table_goal.index: pvt_table_goal = pvt_table_goal.drop(index='รวมทั้งหมด')
    if pvt_table_goal.empty: return pd.DataFrame()
    if 'รวมทั้งหมด' not in pvt_table_goal.columns: pvt_table_goal['รวมทั้งหมด'] = pvt_table_goal.sum(axis=1)
    all_impact_columns_goal = [col for col in pvt_table_goal.columns if col != 'รวมทั้งหมด']
    e_up_columns_goal = [col for col in all_impact_columns_goal if col not in e_up_non_numeric_levels_param and (
                e_up_numeric_levels_param is None or col not in e_up_numeric_levels_param)]
    report_data_goal = []
    for incident_type_goal, row_data_goal in pvt_table_goal.iterrows():
        total_e_up_count_goal = sum(
            row_data_goal[col] for col in e_up_columns_goal if col in row_data_goal and pd.notna(row_data_goal[col]))
        total_all_impacts_goal = row_data_goal['รวมทั้งหมด'] if 'รวมทั้งหมด' in row_data_goal and pd.notna(
            row_data_goal['รวมทั้งหมด']) else 0
        percent_e_up_goal = (total_e_up_count_goal / total_all_impacts_goal * 100) if total_all_impacts_goal > 0 else 0
        report_data_goal.append(
            {'Incident Type': incident_type_goal, 'รวม E-up': total_e_up_count_goal, 'ร้อยละ E-up': percent_e_up_goal})
    report_df_goal = pd.DataFrame(report_data_goal)
    if report_df_goal.empty:
        merged_report_table_goal = pvt_table_goal.reset_index(); merged_report_table_goal['รวม E-up'] = 0;
        merged_report_table_goal['ร้อยละ E-up'] = "0.00%"
    else:
        merged_report_table_goal = pd.merge(pvt_table_goal.reset_index(), report_df_goal, on='Incident Type',
                                            how='outer')
    cols_to_drop_from_display_goal = [col for col in e_up_non_numeric_levels_param if
                                      col in merged_report_table_goal.columns]
    if e_up_numeric_levels_param: cols_to_drop_from_display_goal.extend(
        [col for col in e_up_numeric_levels_param if col in merged_report_table_goal.columns])
    merged_report_table_goal = merged_report_table_goal.drop(columns=cols_to_drop_from_display_goal, errors='ignore')
    merged_report_table_goal.rename(columns={'รวมทั้งหมด': 'รวม A-I'}, inplace=True)
    merged_report_table_goal['Incident Type Name'] = merged_report_table_goal['Incident Type'].map(type_name).fillna(
        merged_report_table_goal['Incident Type'])
    final_columns_goal_order = ['Incident Type Name'] + [col for col in e_up_columns_goal if
                                                         col in merged_report_table_goal.columns] + ['รวม E-up',
                                                                                                     'รวม A-I',
                                                                                                     'ร้อยละ E-up']
    final_columns_present_goal = [col for col in final_columns_goal_order if col in merged_report_table_goal.columns]
    merged_report_table_goal = merged_report_table_goal[final_columns_present_goal]
    if 'ร้อยละ E-up' in merged_report_table_goal.columns and pd.api.types.is_numeric_dtype(
            merged_report_table_goal['ร้อยละ E-up']):
        merged_report_table_goal['ร้อยละ E-up'] = merged_report_table_goal['ร้อยละ E-up'].map('{:.2f}%'.format)
    return merged_report_table_goal.set_index('Incident Type Name')


def create_severity_table(input_df, row_column_name, table_title, specific_row_order=None):
    if input_df.empty:
        st.info(f"ไม่มีข้อมูลสำหรับสร้างตาราง '{table_title}'")
        return None
    if row_column_name not in input_df.columns:
        st.warning(f"ไม่พบคอลัมน์ '{row_column_name}' ที่จะใช้เป็นแถวสำหรับตาราง '{table_title}'")
        return None
    if 'Impact Level' not in input_df.columns:
        st.warning(f"ไม่พบคอลัมน์ 'Impact Level' สำหรับตาราง '{table_title}'")
        return None
    temp_df = input_df.copy()
    temp_df['Impact Level'] = temp_df['Impact Level'].astype(str).replace('N/A', 'ไม่ระบุ')
    try:
        if temp_df[row_column_name].dropna().empty:
            st.info(f"คอลัมน์ '{row_column_name}' ไม่มีข้อมูลที่ใช้ได้สำหรับตาราง '{table_title}'")
            return None
        severity_crosstab = pd.crosstab(temp_df[row_column_name], temp_df['Impact Level'])
    except Exception as e_crosstab:
        st.error(f"เกิดข้อผิดพลาดในการสร้างตาราง Crosstab สำหรับ '{table_title}': {e_crosstab}")
        return None
    impact_level_map_cols = {'1': 'A-B (1)', '2': 'C-D (2)', '3': 'E-F (3)', '4': 'G-H (4)', '5': 'I (5)',
                             'ไม่ระบุ': 'ไม่ระบุ LV'}
    desired_cols_ordered_keys = ['1', '2', '3', '4', '5', 'ไม่ระบุ']
    for col_key in desired_cols_ordered_keys:
        if col_key not in severity_crosstab.columns: severity_crosstab[col_key] = 0
    present_ordered_keys = [key for key in desired_cols_ordered_keys if key in severity_crosstab.columns]
    if not present_ordered_keys:
        st.info(f"ไม่พบข้อมูลระดับความรุนแรงที่กำหนดสำหรับตาราง '{table_title}'")
        return None
    severity_crosstab = severity_crosstab[present_ordered_keys]
    severity_crosstab.rename(columns=impact_level_map_cols, inplace=True)
    final_display_cols = [impact_level_map_cols[key] for key in present_ordered_keys if key in impact_level_map_cols]
    if not final_display_cols:
        st.info(f"ไม่สามารถ map คอลัมน์ระดับความรุนแรงสำหรับตาราง '{table_title}'")
        return None
    severity_crosstab['รวมทุกระดับ'] = severity_crosstab[final_display_cols].sum(axis=1)
    if specific_row_order:
        severity_crosstab = severity_crosstab.reindex(index=specific_row_order).fillna(0)
    if not specific_row_order:
        severity_crosstab = severity_crosstab[severity_crosstab['รวมทุกระดับ'] > 0]
    if severity_crosstab.empty:
        st.info(f"ไม่พบข้อมูลอุบัติการณ์ที่ตรงตามเงื่อนไขสำหรับตาราง '{table_title}' (หลังกรองผลรวม 0)")
        return None
    st.markdown(f"##### {table_title}")
    display_column_order = [impact_level_map_cols.get(key) for key in desired_cols_ordered_keys if
                            impact_level_map_cols.get(key) in severity_crosstab.columns] + ['รวมทุกระดับ']
    display_column_order_present = [col for col in display_column_order if col in severity_crosstab.columns]
    st.dataframe(severity_crosstab[display_column_order_present].astype(int), use_container_width=True)
    return severity_crosstab


# ==============================================================================
# MAIN APPLICATION LOGIC
# ==============================================================================
if uploaded_file:
    # --- STAGE 1: ALL DATA LOADING AND PRE-COMPUTATIONS ---
    df = load_data(uploaded_file);
    df_original_rows = df.shape[0]
    df = df.fillna('None');
    df.rename(columns={'วดป.ที่เกิด': 'Occurrence Date', 'ความรุนแรง': 'Impact'}, inplace=True)
    required_cols_in_upload = ['Incident', 'Occurrence Date', 'Impact']
    if any(col not in df.columns for col in required_cols_in_upload):
        st.error(
            f"ไฟล์อัปโหลดขาดคอลัมน์: {', '.join([col for col in required_cols_in_upload if col not in df.columns])}.")
        st.stop()
    df['Impact_original_value'] = df['Impact']
    df['Impact'] = df['Impact'].astype(str).str.strip();
    df['รหัส'] = df['Incident'].astype(str).str.slice(0, 6).str.strip()
    try:
        df['Occurrence Date'] = pd.to_datetime(df['Occurrence Date'], errors='coerce');
        df.dropna(subset=['Occurrence Date'], inplace=True)
        if df.empty: st.error("ไม่พบข้อมูลวันที่ถูกต้อง"); st.stop()
    except Exception as e:
        st.error(f"Error converting 'Occurrence Date': {e}"); st.stop()

    # --- MODIFIED: Moved display_cols_common definition earlier ---
    display_cols_common = ['Occurrence Date', 'Incident', 'รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง', 'Impact', 'Impact Level',
                           'รายละเอียดการเกิด', 'Resulting Actions']
    # Ensure 'Resulting Actions' exists or handle its absence for display_cols_common if it's critical for all expanders
    if 'Resulting Actions' not in df.columns:
        display_cols_common = [col for col in display_cols_common if col != 'Resulting Actions']

    total_psg9_incidents_for_metric1 = df[df['รหัส'].isin(psg9_r_codes_for_counting)].shape[
        0] if psg9_r_codes_for_counting and 'รหัส' in df.columns else 0
    total_sentinel_incidents_for_metric1 = 0
    if sentinel_composite_keys and 'รหัส' in df.columns and 'Impact' in df.columns:
        df_temp_s1 = df.copy();
        df_temp_s1['Sentinel code for check'] = df_temp_s1[['รหัส', 'Impact']].agg('-'.join, axis=1)
        total_sentinel_incidents_for_metric1 = \
        df_temp_s1[df_temp_s1['Sentinel code for check'].isin(sentinel_composite_keys)].shape[0]

    impact_level_map = {('A', 'B', '1'): '1', ('C', 'D', '2'): '2', ('E', 'F', '3'): '3', ('G', 'H', '4'): '4',
                        ('I', '5'): '5'}


    def map_impact_level(impact_val):
        for k, v_level in impact_level_map.items():
            if impact_val in k: return v_level
        return 'N/A'


    df['Impact Level'] = df['Impact'].apply(map_impact_level)

    severe_impact_levels = ['3', '4', '5']
    df_severe_incidents = df[df['Impact Level'].isin(severe_impact_levels)].copy()
    total_severe_incidents = df_severe_incidents.shape[0]
    total_severe_psg9_incidents = \
    df_severe_incidents[df_severe_incidents['รหัส'].isin(psg9_r_codes_for_counting)].shape[
        0] if psg9_r_codes_for_counting and not df_severe_incidents.empty and 'รหัส' in df_severe_incidents.columns else 0

    total_severe_unresolved_incidents_val = "N/A";
    total_severe_unresolved_psg9_incidents_val = "N/A";
    df_severe_unresolved = pd.DataFrame()
    if 'Resulting Actions' in df.columns:
        df_severe_unresolved = df_severe_incidents[df_severe_incidents['Resulting Actions'] == "None"].copy()
        total_severe_unresolved_incidents_val = df_severe_unresolved.shape[0]
        if psg9_r_codes_for_counting and not df_severe_unresolved.empty and 'รหัส' in df_severe_unresolved.columns:
            total_severe_unresolved_psg9_incidents_val = \
            df_severe_unresolved[df_severe_unresolved['รหัส'].isin(psg9_r_codes_for_counting)].shape[0]
        elif psg9_r_codes_for_counting:
            total_severe_unresolved_psg9_incidents_val = 0

    # --- COMMON DATA PROCESSING FOR ALL SUBSEQUENT ANALYSES (Continued) ---
    cols_to_drop_initial = ['In.HCode', 'วดป.ที่ Import การเกิด', 'รหัสรายงาน'];
    actual_cols_to_drop_initial = [col for col in cols_to_drop_initial if col in df.columns]
    if actual_cols_to_drop_initial: df.drop(columns=actual_cols_to_drop_initial, axis=1, inplace=True)
    df = df.sort_values(by='Occurrence Date', ascending=True, na_position='first')
    total_month = 1
    try:
        if not df.empty:
            max_date_period = df['Occurrence Date'].max().to_period('M');
            min_date_period = df['Occurrence Date'].min().to_period('M')
            total_month = (max_date_period.year - min_date_period.year) * 12 + (
                        max_date_period.month - min_date_period.month) + 1
            st.sidebar.write('Incident count (after cleaning):', df.shape[0]);
            st.sidebar.write('Max month:', max_date_period);
            st.sidebar.write('Min month:', min_date_period)
            st.sidebar.write('Total months:', total_month, 'โปรดตรวจสอบ!')
    except Exception:
        pass
    df['Incident Type'] = df['Incident'].astype(str).apply(lambda x: x[:3] if pd.notnull(x) else 'N/A')
    if 'Occurrence Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Occurrence Date']) and not df.empty:
        df['Month'] = df['Occurrence Date'].dt.month;
        df['เดือน'] = df['Month'].map(month_label);
        df['Year'] = df['Occurrence Date'].dt.year.astype(str)
    else:
        df['Month'] = np.nan; df['เดือน'] = 'N/A'; df['Year'] = 'N/A'

    if not df2.empty and 'รหัส' in df2.columns:
        df = pd.merge(df, df2, on='รหัส', how='left')
        for col_df2_check in ["ชื่ออุบัติการณ์ความเสี่ยง", "กลุ่ม", "หมวด"]:
            if col_df2_check not in df.columns: df[col_df2_check] = 'N/A_df2_missing'
    else:
        for col_df2_check in ["ชื่ออุบัติการณ์ความเสี่ยง", "กลุ่ม", "หมวด"]:
            if col_df2_check not in df.columns: df[col_df2_check] = 'N/A_df2_missing'

    if not PSG9code_df_master.empty:
        PSG9_INCIDENT_CODE_COLUMN_IN_MASTER = 'รหัส'
        PSG9_CATEGORY_COLUMN_IN_MASTER = 'หมวดหมู่PSG'
        if PSG9_INCIDENT_CODE_COLUMN_IN_MASTER in PSG9code_df_master.columns and PSG9_CATEGORY_COLUMN_IN_MASTER in PSG9code_df_master.columns:
            standards_to_merge = PSG9code_df_master[
                [PSG9_INCIDENT_CODE_COLUMN_IN_MASTER, PSG9_CATEGORY_COLUMN_IN_MASTER]].copy()
            standards_to_merge[PSG9_INCIDENT_CODE_COLUMN_IN_MASTER] = standards_to_merge[
                PSG9_INCIDENT_CODE_COLUMN_IN_MASTER].astype(str).str.strip()
            standards_to_merge.drop_duplicates(subset=[PSG9_INCIDENT_CODE_COLUMN_IN_MASTER], keep='first', inplace=True)
            df['รหัส'] = df['รหัส'].astype(str).str.strip()
            df = pd.merge(df, standards_to_merge, left_on='รหัส', right_on=PSG9_INCIDENT_CODE_COLUMN_IN_MASTER,
                          how='left', suffixes=('', '_from_psg9_master'))
            merged_category_col = PSG9_CATEGORY_COLUMN_IN_MASTER
            if PSG9_CATEGORY_COLUMN_IN_MASTER + '_from_psg9_master' in df.columns:
                merged_category_col = PSG9_CATEGORY_COLUMN_IN_MASTER + '_from_psg9_master'
            elif PSG9_CATEGORY_COLUMN_IN_MASTER not in df.columns:
                merged_category_col = None
            if merged_category_col and merged_category_col in df.columns:
                df.rename(columns={merged_category_col: 'หมวดหมู่มาตรฐานสำคัญ'}, inplace=True, errors='ignore')
                if 'หมวดหมู่มาตรฐานสำคัญ' in df.columns: df['หมวดหมู่มาตรฐานสำคัญ'].fillna("ไม่จัดอยู่ใน PSG9 Catalog",
                                                                                           inplace=True)
            else:
                df['หมวดหมู่มาตรฐานสำคัญ'] = "ไม่สามารถระบุ (Merge PSG9 ล้มเหลว)"
            if PSG9_INCIDENT_CODE_COLUMN_IN_MASTER != 'รหัส' and PSG9_INCIDENT_CODE_COLUMN_IN_MASTER in df.columns: df.drop(
                columns=[PSG9_INCIDENT_CODE_COLUMN_IN_MASTER], inplace=True, errors='ignore')
            if PSG9_INCIDENT_CODE_COLUMN_IN_MASTER + '_from_psg9_master' in df.columns and PSG9_INCIDENT_CODE_COLUMN_IN_MASTER != 'รหัส': df.drop(
                columns=[PSG9_INCIDENT_CODE_COLUMN_IN_MASTER + '_from_psg9_master'], inplace=True, errors='ignore')
        else:
            df['หมวดหมู่มาตรฐานสำคัญ'] = "ไม่สามารถระบุ (เช็คคอลัมน์ใน PSG9code.xlsx)"
    else:
        df['หมวดหมู่มาตรฐานสำคัญ'] = "ไม่สามารถระบุ (PSG9code.xlsx ไม่ได้โหลด/ว่างเปล่า)"

    df_freq = pd.DataFrame(columns=['Incident', 'count', 'Incident Rate/mth'])
    if 'Incident' in df.columns and not df.empty:
        df_freq_calc = df['Incident'].value_counts().reset_index();
        df_freq_calc.columns = ['Incident', 'count_from_freq']
        if total_month > 0:
            df_freq_calc['rate_from_freq'] = (df_freq_calc['count_from_freq'] / total_month).round(1)
        else:
            df_freq_calc['rate_from_freq'] = 0
        df = pd.merge(df, df_freq_calc, on="Incident", how='left')
        if 'count_from_freq' in df.columns:
            df['count'] = df['count_from_freq'].fillna(0).astype(int); df.drop(columns=['count_from_freq'],
                                                                               inplace=True, errors='ignore')
        elif 'count_x' in df.columns and 'count_y' in df.columns:
            df['count'] = df['count_y'].fillna(0).astype(int); df.drop(columns=['count_x', 'count_y'], inplace=True,
                                                                       errors='ignore')
        elif 'count' not in df.columns:
            df['count'] = 0
        else:
            df['count'].fillna(0, inplace=True)
        if 'rate_from_freq' in df.columns:
            df['Incident Rate/mth'] = df['rate_from_freq'].fillna(0); df.drop(columns=['rate_from_freq'], inplace=True,
                                                                              errors='ignore')
        elif 'Incident Rate/mth_y' in df.columns:
            df['Incident Rate/mth'] = df['Incident Rate/mth_y'].fillna(0)
            cols_to_drop_r = [col for col in ['Incident Rate/mth_x', 'Incident Rate/mth_y'] if col in df.columns]
            if cols_to_drop_r: df.drop(columns=cols_to_drop_r, inplace=True, errors='ignore')
        elif 'Incident Rate/mth' not in df.columns:
            df['Incident Rate/mth'] = 0
        else:
            df['Incident Rate/mth'].fillna(0, inplace=True)
        current_df_freq_cols = {'Incident': 'Incident'}
        if 'count_from_freq' in df_freq_calc.columns:
            current_df_freq_cols['count_from_freq'] = 'count'
        elif 'count' in df_freq_calc.columns:
            current_df_freq_cols['count'] = 'count'
        if 'rate_from_freq' in df_freq_calc.columns:
            current_df_freq_cols['rate_from_freq'] = 'Incident Rate/mth'
        elif 'Incident Rate/mth' in df_freq_calc.columns:
            current_df_freq_cols['Incident Rate/mth'] = 'Incident Rate/mth'
        if not df_freq_calc.empty:
            df_freq_renamed = df_freq_calc.rename(columns=current_df_freq_cols)
            final_cols_for_df_freq = [col for col in ['Incident', 'count', 'Incident Rate/mth'] if
                                      col in df_freq_renamed.columns]
            if final_cols_for_df_freq: df_freq = df_freq_renamed[final_cols_for_df_freq]
    else:
        df['Incident Rate/mth'] = 0; df['count'] = 0
    df['Count'] = 1
    conditions_freq = [(df['Incident Rate/mth'] < 2.0), (df['Incident Rate/mth'] < 3.9),
                       (df['Incident Rate/mth'] < 6.9), (df['Incident Rate/mth'] < 29.9)]
    choices_freq = ['1', '2', '3', '4'];
    df['Frequency Level'] = np.select(conditions_freq, choices_freq, default='5')
    df['Impact Level'] = df['Impact Level'].astype(str);
    df['Frequency Level'] = df['Frequency Level'].astype(str)
    df['Risk Level'] = df.apply(
        lambda row: row['Impact Level'] + row['Frequency Level'] if row['Impact Level'] not in ['', 'N/A'] and row[
            'Frequency Level'] not in ['', 'N/A'] else 'N/A', axis=1)
    df = pd.merge(df, risk_color_df, on='Risk Level', how='left');
    df['Category Color'].fillna('Undefined', inplace=True)
    # --- `df` IS NOW FULLY PROCESSED ---

    # --- SIDEBAR SELECTION FOR ANALYSIS TYPE ---
    analysis_options = [
        "แดชบอร์ดสรุปภาพรวม",
        "ภาพรวม Risk Matrix & Top 10",
        "กราฟสรุปอุบัติการณ์ (รายมิติ)",
        "ความสัมพันธ์รหัส-หมวดหมู่",
        "วิเคราะห์ตามหมวดหมู่ PSG9",
        "วิเคราะห์อุบัติการณ์ที่ต้องดำเนินการแก้ไข",
        "รายการ Sentinel Events",
        "สรุปอุบัติการณ์ตามเป้าหมายทั่วไป (Goal-based)",
        "ข้อมูลอุบัติการณ์ทั้งหมด (ตารางเต็ม)"  # MODIFIED: Added back as a selectable option
    ]
    selected_analysis = st.sidebar.selectbox("เลือกส่วนที่ต้องการแสดงผล:", analysis_options, index=0,
                                             key="main_analysis_selector")
    st.sidebar.markdown("---")

    # --- CONDITIONAL DISPLAY OF SELECTED ANALYSIS ---
    if selected_analysis == "แดชบอร์ดสรุปภาพรวม":
        st.subheader("สรุปภาพรวมอุบัติการณ์:")
        col1_m1_dash, col2_m1_dash, col3_m1_dash = st.columns(3)
        with col1_m1_dash:
            st.metric("📋 อุบัติการณ์ทั้งหมด (อัปโหลด)", f"{df_original_rows:,}")
        with col2_m1_dash:
            st.metric("🎯PSG9 ทั้งหมด (พบ)", f"{total_psg9_incidents_for_metric1:,}")
        with col3_m1_dash:
            st.metric("🚨 Sentinel ทั้งหมด (พบ)", f"{total_sentinel_incidents_for_metric1:,}")

        col1_m_sev_dash, col2_m_sev_dash, col3_m_sev_dash, col4_m_sev_dash = st.columns(4)
        unresolved_label_suffix_dash = " (ไม่มี 'Resulting Actions')" if 'Resulting Actions' not in df.columns else ""
        with col1_m_sev_dash:
            st.metric("🔺 อุบัติการณ์ E-I & 3-5 ทั้งหมด", f"{total_severe_incidents:,}")
            if total_severe_incidents > 0:
                with st.expander(f"ดูรายการ ({total_severe_incidents})", expanded=False):
                    actual_cols = [col for col in display_cols_common if col in df_severe_incidents.columns]
                    st.dataframe(df_severe_incidents[actual_cols].sort_values(by='Occurrence Date', ascending=False),
                                 hide_index=True, use_container_width=True)
        with col2_m_sev_dash:
            st.metric("📍 อุบัติการณ์ E-I & 3-5 (เฉพาะ PSG9)", f"{total_severe_psg9_incidents:,}")
            if total_severe_psg9_incidents > 0:
                df_severe_psg9_list_dash = df_severe_incidents[
                    df_severe_incidents['รหัส'].isin(psg9_r_codes_for_counting)]
                with st.expander(f"ดูรายการ ({total_severe_psg9_incidents})", expanded=False):
                    actual_cols = [col for col in display_cols_common if col in df_severe_psg9_list_dash.columns]
                    st.dataframe(
                        df_severe_psg9_list_dash[actual_cols].sort_values(by='Occurrence Date', ascending=False),
                        hide_index=True, use_container_width=True)
        with col3_m_sev_dash:
            label_m3_1_dash = f"⏳ อุบัติการณ์ E-I & 3-5 ทั้งหมด ที่ยังไม่ถูกแก้ไข{unresolved_label_suffix_dash}"
            value_m3_1_dash = f"{total_severe_unresolved_incidents_val:,}" if isinstance(
                total_severe_unresolved_incidents_val, int) else total_severe_unresolved_incidents_val
            st.metric(label_m3_1_dash, value_m3_1_dash)
            if isinstance(total_severe_unresolved_incidents_val,
                          int) and total_severe_unresolved_incidents_val > 0 and 'Resulting Actions' in df.columns and 'df_severe_unresolved' in locals() and not df_severe_unresolved.empty:
                with st.expander(f"ดูรายการ ({total_severe_unresolved_incidents_val})", expanded=False):
                    actual_cols = [col for col in display_cols_common if col in df_severe_unresolved.columns]
                    st.dataframe(df_severe_unresolved[actual_cols].sort_values(by='Occurrence Date', ascending=False),
                                 hide_index=True, use_container_width=True)
        with col4_m_sev_dash:
            label_m3_2_dash = f"📝 อุบัติการณ์ E-I & 3-5 PSG9 ที่ยังไม่ถูกแก้ไข{unresolved_label_suffix_dash}"
            value_m3_2_dash = f"{total_severe_unresolved_psg9_incidents_val:,}" if isinstance(
                total_severe_unresolved_psg9_incidents_val, int) else total_severe_unresolved_psg9_incidents_val
            st.metric(label_m3_2_dash, value_m3_2_dash)
            if isinstance(total_severe_unresolved_psg9_incidents_val,
                          int) and total_severe_unresolved_psg9_incidents_val > 0 and 'Resulting Actions' in df.columns:
                df_sev_unresolved_psg9_list_dash = df_severe_incidents[
                    (df_severe_incidents['รหัส'].isin(psg9_r_codes_for_counting)) & (
                                df_severe_incidents['Resulting Actions'] == "None")]
                if not df_sev_unresolved_psg9_list_dash.empty:
                    with st.expander(f"ดูรายการ ({df_sev_unresolved_psg9_list_dash.shape[0]})", expanded=False):
                        actual_cols = [col for col in display_cols_common if
                                       col in df_sev_unresolved_psg9_list_dash.columns]
                        st.dataframe(df_sev_unresolved_psg9_list_dash[actual_cols].sort_values(by='Occurrence Date',
                                                                                               ascending=False),
                                     hide_index=True, use_container_width=True)
        st.markdown("---")

    elif selected_analysis == "ภาพรวม Risk Matrix & Top 10":
        st.subheader("Risk Matrix");
        matrix_data = np.zeros([7, 7], dtype=object);
        matrix_df = pd.DataFrame(matrix_data, columns=['G1', 'W1', 'F1', 'F2', 'F3', 'F4', 'F5'])
        impact_map_matrix = {'5': 0, '4': 1, '3': 2, '2': 3, '1': 4}
        if not df[df['Risk Level'] != 'N/A'].empty:
            risk_counts = df.groupby(['Impact Level', 'Frequency Level']).size().reset_index(name='counts')
            for _, r_c in risk_counts.iterrows():
                il, fl, cv = str(r_c['Impact Level']), str(r_c['Frequency Level']), r_c['counts']
                if il in impact_map_matrix and fl in ['1', '2', '3', '4', '5']:
                    mri, mcn = impact_map_matrix[il], f"F{fl}"
                    if mcn in matrix_df.columns: matrix_df.loc[mri, mcn] = str(cv)
        matrix_df = matrix_df.fillna('0')
        matrix_df.loc[0, 'G1'] = "I/5";
        matrix_df.loc[0, 'W1'] = "Death";
        matrix_df.loc[1, 'G1'] = "G-H/4";
        matrix_df.loc[1, 'W1'] = "Severe";
        matrix_df.loc[2, 'G1'] = "E-F/3";
        matrix_df.loc[2, 'W1'] = "Moderate";
        matrix_df.loc[3, 'G1'] = "C-D/2";
        matrix_df.loc[3, 'W1'] = "Low";
        matrix_df.loc[4, 'G1'] = "A-B/1";
        matrix_df.loc[4, 'W1'] = "No Harm";
        matrix_df.loc[5, 'F1'] = "Remote";
        matrix_df.loc[5, 'F2'] = "Uncommon";
        matrix_df.loc[5, 'F3'] = "Occasional";
        matrix_df.loc[5, 'F4'] = "Probable";
        matrix_df.loc[5, 'F5'] = "Frequent";
        matrix_df.loc[6, 'F1'] = "1/m";
        matrix_df.loc[6, 'F2'] = "2-3/m";
        matrix_df.loc[6, 'F3'] = "4-6/m";
        matrix_df.loc[6, 'F4'] = "7-29/m";
        matrix_df.loc[6, 'F5'] = ">=30/m"
        fig_rm = go.Figure(data=[go.Table(header=None, cells=dict(values=[matrix_df[c] for c in matrix_df.columns],
                                                                  fill_color=colors2.T, align="center",
                                                                  font=dict(size=11), height=33))])
        fig_rm.update_layout(width=750, height=330, margin=dict(l=15, r=5, t=25, b=5));
        st.plotly_chart(fig_rm)
        st.markdown("---")
        st.subheader("Top 10 อุบัติการณ์ (ตามความถี่)")
        if isinstance(df_freq,
                      pd.DataFrame) and not df_freq.empty and 'count' in df_freq.columns and 'Incident' in df_freq.columns:
            df_freq_top10 = df_freq.sort_values(by='count', ascending=False).head(10).copy()
            df_freq_top10['รหัส'] = df_freq_top10['Incident'].astype(str).apply(
                lambda x: x[:6] if pd.notnull(x) else 'N/A')
            if not df2.empty and 'รหัส' in df2.columns:
                df_freq_top10 = pd.merge(df_freq_top10, df2[['รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง', 'กลุ่ม', 'หมวด']],
                                         on='รหัส', how='left', suffixes=('_freq', '_df2'))
            else:
                for cn in ['ชื่ออุบัติการณ์ความเสี่ยง', 'กลุ่ม', 'หมวด']:
                    if cn not in df_freq_top10.columns: df_freq_top10[cn] = 'N/A_df2_missing'
            if 'Incident Rate/mth' not in df_freq_top10.columns: df_freq_top10['Incident Rate/mth'] = 0.0
            display_cols_top10 = {'Incident': 'Incident Code', 'ชื่ออุบัติการณ์ความเสี่ยง': 'ชื่ออุบัติการณ์',
                                  'count': 'จำนวนครั้ง', 'Incident Rate/mth': 'อัตราต่อเดือน'}
            df_freq_display = df_freq_top10[
                [col for col in display_cols_top10.keys() if col in df_freq_top10.columns]].rename(
                columns=display_cols_top10)
            st.dataframe(df_freq_display)
        else:
            st.warning("Cannot display Top 10 Incidents.")

    elif selected_analysis == "กราฟสรุปอุบัติการณ์ (รายมิติ)":
        st.subheader("กราฟสรุปอุบัติการณ์ (แบ่งตามมิติต่างๆ)")
        tab1_v, tab2_v, tab3_v, tab4_v = st.tabs(
            ["By Goals (หมวด)", "By Group (กลุ่ม)", "By Shift (เวร)", "By Place (สถานที่)"])
        df_charts = df.copy()
        chart_cols_map = {'หมวด': 'N/A_df2_missing', 'กลุ่ม': 'N/A_df2_missing', 'ช่วงเวลา/เวร': 'N/A',
                          'ชนิดสถานที่': 'N/A', 'Category Color': 'Undefined', 'Count': 0}
        for c, dv in chart_cols_map.items():
            if c not in df_charts.columns: df_charts[c] = dv
        with tab1_v:
            st.markdown(f"Incidents by Safety Goals ({total_month}m)")
            df_c1 = df_charts.dropna(subset=['หมวด', 'Category Color', 'Count'], how='any')[
                lambda x: x['หมวด'] != 'N/A_df2_missing']
            if not df_c1.empty:
                st.plotly_chart(
                    px.bar(df_c1, x='หมวด', y='Count', color='Category Color', color_discrete_map=color_discrete_map),
                    use_container_width=True)
            else:
                st.warning("No 'หมวด' data for chart.")
        with tab2_v:
            st.markdown(f"Incidents by Group ({total_month}m)")
            df_c2 = df_charts.dropna(subset=['กลุ่ม', 'Category Color', 'Count'], how='any')[
                lambda x: x['กลุ่ม'] != 'N/A_df2_missing']
            if not df_c2.empty:
                st.plotly_chart(
                    px.bar(df_c2, x='กลุ่ม', y='Count', color='Category Color', color_discrete_map=color_discrete_map),
                    use_container_width=True)
            else:
                st.warning("No 'กลุ่ม' data for chart.")
        with tab3_v:
            st.markdown(f"Incidents by Shift ({total_month}m)")
            df_c3 = df_charts.dropna(subset=['ช่วงเวลา/เวร', 'Category Color', 'Count'], how='any')[
                lambda x: x['ช่วงเวลา/เวร'] != 'N/A']
            if not df_c3.empty:
                st.plotly_chart(px.bar(df_c3, x='ช่วงเวลา/เวร', y='Count', color='Category Color',
                                       color_discrete_map=color_discrete_map), use_container_width=True)
            else:
                st.warning("No 'ช่วงเวลา/เวร' data for chart.")
        with tab4_v:
            st.markdown(f"Incidents by Place ({total_month}m)")
            df_c4 = df_charts.dropna(subset=['ชนิดสถานที่', 'Category Color', 'Count'], how='any')[
                lambda x: x['ชนิดสถานที่'] != 'N/A']
            if not df_c4.empty:
                st.plotly_chart(px.bar(df_c4, x='ชนิดสถานที่', y='Count', color='Category Color',
                                       color_discrete_map=color_discrete_map), use_container_width=True)
            else:
                st.warning("No 'ชนิดสถานที่' data for chart.")

    elif selected_analysis == "ความสัมพันธ์รหัส-หมวดหมู่":
        st.subheader("ความสัมพันธ์ระหว่างรหัสอุบัติการณ์และหมวดหมู่ (Scatter Plot)")
        scatter_cols = ['รหัส', 'หมวด', 'Category Color', 'Incident Rate/mth']
        if all(col in df.columns for col in scatter_cols):
            df_sc = df.dropna(subset=scatter_cols, how='any')
            if not df_sc.empty:
                h_data = ['Incident']
                if 'ชื่ออุบัติการณ์ความเสี่ยง' in df_sc.columns and df_sc[
                    'ชื่ออุบัติการณ์ความเสี่ยง'].notna().any(): h_data.append('ชื่ออุบัติการณ์ความเสี่ยง')
                fig_sc = px.scatter(df_sc, x='รหัส', y='หมวด', color='Category Color', size='Incident Rate/mth',
                                    hover_data=h_data, size_max=30, color_discrete_map=color_discrete_map)
                st.plotly_chart(fig_sc, theme="streamlit", use_container_width=True)
            else:
                st.warning("Insufficient data for Scatter Plot after filtering.")
        else:
            st.warning(f"Missing one or more required columns for Scatter Plot: {scatter_cols}")

    elif selected_analysis == "วิเคราะห์ตามหมวดหมู่ PSG9":
        st.subheader("📊 ผลการวิเคราะห์อุบัติการณ์ตามหมวดหมู่ PSG9 และระดับความเสี่ยง")
        if 'หมวดหมู่มาตรฐานสำคัญ' not in df.columns or \
                df['หมวดหมู่มาตรฐานสำคัญ'].dropna().empty or \
                df['หมวดหมู่มาตรฐานสำคัญ'].nunique() == 0 or \
                (df['หมวดหมู่มาตรฐานสำคัญ'].nunique() == 1 and df['หมวดหมู่มาตรฐานสำคัญ'].unique()[0] in [
                    "ไม่สามารถระบุ (เช็คคอลัมน์ใน PSG9code.xlsx)", "ไม่สามารถระบุ (PSG9code.xlsx ไม่ได้โหลด/ว่างเปล่า)",
                    "ไม่จัดอยู่ใน PSG9 Catalog"]):
            st.warning(
                "ไม่สามารถดำเนินการวิเคราะห์ตามหมวดหมู่ PSG9 ได้ กรุณาตรวจสอบไฟล์ PSG9code.xlsx และการตั้งค่าการ Merge ข้อมูล")
        elif 'Category Color' not in df.columns:
            st.error("คอลัมน์ 'Category Color' ที่จำเป็น (จาก Risk Level) ไม่พบในข้อมูลหลัก")
        else:
            df_psg9_analysis = df[df['หมวดหมู่มาตรฐานสำคัญ'] != "ไม่จัดอยู่ใน PSG9 Catalog"].copy()
            df_psg9_analysis = df_psg9_analysis[
                ~df_psg9_analysis['หมวดหมู่มาตรฐานสำคัญ'].str.startswith("ไม่สามารถระบุ", na=False)]
            if df_psg9_analysis.empty or df_psg9_analysis['หมวดหมู่มาตรฐานสำคัญ'].dropna().empty:
                st.info("ไม่พบอุบัติการณ์ที่เกี่ยวข้องกับหมวดหมู่ PSG9 ที่ระบุ (หลังจากกรองค่าที่ไม่ถูกต้อง)")
            else:
                unique_psg9_cats = sorted(df_psg9_analysis['หมวดหมู่มาตรฐานสำคัญ'].dropna().unique())
                results_psg9_list = []
                for psg9_cat_name in unique_psg9_cats:
                    df_current_psg9_cat = df_psg9_analysis[df_psg9_analysis['หมวดหมู่มาตรฐานสำคัญ'] == psg9_cat_name]
                    total_for_psg9_cat = df_current_psg9_cat.shape[0]
                    color_counts_psg9 = {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0, 'Undefined': 0}
                    if not df_current_psg9_cat.empty:
                        actual_counts = df_current_psg9_cat['Category Color'].value_counts().to_dict()
                        for color, num in actual_counts.items():
                            if color in color_counts_psg9:
                                color_counts_psg9[color] = num
                            else:
                                color_counts_psg9['Undefined'] += num
                    results_psg9_list.append(
                        {"หมวดหมู่ PSG9": psg9_cat_name, "🔴 (Critical)": color_counts_psg9.get('Critical', 0),
                         "🟠 (High)": color_counts_psg9.get('High', 0), "🟡 (Medium)": color_counts_psg9.get('Medium', 0),
                         "🟢 (Low)": color_counts_psg9.get('Low', 0), "ไม่ระบุสี": color_counts_psg9.get('Undefined', 0),
                         "รวมทั้งหมด": total_for_psg9_cat})

                if results_psg9_list:
                    results_psg9_df = pd.DataFrame(results_psg9_list)
                    st.markdown("##### ตารางสรุปจำนวนอุบัติการณ์ตามหมวดหมู่ PSG9 และระดับความเสี่ยง:")
                    st.dataframe(results_psg9_df.set_index("หมวดหมู่ PSG9"), use_container_width=True)
                    if not results_psg9_df.empty:
                        plot_psg9_df_melted = results_psg9_df.melt(id_vars=["หมวดหมู่ PSG9", "รวมทั้งหมด", "ไม่ระบุสี"],
                                                                   value_vars=["🔴 (Critical)", "🟠 (High)", "🟡 (Medium)",
                                                                               "🟢 (Low)"], var_name="ระดับความเสี่ยง",
                                                                   value_name="จำนวนอุบัติการณ์")
                        plot_psg9_df_melted = plot_psg9_df_melted[plot_psg9_df_melted["จำนวนอุบัติการณ์"] > 0]
                        if not plot_psg9_df_melted.empty:
                            pastel_color_map_psg9_updated = {"🔴 (Critical)": "#FF9999", "🟠 (High)": "#FFCC99",
                                                             "🟡 (Medium)": "#FFD900", "🟢 (Low)": "#99FF99"}
                            fig_psg9 = px.bar(plot_psg9_df_melted, x="หมวดหมู่ PSG9", y="จำนวนอุบัติการณ์",
                                              color="ระดับความเสี่ยง",
                                              title="สัดส่วนอุบัติการณ์ตามหมวดหมู่ PSG9 และระดับความเสี่ยง",
                                              color_discrete_map=pastel_color_map_psg9_updated, barmode="stack")
                            fig_psg9.update_layout(xaxis_title="หมวดหมู่ PSG9", yaxis_title="จำนวนอุบัติการณ์",
                                                   xaxis_tickangle=-30, bargap=0.2, )
                            for trace in fig_psg9.data:
                                if isinstance(trace, go.Bar): trace.marker.cornerradius = 5
                            st.plotly_chart(fig_psg9, use_container_width=True)
                        else:
                            st.info("ไม่มีข้อมูลอุบัติการณ์ PSG9 ที่มีระดับความเสี่ยงสำหรับแสดงในกราฟ")
                else:
                    st.info("ไม่สามารถประมวลผลข้อมูลสำหรับตารางสรุปตามหมวดหมู่ PSG9 ได้")

    # ===============================================================================================
    # START OF MODIFIED SECTION for "วิเคราะห์อุบัติการณ์ที่ต้องดำเนินการแก้ไข"
    # ===============================================================================================
    elif selected_analysis == "วิเคราะห์อุบัติการณ์ที่ต้องดำเนินการแก้ไข":
        st.subheader("🔎 วิเคราะห์อุบัติการณ์: ภาพรวมตามกลุ่ม, มาตรฐานสำคัญ และสถานะการแก้ไข")

        # --- Part 1: ตารางกลุ่มอุบัติการณ์ทางคลินิก (C) และทั่วไป (G) ---
        st.markdown("#### 1. สรุปอุบัติการณ์ตามกลุ่มหลักและระดับความรุนแรง")
        clinical_incident_codes = {'CPE101', 'CPE201', 'CPE202', 'CPE203', 'CPE204', 'CPE301', 'CPE302', 'CPE303',
                                   'CPE304', 'CPE305', 'CPE306', 'CPE401', 'CPE402', 'CPE403', 'CPE404', 'CPE405',
                                   'CPE406', 'CPE407', 'CPE408', 'CPE409', 'CPE410', 'CPE411', 'CPI101', 'CPI201',
                                   'CPI202', 'CPI203', 'CPI204', 'CPI205', 'CPI301', 'CPI302', 'CPI303', 'CPI401',
                                   'CPL101', 'CPL102', 'CPL103', 'CPL201', 'CPL202', 'CPL203', 'CPM101', 'CPM102',
                                   'CPM103', 'CPM104', 'CPM105', 'CPM106', 'CPM107', 'CPM201', 'CPM202', 'CPM203',
                                   'CPM204', 'CPM205', 'CPM206', 'CPM207', 'CPM208', 'CPM301', 'CPM302', 'CPM303',
                                   'CPM304', 'CPM401', 'CPM402', 'CPM403', 'CPM404', 'CPM501', 'CPM502', 'CPM503',
                                   'CPM504', 'CPM505', 'CPM506', 'CPP101', 'CPP201', 'CPP202', 'CPP203', 'CPP204',
                                   'CPP205', 'CPP206', 'CPP207', 'CPP301', 'CPP302', 'CPP303', 'CPP304', 'CPP305',
                                   'CPP306', 'CPP307', 'CPP308', 'CPP309', 'CPP310', 'CPP311', 'CPP401', 'CPP402',
                                   'CPP403', 'CPP404', 'CPP405', 'CPP406', 'CPP407', 'CPP501', 'CPP502', 'CPP503',
                                   'CPP504', 'CPP505', 'CPP506', 'CPP601', 'CPP602', 'CPS101', 'CPS102', 'CPS103',
                                   'CPS104', 'CPS105', 'CPS106', 'CPS107', 'CPS108', 'CPS109', 'CPS110', 'CPS111',
                                   'CPS112', 'CPS113', 'CPS114', 'CPS115', 'CPS116', 'CPS117', 'CPS118', 'CPS201',
                                   'CPS202', 'CPS203', 'CPS301', 'CPS302', 'CPS303', 'CPS304', 'CPS305', 'CPS306',
                                   'CPS307', 'CPS308', 'CSD101', 'CSD102', 'CSD103', 'CSD104', 'CSD105', 'CSD106',
                                   'CSD107', 'CSD108', 'CSD109', 'CSD110', 'CSD111', 'CSE101', 'CSE102', 'CSE103',
                                   'CSE104', 'CSE105', 'CSE106', 'CSE107', 'CSE108', 'CSE201', 'CSE202', 'CSE203',
                                   'CSG101', 'CSG102', 'CSG103', 'CSG104', 'CSG105', 'CSG106', 'CSG107', 'CSG201',
                                   'CSG301', 'CSG302', 'CSG303', 'CSG304', 'CSG305', 'CSG306', 'CSM101', 'CSM102',
                                   'CSM103', 'CSM104', 'CSM105', 'CSM106', 'CSM107', 'CSM201', 'CSM301', 'CSM302',
                                   'CSM303', 'CSM401', 'CSM402', 'CSM403', 'CSM404', 'CSM501', 'CSM502', 'CSM503',
                                   'CSM504', 'CSM601', 'CSM602', 'CSM603', 'CSM604', 'CSM605', 'CSM606', 'CSM607',
                                   'CSM608', 'CSM609', 'CSO101', 'CSO102', 'CSO103', 'CSO104', 'CSO105', 'CSO106',
                                   'CSP101', 'CSP102', 'CSP103', 'CSP104', 'CSP105', 'CSP201', 'CSP202', 'CSP203',
                                   'CSS101', 'CSS102', 'CSS103', 'CSS104', 'CSS105', 'CSS106', 'CSS107', 'CSS108',
                                   'CSS201', 'CSS202', 'CSS203'}
        general_incident_codes = {'GOE101', 'GOE201', 'GOI101', 'GOI102', 'GOI103', 'GOI104', 'GOI105', 'GOI106',
                                  'GOI107', 'GOI108', 'GOI201', 'GOI202', 'GOI203', 'GOI204', 'GOI205', 'GOI206',
                                  'GOI207', 'GOL101', 'GOL102', 'GOM101', 'GOM102', 'GOM103', 'GOM201', 'GOP101',
                                  'GOP201', 'GOS101', 'GOS102', 'GOS103', 'GOS201', 'GOS202', 'GOS301', 'GPE101',
                                  'GPE102', 'GPE201', 'GPE202', 'GPE203', 'GPE204', 'GPE205', 'GPE206', 'GPE207',
                                  'GPE301', 'GPE302', 'GPE303', 'GPE304', 'GPE305', 'GPI101', 'GPI102', 'GPI103',
                                  'GPI104', 'GPI201', 'GPI202', 'GPI203', 'GPI204', 'GPL101', 'GPL102', 'GPL103',
                                  'GPL104', 'GPL105', 'GPL106', 'GPL201', 'GPL202', 'GPL203', 'GPL204', 'GPL205',
                                  'GPM101', 'GPM102', 'GPM103', 'GPM104', 'GPM203', 'GPM204', 'GPM205', 'GPM206',
                                  'GPM207', 'GPM208', 'GPP101', 'GPP102', 'GPP103', 'GPP201', 'GPP202', 'GPP203',
                                  'GPP204', 'GPP205', 'GPP206', 'GPP207', 'GPP208', 'GPP209', 'GPP210', 'GPP211',
                                  'GPP212', 'GPP301', 'GPP302', 'GPP303', 'GPS101', 'GPS102', 'GPS103', 'GPS104',
                                  'GPS105', 'GPS106', 'GPS201', 'GPS202', 'GPS203', 'GPS204'}

        if 'รหัส' in df.columns and 'หมวด' in df.columns and 'Impact Level' in df.columns:
            df_clinical_group = df[df['รหัส'].isin(clinical_incident_codes)].copy()
            create_severity_table(df_clinical_group, 'หมวด', "กลุ่มอุบัติการณ์ทางคลินิก (C)")
            st.markdown("---")

            df_general_group = df[df['รหัส'].isin(general_incident_codes)].copy()
            create_severity_table(df_general_group, 'หมวด', "กลุ่มอุบัติการณ์ทั่วไป (G)")
        else:
            st.warning(
                "ไม่พบคอลัมน์ 'รหัส', 'หมวด', หรือ 'Impact Level' ในข้อมูล จึงไม่สามารถสร้างตารางสรุปตามกลุ่มหลักได้")
        st.markdown("---")

        # --- Part 2: ตารางสรุปอุบัติการณ์ตามมาตรฐานสำคัญจำเป็น 9 ข้อ ---
        st.markdown("#### 2. สรุปอุบัติการณ์ตามมาตรฐานสำคัญ 9 ข้อและระดับความรุนแรง")
        if 'หมวดหมู่มาตรฐานสำคัญ' in df.columns and 'Impact Level' in df.columns:
            df_psg9_for_table = df[
                ~df['หมวดหมู่มาตรฐานสำคัญ'].isin([
                    "ไม่จัดอยู่ใน PSG9 Catalog",
                    "ไม่สามารถระบุ (Merge PSG9 ล้มเหลว)",
                    "ไม่สามารถระบุ (เช็คคอลัมน์ใน PSG9code.xlsx)",
                    "ไม่สามารถระบุ (PSG9code.xlsx ไม่ได้โหลด/ว่างเปล่า)"
                ]) & df['หมวดหมู่มาตรฐานสำคัญ'].notna()
                ].copy()

            if not df_psg9_for_table.empty and not df_psg9_for_table['หมวดหมู่มาตรฐานสำคัญ'].dropna().empty:
                psg9_standard_order_from_user = [
                    "01 ผ่าตัดผิดข้าง ผิดคน ผิดตำแหน่ง ผิดหัตถการ", "02 บุคลากรติดเชื้อจากการปฏิบัติหน้าที่",
                    "03 การติดเชื้อสำคัญ (SSI, VAP,CAUTI, CLABSI)",
                    "04 การเกิด Medication Error และ Adverse Drug Event",
                    "05 การให้เลือดผิดคน ผิดหมู่ ผิดชนิด",
                    "06 การรายงานผลการตรวจทางห้องปฏิบัติการ/พยาธิวิทยา คลาดเคลื่อน",
                    "07 การระบุตัวผู้ป่วยผิดพลาด", "08 ความคลาดเคลื่อนในการวินิจฉัยโรค",
                    "09 การคัดกรองที่ห้องฉุกเฉินคลาดเคลื่อน"
                ]
                create_severity_table(df_psg9_for_table, 'หมวดหมู่มาตรฐานสำคัญ', "อุบัติการณ์ตามมาตรฐานสำคัญ 9 ข้อ",
                                      specific_row_order=psg9_standard_order_from_user)
            else:
                st.info("ไม่พบอุบัติการณ์ที่เกี่ยวข้องกับมาตรฐานสำคัญ 9 ข้อ (หลังจากกรองค่าที่ไม่เกี่ยวข้องออก)")
        else:
            st.warning("ไม่พบคอลัมน์ 'หมวดหมู่มาตรฐานสำคัญ' หรือ 'Impact Level' สำหรับการวิเคราะห์ตามมาตรฐาน 9 ข้อ")
        st.markdown("---")

        # --- Part 3: ตารางสรุป % การแก้ไขอุบัติการณ์รุนแรง ---
        val_row1_total_sum = f"{total_severe_incidents:,}"
        val_row1_psg9_sum = f"{total_severe_psg9_incidents:,}"
        val_row2_total_sum = "N/A";
        val_row2_psg9_sum = "N/A"
        val_row3_total_pct_str_sum = "N/A";
        val_row3_psg9_pct_str_sum = "N/A"
        has_resulting_actions_column = 'Resulting Actions' in df.columns
        if has_resulting_actions_column:
            if isinstance(total_severe_unresolved_incidents_val, int):
                val_row2_total_sum = f"{total_severe_unresolved_incidents_val:,}"
                if total_severe_incidents > 0:
                    val_row3_total_pct_str_sum = f"{(total_severe_unresolved_incidents_val / total_severe_incidents) * 100:.2f}%"
                elif total_severe_unresolved_incidents_val == 0:
                    val_row3_total_pct_str_sum = "0.00%"
                else:
                    val_row3_total_pct_str_sum = "คำนวณไม่ได้"
            if isinstance(total_severe_unresolved_psg9_incidents_val, int):
                val_row2_psg9_sum = f"{total_severe_unresolved_psg9_incidents_val:,}"
                if total_severe_psg9_incidents > 0:
                    val_row3_psg9_pct_str_sum = f"{(total_severe_unresolved_psg9_incidents_val / total_severe_psg9_incidents) * 100:.2f}%"
                elif total_severe_unresolved_psg9_incidents_val == 0:
                    val_row3_psg9_pct_str_sum = "0.00%"
                else:
                    val_row3_psg9_pct_str_sum = "คำนวณไม่ได้"
        else:
            st.warning("ไม่พบคอลัมน์ 'Resulting Actions', ไม่สามารถคำนวณข้อมูล 'ที่ยังไม่ถูกแก้ไข' ได้", icon="⚠️")
        summary_action_data = [
            {"รายละเอียด": "1. จำนวนอุบัติการณ์รุนแรง E-I & 3-5", "ทั้งหมด": val_row1_total_sum,
             "เฉพาะ PSG9": val_row1_psg9_sum},
            {"รายละเอียด": "2. อุบัติการณ์ E-I & 3-5 ที่ยังไม่ได้รับการแก้ไข", "ทั้งหมด": val_row2_total_sum,
             "เฉพาะ PSG9": val_row2_psg9_sum},
            {"รายละเอียด": "3. % อุบัติการณ์ E-I & 3-5 ที่ยังไม่ได้รับการแก้ไข", "ทั้งหมด": val_row3_total_pct_str_sum,
             "เฉพาะ PSG9": val_row3_psg9_pct_str_sum},
        ]
        summary_action_df = pd.DataFrame(summary_action_data)
        st.markdown("##### 3. ตารางสรุปเปอร์เซ็นต์การแก้ไขอุบัติการณ์รุนแรง (E-I & 3-5):")
        st.dataframe(summary_action_df.set_index('รายละเอียด'), use_container_width=True)
        st.markdown("---")

        if has_resulting_actions_column:
            if 'df_severe_unresolved' in locals() and isinstance(df_severe_unresolved,
                                                                 pd.DataFrame) and not df_severe_unresolved.empty:
                expander1_label = f"ดูรายการอุบัติการณ์ระดับ E-I & 3-5 ทั้งหมด ที่ยังไม่ถูกแก้ไข ({total_severe_unresolved_incidents_val} รายการ)"
                with st.expander(expander1_label, expanded=False):
                    actual_cols_exp1 = [col for col in display_cols_common if col in df_severe_unresolved.columns]
                    st.dataframe(
                        df_severe_unresolved[actual_cols_exp1].sort_values(by='Occurrence Date', ascending=False),
                        hide_index=True, use_container_width=True)
            else:
                st.info(
                    "ไม่พบข้อมูลอุบัติการณ์รุนแรงทั้งหมดที่ยังไม่ถูกแก้ไข (df_severe_unresolved ว่างเปล่า หรือไม่ได้ถูกสร้าง)")

            df_severe_unresolved_psg9_for_expander = pd.DataFrame()
            count_for_label_psg9_unresolved = 0
            if 'df_severe_unresolved' in locals() and isinstance(df_severe_unresolved,
                                                                 pd.DataFrame) and not df_severe_unresolved.empty and \
                    psg9_r_codes_for_counting and 'รหัส' in df_severe_unresolved.columns:
                df_severe_unresolved_psg9_for_expander = df_severe_unresolved[
                    df_severe_unresolved['รหัส'].isin(psg9_r_codes_for_counting)].copy()

            if isinstance(total_severe_unresolved_psg9_incidents_val, int):
                count_for_label_psg9_unresolved = total_severe_unresolved_psg9_incidents_val
            elif not df_severe_unresolved_psg9_for_expander.empty:
                count_for_label_psg9_unresolved = df_severe_unresolved_psg9_for_expander.shape[0]

            expander2_label = f"ดูรายการอุบัติการณ์ระดับ E-I & 3-5 เฉพาะ PSG9 ที่ยังไม่ถูกแก้ไข ({count_for_label_psg9_unresolved} รายการ)"
            with st.expander(expander2_label, expanded=False):
                if not df_severe_unresolved_psg9_for_expander.empty:
                    actual_cols_exp2 = [col for col in display_cols_common if
                                        col in df_severe_unresolved_psg9_for_expander.columns]
                    df_display_exp2 = df_severe_unresolved_psg9_for_expander.copy()
                    if 'PSG9code_df_master' in locals() and isinstance(PSG9code_df_master,
                                                                       pd.DataFrame) and not PSG9code_df_master.empty:
                        cols_to_merge_psg_detail = ['รหัส']
                        if 'หมวดหมู่PSG' in PSG9code_df_master.columns: cols_to_merge_psg_detail.append('หมวดหมู่PSG')
                        if 'ชื่ออุบัติการณ์ย่อย' in PSG9code_df_master.columns: cols_to_merge_psg_detail.append(
                            'ชื่ออุบัติการณ์ย่อย')
                        if len(cols_to_merge_psg_detail) > 1:
                            df_display_exp2 = pd.merge(df_display_exp2,
                                                       PSG9code_df_master[cols_to_merge_psg_detail].drop_duplicates(
                                                           subset=['รหัส']), on='รหัส', how='left',
                                                       suffixes=('', '_detail'))

                    final_cols_exp2_check = ['Occurrence Date', 'Incident', 'รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง',
                                             'Impact', 'Impact Level', 'รายละเอียดการเกิด', 'Resulting Actions',
                                             'หมวดหมู่PSG', 'ชื่ออุบัติการณ์ย่อย', 'หมวดหมู่มาตรฐานสำคัญ']
                    final_cols_exp2 = [col for col in final_cols_exp2_check if col in df_display_exp2.columns]
                    st.dataframe(df_display_exp2[final_cols_exp2].sort_values(by='Occurrence Date', ascending=False),
                                 hide_index=True, use_container_width=True)
                else:
                    st.info("ไม่พบอุบัติการณ์รุนแรง PSG9 ที่ยังไม่ถูกแก้ไข")
        else:
            st.info("ไม่สามารถแสดงรายการอุบัติการณ์ที่ยังไม่แก้ไขได้ เนื่องจากไม่พบคอลัมน์ 'Resulting Actions'")
    # ===============================================================================================
    # END OF CORRECTED SECTION for "วิเคราะห์อุบัติการณ์ที่ต้องดำเนินการแก้ไข"
    # ===============================================================================================

    elif selected_analysis == "รายการ Sentinel Events":
        st.subheader("รายการ Sentinel Events ที่ตรวจพบ")
        df_sent_an = df.copy();
        df_sent_an['รหัส'] = df_sent_an['รหัส'].astype(str);
        df_sent_an['Impact'] = df_sent_an['Impact'].astype(str)
        df_sent_an['Sentinel code for analysis'] = df_sent_an['รหัส'] + '-' + df_sent_an['Impact']
        sent_rec_found = df_sent_an[df_sent_an['Sentinel code for analysis'].isin(sentinel_composite_keys)]
        if 'Sentinel2024_df' in locals() and isinstance(Sentinel2024_df,
                                                        pd.DataFrame) and not Sentinel2024_df.empty and 'Sentinel code' in Sentinel2024_df.columns and 'ชื่ออุบัติการณ์ความเสี่ยง' in Sentinel2024_df.columns:
            sent_rec_found = pd.merge(sent_rec_found, Sentinel2024_df[['Sentinel code', 'ชื่ออุบัติการณ์ความเสี่ยง']],
                                      left_on='Sentinel code for analysis', right_on='Sentinel code', how='left',
                                      suffixes=('', '_m'))
        else:
            sent_rec_found['ชื่ออุบัติการณ์ความเสี่ยง_m'] = 'N/A (Def Missing)'
        sent_disp_cols = ['รหัส', 'Incident', 'Impact', 'Occurrence Date', 'รายละเอียดการเกิด',
                          'ชื่ออุบัติการณ์ความเสี่ยง_m']
        act_sent_cols = [c for c in sent_disp_cols if c in sent_rec_found.columns]
        sent_rec_disp = sent_rec_found[act_sent_cols]
        if 'ชื่ออุบัติการณ์ความเสี่ยง_m' in sent_rec_disp.columns: sent_rec_disp = sent_rec_disp.rename(
            columns={'ชื่ออุบัติการณ์ความเสี่ยง_m': 'Sentinel Event Name'})
        sent_rec_disp = sent_rec_disp.sort_values(by='Occurrence Date', ascending=False, na_position='first')
        st.dataframe(sent_rec_disp, hide_index=True);
        st.write(f"ตรวจพบ Sentinel Events ทั้งหมด: {sent_rec_disp.shape[0]} รายการ")

    elif selected_analysis == "สรุปอุบัติการณ์ตามเป้าหมายทั่วไป (Goal-based)":
        st.subheader("สรุปอุบัติการณ์ตามเป้าหมายทั่วไป (ใช้คอลัมน์ 'หมวด' จาก AllCode)")
        if 'หมวด' in df.columns and not df['หมวด'].dropna().empty:
            type_list_g = sorted(df['หมวด'].dropna().unique())
            tab_gpsg, tab_gscr, tab_gpers, tab_gorg = st.tabs(
                ["Patient Safety (ทั่วไป)", "Clinical Risk (ทั่วไป)", "Personnel Safety (ทั่วไป)",
                 "Organization Safety (ทั่วไป)"])
            goal_map_general = {"Patient Safety (ทั่วไป)": "หมวดPSจากAllCode",
                                "Clinical Risk (ทั่วไป)": "หมวดCRจากAllCode",
                                "Personnel Safety (ทั่วไป)": "หมวดPERSจากAllCode",
                                "Organization Safety (ทั่วไป)": "หมวดORGจากAllCode"}  # !!! แก้ไขค่า value ตามข้อมูลจริง !!!
            with tab_gpsg:
                idx_psg = type_list_g.index(goal_map_general["Patient Safety (ทั่วไป)"]) if goal_map_general[
                                                                                                "Patient Safety (ทั่วไป)"] in type_list_g else 0
                cat_psg = st.selectbox("หมวด Patient Safety (ทั่วไป):", type_list_g, key="psg_gen_sel", index=idx_psg)
                if cat_psg in type_list_g:
                    sum_psg = create_goal_summary_table(df, cat_psg, e_up_non_numeric_levels_param=['A', 'B', 'C', 'D'])
                    if not sum_psg.empty:
                        st.dataframe(sum_psg)
                    else:
                        st.info(f"No data for: {cat_psg}")
            with tab_gscr:
                idx_scr = type_list_g.index(goal_map_general["Clinical Risk (ทั่วไป)"]) if goal_map_general[
                                                                                               "Clinical Risk (ทั่วไป)"] in type_list_g else 0
                cat_scr = st.selectbox("หมวด Clinical Risk (ทั่วไป):", type_list_g, key="scr_gen_sel", index=idx_scr)
                if cat_scr in type_list_g:
                    sum_scr = create_goal_summary_table(df, cat_scr, e_up_non_numeric_levels_param=['A', 'B', 'C', 'D'])
                    if not sum_scr.empty:
                        st.dataframe(sum_scr)
                    else:
                        st.info(f"No data for: {cat_scr}")
            with tab_gpers:
                idx_pers = type_list_g.index(goal_map_general["Personnel Safety (ทั่วไป)"]) if goal_map_general[
                                                                                                   "Personnel Safety (ทั่วไป)"] in type_list_g else 0
                cat_pers = st.selectbox("หมวด Personnel Safety (ทั่วไป):", type_list_g, key="pers_gen_sel",
                                        index=idx_pers)
                if cat_pers in type_list_g:
                    sum_pers = create_goal_summary_table(df, cat_pers,
                                                         e_up_non_numeric_levels_param=['A', 'B', 'C', 'D'])
                    if not sum_pers.empty:
                        st.dataframe(sum_pers)
                    else:
                        st.info(f"No data for: {cat_pers}")
            with tab_gorg:
                idx_org = type_list_g.index(goal_map_general["Organization Safety (ทั่วไป)"]) if goal_map_general[
                                                                                                     "Organization Safety (ทั่วไป)"] in type_list_g else 0
                cat_org = st.selectbox("หมวด Organization Safety (ทั่วไป):", type_list_g, key="org_gen_sel",
                                       index=idx_org)
                if cat_org in type_list_g:
                    sum_org = create_goal_summary_table(df, cat_org, e_up_non_numeric_levels_param=[],
                                                        e_up_numeric_levels_param=['1', '2'])
                    if not sum_org.empty:
                        st.dataframe(sum_org)
                    else:
                        st.info(f"No data for: {cat_org}")
        else:
            st.warning("Column 'หมวด' (จาก AllCode) not found or no valid data for General Goal-based Analysis.")

    elif selected_analysis == "ข้อมูลอุบัติการณ์ทั้งหมด (ตารางเต็ม)":
        st.subheader("ข้อมูลอุบัติการณ์ทั้งหมด (Fully Processed Table)")
        st.markdown("ตารางนี้แสดงข้อมูลอุบัติการณ์ทั้งหมดที่ผ่านการประมวลผลครบทุกขั้นตอนแล้ว")
        with st.expander("คลิกเพื่อดู/ซ่อน ตารางข้อมูลทั้งหมด", expanded=False):  # Set expanded to False by default
            st.dataframe(df, hide_index=False, use_container_width=True)
else:
    st.info("กรุณาอัปโหลดไฟล์ Excel (.xlsx) เพื่อเริ่มการวิเคราะห์")

# ==============================================================================
# END OF SCRIPT
# ==============================================================================
import streamlit as st
import pandas as pd
import os
import plotly.express as px

# ------------------ Login ------------------ #
def check_login():
    st.sidebar.title("🔐 Login Pegawai")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if username == "admin" and password == "pln123":
            st.session_state['logged_in'] = True
            st.success("Login berhasil!")
        else:
            st.error("Username/password salah")

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

check_login()
if not st.session_state['logged_in']:
    st.stop()

# ------------------ Setup ------------------ #
st.set_page_config(page_title="Dashboard TO AMR", layout="wide")
st.title("📊 Dashboard Target Operasi AMR - P2TL")
st.markdown("---")

# ------------------ Parameter Filter Dinamis ------------------ #
st.sidebar.header("⚙️ Parameter Filter")
param = {
    'cos_phi_max': st.sidebar.number_input("Max Cos Phi", value=0.85),
    'v_over_tm': st.sidebar.number_input("Over Voltage TM", value=240.0),
    'v_over_tr': st.sidebar.number_input("Over Voltage TR", value=241.0),
    'i_over': st.sidebar.number_input("Over Current Threshold", value=100.0),
    'i_max': st.sidebar.number_input("Arus Maksimum", value=120.0),
    'v_drop_min': st.sidebar.number_input("Tegangan Drop Minimum (V)", value=10.0),
    'unbalance_tol': st.sidebar.number_input("% Toleransi Unbalance", value=0.15)
}

# ------------------ Fungsi Cek ------------------ #
def cek_indikator(row):
    indikator = {}
    indikator['arus_hilang'] = all([row['CURRENT_L1'] == 0, row['CURRENT_L2'] == 0, row['CURRENT_L3'] == 0])
    indikator['over_current'] = any([row['CURRENT_L1'] > param['i_over'], row['CURRENT_L2'] > param['i_over'], row['CURRENT_L3'] > param['i_over']])
    indikator['over_voltage'] = any([row['VOLTAGE_L1'] > param['v_over_tm'], row['VOLTAGE_L2'] > param['v_over_tm'], row['VOLTAGE_L3'] > param['v_over_tm']])
    v = [row['VOLTAGE_L1'], row['VOLTAGE_L2'], row['VOLTAGE_L3']]
    indikator['v_drop'] = max(v) - min(v) > param['v_drop_min']
    indikator['cos_phi_kecil'] = any([row.get(f'POWER_FACTOR_L{i}', 1) < param['cos_phi_max'] for i in range(1, 4)])
    indikator['active_power_negative'] = any([row.get(f'ACTIVE_POWER_L{i}', 0) < 0 for i in range(1, 4)])
    indikator['arus_kecil_teg_kecil'] = all([
        all([row['CURRENT_L1'] < 1, row['CURRENT_L2'] < 1, row['CURRENT_L3'] < 1]),
        all([row['VOLTAGE_L1'] < 180, row['VOLTAGE_L2'] < 180, row['VOLTAGE_L3'] < 180]),
        any([row.get(f'ACTIVE_POWER_L{i}', 0) > 10 for i in range(1, 4)])
    ])
    arus = [row['CURRENT_L1'], row['CURRENT_L2'], row['CURRENT_L3']]
    max_i, min_i = max(arus), min(arus)
    indikator['unbalance_I'] = (max_i - min_i) / max_i > param['unbalance_tol'] if max_i > 0 else False
    indikator['v_lost'] = row.get('VOLTAGE_L1', 0) == 0 or row.get('VOLTAGE_L2', 0) == 0 or row.get('VOLTAGE_L3', 0) == 0
    indikator['In_more_Imax'] = any([row['CURRENT_L1'] > param['i_max'], row['CURRENT_L2'] > param['i_max'], row['CURRENT_L3'] > param['i_max']])
    indikator['active_power_negative_siang'] = row.get('ACTIVE_POWER_SIANG', 0) < 0
    indikator['active_power_negative_malam'] = row.get('ACTIVE_POWER_MALAM', 0) < 0
    indikator['active_p_lost'] = row.get('ACTIVE_POWER_L1', 0) == 0 and row.get('ACTIVE_POWER_L2', 0) == 0 and row.get('ACTIVE_POWER_L3', 0) == 0
    indikator['current_loop'] = row.get('CURRENT_LOOP', 0) == 1
    indikator['freeze'] = row.get('FREEZE', 0) == 1
    return indikator

# ------------------ Navigasi ------------------ #
tab1, tab2 = st.tabs(["📂 Data Historis", "➕ Upload Data Baru"])

# ------------------ Tab 1: Data Historis ------------------ #
with tab1:
    data_path = "data_harian.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        indikator_list = df.apply(cek_indikator, axis=1)
        indikator_df = pd.DataFrame(indikator_list.tolist())
        result = pd.concat([df[['LOCATION_CODE']], indikator_df], axis=1)
        result['Jumlah Potensi TO'] = indikator_df.sum(axis=1)
        top50 = result.sort_values(by='Jumlah Potensi TO', ascending=False).head(50)

        col1, col2, col3 = st.columns(3)
        col1.metric("📄 Total Data", len(df))
        col2.metric("🔢 Total IDPEL Unik", df['LOCATION_CODE'].nunique())
        col3.metric("🎯 Potensi Target Operasi", sum(result['Jumlah Potensi TO'] > 0))

        st.subheader("🏆 Top 50 Rekomendasi Target Operasi")
        st.dataframe(top50, use_container_width=True)

        st.subheader("📈 Visualisasi Indikator Anomali")
        indikator_counts = indikator_df.sum().sort_values(ascending=False).reset_index()
        indikator_counts.columns = ['Indikator', 'Jumlah']
        fig = px.bar(indikator_counts, x='Indikator', y='Jumlah', text='Jumlah', color='Indikator')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Belum ada data historis. Silakan upload pada tab berikutnya.")

# ------------------ Tab 2: Upload Data ------------------ #
with tab2:
    uploaded_file = st.file_uploader("📥 Upload File Excel AMR Harian", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file, sheet_name=0)
        df = df.dropna(subset=['LOCATION_CODE'])

        num_cols = [
            'CURRENT_L1', 'CURRENT_L2', 'CURRENT_L3',
            'VOLTAGE_L1', 'VOLTAGE_L2', 'VOLTAGE_L3',
            'ACTIVE_POWER_L1', 'ACTIVE_POWER_L2', 'ACTIVE_POWER_L3',
            'POWER_FACTOR_L1', 'POWER_FACTOR_L2', 'POWER_FACTOR_L3',
            'ACTIVE_POWER_SIANG', 'ACTIVE_POWER_MALAM', 'CURRENT_LOOP', 'FREEZE'
        ]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        if os.path.exists(data_path):
            df_hist = pd.read_csv(data_path)
            df = pd.concat([df_hist, df], ignore_index=True).drop_duplicates()
        df.to_csv(data_path, index=False)
        st.success("Data berhasil ditambahkan ke histori.")

    if st.button("🗑️ Hapus Semua Data Historis"):
        if os.path.exists(data_path):
            os.remove(data_path)
            st.success("Data historis berhasil dihapus.")
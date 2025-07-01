import streamlit as st
import pandas as pd
import os
import plotly.express as px

# ------------------ Login ------------------ #
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    with st.sidebar:
        st.subheader("Login Pegawai")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == "admin" and password == "pln123":
                st.session_state['logged_in'] = True
                st.success("Login berhasil!")
                st.rerun()
            else:
                st.error("Username/password salah")
    st.stop()

# ------------------ Tombol Logout ------------------ #
st.markdown("""
    <style>
    .logout-button {
        position: absolute;
        top: 10px;
        right: 16px;
        background-color: #f44336;
        color: white;
        border: none;
        padding: 6px 12px;
        border-radius: 6px;
        cursor: pointer;
    }
    </style>
    <form action="#" method="post">
        <button class="logout-button" onclick="window.location.reload();">Logout</button>
    </form>
""", unsafe_allow_html=True)

# ------------------ Setup ------------------ #
st.set_page_config(page_title="Dashboard TO AMR", layout="wide")

with st.expander("📊 AMR", expanded=False):
    st.title("📊 Dashboard Target Operasi AMR - P2TL")
    st.markdown("---")
# ------------------ Ambil semua parameter threshold dari session state ------------------ #
param = {k: v for k, v in st.session_state.items() if isinstance(v, (int, float, float))}

# ------------------ Parameter Threshold Section ------------------ #
 with st.expander("⚙️ Setting Parameter"):
        st.markdown("""
        Operasi Logika yang digunakan di sini adalah **OR**. Dengan demikian, indikator yang sesuai dengan salah satu spesifikasi aturan tersebut akan di-highlight berwarna hijau cerah dan berkontribusi pada perhitungan potensi TO.
        """)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("#### Tegangan Drop")
            st.number_input("Set Batas Atas Tegangan Menengah (tm)", key="v_tm_max", value=56.0)
            st.number_input("Set Batas Atas Tegangan Rendah (tr)", key="v_tr_max", value=180.0)
            st.number_input("Set Batas Bawah Arus Besar tm", key="i_tm_min", value=0.5)
            st.number_input("Set Batas Bawah Arus Besar tr", key="i_tr_min", value=0.5)

            st.markdown("#### Arus Netral vs Arus Maks")
            st.number_input("Set Batas Bawah Arus Netral tm", key="neutral_tm", value=1.0)
            st.number_input("Set Batas Bawah Arus Netral tr", key="neutral_tr", value=10.0)

            st.markdown("#### Reverse Power")
            st.number_input("Set Non Aktif Power TM", key="reverse_p_tm", value=0.0)
            st.number_input("Set Non Aktif Power TR", key="reverse_p_tr", value=0.0)
            st.number_input("Set Batas Bawah Arus Reverse Power TM", key="reverse_i_tm", value=0.5)
            st.number_input("Set Batas Bawah Arus Reverse Power TR", key="reverse_i_tr", value=0.7)

        with col2:
            st.markdown("#### Tegangan Hilang")
            st.number_input("Nilai Tegangan Menengah Hilang (tm)", key="v_tm_zero", value=0.0)
            st.number_input("Nilai Tegangan Rendah Hilang (tr)", key="v_tr_zero", value=0.0)
            st.number_input("Set Batas Bawah Arus Besar tm", key="loss_tm_i", value=-1.0)
            st.number_input("Set Batas Bawah Arus Besar tr", key="loss_tr_i", value=-1.0)

            st.markdown("#### Arus Unbalance")
            st.number_input("Toleransi Unbalance TM", key="unbal_tol_tm", value=0.5)
            st.number_input("Toleransi Unbalance TR", key="unbal_tol_tr", value=0.5)
            st.number_input("Set Batas Bawah Arus Unbalance TM", key="unbal_i_tm", value=0.5)
            st.number_input("Set Batas Bawah Arus Unbalance TR", key="unbal_i_tr", value=1.0)

            st.markdown("#### Active Power Lost")
            st.number_input("Set Batas Bawah Arus P Lost", key="plost_i_min", value=0.5)

        with col3:
            st.markdown("#### Cos Phi Kecil")
            st.number_input("Cos Phi Max TM", key="cos_phi_tm", value=0.4)
            st.number_input("Cos Phi Max TR", key="cos_phi_tr", value=0.4)
            st.number_input("Set Batas Arus Besar tm", key="cos_i_tm", value=0.8)
            st.number_input("Set Batas Arus Besar tr", key="cos_i_tr", value=0.8)

            st.markdown("#### Arus < Tegangan Kecil")
            st.number_input("Set Selisih Tegangan TM", key="low_v_diff_tm", value=2.0)
            st.number_input("Set Selisih Tegangan TR", key="low_v_diff_tr", value=8.0)

        with col4:
            st.markdown("#### Arus Hilang")
            st.number_input("Set Batas Arus Hilang pada TM", key="loss_i_tm", value=0.02)
            st.number_input("Set Batas Arus Hilang pada TR", key="loss_i_tr", value=0.02)
            st.number_input("Set Batas Bawah Arus Maksimum tm", key="max_i_tm", value=1.0)
            st.number_input("Set Batas Bawah Arus Maksimum tr", key="max_i_tr", value=1.0)

            st.markdown("#### Over Current (Tak Langsung)")
            st.number_input("Set Batas bawah Arus Maks pada TM", key="over_i_tm", value=5.0)
            st.number_input("Set Batas bawah Arus Maks pada TR", key="over_i_tr", value=5.0)

            st.markdown("#### Over Voltage")
            st.number_input("Tegangan Maksimum TM", key="vmax_tm", value=62.0)
            st.number_input("Tegangan Maksimum TR", key="vmax_tr", value=241.0)

        st.markdown("---")
        st.markdown("### Kriteria TO")
        st.number_input("Jumlah Indikator ≥", key="min_indicator", value=1)
        st.number_input("Jumlah Bobot ≥", key="min_weight", value=2)
        st.number_input("Banyak Data yang Ditampilkan", key="top_limit", value=50)

# ------------------ Fungsi Cek ------------------ #
def cek_indikator(row):
    # fungsi untuk mendeteksi anomali teknis pelanggan AMR
    indikator = {}
    indikator['arus_hilang'] = all([row['CURRENT_L1'] == 0, row['CURRENT_L2'] == 0, row['CURRENT_L3'] == 0])
    indikator['over_current'] = any([
        row['CURRENT_L1'] > param.get('over_i_tm', 5.0),
        row['CURRENT_L2'] > param.get('over_i_tm', 5.0),
        row['CURRENT_L3'] > param.get('over_i_tm', 5.0)
    ])
    indikator['over_voltage'] = any([
        row['VOLTAGE_L1'] > param.get('vmax_tm', 62.0),
        row['VOLTAGE_L2'] > param.get('vmax_tm', 62.0),
        row['VOLTAGE_L3'] > param.get('vmax_tm', 62.0)
    ])
    v = [row['VOLTAGE_L1'], row['VOLTAGE_L2'], row['VOLTAGE_L3']]
    indikator['v_drop'] = max(v) - min(v) > param.get('low_v_diff_tm', 2.0)
    indikator['cos_phi_kecil'] = any([
        row.get(f'POWER_FACTOR_L{i}', 1) < param.get('cos_phi_tm', 0.4)
        for i in range(1, 4)
    ])
    indikator['active_power_negative'] = any([
        row.get(f'ACTIVE_POWER_L{i}', 0) < 0
        for i in range(1, 4)
    ])
    indikator['arus_kecil_teg_kecil'] = all([
        all([
            row['CURRENT_L1'] < 1,
            row['CURRENT_L2'] < 1,
            row['CURRENT_L3'] < 1
        ]),
        all([
            row['VOLTAGE_L1'] < 180,
            row['VOLTAGE_L2'] < 180,
            row['VOLTAGE_L3'] < 180
        ]),
        any([
            row.get(f'ACTIVE_POWER_L{i}', 0) > 10
            for i in range(1, 4)
        ])
    ])
    arus = [row['CURRENT_L1'], row['CURRENT_L2'], row['CURRENT_L3']]
    max_i, min_i = max(arus), min(arus)
    indikator['unbalance_I'] = (max_i - min_i) / max_i > param.get('unbal_tol_tm', 0.5) if max_i > 0 else False
    indikator['v_lost'] = (
        row.get('VOLTAGE_L1', 0) == 0 or
        row.get('VOLTAGE_L2', 0) == 0 or
        row.get('VOLTAGE_L3', 0) == 0
    )
    indikator['In_more_Imax'] = any([
        row['CURRENT_L1'] > param.get('max_i_tm', 1.0),
        row['CURRENT_L2'] > param.get('max_i_tm', 1.0),
        row['CURRENT_L3'] > param.get('max_i_tm', 1.0)
    ])
    indikator['active_power_negative_siang'] = row.get('ACTIVE_POWER_SIANG', 0) < 0
    indikator['active_power_negative_malam'] = row.get('ACTIVE_POWER_MALAM', 0) < 0
    indikator['active_p_lost'] = (
        row.get('ACTIVE_POWER_L1', 0) == 0 and
        row.get('ACTIVE_POWER_L2', 0) == 0 and
        row.get('ACTIVE_POWER_L3', 0) == 0
    )
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

        # Tambahkan kolom jumlah kemunculan LOCATION_CODE
        df['Jumlah Berulang'] = df.groupby('LOCATION_CODE')['LOCATION_CODE'].transform('count')

        indikator_list = df.apply(cek_indikator, axis=1)
        indikator_df = pd.DataFrame(indikator_list.tolist())
        indikator_df['Jumlah Berulang'] = df['Jumlah Berulang']

        result = pd.concat([df[['LOCATION_CODE']], indikator_df], axis=1)
        result['Jumlah Potensi TO'] = indikator_df.drop(columns='Jumlah Berulang').sum(axis=1)

        # Hilangkan duplikat LOCATION_CODE
        result_unique = result.drop_duplicates(subset='LOCATION_CODE')
        top50 = result_unique.sort_values(by='Jumlah Potensi TO', ascending=False).head(50)

        col1, col2, col3 = st.columns(3)
        col1.metric("📄 Total Data", len(df))
        col2.metric("🔢 Total IDPEL Unik", df['LOCATION_CODE'].nunique())
        col3.metric("🎯 Potensi Target Operasi", sum(result_unique['Jumlah Potensi TO'] > 0))

        st.subheader("🏆 Top 50 Rekomendasi Target Operasi")
        st.dataframe(top50, use_container_width=True)

        st.subheader("📈 Visualisasi Indikator Anomali")
        indikator_counts = indikator_df.drop(columns='Jumlah Berulang').sum().sort_values(ascending=False).reset_index()
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
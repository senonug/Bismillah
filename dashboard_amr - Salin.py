import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px

# ------------------ Login ------------------ #
st.set_page_config(page_title="T-Energy", layout="centered", page_icon="⚡")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    with st.container():
        st.markdown("<h1 style='text-align:center; color:#005aa7;'>T-Energy</h1>", unsafe_allow_html=True)
        with st.form("login_form"):
            st.subheader("Masuk ke Dashboard")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("🔒 Sign In with IAM PLN")
            if submitted:
                if username == "admin" and password == "pln123":
                    st.session_state["logged_in"] = True
                    st.success("Login berhasil! Selamat datang di T-Energy.")
                    st.rerun()
                else:
                    st.error("Username atau password salah")
        st.markdown("<hr><div style='text-align:center; font-size:0.85rem;'>© 2025 PT PLN (Persero). All rights reserved.</div>", unsafe_allow_html=True)
    st.stop()


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
with st.expander("📦 AMR", expanded=False):
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

# ------------------ TABS LAIN ------------------ #
tab_pasca, tab_prabayar = st.tabs(["💳 Pascabayar", "💡 Prabayar"])

with tab_pasca:
    st.title("📊 Dashboard Target Operasi Pascabayar")
    st.markdown("---")
    olap_path = "olap_pascabayar.csv"
    uploaded_file = st.file_uploader("📥 Upload File OLAP Pascabayar Bulanan", type=["xlsx"], key="pasca")

    if uploaded_file:
        try:
            df_new = pd.read_excel(uploaded_file)
            required_cols = ["THBLREK", "IDPEL", "NAMA", "ALAMAT", "NAMAGARDU", "KDDK", "PEMKWH", "JAMNYALA"]
            if not set(required_cols).issubset(df_new.columns):
                st.error("Kolom yang dibutuhkan tidak lengkap dalam file.")
            else:
                df_new = df_new[required_cols].dropna(subset=["IDPEL"])
                df_hist = pd.read_csv(olap_path) if os.path.exists(olap_path) else pd.DataFrame()
                df_all = pd.concat([df_hist, df_new]).drop_duplicates(subset=["THBLREK", "IDPEL"])
                df_all.to_csv(olap_path, index=False)
                st.success("Data berhasil ditambahkan ke histori OLAP Pascabayar.")
        except Exception as e:
            st.error(f"Gagal memproses file: {e}")

    if st.button("🗑 Hapus Histori OLAP Pascabayar"):
        if os.path.exists(olap_path):
            os.remove(olap_path)
            st.success("Histori OLAP berhasil dihapus.")

    if os.path.exists(olap_path):
        df = pd.read_csv(olap_path)

        with st.expander("📁 Tabel PEMKWH Bulanan"):
            df_pivot_kwh = df.pivot(index="IDPEL", columns="THBLREK", values="PEMKWH")
            st.dataframe(df_pivot_kwh, use_container_width=True)

        with st.expander("📁 Tabel JAMNYALA Bulanan"):
            df_pivot_jam = df.pivot(index="IDPEL", columns="THBLREK", values="JAMNYALA")
            st.dataframe(df_pivot_jam, use_container_width=True)
    else:
        df = pd.DataFrame()

    if not df.empty:
        st.subheader("🎯 Rekomendasi Target Operasi")

        thblrek_options = sorted(df["THBLREK"].unique())
        selected_thblrek = st.selectbox("Filter Bulan (THBLREK)", ["Semua"] + thblrek_options)
        if selected_thblrek != "Semua":
            df = df[df["THBLREK"] == selected_thblrek]

        idpel_selected = st.selectbox("Pilih IDPEL untuk Analisis Detail (Opsional)", ["Semua"] + df["IDPEL"].unique().tolist())

        with st.expander("⚙️ Parameter Indikator Risiko (Opsional)"):
            min_jamnyala = st.number_input("Jam Nyala Minimum", value=50)
            min_kwh_mean = st.number_input("Rata-Rata KWH Minimum", value=50)
            max_std = st.number_input("Standar Deviasi Maksimum", value=200)
            threshold_kwh_tinggi = st.number_input("Threshold KWH Tinggi", value=300)
            threshold_jamnyala_kecil = st.number_input("Threshold Jam Nyala Kecil", value=100)
            threshold_drop = st.number_input("Penurunan 3 Bulan Terakhir (%)", value=30)

        risk_df = df.groupby("IDPEL").agg(
            nama=("NAMA", "first"),
            alamat=("ALAMAT", "first"),
            std_kwh=("PEMKWH", "std"),
            mean_kwh=("PEMKWH", "mean"),
            min_kwh=("PEMKWH", "min"),
            max_kwh=("PEMKWH", "max"),
            zero_count=("PEMKWH", lambda x: (x == 0).sum()),
            count_months=("PEMKWH", "count"),
            mean_jamnyala=("JAMNYALA", "mean")
        ).reset_index()

        risk_df["pemakaian_zero_3x"] = risk_df["zero_count"] >= 3
        risk_df["jamnyala_abnormal"] = risk_df["mean_jamnyala"] < min_jamnyala
        risk_df["min_kwh_zero"] = risk_df["min_kwh"] == 0
        risk_df["rendah_rata"] = risk_df["mean_kwh"] < min_kwh_mean
        risk_df["variasi_tinggi"] = risk_df["std_kwh"] > max_std
        risk_df["jamnyala_kecil_tapi_kwh_tinggi"] = (risk_df["mean_jamnyala"] < threshold_jamnyala_kecil) & (risk_df["mean_kwh"] > threshold_kwh_tinggi)

        indikator_cols = [
            "pemakaian_zero_3x",
            "jamnyala_abnormal",
            "min_kwh_zero",
            "rendah_rata",
            "variasi_tinggi",
            "jamnyala_kecil_tapi_kwh_tinggi"
        ]
        risk_df["skor"] = risk_df[indikator_cols].sum(axis=1)

        skor_threshold = st.slider("Minimal Skor Risiko untuk TO", 1, len(indikator_cols), 3)
        df_to = risk_df[risk_df["skor"] >= skor_threshold].sort_values("skor", ascending=False)

        st.metric("Pelanggan Berpotensi TO", len(df_to))
        st.dataframe(df_to.head(1000), use_container_width=True)
        fig_risk = px.histogram(df_to, x="skor", nbins=len(indikator_cols), title="Distribusi Skor Risiko Pelanggan Pascabayar")
        st.plotly_chart(fig_risk, use_container_width=True)
        st.download_button("📤 Download Target Operasi Pascabayar", df_to.to_csv(index=False).encode(), file_name="target_operasi_pascabayar.csv", mime="text/csv")

        if idpel_selected != "Semua":
            st.subheader(f"📈 Riwayat Konsumsi Pelanggan {idpel_selected}")
            df_idpel = df[df["IDPEL"] == idpel_selected].sort_values("THBLREK")
            fig_line = px.line(df_idpel, x="THBLREK", y="PEMKWH", title="Grafik Konsumsi KWH Bulanan")
            st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Belum ada data histori OLAP pascabayar. Silakan upload terlebih dahulu.")
with tab_prabayar:
    st.title("📊 Dashboard Target Operasi Prabayar")
    st.markdown("---")
    uploaded_file = st.file_uploader("📥 Upload File Excel Prabayar", type=["xlsx"], key="prabayar")
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        df["skor_indikator"] = df.filter(like="POWER").gt(0).sum(axis=1)
        df["skor_risiko"] = df["skor_indikator"].apply(lambda x: "Tinggi" if x > 5 else "Sedang" if x > 2 else "Rendah")
        st.dataframe(df.head(50), use_container_width=True)
        fig = px.histogram(df, x="skor_risiko", color="skor_risiko", title="Distribusi Risiko Prabayar")
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("📤 Download Excel", df.to_csv(index=False).encode(), file_name="hasil_prabayar.csv", mime="text/csv")
    else:
        st.info("Silakan upload file Excel pelanggan prabayar.")
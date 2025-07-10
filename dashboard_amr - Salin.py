import streamlit as st
import pandas as pd
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
# Definisi path file
data_path = "amr_harian_histori.csv"
olap_path = "olap_pascabayar.csv"

# ------------------ Fungsi Aman Pivot ------------------ #
def safe_pivot_table(df, index, columns, values):
    try:
        return df.pivot(index=index, columns=columns, values=values)
    except ValueError:
        st.warning(f"Duplikasi ditemukan untuk {index}/{columns}. Menggunakan pivot_table dengan agregasi rata-rata.")
        try:
            return df.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
        except Exception as e:
            st.error(f"Gagal membuat pivot table: {e}")
            return pd.DataFrame()

# ------------------ TABS ------------------ #
tab2, tab_pasca, tab_prabayar = st.tabs(["📥 AMR Harian", "💳 Pascabayar", "💡 Prabayar"])

# ------------------ Tab AMR Harian ------------------ #
with tab2:
    uploaded_file = st.file_uploader("📥 Upload File Excel AMR Harian", type=["xlsx"])
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file, sheet_name=0)
            required_cols = ['LOCATION_CODE'] + [
                'CURRENT_L1', 'CURRENT_L2', 'CURRENT_L3',
                'VOLTAGE_L1', 'VOLTAGE_L2', 'VOLTAGE_L3',
                'ACTIVE_POWER_L1', 'ACTIVE_POWER_L2', 'ACTIVE_POWER_L3',
                'POWER_FACTOR_L1', 'POWER_FACTOR_L2', 'POWER_FACTOR_L3',
                'ACTIVE_POWER_SIANG', 'ACTIVE_POWER_MALAM', 'CURRENT_LOOP', 'FREEZE'
            ]
            if not set(required_cols).issubset(df.columns):
                st.error("File Excel tidak memiliki semua kolom yang diperlukan.")
                st.stop()
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
                df = pd.concat([df_hist, df], ignore_index=True).drop_duplicates(subset=['LOCATION_CODE'], keep='last')
            df.to_csv(data_path, index=False)
            st.success("Data berhasil ditambahkan ke histori.")
        except Exception as e:
            st.error(f"Gagal memproses file: {e}")
            st.stop()

    if st.button("🗑️ Hapus Semua Data Historis"):
        if os.path.exists(data_path):
            if st.checkbox("Konfirmasi penghapusan data historis"):
                os.remove(data_path)
                st.success("Data historis berhasil dihapus.")
            else:
                st.warning("Centang kotak konfirmasi untuk menghapus data.")

# ------------------ Tab Pascabayar ------------------ #
with tab_pasca:
    st.title("📊 Dashboard Target Operasi Pascabayar")
    st.markdown("---")
    uploaded_file = st.file_uploader("📥 Upload File OLAP Pascabayar Bulanan", type=["xlsx"], key="pasca")

    if uploaded_file:
        try:
            df_new = pd.read_excel(uploaded_file)
            required_cols = ["THBLREK", "IDPEL", "NAMA", "ALAMAT", "NAMAGARDU", "KDDK", "PEMKWH", "JAMNYALA"]
            if not set(required_cols).issubset(df_new.columns):
                st.error("Kolom yang dibutuhkan tidak lengkap dalam file.")
                st.stop()
            df_new = df_new[required_cols].dropna(subset=["IDPEL"])
            df_new["IDPEL"] = df_new["IDPEL"].astype(str)
            df_hist = pd.read_csv(olap_path) if os.path.exists(olap_path) else pd.DataFrame()
            df_all = pd.concat([df_hist, df_new]).drop_duplicates(subset=["THBLREK", "IDPEL"], keep="last")
            df_all.to_csv(olap_path, index=False)
            st.success("Data berhasil ditambahkan ke histori OLAP Pascabayar.")
        except Exception as e:
            st.error(f"Gagal memproses file: {e}")
            st.stop()

    if st.button("🗑 Hapus Histori OLAP Pascabayar"):
        if os.path.exists(olap_path):
            if st.checkbox("Konfirmasi penghapusan histori OLAP Pascabayar"):
                os.remove(olap_path)
                st.success("Histori OLAP berhasil dihapus.")
            else:
                st.warning("Centang kotak konfirmasi untuk menghapus data.")

    if os.path.exists(olap_path):
        df = pd.read_csv(olap_path)
        df["IDPEL"] = df["IDPEL"].astype(str)
        df = df.drop_duplicates(subset=["IDPEL", "THBLREK"], keep="last")

        if df.duplicated(subset=["IDPEL", "THBLREK"]).any():
            st.warning("⚠️ Terdapat duplikat kombinasi IDPEL dan THBLREK. Data telah dibersihkan.")

        with st.expander("📁 Tabel PEMKWH Bulanan"):
            df_pivot_kwh = safe_pivot_table(df, index="IDPEL", columns="THBLREK", values="PEMKWH")
            st.dataframe(df_pivot_kwh, use_container_width=True)

        with st.expander("📁 Tabel JAMNYALA Bulanan"):
            df_pivot_jam = safe_pivot_table(df, index="IDPEL", columns="THBLREK", values="JAMNYALA")
            st.dataframe(df_pivot_jam, use_container_width=True)
    else:
        df = pd.DataFrame()

    if not df.empty:
        st.subheader("🎯 Rekomendasi Target Operasi")
        thblrek_options = sorted(df["THBLREK"].dropna().unique().astype(str))
        selected_thblrek = st.selectbox("Filter Bulan (THBLREK)", ["Semua"] + thblrek_options)
        if selected_thblrek != "Semua":
            df = df[df["THBLREK"] == selected_thblrek]

        selected_idpel = st.selectbox(
            "🔍 Pilih IDPEL untuk Tabel & Grafik",
            ["Semua"] + sorted(df["IDPEL"].unique().tolist())
        )

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

        indikator_cols = [
            "pemakaian_zero_3x",
            "jamnyala_abnormal",
            "min_kwh_zero",
            "rendah_rata",
            "variasi_tinggi"
        ]

        risk_df["pemakaian_zero_3x"] = risk_df["zero_count"] >= 3
        risk_df["jamnyala_abnormal"] = risk_df["mean_jamnyala"] < 50
        risk_df["min_kwh_zero"] = risk_df["min_kwh"] == 0
        risk_df["rendah_rata"] = risk_df["mean_kwh"] < 50
        risk_df["variasi_tinggi"] = risk_df["std_kwh"] > 200

        risk_df["skor"] = risk_df[indikator_cols].sum(axis=1)
        skor_threshold = st.slider("Minimal Skor Risiko untuk TO", 1, len(indikator_cols), 3)
        df_to = risk_df[risk_df["skor"] >= skor_threshold].sort_values("skor", ascending=False)

        st.metric("Pelanggan Berpotensi TO", len(df_to))
        st.dataframe(df_to.head(1000), use_container_width=True)
        fig_risk = px.histogram(df_to, x="skor", nbins=len(indikator_cols), title="Distribusi Skor Risiko Pelanggan Pascabayar")
        st.plotly_chart(fig_risk, use_container_width=True)

        if selected_idpel != "Semua":
            st.subheader(f"📈 Riwayat Konsumsi Pelanggan {selected_idpel}")
            df_idpel = df[df["IDPEL"] == selected_idpel].sort_values("THBLREK")
            fig_line = px.line(df_idpel, x="THBLREK", y="PEMKWH", title="Grafik Konsumsi KWH Bulanan")
            st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Belum ada data histori OLAP pascabayar. Silakan upload terlebih dahulu.")

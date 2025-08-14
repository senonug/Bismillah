
import streamlit as st
import pandas as pd
import os
import plotly.express as px

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# ------------------ Utilities: ML & Similarity ------------------ #
def detect_to_ml(df):
    fitur_teknis = [
        "CURRENT_L1", "CURRENT_L2", "CURRENT_L3",
        "VOLTAGE_L1", "VOLTAGE_L2", "VOLTAGE_L3",
        "ACTIVE_POWER_L1", "ACTIVE_POWER_L2", "ACTIVE_POWER_L3",
        "ACTIVE_POWER_SIANG", "ACTIVE_POWER_MALAM",
        "POWER_FACTOR_L1", "POWER_FACTOR_L2", "POWER_FACTOR_L3",
        "CURRENT_LOOP", "FREEZE"
    ]
    df_clean = df.copy()
    for col in fitur_teknis:
        if col not in df_clean.columns:
            df_clean[col] = 0
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)

    X = df_clean[fitur_teknis]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    model.fit(X_scaled)

    df_result = df.copy()
    df_result['anomaly_score'] = model.decision_function(X_scaled)
    df_result['TO_PRED'] = model.predict(X_scaled)
    df_result['TO_PRED'] = df_result['TO_PRED'].map({1: 0, -1: 1})
    return df_result


def cari_pelanggan_mirip(df, idpel_target, n_tetangga=10):
    fitur = [
        "CURRENT_L1", "CURRENT_L2", "CURRENT_L3",
        "VOLTAGE_L1", "VOLTAGE_L2", "VOLTAGE_L3",
        "ACTIVE_POWER_L1", "ACTIVE_POWER_L2", "ACTIVE_POWER_L3",
        "ACTIVE_POWER_SIANG", "ACTIVE_POWER_MALAM",
        "POWER_FACTOR_L1", "POWER_FACTOR_L2", "POWER_FACTOR_L3",
        "CURRENT_LOOP", "FREEZE"
    ]
    df_clean = df.copy()
    for col in fitur:
        if col not in df_clean.columns:
            df_clean[col] = 0
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[fitur])

    model = NearestNeighbors(n_neighbors=n_tetangga + 1)
    model.fit(X_scaled)

    if idpel_target not in df_clean['LOCATION_CODE'].astype(str).values:
        return pd.DataFrame()

    idx_target = df_clean[df_clean['LOCATION_CODE'].astype(str) == idpel_target].index[0]
    distances, indices = model.kneighbors([X_scaled[idx_target]])

    hasil_mirip = df_clean.iloc[indices[0][1:]].copy()
    hasil_mirip['Distance'] = distances[0][1:]

    # Buang IDPEL yang sama dengan target & duplikasi LOCATION_CODE
    hasil_mirip = hasil_mirip[hasil_mirip['LOCATION_CODE'].astype(str) != idpel_target]
    hasil_mirip = hasil_mirip.drop_duplicates(subset='LOCATION_CODE')
    return hasil_mirip


# ------------------ App Config ------------------ #
st.set_page_config(page_title="T-Energy", layout="wide", page_icon="⚡")

# ------------------ Session & Login ------------------ #
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

st.markdown("""
    <style>
    .logout-button {
        background-color: #f44336;
        color: white;
        border: none;
        padding: 6px 12px;
        border-radius: 6px;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

if st.button("🔒 Logout", key="logout_button", help="Keluar dari dashboard"):
    st.session_state["logged_in"] = False
    st.success("Logout berhasil!")
    st.rerun()

# ------------------ Tabs ------------------ #
tab_amr, tab_pasca, tab_prabayar = st.tabs(["📥 AMR Harian", "💳 Pascabayar", "💡 Prabayar"])

# ------------------ Tab AMR Harian ------------------ #
with tab_amr: 
    st.title("📊 Dashboard Target Operasi AMR - P2TL")
    st.markdown("---")

    # shared paths (digunakan oleh tab Data & Upload)
    data_path = "data_harian.csv"

    # ------------------ Parameter Threshold Section ------------------ #
    with st.expander("⚙️ Setting Parameter"):
        st.markdown("""
        Operasi Logika yang digunakan di sini adalah **OR**. Dengan demikian, indikator yang sesuai dengan salah satu spesifikasi aturan tersebut akan di-highlight berwarna hijau cerah dan berkontribusi pada perhitungan potensi TO.
        """)

        colA, colB = st.columns(2)
        with colA:
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

        with colB:
            st.markdown("#### Cos Phi Kecil")
            st.number_input("Cos Phi Max TM", key="cos_phi_tm", value=0.4)
            st.number_input("Cos Phi Max TR", key="cos_phi_tr", value=0.4)
            st.number_input("Set Batas Arus Besar tm", key="cos_i_tm", value=0.8)
            st.number_input("Set Batas Arus Besar tr", key="cos_i_tr", value=0.8)

            st.markdown("#### Arus < Tegangan Kecil")
            st.number_input("Set Selisih Tegangan TM", key="low_v_diff_tm", value=2.0)
            st.number_input("Set Selisih Tegangan TR", key="low_v_diff_tr", value=8.0)

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

    # ------------------ Fungsi Cek Indikator ------------------ #
    def cek_indikator(row):
        indikator = {}
        indikator['arus_hilang'] = all([row['CURRENT_L1'] == 0, row['CURRENT_L2'] == 0, row['CURRENT_L3'] == 0])
        indikator['over_current'] = any([
            row['CURRENT_L1'] > st.session_state.get('over_i_tm', 5.0),
            row['CURRENT_L2'] > st.session_state.get('over_i_tm', 5.0),
            row['CURRENT_L3'] > st.session_state.get('over_i_tm', 5.0)
        ])
        indikator['over_voltage'] = any([
            row['VOLTAGE_L1'] > st.session_state.get('vmax_tm', 62.0),
            row['VOLTAGE_L2'] > st.session_state.get('vmax_tm', 62.0),
            row['VOLTAGE_L3'] > st.session_state.get('vmax_tm', 62.0)
        ])
        v = [row['VOLTAGE_L1'], row['VOLTAGE_L2'], row['VOLTAGE_L3']]
        indikator['v_drop'] = max(v) - min(v) > st.session_state.get('low_v_diff_tm', 2.0)
        indikator['cos_phi_kecil'] = any([
            row.get(f'POWER_FACTOR_L{i}', 1) < st.session_state.get('cos_phi_tm', 0.4)
            for i in range(1, 4)
        ])
        indikator['active_power_negative'] = any([
            row.get(f'ACTIVE_POWER_L{i}', 0) < 0
            for i in range(1, 4)
        ])
        indikator['arus_kecil_teg_kecil'] = all([
            all([row['CURRENT_L1'] < 1, row['CURRENT_L2'] < 1, row['CURRENT_L3'] < 1]),
            all([row['VOLTAGE_L1'] < 180, row['VOLTAGE_L2'] < 180, row['VOLTAGE_L3'] < 180]),
            any([row.get(f'ACTIVE_POWER_L{i}', 0) > 10 for i in range(1, 4)])
        ])
        arus = [row['CURRENT_L1'], row['CURRENT_L2'], row['CURRENT_L3']]
        max_i, min_i = max(arus), min(arus)
        indikator['unbalance_I'] = (max_i - min_i) / max_i > st.session_state.get('unbal_tol_tm', 0.5) if max_i > 0 else False
        indikator['v_lost'] = (row.get('VOLTAGE_L1', 0) == 0 or row.get('VOLTAGE_L2', 0) == 0 or row.get('VOLTAGE_L3', 0) == 0)
        indikator['In_more_Imax'] = any([
            row['CURRENT_L1'] > st.session_state.get('max_i_tm', 1.0),
            row['CURRENT_L2'] > st.session_state.get('max_i_tm', 1.0),
            row['CURRENT_L3'] > st.session_state.get('max_i_tm', 1.0)
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

    # ------------------ Sub Tabs ------------------ #
    sub_data, sub_upload = st.tabs(["📂 Data Historis", "➕ Upload Data Baru"])

    # ------------------ Data Historis ------------------ #
    with sub_data:
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)

            # --- Normalisasi dasar ---
            df = df.dropna(subset=['LOCATION_CODE']).copy()
            df['LOCATION_CODE'] = df['LOCATION_CODE'].astype(str).str.strip()

            # Tambahkan kolom pelanggan jika tidak ada
            for col in ['NAMA_PELANGGAN', 'TARIFF', 'POWER']:
                if col not in df.columns:
                    df[col] = "-"

            # Hitung jumlah kemunculan per IDPEL (untuk informasi 'Jumlah Berulang')
            df['Jumlah Berulang'] = df.groupby('LOCATION_CODE')['LOCATION_CODE'].transform('count')

            # Hitung indikator per baris
            indikator_list = df.apply(cek_indikator, axis=1)
            indikator_df = pd.DataFrame(indikator_list.tolist())
            indikator_df['LOCATION_CODE'] = df['LOCATION_CODE']
            indikator_df['Jumlah Berulang'] = df['Jumlah Berulang']

            # Agregasi indikator per LOCATION_CODE (OR/any)
            boolean_cols = [c for c in indikator_df.columns if c not in ['LOCATION_CODE', 'Jumlah Berulang']]
            agg_dict = {c: 'any' for c in boolean_cols}
            agg_dict['Jumlah Berulang'] = 'max'
            indikator_agg = indikator_df.groupby('LOCATION_CODE', as_index=False).agg(agg_dict)

            # Bobot & skor
            indikator_bobot = {
                'arus_hilang': 2, 'over_current': 1, 'over_voltage': 1, 'v_drop': 1,
                'cos_phi_kecil': 1, 'active_power_negative': 2, 'arus_kecil_teg_kecil': 1,
                'unbalance_I': 1, 'v_lost': 2, 'In_more_Imax': 1,
                'active_power_negative_siang': 2, 'active_power_negative_malam': 2,
                'active_p_lost': 2, 'current_loop': 2, 'freeze': 2
            }
            indikator_agg['Jumlah Indikator'] = indikator_agg[boolean_cols].sum(axis=1)

            def hitung_skor(row):
                s = 0
                for k, w in indikator_bobot.items():
                    if k in row and bool(row[k]):
                        s += w
                return s
            indikator_agg['Skor'] = indikator_agg.apply(hitung_skor, axis=1)

            # Info pelanggan terbaru per LOCATION_CODE
            if 'TANGGAL' in df.columns:
                df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], errors='coerce')
                df_info = df.sort_values('TANGGAL').dropna(subset=['TANGGAL']).groupby('LOCATION_CODE').tail(1)
            else:
                df_info = df.drop_duplicates(subset='LOCATION_CODE', keep='last')

            df_info = df_info[['LOCATION_CODE', 'NAMA_PELANGGAN', 'TARIF' if 'TARIF' in df_info.columns else 'TARIFF', 'POWER']].rename(
                columns={'NAMA_PELANGGAN': 'NAMA', 'TARIFF': 'TARIF', 'POWER': 'DAYA'}
            )
            if 'TARIF' not in df_info.columns:
                df_info['TARIF'] = '-'  # fallback kalau kolom tariff tidak ada

            # Merge rapi
            result = df_info.merge(indikator_agg, on='LOCATION_CODE', how='right')
            for c in ['NAMA', 'TARIF', 'DAYA']:
                if c in result.columns:
                    result[c] = result[c].fillna('-')

            # Tampilkan
            top_limit = st.session_state.get('top_limit', 50)
            top50 = (result.drop_duplicates(subset='LOCATION_CODE')
                 .sort_values(by='Skor', ascending=False)
                 .head(top_limit))

            col1, col2, col3 = st.columns([1.2, 1.2, 1])
            col1.metric("📄 Total Record Histori", len(df))
            col3.metric("🎯 Pelanggan Potensial TO", int((result['Skor'] > 0).sum()))

            st.subheader("🏆 Top 50 Rekomendasi Target Operasi")
            st.download_button(
                label="📥 Download Hasil Lengkap (Excel)",
                data=result.to_csv(index=False).encode('utf-8'),
                file_name="hasil_target_operasi_amr.csv",
                mime="text/csv"
            )

            kolom_tampil = ['LOCATION_CODE', 'NAMA', 'TARIF', 'DAYA'] + \
                           [k for k in indikator_bobot.keys() if k in result.columns] + \
                           ['Jumlah Berulang', 'Jumlah Indikator', 'Skor']
            st.dataframe(top50[kolom_tampil], use_container_width=True, height=600)

            # Visualisasi
            indikator_counts = indikator_agg[[c for c in indikator_bobot.keys() if c in indikator_agg.columns]].sum().sort_values(ascending=False).reset_index()
            indikator_counts.columns = ['Indikator', 'Jumlah']
            fig = px.bar(indikator_counts, x='Indikator', y='Jumlah', text='Jumlah', color='Indikator')

            # ==== Interaktif: klik bar untuk melihat IDPEL pada indikator terpilih ====
            selected_indicator = None
            try:
                from streamlit_plotly_events import plotly_events  # type: ignore
                st.markdown("**Klik salah satu batang (indikator) untuk melihat daftar IDPEL.**")
                selected_points = plotly_events(fig, click_event=True, hover_event=False, select_event=False, override_height=500)
                if selected_points:
                    selected_indicator = selected_points[0].get('x')
            except Exception:
                st.plotly_chart(fig, use_container_width=True)
                st.info("Interaksi klik membutuhkan paket `streamlit-plotly-events`. Karena belum tersedia, gunakan pemilih di bawah ini.")
                selected_indicator = st.selectbox("Pilih indikator untuk melihat daftar IDPEL:", indikator_counts['Indikator'].tolist())

            if selected_indicator:
                if selected_indicator in indikator_agg.columns:
                    id_list = indikator_agg.loc[indikator_agg[selected_indicator] == True, "LOCATION_CODE"].astype(str).unique().tolist()
                    detail_df = result[result["LOCATION_CODE"].astype(str).isin(id_list)][["LOCATION_CODE", "NAMA", "TARIF", "DAYA"]].drop_duplicates(subset="LOCATION_CODE")
                    st.subheader(f"📋 Daftar IDPEL untuk indikator: **{selected_indicator}** (unik: {len(detail_df)})")
                    st.dataframe(detail_df, use_container_width=True, height=400)
                    st.download_button(
                        "📥 Download daftar IDPEL (CSV)",
                        detail_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"idpel_{selected_indicator}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("Indikator tidak ditemukan di data.")
            # (rendered below in interactive block)

            # ML
            with st.expander("🤖 Prediksi TO dengan Machine Learning"):
                if st.button("🔍 Jalankan Prediksi TO (ML)"):
                    df_ml = detect_to_ml(df)
                    hasil_ml = df_ml[df_ml["TO_PRED"] == 1].copy()
                    st.metric("Jumlah Pelanggan Terindikasi TO (ML)", len(hasil_ml))
                    st.dataframe(hasil_ml[["LOCATION_CODE", "NAMA_PELANGGAN", "TARIFF" if "TARIFF" in hasil_ml.columns else "TARIF", "POWER", "TO_PRED", "anomaly_score"]].head(100), use_container_width=True)
                    st.download_button(
                        label="📥 Download Hasil Deteksi ML",
                        data=hasil_ml.to_csv(index=False).encode(),
                        file_name="hasil_ml_target_operasi.csv",
                        mime="text/csv"
                    )

            # Similarity
            with st.expander("🧬 Analisa Kemiripan Pelanggan dengan TO Terbukti"):
                idpel_input = st.text_input("Masukkan IDPEL Pelanggan yang Terbukti TO")
                if st.button("🔍 Cari Pelanggan Mirip"):
                    if idpel_input and idpel_input in df["LOCATION_CODE"].astype(str).values:
                        hasil_mirip = cari_pelanggan_mirip(df, idpel_input)
                        if not hasil_mirip.empty:
                            st.metric("Pelanggan Mirip Ditemukan", len(hasil_mirip))
                            view_cols = ["LOCATION_CODE", "NAMA_PELANGGAN", "TARIFF" if "TARIFF" in hasil_mirip.columns else "TARIF", "POWER", "Distance"]
                            view_cols = [c for c in view_cols if c in hasil_mirip.columns]
                            st.dataframe(hasil_mirip[view_cols].head(20), use_container_width=True)
                            st.download_button(
                                label="📥 Download Hasil Kemiripan",
                                data=hasil_mirip.to_csv(index=False).encode(),
                                file_name="hasil_kemiripan_pelanggan.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("IDPEL tidak ditemukan di data.")
                    else:
                        st.warning("Silakan masukkan IDPEL yang valid.")
        else:
            st.warning("Belum ada data historis. Silakan upload pada tab berikutnya.")

    # ------------------ Upload Data ------------------ #
    with sub_upload:
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

        # Konfirmasi hapus
        if st.button("🗑️ Hapus Semua Data Historis"):
            st.session_state['_confirm_del_amr'] = True

        if st.session_state.get('_confirm_del_amr'):
            st.warning("Konfirmasi penghapusan data historis AMR.")
            col_del1, col_del2 = st.columns(2)
            with col_del1:
                if st.button("✅ Ya, hapus"):
                    if os.path.exists(data_path):
                        os.remove(data_path)
                        st.success("Data historis berhasil dihapus.")
                    st.session_state['_confirm_del_amr'] = False
            with col_del2:
                if st.button("❌ Batal"):
                    st.session_state['_confirm_del_amr'] = False

# ------------------ Tab Pascabayar ------------------ #
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

    # Konfirmasi hapus OLAP
    if st.button("🗑 Hapus Histori OLAP Pascabayar"):
        st.session_state['_confirm_del_olap'] = True
    if st.session_state.get('_confirm_del_olap'):
        st.warning("Konfirmasi penghapusan histori OLAP Pascabayar.")
        colx, coly = st.columns(2)
        with colx:
            if st.button("✅ Ya, hapus OLAP"):
                if os.path.exists(olap_path):
                    os.remove(olap_path)
                    st.success("Histori OLAP berhasil dihapus.")
                st.session_state['_confirm_del_olap'] = False
        with coly:
            if st.button("❌ Batal"):
                st.session_state['_confirm_del_olap'] = False

    if os.path.exists(olap_path):
        df = pd.read_csv(olap_path)

        if df.duplicated(subset=["IDPEL", "THBLREK"]).any():
            st.warning("⚠️ Terdapat duplikat kombinasi IDPEL dan THBLREK. Data akan dirata-ratakan.")

        # Input nilai penurunan terlebih dahulu
        drop_threshold = st.number_input("Penurunan Pemakaian kWh 3 Bulan Terakhir (%)", value=30, min_value=0, max_value=100, step=1)
        use_drop_threshold = st.checkbox("Aktifkan Filter Penurunan Pemakaian kWh", value=False)

        selected_idpel = st.selectbox(
            "🔍 Pilih IDPEL untuk Tabel & Grafik",
            ["Semua"] + sorted(df["IDPEL"].astype(str).unique().tolist())
        )

        if selected_idpel != "Semua":
            df_temp = df[df["IDPEL"].astype(str) == selected_idpel].copy()
            if use_drop_threshold and len(df_temp) >= 3:
                df_temp = df_temp.sort_values("THBLREK", ascending=False)
                recent_3 = df_temp.head(3)["PEMKWH"].mean()
                previous = df_temp.tail(len(df_temp)-3)["PEMKWH"].mean() if len(df_temp) > 3 else df_temp["PEMKWH"].mean()
                if previous > 0:
                    drop_percent = ((previous - recent_3) / previous) * 100
                    if drop_percent < drop_threshold:
                        df_filtered = pd.DataFrame()
                    else:
                        df_filtered = df_temp
                else:
                    df_filtered = df_temp
            else:
                df_filtered = df_temp
        else:
            df_filtered = df.copy()

        with st.expander("📁 Tabel PEMKWH Bulanan"):
            df_pivot_kwh = df_filtered.pivot_table(index="IDPEL", columns="THBLREK", values="PEMKWH", aggfunc="mean")
            st.dataframe(df_pivot_kwh, use_container_width=True)

        with st.expander("📁 Tabel JAMNYALA Bulanan"):
            df_pivot_jam = df_filtered.pivot_table(index="IDPEL", columns="THBLREK", values="JAMNYALA", aggfunc="mean")
            st.dataframe(df_pivot_jam, use_container_width=True)

        if selected_idpel != "Semua" and not df_filtered.empty:
            st.subheader(f"📈 Riwayat Konsumsi Pelanggan {selected_idpel}")
            df_idpel = df[df["IDPEL"].astype(str) == selected_idpel].sort_values("THBLREK")
            df_idpel = df_idpel.dropna(subset=["THBLREK", "PEMKWH"])
            if df_idpel.empty:
                st.warning("Tidak ada data konsumsi untuk IDPEL yang dipilih.")
            else:
                df_idpel["THBLREK"] = pd.to_datetime(df_idpel["THBLREK"], format="%Y%m").dt.strftime("%b %Y")
                df_idpel["Moving_Avg"] = df_idpel["PEMKWH"].rolling(window=3, min_periods=1).mean()
                fig_line = px.line(df_idpel, x="THBLREK", y=["PEMKWH", "Moving_Avg"], title="Grafik Konsumsi KWH Bulanan",
                                  labels={"THBLREK": "Bulan", "value": "Konsumsi kWh (kWh)", "variable": "Metrik"},
                                  hover_data=["NAMA", "ALAMAT"])
                st.plotly_chart(fig_line, use_container_width=True)

        st.subheader("🎯 Rekomendasi Target Operasi")

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

        indikator_cols = [
            "pemakaian_zero_3x",
            "jamnyala_abnormal",
            "min_kwh_zero",
            "rendah_rata",
            "variasi_tinggi",
        ]
        risk_df["skor"] = risk_df[indikator_cols].sum(axis=1)

        skor_threshold = st.slider("Minimal Skor Risiko untuk TO", 1, len(indikator_cols), 3)
        df_to = risk_df[risk_df["skor"] >= skor_threshold].sort_values("skor", ascending=False)

        st.metric("Pelanggan Berpotensi TO", len(df_to))
        st.dataframe(df_to.head(1000), use_container_width=True)
        st.download_button("📄 Download Target Operasi Pascabayar", df_to.to_csv(index=False).encode(), file_name="target_operasi_pascabayar.csv", mime="text/csv")

    else:
        st.info("Belum ada data histori OLAP pascabayar. Silakan upload terlebih dahulu.")

# ------------------ Tab Prabayar ------------------ #
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

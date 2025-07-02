
import streamlit as st
import pandas as pd
import os
import plotly.express as px

# ------------------ Tampilan Login Profesional ------------------ #
st.set_page_config(page_title="T-Energy Dashboard", layout="centered", page_icon="⚡")
st.markdown("""
    <style>
    .stApp {
        background-color: #f9fcff;
    }
    h1, h2, h3 {
        color: #005aa7;
    }
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    with st.container():
        st.markdown("<h1 style='text-align:center; color:#005aa7;'>T-Energy</h1>", unsafe_allow_html=True)
        st.image("https://upload.wikimedia.org/wikipedia/commons/1/19/Logo_PLN.png", width=100)

        with st.form("login_form"):
            st.subheader("Masuk ke Dashboard")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("🔒 Sign In with IAM PLN")

            if submitted:
                if username == "admin" and password == "pln123":
                    st.session_state['logged_in'] = True
                    st.success("Login berhasil! Selamat datang di T-Energy.")
                    st.rerun()
                else:
                    st.error("Username atau password salah.")

        st.markdown("""
            <hr>
            <div style='text-align:center; font-size:0.9rem; color:#666;'>
            © 2025 PT PLN (Persero). All rights reserved.
            </div>
        """, unsafe_allow_html=True)
    st.stop()

# ------------------ Navigasi Tab ------------------ #
tab_amr, tab_prabayar, tab_pascabayar = st.tabs(["🔌 AMR", "💡 Prabayar", "📥 Pascabayar"])

with tab_amr:
    st.subheader("📦 Analisis Data AMR")

    uploaded_file = st.file_uploader("📥 Upload File Excel AMR Harian", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        df = df.dropna(subset=['LOCATION_CODE'])

        # Skoring risiko berdasarkan total arus & tegangan abnormal
        df['RISK_SCORE'] = (
            (df['CURRENT_L1'] > 80).astype(int) +
            (df['CURRENT_L2'] > 80).astype(int) +
            (df['CURRENT_L3'] > 80).astype(int) +
            (df['VOLTAGE_L1'] < 180).astype(int) +
            (df['VOLTAGE_L2'] < 180).astype(int) +
            (df['VOLTAGE_L3'] < 180).astype(int)
        )

        df['LEVEL'] = pd.cut(df['RISK_SCORE'], bins=[-1,1,3,6], labels=['Rendah','Sedang','Tinggi'])

        st.success(f"Data terdeteksi: {len(df)} baris")
        st.dataframe(df[['LOCATION_CODE', 'RISK_SCORE', 'LEVEL']].sort_values(by='RISK_SCORE', ascending=False))

        # Tombol Export
        import io
        from datetime import datetime

        to_export = df[['LOCATION_CODE', 'RISK_SCORE', 'LEVEL']]
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            to_export.to_excel(writer, index=False, sheet_name='Skoring')
        st.download_button("⬇️ Export Skoring ke Excel", data=buffer.getvalue(), file_name=f"skoring_amr_{datetime.now().date()}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("Silakan unggah file AMR harian (.xlsx) untuk mulai analisis skoring.")

with tab_prabayar:
    st.subheader("💡 Analisis Data Prabayar")
    st.warning("Fitur prabayar sedang dalam pengembangan. Silakan siapkan data token, arus, dan tegangan.")

with tab_pascabayar:
    st.subheader("📥 Integrasi Data Lapangan (DIL / Target Operasi)")
    st.info("Unggah file DIL (.xlsx) untuk mencocokkan ID pelanggan AMR dengan hasil temuan lapangan.")

    uploaded_dil = st.file_uploader("📎 Upload File DIL (XLSX)", type=["xlsx"], key="dil")
    if uploaded_dil:
        df_dil = pd.read_excel(uploaded_dil)
        st.success(f"Data DIL berhasil dimuat: {len(df_dil)} baris")
        st.dataframe(df_dil)

        if 'df' in locals():
            matched = df.merge(df_dil, left_on='LOCATION_CODE', right_on='IDPEL', how='inner')
            st.markdown("### 🔍 Hasil Pencocokan AMR vs DIL")
            st.dataframe(matched[['LOCATION_CODE', 'RISK_SCORE', 'LEVEL'] + [col for col in df_dil.columns if col != 'IDPEL']])
        else:
            st.warning("Harap unggah juga data AMR terlebih dahulu pada tab AMR.")

st.markdown("<hr><div style='text-align:center; font-size:0.8rem;'>© 2025 PT PLN (Persero)</div>", unsafe_allow_html=True)
st.title("📊 Dashboard T-Energy (AMR / Prabayar / Pascabayar)")
st.markdown("Silakan pilih tab dan unggah data untuk mulai analisis.")


# ------------------ FUNGSI CEK INDIKATOR ------------------ #
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
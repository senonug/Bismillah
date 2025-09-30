from io import BytesIO
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors

# ===================== Helpers & Config ===================== #

st.set_page_config(page_title="T-Energy", layout="wide", page_icon="⚡")

DATA_PATH_AMR = "data_harian.csv"
DATA_PATH_OLAP = "olap_pascabayar.csv"

# ---------- Caching ---------- #
@st.cache_data(show_spinner=False)
def load_csv_safe(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def _filter_customer_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only LOCATION_TYPE == 'Customer' or 'COSTUMER' (case-insensitive)."""
    if 'LOCATION_TYPE' not in df.columns:
        st.warning("Kolom LOCATION_TYPE tidak ditemukan. Tidak dapat memfilter ke Customer saja.")
        return df
    lt = df['LOCATION_TYPE'].astype(str).str.strip().str.lower()
    allowed = {'customer', 'costumer'}
    return df[lt.isin(allowed)].copy()

def _ensure_customer_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in ['NAMA_PELANGGAN', 'TARIF', 'TARIFF', 'POWER']:
        if col not in df.columns:
            df[col] = "-"
    return df

def _numericize
TECH_COLS = [
    "VOLTAGE_L1","VOLTAGE_L2","VOLTAGE_L3",
    "CURRENT_L1","CURRENT_L2","CURRENT_L3","CURRENT_N",
    "VOLTAGE_ANGLE_L1","VOLTAGE_ANGLE_L2","VOLTAGE_ANGLE_L3",
    "CURRENT_ANGLE_L1","CURRENT_ANGLE_L2","CURRENT_ANGLE_L3",
    "POWER_FACTOR_L1","POWER_FACTOR_L2","POWER_FACTOR_L3",
    "ACTIVE_POWER_L1","ACTIVE_POWER_L2","ACTIVE_POWER_L3","ACTIVE_POWER_TOTAL",
    "KWH_ABS_TOTAL",
    "APPARENT_POWER_L1","APPARENT_POWER_L2","APPARENT_POWER_L3",
    "BILL_REFF_KWH"
]

def make_xlsx_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    """Return DataFrame as XLSX bytes with reasonable column widths."""
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        ws = writer.sheets[sheet_name]
        for i, col in enumerate(df.columns):
            try:
                width = min(40, max(10, int(df[col].astype(str).str.len().quantile(0.9)) + 2))
            except Exception:
                width = 16
            ws.set_column(i, i, width)
    return out.getvalue()
(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
        else:
            df[c] = 0.0
    return df

# ---------- ML utilities ---------- #
FITUR_TEKNIS = [
    "CURRENT_L1","CURRENT_L2","CURRENT_L3",
    "VOLTAGE_L1","VOLTAGE_L2","VOLTAGE_L3",
    "ACTIVE_POWER_L1","ACTIVE_POWER_L2","ACTIVE_POWER_L3",
    "ACTIVE_POWER_SIANG","ACTIVE_POWER_MALAM",
    "POWER_FACTOR_L1","POWER_FACTOR_L2","POWER_FACTOR_L3",
    "CURRENT_LOOP","FREEZE"
]

def robust_zscores(X: np.ndarray) -> np.ndarray:
    """Robust Z using median and MAD."""
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    mad_safe = np.where(mad==0, 1.0, mad)
    z = (X - med) / (1.4826 * mad_safe)  # consistency constant for normal
    return z

def top_feature_reasons(X: np.ndarray, feat_names: list, k: int = 3) -> list:
    z = robust_zscores(X)
    reasons = []
    for i in range(z.shape[0]):
        idx = np.argsort(-np.abs(z[i]))[:k]
        tags = []
        for j in idx:
            arrow = "↑" if z[i, j] > 0 else "↓"
            tags.append(f"{feat_names[j]}{arrow}")
        reasons.append(", ".join(tags))
    return reasons

def aggregate_features(df: pd.DataFrame, how: str = "median") -> pd.DataFrame:
    """Aggregate per LOCATION_CODE into one row per IDPEL, including features & last info cols."""
    # numeric features
    df_feat = _numericize(df.copy(), FITUR_TEKNIS)
    how = how.lower()
    if how == "mean":
        aggfun = "mean"
    elif how == "p95":
        aggfun = lambda x: np.percentile(x, 95)
    else:
        aggfun = "median"
    feat_agg = df_feat.groupby("LOCATION_CODE")[FITUR_TEKNIS].agg(aggfun).reset_index()

    # latest info row per IDPEL
    if 'TANGGAL' in df.columns:
        df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], errors='coerce')
        info = df.sort_values('TANGGAL').dropna(subset=['TANGGAL']).groupby('LOCATION_CODE').tail(1)
    else:
        info = df.drop_duplicates(subset='LOCATION_CODE', keep='last')
    # map tariff/power/nama
    pick_cols = ["LOCATION_CODE"]
    if "NAMA_PELANGGAN" in info.columns:
        pick_cols.append("NAMA_PELANGGAN")
    if "TARIF" in info.columns:
        pick_cols.append("TARIF")
    elif "TARIFF" in info.columns:
        pick_cols.append("TARIFF")
    if "POWER" in info.columns:
        pick_cols.append("POWER")
    info = info[pick_cols].copy()
    info = info.rename(columns={"NAMA_PELANGGAN": "NAMA", "TARIFF": "TARIF", "POWER": "DAYA"})
    if "TARIF" not in info.columns:
        info["TARIF"] = "-"
    if "DAYA" not in info.columns:
        info["DAYA"] = "-"

    merged = feat_agg.merge(info, on="LOCATION_CODE", how="left")
    return merged

def run_iforest(X: np.ndarray, contamination: float = 0.1, random_state: int = 42):
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=random_state)
    model.fit(X)
    score = model.decision_function(X)  # higher = more normal
    label = model.predict(X)            # -1 outlier, 1 inlier
    return model, score, label

# ---------- Indicators (threshold rules) ---------- #
def cek_indikator_row(row, session):
    indikator = {}
    indikator['arus_hilang'] = all([row['CURRENT_L1'] == 0, row['CURRENT_L2'] == 0, row['CURRENT_L3'] == 0])
    indikator['over_current'] = any([
        row['CURRENT_L1'] > session.get('over_i_tm', 5.0),
        row['CURRENT_L2'] > session.get('over_i_tm', 5.0),
        row['CURRENT_L3'] > session.get('over_i_tm', 5.0)
    ])
    indikator['over_voltage'] = any([
        row['VOLTAGE_L1'] > session.get('vmax_tm', 62.0),
        row['VOLTAGE_L2'] > session.get('vmax_tm', 62.0),
        row['VOLTAGE_L3'] > session.get('vmax_tm', 62.0)
    ])
    v = [row['VOLTAGE_L1'], row['VOLTAGE_L2'], row['VOLTAGE_L3']]
    indikator['v_drop'] = max(v) - min(v) > session.get('low_v_diff_tm', 2.0)
    indikator['cos_phi_kecil'] = any([row.get(f'POWER_FACTOR_L{i}', 1) < session.get('cos_phi_tm', 0.4) for i in range(1,4)])
    indikator['active_power_negative'] = any([row.get(f'ACTIVE_POWER_L{i}', 0) < 0 for i in range(1,4)])
    indikator['arus_kecil_teg_kecil'] = all([
        all([row['CURRENT_L1'] < 1, row['CURRENT_L2'] < 1, row['CURRENT_L3'] < 1]),
        all([row['VOLTAGE_L1'] < 180, row['VOLTAGE_L2'] < 180, row['VOLTAGE_L3'] < 180]),
        any([row.get(f'ACTIVE_POWER_L{i}', 0) > 10 for i in range(1,4)])
    ])
    arus = [row['CURRENT_L1'], row['CURRENT_L2'], row['CURRENT_L3']]
    max_i, min_i = max(arus), min(arus)
    indikator['unbalance_I'] = (max_i - min_i) / max_i > session.get('unbal_tol_tm', 0.5) if max_i > 0 else False
    indikator['v_lost'] = (row.get('VOLTAGE_L1', 0) == 0 or row.get('VOLTAGE_L2', 0) == 0 or row.get('VOLTAGE_L3', 0) == 0)
    indikator['In_more_Imax'] = any([
        row['CURRENT_L1'] > session.get('max_i_tm', 1.0),
        row['CURRENT_L2'] > session.get('max_i_tm', 1.0),
        row['CURRENT_L3'] > session.get('max_i_tm', 1.0)
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

INDIKATOR_BOBOT = {
    'arus_hilang': 2, 'over_current': 1, 'over_voltage': 1, 'v_drop': 1,
    'cos_phi_kecil': 1, 'active_power_negative': 2, 'arus_kecil_teg_kecil': 1,
    'unbalance_I': 1, 'v_lost': 2, 'In_more_Imax': 1,
    'active_power_negative_siang': 2, 'active_power_negative_malam': 2,
    'active_p_lost': 2, 'current_loop': 2, 'freeze': 2
}

# ===================== Auth ===================== #
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

if st.button("🔒 Logout", key="logout_button", help="Keluar dari dashboard"):
    st.session_state["logged_in"] = False
    st.success("Logout berhasil!")
    st.rerun()

# ===================== Tabs ===================== #
tab_amr, tab_pasca, tab_prabayar = st.tabs(["📥 AMR Harian", "💳 Pascabayar", "💡 Prabayar"])

# ===================== AMR Harian ===================== #
with tab_amr:
    st.title("📊 Dashboard Target Operasi AMR - P2TL")
    st.markdown("---")

    # --- Parameter Threshold (dipakai di tab Threshold) --- #
    with st.expander("⚙️ Setting Parameter Threshold"):
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
        with colB:
            st.markdown("#### Active Power Lost")
            st.number_input("Set Batas Bawah Arus P Lost", key="plost_i_min", value=0.5)
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
            st.markdown("#### Over Current (Tak Langsung)")
            st.number_input("Set Batas bawah Arus Maks pada TM", key="over_i_tm", value=5.0)
            st.number_input("Set Batas bawah Arus Maks pada TR", key="over_i_tr", value=5.0)
            st.markdown("#### Over Voltage")
            st.number_input("Tegangan Maksimum TM", key="vmax_tm", value=62.0)
            st.number_input("Tegangan Maksimum TR", key="vmax_tr", value=241.0)
        st.markdown("---")
        st.number_input("Jumlah Indikator ≥", key="min_indicator", value=1)
        st.number_input("Jumlah Bobot ≥", key="min_weight", value=2)
        st.number_input("Banyak Data yang Ditampilkan", key="top_limit", value=50)

    # Load data once (used by both sub-tabs)
    df_raw = load_csv_safe(DATA_PATH_AMR)

    sub_threshold, sub_ml, sub_upload = st.tabs(["🔎 Deteksi Threshold", "🤖 Deteksi ML (Eksperimental)", "➕ Upload Data"])

    # --------- Sub-tab Threshold --------- #
    with sub_threshold:
        if df_raw.empty:
            st.warning("Belum ada data historis. Silakan upload pada tab 'Upload Data'.")
        else:
            df = df_raw.copy()
            df = df.dropna(subset=['LOCATION_CODE']).copy()
            df['LOCATION_CODE'] = df['LOCATION_CODE'].astype(str).str.strip()
            df = _filter_customer_only(df)
            df = _ensure_customer_cols(df)

            # Hitung jumlah kemunculan per IDPEL
            df['Jumlah Berulang'] = df.groupby('LOCATION_CODE')['LOCATION_CODE'].transform('count')

            # Hitung indikator per baris
            num_cols = list(set(FITUR_TEKNIS))
            df = _numericize(df, num_cols)
            indikator_list = df.apply(lambda r: cek_indikator_row(r, st.session_state), axis=1)
            indikator_df = pd.DataFrame(indikator_list.tolist())
            indikator_df['LOCATION_CODE'] = df['LOCATION_CODE']
            indikator_df['Jumlah Berulang'] = df['Jumlah Berulang']

            # Agregasi indikator per LOCATION_CODE (OR/any)
            boolean_cols = [c for c in indikator_df.columns if c not in ['LOCATION_CODE', 'Jumlah Berulang']]
            agg_dict = {c: 'any' for c in boolean_cols}
            agg_dict['Jumlah Berulang'] = 'max'
            indikator_agg = indikator_df.groupby('LOCATION_CODE', as_index=False).agg(agg_dict)

            # Bobot & skor
            indikator_agg['Jumlah Indikator'] = indikator_agg[boolean_cols].sum(axis=1)
            def hitung_skor(row):
                s = 0
                for k, w in INDIKATOR_BOBOT.items():
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

            # Konsistensi nama kolom
	    if 'TARIF' in df_info.columns:
   					tarif_col = 'TARIF'
	    else:
    					tarif_col = 'TARIFF' if 'TARIFF' in df_info.columns else None

	    tech_cols_present = [c for c in TECH_COLS if c in df_info.columns]

	    use_cols = ['LOCATION_CODE', 'NAMA_PELANGGAN'] + ([tarif_col] if tarif_col else []) + ['POWER'] + tech_cols_present
	    df_info = df_info[use_cols].rename(columns={'NAMA_PELANGGAN': 'NAMA','TARIFF':'TARIF','POWER':'DAYA'})

	    # pastikan ada kolom TARIF & DAYA walau kosong
	    if 'TARIF' not in df_info.columns: df_info['TARIF'] = '-'
	    if 'DAYA' not in df_info.columns: df_info['DAYA'] = '-'


            result = df_info.merge(indikator_agg, on='LOCATION_CODE', how='right')
            for c in ['NAMA','TARIF','DAYA']:
                if c in result.columns: result[c] = result[c].fillna('-')

            top_limit = st.session_state.get('top_limit', 50)
            top50 = (result.drop_duplicates(subset='LOCATION_CODE')
                            .sort_values(by='Skor', ascending=False)
                            .head(top_limit))

            col1, col2, col3 = st.columns([1.2,1.2,1])
            col1.metric("📄 Total Record (Customer)", len(df))
            col3.metric("🎯 Pelanggan Potensial TO", int((result['Skor'] > 0).sum()))

            st.subheader("🏆 Top 50 Rekomendasi Target Operasi (unik per IDPEL)")

	    # susun kolom ekspor: identitas + teknis + indikator + metrik
	    tech_cols_present = [c for c in TECH_COLS if c in result.columns]
	    indikator_cols_order = [k for k in INDIKATOR_BOBOT.keys() if k in result.columns]
	    export_cols = (['LOCATION_CODE','NAMA','TARIF','DAYA'] + tech_cols_present +
               								indikator_cols_order + ['Jumlah Berulang','Jumlah Indikator','Skor'])

	    result_export = result.copy()
	    for c in export_cols:
    					if c not in result_export.columns:
       						 result_export[c] = np.nan
	    result_export = result_export[export_cols]

	    st.download_button("📥 Download Hasil Lengkap (XLSX)",
                   								data=make_xlsx_bytes(result_export, sheet_name="TO_AMR"),
                   								file_name="hasil_target_operasi_amr.xlsx",
                  								 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            kolom_tampil = ['LOCATION_CODE','NAMA','TARIF','DAYA'] + \
                           [k for k in INDIKATOR_BOBOT.keys() if k in result.columns] + \
                           ['Jumlah Berulang','Jumlah Indikator','Skor']
            st.dataframe(top50[kolom_tampil], use_container_width=True, height=520)

            # Visualisasi indikator per-ID
            indikator_counts = indikator_agg[[c for c in INDIKATOR_BOBOT.keys() if c in indikator_agg.columns]].sum().sort_values(ascending=False).reset_index()
            indikator_counts.columns = ['Indikator','Jumlah']
            fig = px.bar(indikator_counts, x='Indikator', y='Jumlah', text='Jumlah', color='Indikator')

            # Interaksional: klik bar (butuh streamlit-plotly-events); fallback dropdown
            selected_indicator = None
            try:
                from streamlit_plotly_events import plotly_events  # type: ignore
                st.markdown("**Klik batang indikator untuk melihat daftar IDPEL.**")
                selected_points = plotly_events(fig, click_event=True, hover_event=False, select_event=False, override_height=500)
                if selected_points:
                    selected_indicator = selected_points[0].get('x')
            except Exception:
                st.plotly_chart(fig, use_container_width=True)
                st.info("Interaksi klik membutuhkan paket `streamlit-plotly-events`. Gunakan pemilih berikut jika paket belum terpasang.")
                selected_indicator = st.selectbox("Pilih indikator:", indikator_counts['Indikator'].tolist())

            if selected_indicator:
                if selected_indicator in indikator_agg.columns:
                    id_list = indikator_agg.loc[indikator_agg[selected_indicator] == True, "LOCATION_CODE"].astype(str).unique().tolist()
                    tech_cols_present = [c for c in TECH_COLS if c in result.columns]
		    base_cols = ["LOCATION_CODE","NAMA","TARIF","DAYA"]
		    cols_daftar = base_cols + tech_cols_present

		    detail_df = (result[result["LOCATION_CODE"].astype(str).isin(id_list)][cols_daftar]
             										.drop_duplicates(subset="LOCATION_CODE"))

		    st.download_button("📥 Download daftar IDPEL (XLSX)",
                   											data=make_xlsx_bytes(detail_df, sheet_name=f"IDPEL_{selected_indicator}"),
                   											file_name=f"idpel_{selected_indicator}.xlsx",
                  											mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                else:
                    st.warning("Indikator tidak ditemukan di data.")

    # --------- Sub-tab ML (Eksperimental) --------- #
    with sub_ml:
        st.info("ML mencari pola tak biasa (outlier) pada data **Customer**. Gunakan sebagai pendukung, bukan pengganti inspeksi lapangan.")
        if df_raw.empty:
            st.warning("Belum ada data historis. Silakan upload pada tab 'Upload Data'.")
        else:
            df = df_raw.copy()
            df = df.dropna(subset=['LOCATION_CODE']).copy()
            df['LOCATION_CODE'] = df['LOCATION_CODE'].astype(str).str.strip()
            df = _filter_customer_only(df)
            df = _ensure_customer_cols(df)
            df = _numericize(df, FITUR_TEKNIS)

            # Langkah 1 — Data & Agregasi
            st.markdown("### Langkah 1 – Data & Agregasi")
            unit_opt = st.radio("Unit analisis", ["Per IDPEL (agregat)", "Per baris histori"], horizontal=True, index=0)
            agg_opt = None
            if unit_opt == "Per IDPEL (agregat)":
                agg_opt = st.selectbox("Metode agregasi fitur per IDPEL", ["median", "mean", "p95"], index=0, help="Median disarankan (lebih stabil).")
                df_ml = aggregate_features(df, how=agg_opt)
            else:
                # per baris histori: keep features as-is + info cols
                info_cols = []
                if "NAMA_PELANGGAN" in df.columns: info_cols.append("NAMA_PELANGGAN")
                if "TARIF" in df.columns: info_cols.append("TARIF")
                elif "TARIFF" in df.columns: info_cols.append("TARIFF")
                if "POWER" in df.columns: info_cols.append("POWER")
                df_ml = df[["LOCATION_CODE"] + FITUR_TEKNIS + info_cols].copy()
                df_ml = df_ml.rename(columns={"NAMA_PELANGGAN":"NAMA","TARIFF":"TARIF","POWER":"DAYA"})
                if "TARIF" not in df_ml.columns: df_ml["TARIF"] = "-"
                if "DAYA" not in df_ml.columns: df_ml["DAYA"] = "-"

            st.caption(f"Data: Customer saja • Baris analisis: {len(df_ml):,} • Fitur: {len(FITUR_TEKNIS)}")

            # Langkah 2 — Parameter sederhana
            st.markdown("### Langkah 2 – Parameter")
            colp, cols, coln = st.columns([1,1,1])
            with colp:
                contam_pct = st.slider("Proporsi anomali (kuota investigasi) – %", min_value=1, max_value=30, value=10, step=1) / 100.0
            with cols:
                scaler_choice = st.selectbox("Stabilisasi skala fitur", ["RobustScaler (disarankan)", "StandardScaler"], index=0)
            with coln:
                max_rows = st.number_input("Tampilkan maksimal (baris)", min_value=50, max_value=5000, value=300, step=50)

            with st.expander("Pengaturan Lanjutan"):
                seed = st.number_input("Random state", value=42, step=1)
                pilih_fitur = st.multiselect("Pilih fitur ML (kosongkan = semua)", FITUR_TEKNIS, default=[])

            fitur_pakai = pilih_fitur if len(pilih_fitur) > 0 else FITUR_TEKNIS
            X = df_ml[fitur_pakai].values.astype(float)

            # Skaler
            if scaler_choice.startswith("Robust"):
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            Xs = scaler.fit_transform(X)

            # Langkah 3 — Jalankan
            st.markdown("### Langkah 3 – Hasil & Penjelasan")
            if st.button("🚀 Jalankan Deteksi"):
                model, score, label = run_iforest(Xs, contamination=contam_pct, random_state=seed)
                outlier_mask = (label == -1)
                df_res = df_ml.copy()
                df_res["skor_anomali"] = score  # lebih kecil = lebih anomali
                df_res["is_anomali"] = outlier_mask.astype(int)

                # alasan (top 3 robust-z) — hitung pada seluruh X dan ambil utk baris outlier
                reasons = top_feature_reasons(X, fitur_pakai, k=3)
                df_res["alasan"] = reasons

                # urutkan dari paling anomali
                df_res = df_res.sort_values("skor_anomali", ascending=True)

                # ringkasan
                total = len(df_res)
                jml_out = int(outlier_mask.sum())
                st.success(f"Terpilih {jml_out:,}/{total:,} ({int(contam_pct*100)}%) kandidat investigasi.")

                # kolom tampilan
                show_cols = [c for c in ["LOCATION_CODE","NAMA","TARIF","DAYA"] if c in df_res.columns] + \
                            ["skor_anomali","alasan"]
                df_show = df_res[df_res["is_anomali"]==1][show_cols].head(int(max_rows))

                # warna baris (opsional: gunakan dataframe biasa)
                st.dataframe(df_show, use_container_width=True, height=520)

                # simpan & unduh
                st.download_button("📥 Download Hasil Anomali (CSV)",
                                   df_res[df_res["is_anomali"]==1][["LOCATION_CODE","skor_anomali","alasan"] + [c for c in ["NAMA","TARIF","DAYA"] if c in df_res.columns]].to_csv(index=False).encode('utf-8'),
                                   file_name="hasil_ml_anomali.csv",
                                   mime="text/csv")

                # Daftar TO dari ML (opsional)
                if "to_list_ml" not in st.session_state:
                    st.session_state["to_list_ml"] = set()
                if st.button("➕ Tambahkan baris ditampilkan ke Daftar TO (ML)"):
                    st.session_state["to_list_ml"].update(df_show["LOCATION_CODE"].astype(str).tolist())
                    st.success(f"Ditambahkan {len(df_show)} IDPEL ke daftar TO (ML).")

                if st.session_state.get("to_list_ml"):
                    list_to = sorted(list(st.session_state["to_list_ml"]))
                    st.info(f"Daftar TO (ML) saat ini: {len(list_to)} IDPEL.")
                    df_to_view = pd.DataFrame({"LOCATION_CODE": list_to})
                    st.dataframe(df_to_view, use_container_width=True, height=200)
                    st.download_button("📥 Download Daftar TO (ML)",
                                       df_to_view.to_csv(index=False).encode('utf-8'),
                                       file_name="daftar_to_ml.csv",
                                       mime="text/csv")
                    if st.button("🗑️ Kosongkan Daftar TO (ML)"):
                        st.session_state["to_list_ml"] = set()
                        st.success("Daftar TO (ML) dikosongkan.")

    # --------- Sub-tab Upload --------- #
    with sub_upload:
        uploaded_file = st.file_uploader("📥 Upload File Excel AMR Harian", type=["xlsx"])
        if uploaded_file:
            df_up = pd.read_excel(uploaded_file, sheet_name=0)
            df_up = df_up.dropna(subset=['LOCATION_CODE'])
            df_up['LOCATION_CODE'] = df_up['LOCATION_CODE'].astype(str).str.strip()
            df_up = _filter_customer_only(df_up)
            df_up = _ensure_customer_cols(df_up)
            df_up = _numericize(df_up, FITUR_TEKNIS)

            df_hist = load_csv_safe(DATA_PATH_AMR)
            if not df_hist.empty:
                df_new = pd.concat([df_hist, df_up], ignore_index=True).drop_duplicates()
            else:
                df_new = df_up
            df_new.to_csv(DATA_PATH_AMR, index=False)
            st.cache_data.clear()
            st.success("Data (Customer saja) berhasil ditambahkan ke histori.")

        if st.button("🗑️ Hapus Semua Data Historis"):
            if os.path.exists(DATA_PATH_AMR):
                os.remove(DATA_PATH_AMR)
                st.cache_data.clear()
                st.success("Data historis berhasil dihapus.")

# ===================== Pascabayar & Prabayar (ringkas, tak diubah) ===================== #
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
            else:
                df_new = df_new[required_cols].dropna(subset=["IDPEL"])
                df_hist = load_csv_safe(DATA_PATH_OLAP)
                df_all = pd.concat([df_hist, df_new]).drop_duplicates(subset=["THBLREK", "IDPEL"])
                df_all.to_csv(DATA_PATH_OLAP, index=False)
                st.cache_data.clear()
                st.success("Data berhasil ditambahkan ke histori OLAP Pascabayar.")
        except Exception as e:
            st.error(f"Gagal memproses file: {e}")

    if st.button("🗑 Hapus Histori OLAP Pascabayar"):
        if os.path.exists(DATA_PATH_OLAP):
            os.remove(DATA_PATH_OLAP)
            st.cache_data.clear()
            st.success("Histori OLAP berhasil dihapus.")

    df = load_csv_safe(DATA_PATH_OLAP)
    if not df.empty:
        if df.duplicated(subset=["IDPEL", "THBLREK"]).any():
            st.warning("⚠️ Terdapat duplikat kombinasi IDPEL dan THBLREK. Data akan dirata-ratakan.")

        drop_threshold = st.number_input("Penurunan Pemakaian kWh 3 Bulan Terakhir (%)", value=30, min_value=0, max_value=100, step=1)
        use_drop_threshold = st.checkbox("Aktifkan Filter Penurunan Pemakaian kWh", value=False)

        selected_idpel = st.selectbox("🔍 Pilih IDPEL untuk Tabel & Grafik", ["Semua"] + sorted(df["IDPEL"].astype(str).unique().tolist()))

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
        indikator_cols = ["pemakaian_zero_3x","jamnyala_abnormal","min_kwh_zero","rendah_rata","variasi_tinggi"]
        risk_df["skor"] = risk_df[indikator_cols].sum(axis=1)
        skor_threshold = st.slider("Minimal Skor Risiko untuk TO", 1, len(indikator_cols), 3)
        df_to = risk_df[risk_df["skor"] >= skor_threshold].sort_values("skor", ascending=False)
        st.metric("Pelanggan Berpotensi TO", len(df_to))
        st.dataframe(df_to.head(1000), use_container_width=True)
        st.download_button("📄 Download Target Operasi Pascabayar",
                           df_to.to_csv(index=False).encode(),
                           file_name="target_operasi_pascabayar.csv",
                           mime="text/csv")
    else:
        st.info("Belum ada data histori OLAP pascabayar. Silakan upload terlebih dahulu.")

# ===================== Prabayar ===================== #
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

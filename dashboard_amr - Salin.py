
import streamlit as st
import pandas as pd
import numpy as np
import os, pickle
import plotly.express as px
from datetime import datetime

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance

# ===================== App Config ===================== #
st.set_page_config(page_title="T-Energy", layout="wide", page_icon="⚡")

DATA_PATH_AMR   = "data_harian.csv"
DATA_PATH_OLAP  = "olap_pascabayar.csv"
LABELS_STORE    = "labels_store.csv"     # simpan label hasil lapangan
MODEL_STORE     = "model_rf.pkl"         # simpan model supervised

# ===================== Helpers & Cache ===================== #
@st.cache_data(show_spinner=False)
def load_csv_safe(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_labels_store(path: str = LABELS_STORE) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if 'TANGGAL_INSPEKSI' in df.columns:
                df['TANGGAL_INSPEKSI'] = pd.to_datetime(df['TANGGAL_INSPEKSI'], errors='coerce')
            return df
        except Exception:
            return pd.DataFrame(columns=['LOCATION_CODE','TANGGAL_INSPEKSI','LABEL_TO'])
    else:
        return pd.DataFrame(columns=['LOCATION_CODE','TANGGAL_INSPEKSI','LABEL_TO'])

def save_labels_to_store(new_labels: pd.DataFrame, path: str = LABELS_STORE) -> pd.DataFrame:
    """Gabungkan label baru ke store, anti-duplikat (LOCATION_CODE + TANGGAL_INSPEKSI)."""
    store = load_labels_store(path)
    keep_cols = ['LOCATION_CODE','TANGGAL_INSPEKSI','LABEL_TO']
    for c in keep_cols:
        if c not in new_labels.columns:
            new_labels[c] = np.nan
    new_labels = new_labels[keep_cols].copy()
    new_labels['LOCATION_CODE'] = new_labels['LOCATION_CODE'].astype(str).str.strip()
    new_labels['TANGGAL_INSPEKSI'] = pd.to_datetime(new_labels['TANGGAL_INSPEKSI'], errors='coerce')
    new_labels['LABEL_TO'] = pd.to_numeric(new_labels['LABEL_TO'], errors='coerce').astype('Int64')
    new_labels = new_labels.dropna(subset=['LOCATION_CODE','TANGGAL_INSPEKSI','LABEL_TO'])

    all_lab = pd.concat([store, new_labels], ignore_index=True)
    all_lab = all_lab.drop_duplicates(subset=['LOCATION_CODE','TANGGAL_INSPEKSI']).reset_index(drop=True)
    all_lab.to_csv(path, index=False)
    load_labels_store.clear()
    return all_lab

def _filter_customer_only(df: pd.DataFrame) -> pd.DataFrame:
    if 'LOCATION_TYPE' not in df.columns:
        return df
    lt = df['LOCATION_TYPE'].astype(str).str.strip().str.lower()
    allowed = {'customer', 'costumer'}
    return df[lt.isin(allowed)].copy()

def _ensure_customer_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in ['NAMA_PELANGGAN', 'TARIF', 'TARIFF', 'POWER']:
        if col not in df.columns:
            df[col] = "-"
    return df

def _numericize(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
        else:
            df[c] = 0.0
    return df

def _ensure_date_col(df: pd.DataFrame, target_col: str = "TANGGAL") -> pd.DataFrame:
    """Pastikan df punya kolom datetime 'TANGGAL'. Jika tidak ada, dan ada 'READ_DATE', pakai itu."""
    df = df.copy()
    if target_col in df.columns:
        df[target_col] = pd.to_datetime(df[target_col], errors='coerce')
    elif 'READ_DATE' in df.columns:
        df[target_col] = pd.to_datetime(df['READ_DATE'], errors='coerce')
    return df

# ---------- Threshold indicator logic ---------- #
FITUR_TEKNIS = [
    "CURRENT_L1","CURRENT_L2","CURRENT_L3",
    "VOLTAGE_L1","VOLTAGE_L2","VOLTAGE_L3",
    "ACTIVE_POWER_L1","ACTIVE_POWER_L2","ACTIVE_POWER_L3",
    "ACTIVE_POWER_SIANG","ACTIVE_POWER_MALAM",
    "POWER_FACTOR_L1","POWER_FACTOR_L2","POWER_FACTOR_L3",
    "CURRENT_LOOP","FREEZE"
]

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
    indikator['cos_phi_kecil'] = any([row.get(f'POWER_FACTOR_L{i}', 1) < session.get('cos_phi_tm', 0.4) for i in range(1, 4)])
    indikator['active_power_negative'] = any([row.get(f'ACTIVE_POWER_L{i}', 0) < 0 for i in range(1, 4)])
    indikator['arus_kecil_teg_kecil'] = all([
        all([row['CURRENT_L1'] < 1, row['CURRENT_L2'] < 1, row['CURRENT_L3'] < 1]),
        all([row['VOLTAGE_L1'] < 180, row['VOLTAGE_L2'] < 180, row['VOLTAGE_L3'] < 180]),
        any([row.get(f'ACTIVE_POWER_L{i}', 0) > 10 for i in range(1, 4)])
    ])
    arus = [row['CURRENT_L1'], row['CURRENT_L2'], row['CURRENT_L3']]
    max_i, min_i = max(arus), min(arus)
    indikator['unbalance_I'] = (max_i - min_i) / max_i > session.get('unbal_tol_tm', 0.5) if max_i > 0 else False
    indikator['v_lost'] = (row.get('VOLTAGE_L1', 0) == 0 or row.get('VOLTAGE_L2', 0) == 0 or row.get('VOLTAGE_L3', 0) == 0)
    indikator['current_loop'] = row.get('CURRENT_LOOP', 0) == 1
    indikator['freeze'] = row.get('FREEZE', 0) == 1
    return indikator

INDIKATOR_BOBOT = {
    'arus_hilang': 2, 'over_current': 1, 'over_voltage': 1, 'v_drop': 1,
    'cos_phi_kecil': 1, 'active_power_negative': 2, 'arus_kecil_teg_kecil': 1,
    'unbalance_I': 1, 'v_lost': 2, 'current_loop': 2, 'freeze': 2
}

# ---------- Common ML utils ---------- #
def robust_zscores(X: np.ndarray) -> np.ndarray:
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    mad_safe = np.where(mad==0, 1.0, mad)
    return (X - med) / (1.4826 * mad_safe)

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
    df = _ensure_date_col(df)  # support READ_DATE
    df_feat = _numericize(df.copy(), FITUR_TEKNIS)
    how = how.lower()
    if how == "mean":
        aggfun = "mean"
    elif how == "p95":
        aggfun = lambda x: np.percentile(x, 95)
    else:
        aggfun = "median"
    feat_agg = df_feat.groupby("LOCATION_CODE")[FITUR_TEKNIS].agg(aggfun).reset_index()

    if 'TANGGAL' in df.columns and pd.api.types.is_datetime64_any_dtype(df['TANGGAL']):
        info = df.sort_values('TANGGAL').dropna(subset=['TANGGAL']).groupby('LOCATION_CODE').tail(1)
    else:
        info = df.drop_duplicates(subset='LOCATION_CODE', keep='last')
    pick_cols = ["LOCATION_CODE"]
    if "NAMA_PELANGGAN" in info.columns: pick_cols.append("NAMA_PELANGGAN")
    if "TARIF" in info.columns: pick_cols.append("TARIF")
    elif "TARIFF" in info.columns: pick_cols.append("TARIFF")
    if "POWER" in info.columns: pick_cols.append("POWER")

    info = info[pick_cols].copy().rename(columns={"NAMA_PELANGGAN":"NAMA","TARIFF":"TARIF","POWER":"DAYA"})
    if "TARIF" not in info.columns: info["TARIF"] = "-"
    if "DAYA" not in info.columns: info["DAYA"] = "-"
    merged = feat_agg.merge(info, on="LOCATION_CODE", how="left")
    return merged

# ============ Auth ============ #
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

    # --- Parameter Threshold --- #
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

    # Load AMR once
    df_raw = load_csv_safe(DATA_PATH_AMR)

    # Sub-tabs: Threshold, ML Auto, Upload
    sub_threshold, sub_ml_auto, sub_upload = st.tabs(
        ["🔎 Deteksi Threshold", "🤖 Deteksi ML – Otomatis", "➕ Upload Data"]
    )

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
            df = _ensure_date_col(df)

            df['Jumlah Berulang'] = df.groupby('LOCATION_CODE')['LOCATION_CODE'].transform('count')
            df = _numericize(df, FITUR_TEKNIS)
            indikator_list = df.apply(lambda r: cek_indikator_row(r, st.session_state), axis=1)
            indikator_df = pd.DataFrame(indikator_list.tolist())
            indikator_df['LOCATION_CODE'] = df['LOCATION_CODE']
            indikator_df['Jumlah Berulang'] = df['Jumlah Berulang']

            boolean_cols = [c for c in indikator_df.columns if c not in ['LOCATION_CODE', 'Jumlah Berulang']]
            agg_dict = {c: 'any' for c in boolean_cols}
            agg_dict['Jumlah Berulang'] = 'max'
            indikator_agg = indikator_df.groupby('LOCATION_CODE', as_index=False).agg(agg_dict)

            indikator_agg['Jumlah Indikator'] = indikator_agg[boolean_cols].sum(axis=1)
            indikator_agg['Skor'] = indikator_agg.apply(lambda row: sum(INDIKATOR_BOBOT[k] for k in INDIKATOR_BOBOT.keys() if k in row and bool(row[k])), axis=1)

            if 'TANGGAL' in df.columns and pd.api.types.is_datetime64_any_dtype(df['TANGGAL']):
                df_info = df.sort_values('TANGGAL').dropna(subset=['TANGGAL']).groupby('LOCATION_CODE').tail(1)
            else:
                df_info = df.drop_duplicates(subset='LOCATION_CODE', keep='last')

            if 'TARIF' in df_info.columns: tarif_col = 'TARIF'
            else: tarif_col = 'TARIFF' if 'TARIFF' in df_info.columns else None
            use_cols = ['LOCATION_CODE', 'NAMA_PELANGGAN'] + ([tarif_col] if tarif_col else []) + ['POWER']
            df_info = df_info[use_cols].rename(columns={'NAMA_PELANGGAN': 'NAMA', 'TARIFF': 'TARIF', 'POWER': 'DAYA'})
            if 'TARIF' not in df_info.columns: df_info['TARIF'] = '-'
            if 'DAYA' not in df_info.columns: df_info['DAYA'] = '-'

            result = df_info.merge(indikator_agg, on='LOCATION_CODE', how='right')
            for c in ['NAMA', 'TARIF', 'DAYA']:
                if c in result.columns: result[c] = result[c].fillna('-')

            top_limit = st.session_state.get('top_limit', 50)
            top50 = (result.drop_duplicates(subset='LOCATION_CODE').sort_values(by='Skor', ascending=False).head(top_limit))

            col1, col2, col3 = st.columns([1.2,1.2,1])
            col1.metric("📄 Total Record (Customer)", len(df))
            col3.metric("🎯 Pelanggan Potensial TO", int((result['Skor'] > 0).sum()))

            st.subheader("🏆 Top 50 Rekomendasi Target Operasi (unik per IDPEL)")
            st.download_button("📥 Download Hasil Lengkap (Excel)",
                               data=result.to_csv(index=False).encode('utf-8'),
                               file_name="hasil_target_operasi_amr.csv",
                               mime="text/csv")
            kolom_tampil = ['LOCATION_CODE','NAMA','TARIF','DAYA'] + [k for k in INDIKATOR_BOBOT.keys() if k in result.columns] + ['Jumlah Berulang','Jumlah Indikator','Skor']
            st.dataframe(top50[kolom_tampil], use_container_width=True, height=520)

            indikator_counts = indikator_agg[[c for c in INDIKATOR_BOBOT.keys() if c in indikator_agg.columns]].sum().sort_values(ascending=False).reset_index()
            indikator_counts.columns = ['Indikator','Jumlah']
            fig = px.bar(indikator_counts, x='Indikator', y='Jumlah', text='Jumlah', color='Indikator')
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
                    id_list = indikator_agg.loc[indikator_agg[selected_indicator] == True, 'LOCATION_CODE'].astype(str).unique().tolist()
                    detail_df = result[result['LOCATION_CODE'].astype(str).isin(id_list)][['LOCATION_CODE','NAMA','TARIF','DAYA']].drop_duplicates(subset='LOCATION_CODE')
                    st.subheader(f"📋 Daftar IDPEL untuk indikator: **{selected_indicator}** (unik: {len(detail_df)})")
                    st.dataframe(detail_df, use_container_width=True, height=400)
                    st.download_button("📥 Download daftar IDPEL (CSV)",
                                       detail_df.to_csv(index=False).encode('utf-8'),
                                       file_name=f"idpel_{selected_indicator}.csv",
                                       mime="text/csv")
                else:
                    st.warning("Indikator tidak ditemukan di data.")

    # --------- Sub-tab ML – Otomatis dengan Penyimpanan Persisten --------- #
    with sub_ml_auto:
        st.info("Mode otomatis: Tanpa label → anomali. Jika Anda unggah label, sistem melatih supervised dan menyimpan model agar 'terus belajar'.")

        if df_raw.empty:
            st.warning("Belum ada data historis. Silakan upload pada tab 'Upload Data'.")
        else:
            df = df_raw.copy()
            df = df.dropna(subset=['LOCATION_CODE']).copy()
            df['LOCATION_CODE'] = df['LOCATION_CODE'].astype(str).str.strip()
            df = _filter_customer_only(df)
            df = _ensure_customer_cols(df)
            df = _numericize(df, FITUR_TEKNIS)
            df = _ensure_date_col(df)

            colL, colR = st.columns([1,1])
            with colL:
                if "TARIF" not in df.columns and "TARIFF" in df.columns: df["TARIF"] = df["TARIFF"]
                tarif_vals = sorted([t for t in df["TARIF"].astype(str).str.upper().unique() if t != "-"])
                prefixes = sorted(list(set([t[:1] for t in tarif_vals if len(t)>0])))
                pilih_prefix = st.multiselect("Prefix Tarif", options=prefixes, default=[])
            with colR:
                if "POWER" in df.columns:
                    daya_num = pd.to_numeric(df["POWER"], errors="coerce")
                    dmin = int(np.nanmin(daya_num)) if not np.isnan(daya_num.min()) else 0
                    dmax = int(np.nanmax(daya_num)) if not np.isnan(daya_num.max()) else 10000
                else:
                    dmin, dmax = 0, 10000
                range_daya = st.slider("Rentang Daya (VA)", min_value=int(dmin), max_value=int(max(dmax, dmin+1)), value=(int(dmin), int(max(dmin+1000, min(dmax, dmin+5000)))))

            seg_df = df.copy()
            if len(pilih_prefix) > 0 dan "TARIFF" in seg_df.columns:
                seg_df = seg_df[seg_df["TARIF"].astype(str).str.upper().str.startswith(tuple(pilih_prefix))]
            if "POWER" in seg_df.columns:
                pn = pd.to_numeric(seg_df["POWER"], errors="coerce").fillna(0)
                seg_df = seg_df[(pn >= range_daya[0]) & (pn <= range_daya[1])]

            st.caption(f"Segmen aktif → Baris historis: {len(seg_df):,} • IDPEL unik: {seg_df['LOCATION_CODE'].nunique():,}")

            st.markdown("### Unit Analisis")
            unit_opt = st.radio("Pilih unit", ["Per IDPEL (agregat)", "Per baris histori"], horizontal=True, index=0)
            if unit_opt == "Per IDPEL (agregat)":
                agg_opt = st.selectbox("Metode agregasi fitur per IDPEL", ["median", "mean", "p95"], index=0)
                df_ml = aggregate_features(seg_df, how=agg_opt)
            else:
                info_cols = []
                if "NAMA_PELANGGAN" in seg_df.columns: info_cols.append("NAMA_PELANGGAN")
                if "TARIF" in seg_df.columns: info_cols.append("TARIF")
                elif "TARIFF" in seg_df.columns: info_cols.append("TARIFF")
                if "POWER" in seg_df.columns: info_cols.append("POWER")
                df_ml = seg_df[["LOCATION_CODE"] + FITUR_TEKNIS + info_cols].copy()
                df_ml = df_ml.rename(columns={"NAMA_PELANGGAN":"NAMA","TARIFF":"TARIF","POWER":"DAYA"})
                if "TARIF" not in df_ml.columns: df_ml["TARIF"] = "-"
                if "DAYA" not in df_ml.columns: df_ml["DAYA"] = "-"

            st.caption(f"Baris analisis: {len(df_ml):,} • Fitur: {len(FITUR_TEKNIS)}")

            # Parameter umum
            colp, cols, colk = st.columns([1,1,1])
            with colp:
                contam_pct = st.slider("Proporsi anomali (%, alternatif)", min_value=1, max_value=30, value=10, step=1) / 100.0
            with cols:
                scaler_choice = st.selectbox("Stabilisasi skala fitur", ["RobustScaler (disarankan)", "StandardScaler"], index=0)
            with colk:
                topk = st.number_input("Top-K kapasitas inspeksi (prioritas)", min_value=10, max_value=10000, value=200, step=10)

            with st.expander("Pengaturan Lanjutan"):
                seed = st.number_input("Random state", value=42, step=1)
                pilih_fitur = st.multiselect("Pilih fitur ML (kosongkan = semua)", FITUR_TEKNIS, default=[])
                window_days = st.number_input("Jendela fitur supervised (hari)", value=30, min_value=7, max_value=120, step=1)

            fitur_pakai = pilih_fitur if len(pilih_fitur) > 0 else FITUR_TEKNIS
            X = df_ml[fitur_pakai].values.astype(float)
            scaler = RobustScaler() if scaler_choice.startswith("Robust") else StandardScaler()
            Xs = scaler.fit_transform(X)

            # ====== Upload Label (opsional) ======
            st.markdown("### (Opsional) Unggah Label / Hasil Lapangan")
            labels_file = st.file_uploader("Label TO (CSV: LOCATION_CODE, TANGGAL_INSPEKSI, LABEL_TO)", type=["csv"], key="labels_upload_auto")
            if labels_file is not None:
                try:
                    labels_new = pd.read_csv(labels_file)
                    store_df = save_labels_to_store(labels_new)
                    st.success(f"Label disimpan. Total label tersimpan: {len(store_df)} baris unik.")
                except Exception as e:
                    st.error(f"Gagal memproses label: {e}")

            # Tampilkan ringkasan store & opsi
            store_df = load_labels_store()
            has_model = os.path.exists(MODEL_STORE)
            cA, cB, cC = st.columns(3)
            cA.metric("Label tersimpan", len(store_df))
            cB.metric("Model tersimpan", "Ada ✅" if has_model else "Belum ada")
            if not store_df.empty:
                cC.download_button("📥 Unduh seluruh label", data=store_df.to_csv(index=False).encode('utf-8'),
                                   file_name="labels_store.csv", mime="text/csv")

            # ====== Tombol Jalankan ======
            st.markdown("### Jalankan")
            colU, colS1, colS2 = st.columns([1.1,1,1])
            run_unsup = colU.button("🚀 Deteksi Tanpa Label (Anomali)")
            train_supervised = colS1.button("🧠 Latih Ulang & Simpan Model Supervised")
            score_with_saved = colS2.button("🎯 Skoring dengan Model Tersimpan")

            # ====== Unsupervised ======
            if run_unsup:
                N = len(df_ml)
                cont_from_k = max(0.001, min(0.5, topk / max(1, N)))
                contamination = cont_from_k if topk else contam_pct
                model = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
                model.fit(Xs)
                score = model.decision_function(Xs)
                label = model.predict(Xs)

                df_res = df_ml.copy()
                df_res["skor_anomali"] = score
                df_res["is_anomali"] = (label == -1).astype(int)
                df_res["alasan"] = top_feature_reasons(X, fitur_pakai, k=3)
                df_res = df_res.sort_values("skor_anomali", ascending=True)
                outliers = df_res[df_res["is_anomali"] == 1]
                outliers_topk = outliers.head(min(topk, len(outliers))).copy()

                st.success(f"[Unsupervised] Kandidat investigasi: {len(outliers_topk):,} dari total {N:,}.")
                show_cols = [c for c in ["LOCATION_CODE","NAMA","TARIF","DAYA"] if c in outliers_topk.columns] + ["skor_anomali","alasan"]
                st.dataframe(outliers_topk[show_cols], use_container_width=True, height=520)

                plan_cols = ["LOCATION_CODE","NAMA","TARIF","DAYA","skor_anomali","alasan","TANGGAL_INSPEKSI","LABEL_TO","CATATAN","PETUGAS"]
                plan_df = outliers_topk.copy()
                for c in ["TANGGAL_INSPEKSI","LABEL_TO","CATATAN","PETUGAS"]: plan_df[c] = ""
                for c in plan_cols:
                    if c not in plan_df.columns: plan_df[c] = ""
                st.download_button("📥 Export Inspection Plan (Top-K)",
                                   plan_df[plan_cols].to_csv(index=False).encode("utf-8"),
                                   file_name="inspection_plan_topk.csv",
                                   mime="text/csv")

            # Helper build supervised table
            def build_supervised_table(df_amr: pd.DataFrame, labels_df: pd.DataFrame, window_days: int) -> pd.DataFrame:
                feats = []
                df_amr = _ensure_date_col(df_amr)
                valid_ids = set(seg_df['LOCATION_CODE'].astype(str).unique().tolist())
                for _, row in labels_df.iterrows():
                    loc = str(row['LOCATION_CODE']).strip()
                    if loc not in valid_ids:
                        continue
                    t_ins = pd.to_datetime(row['TANGGAL_INSPEKSI'], errors='coerce')
                    if pd.isna(t_ins):
                        continue
                    t0 = t_ins - pd.Timedelta(days=window_days)
                    sub = df_amr[(df_amr['LOCATION_CODE'].astype(str) == loc) & (df_amr['TANGGAL'] >= t0) & (df_amr['TANGGAL'] <= t_ins)]
                    if len(sub) == 0:
                        continue
                    med = sub[FITUR_TEKNIS].median()
                    std = sub[FITUR_TEKNIS].std().fillna(0.0)
                    p95 = sub[FITUR_TEKNIS].quantile(0.95)
                    feat = {f"MED_{c}": med[c] for c in FITUR_TEKNIS}
                    feat.update({f"STD_{c}": std[c] for c in FITUR_TEKNIS})
                    feat.update({f"P95_{c}": p95[c] for c in FITUR_TEKNIS})
                    feat['LOCATION_CODE'] = loc
                    feat['LABEL_TO'] = int(row['LABEL_TO'])
                    feat['TANGGAL_INSPEKSI'] = t_ins
                    feats.append(feat)
                return pd.DataFrame(feats)

            # ====== Train Supervised & Save Model ======
            if train_supervised:
                store_df = load_labels_store()
                if store_df.empty:
                    st.warning("Belum ada label tersimpan. Unggah label dulu.")
                else:
                    st.info("Mempersiapkan fitur dari label tersimpan...")
                    sup_df = build_supervised_table(df, store_df, int(window_days))
                    if sup_df.empty:
                        st.error("Tidak ada fitur yang terbentuk dari label (cek kecocokan IDPEL/tanggal).")
                    else:
                        st.success(f"Fitur terbentuk: {len(sup_df)} baris, {sup_df.shape[1]-3} kolom fitur.")
                        train_df = sup_df.sort_values('TANGGAL_INSPEKSI')
                        cut_idx = max(1, int(len(train_df)*0.8))
                        tr, va = train_df.iloc[:cut_idx], train_df.iloc[cut_idx:]
                        feat_cols = [c for c in sup_df.columns if c not in ['LOCATION_CODE','LABEL_TO','TANGGAL_INSPEKSI']]
                        if tr.empty or va.empty:
                            st.error("Data train/valid kosong. Tambah label.")
                        else:
                            X_tr = tr[feat_cols].fillna(0.0).values; y_tr = tr['LABEL_TO'].values
                            X_va = va[feat_cols].fillna(0.0).values; y_va = va['LABEL_TO'].values
                            clf = RandomForestClassifier(n_estimators=350, class_weight='balanced', n_jobs=-1, random_state=42)
                            clf.fit(X_tr, y_tr)
                            proba = clf.predict_proba(X_va)[:,1]
                            roc = roc_auc_score(y_va, proba) if len(np.unique(y_va))>1 else float('nan')
                            pr  = average_precision_score(y_va, proba) if len(np.unique(y_va))>1 else float('nan')
                            st.success(f"Model dilatih. ROC-AUC={roc:.3f}  PR-AUC={pr:.3f}")
                            artefact = {
                                "model": clf,
                                "feat_cols": feat_cols,
                                "window_days": int(window_days),
                                "timestamp": datetime.now().isoformat(timespec='seconds'),
                                "segment": {
                                    "prefix": pilih_prefix,
                                    "range_daya": list(range_daya),
                                    "unit_opt": unit_opt
                                }
                            }
                            with open(MODEL_STORE, "wb") as f:
                                pickle.dump(artefact, f)
                            st.success("✅ Model tersimpan ke file: {}".format(MODEL_STORE))

            # ====== Score with saved model ======
            if score_with_saved:
                if not os.path.exists(MODEL_STORE):
                    st.error("Belum ada model tersimpan.")
                else:
                    with open(MODEL_STORE, "rb") as f:
                        artefact = pickle.load(f)
                    feat_cols = artefact.get("feat_cols", [])
                    wdays = int(artefact.get("window_days", 30))
                    clf = artefact["model"]
                    max_date = df['TANGGAL'].max()
                    t0 = max_date - pd.Timedelta(days=wdays)
                    recent = df[df['TANGGAL'].between(t0, max_date)].copy()
                    med = recent.groupby('LOCATION_CODE')[FITUR_TEKNIS].median().add_prefix('MED_')
                    std = recent.groupby('LOCATION_CODE')[FITUR_TEKNIS].std().fillna(0.0).add_prefix('STD_')
                    p95 = recent.groupby('LOCATION_CODE')[FITUR_TEKNIS].quantile(0.95).add_prefix('P95_')
                    feat_now = pd.concat([med, std, p95], axis=1).reset_index()
                    for c in feat_cols:
                        if c not in feat_now.columns: feat_now[c] = 0.0
                    X_now = feat_now[feat_cols].fillna(0.0).values
                    feat_now['proba_TO'] = clf.predict_proba(X_now)[:,1]
                    info_cols = []
                    if "NAMA_PELANGGAN" in df.columns: info_cols.append("NAMA_PELANGGAN")
                    if "TARIF" in df.columns: info_cols.append("TARIF")
                    elif "TARIFF" in df.columns: info_cols.append("TARIFF")
                    if "POWER" in df.columns: info_cols.append("POWER")
                    info = df.sort_values('TANGGAL').groupby('LOCATION_CODE').tail(1)[['LOCATION_CODE']+info_cols].copy()
                    info = info.rename(columns={"NAMA_PELANGGAN":"NAMA","TARIFF":"TARIF","POWER":"DAYA"})
                    if "TARIF" not in info.columns: info["TARIF"] = "-"
                    if "DAYA" not in info.columns: info["DAYA"] = "-"
                    feat_now = feat_now.merge(info, on='LOCATION_CODE', how='left').sort_values('proba_TO', ascending=False)
                    topk_df = feat_now.head(int(topk))[['LOCATION_CODE','NAMA','TARIF','DAYA','proba_TO']]
                    st.subheader("🎯 Rekomendasi Operasional (Model Tersimpan)")
                    st.dataframe(topk_df, use_container_width=True, height=420)
                    st.download_button("📥 Unduh Rekomendasi Operasional (Top-K)",
                                       topk_df.to_csv(index=False).encode('utf-8'),
                                       file_name="rekomendasi_operasional_supervised_topk.csv",
                                       mime="text/csv")

            st.markdown("---")
            with st.expander("🧹 Manajemen Penyimpanan (Lanjutan)"):
                colm1, colm2 = st.columns(2)
                if colm1.button("Hapus labels_store.csv"):
                    if os.path.exists(LABELS_STORE):
                        os.remove(LABELS_STORE)
                        load_labels_store.clear()
                        st.success("labels_store.csv dihapus.")
                    else:
                        st.info("labels_store.csv belum ada.")
                if colm2.button("Hapus model_rf.pkl"):
                    if os.path.exists(MODEL_STORE):
                        os.remove(MODEL_STORE)
                        st.success("model_rf.pkl dihapus.")
                    else:
                        st.info("model_rf.pkl belum ada.")

    # --------- Sub-tab Upload --------- #
    with sub_upload:
        uploaded_file = st.file_uploader("📥 Upload File Excel AMR Harian", type=["xlsx"])
        if uploaded_file:
            df_up = pd.read_excel(uploaded_file, sheet_name=0)
            df_up = df_up.dropna(subset=['LOCATION_CODE'])
            df_up['LOCATION_CODE'] = df_up['LOCATION_CODE'].astype(str).str.strip()
            if 'TANGGAL' in df_up.columns:
                df_up['TANGGAL'] = pd.to_datetime(df_up['TANGGAL'], errors='coerce')
            elif 'READ_DATE' in df_up.columns:
                df_up['TANGGAL'] = pd.to_datetime(df_up['READ_DATE'], errors='coerce')
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

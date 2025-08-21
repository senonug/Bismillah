import streamlit as st
import pandas as pd
import numpy as np
import os, pickle
import plotly.express as px
from datetime import datetime

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

st.set_page_config(page_title="T-Energy", layout="wide", page_icon="⚡")

DATA_PATH_AMR   = "data_harian.csv"
DATA_PATH_OLAP  = "olap_pascabayar.csv"
LABELS_STORE    = "labels_store.csv"
MODEL_STORE     = "model_rf.pkl"

# ========== Helpers & Cache ========== #
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
    df = df.copy()
    if target_col in df.columns:
        df[target_col] = pd.to_datetime(df[target_col], errors='coerce')
    elif 'READ_DATE' in df.columns:
        df[target_col] = pd.to_datetime(df['READ_DATE'], errors='coerce')
    return df

# ---------- Technical features & rules ---------- #
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
    df = _ensure_date_col(df)
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

# ========== IDPEL Explanation Utilities ========== #
def explain_idpel(df_all: pd.DataFrame, idpel: str) -> dict:
    res = {"exists": False}
    if df_all.empty:
        return res
    df_all = df_all.copy()
    df_all['LOCATION_CODE'] = df_all['LOCATION_CODE'].astype(str).str.strip()
    idpel = str(idpel).strip()
    df_all = _ensure_date_col(df_all)
    df_id = df_all[df_all['LOCATION_CODE'] == idpel].copy()
    if df_id.empty:
        return res
    res["exists"] = True
    latest = df_id.sort_values('TANGGAL').tail(1)
    nama = latest['NAMA_PELANGGAN'].iloc[0] if 'NAMA_PELANGGAN' in latest.columns else "-"
    tarif = latest['TARIF'].iloc[0] if 'TARIF' in latest.columns else (latest['TARIFF'].iloc[0] if 'TARIFF' in latest.columns else "-")
    daya  = latest['POWER'].iloc[0] if 'POWER' in latest.columns else "-"
    res["identity"] = {"LOCATION_CODE": idpel, "NAMA": nama, "TARIF": tarif, "DAYA": daya}

    df_num = _numericize(df_id.copy(), FITUR_TEKNIS)
    ind_rows = df_num.apply(lambda r: cek_indikator_row(r, st.session_state), axis=1)
    ind_df = pd.DataFrame(ind_rows.tolist())
    ind_df['TANGGAL'] = df_id['TANGGAL']
    res["indicator_daily"] = ind_df
    bool_cols = ind_df.columns.drop('TANGGAL')
    res["indicator_summary"] = ind_df[bool_cols].sum().sort_values(ascending=False)

    agg_all = aggregate_features(df_all, how="median")
    feat_names = FITUR_TEKNIS
    Xagg = agg_all[feat_names].fillna(0.0).values.astype(float)
    z = robust_zscores(Xagg)
    try:
        idx = agg_all.index[agg_all['LOCATION_CODE'].astype(str)==idpel][0]
        z_row = z[idx]
        order = np.argsort(-np.abs(z_row))[:6]
        reasons = []
        for j in order:
            arrow = "↑" if z_row[j] > 0 else "↓"
            reasons.append({"feature": feat_names[j], "z": float(z_row[j]), "direction": arrow})
        res["unsup_reasons"] = reasons
    except IndexError:
        res["unsup_reasons"] = []

    try:
        model = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
        model.fit(Xagg)
        score = model.decision_function(Xagg)
        res["unsup_score"] = float(score[idx])
    except Exception:
        res["unsup_score"] = None

    if os.path.exists(MODEL_STORE):
        try:
            with open(MODEL_STORE, "rb") as f:
                artefact = pickle.load(f)
            feat_cols = artefact.get("feat_cols", [])
            wdays = int(artefact.get("window_days", 30))
            clf = artefact["model"]
            max_date = df_all['TANGGAL'].max()
            t0 = max_date - pd.Timedelta(days=wdays)
            recent = df_all[df_all['TANGGAL'].between(t0, max_date)].copy()
            med = recent.groupby('LOCATION_CODE')[FITUR_TEKNIS].median().add_prefix('MED_')
            std = recent.groupby('LOCATION_CODE')[FITUR_TEKNIS].std().fillna(0.0).add_prefix('STD_')
            p95 = recent.groupby('LOCATION_CODE')[FITUR_TEKNIS].quantile(0.95).add_prefix('P95_')
            feat_now = pd.concat([med, std, p95], axis=1).reset_index()
            if idpel not in feat_now['LOCATION_CODE'].astype(str).tolist():
                res["sup_available"] = False
                return res
            row_now = feat_now[feat_now['LOCATION_CODE'].astype(str) == idpel].copy()
            for c in feat_cols:
                if c not in row_now.columns: row_now[c] = 0.0
            X_now = row_now[feat_cols].fillna(0.0).values
            proba = clf.predict_proba(X_now)[:,1][0]
            res["sup_available"] = True
            res["proba_TO"] = float(proba)

            imp = getattr(clf, "feature_importances_", None)
            if imp is None:
                res["sup_top_features"] = []
            else:
                M = feat_now[feat_cols].fillna(0.0).values
                zM = robust_zscores(M)
                ridx = feat_now.index[feat_now['LOCATION_CODE'].astype(str)==idpel][0]
                zrow = zM[ridx]
                contrib = np.abs(zrow) * imp
                order = np.argsort(-contrib)[:8]
                res["sup_top_features"] = [{"feature": feat_cols[j], "z": float(zrow[j]), "importance": float(imp[j])} for j in order]
        except Exception:
            res["sup_available"] = False
    else:
        res["sup_available"] = False
    return res

# ========== Auth ========== #
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

# ========== Tabs ========== #
tab_amr, tab_pasca, tab_prabayar = st.tabs(["📥 AMR Harian", "💳 Pascabayar", "💡 Prabayar"])

with tab_amr:
    st.title("📊 Dashboard Target Operasi AMR - P2TL")
    st.markdown("---")

    with st.expander("⚙️ Setting Parameter Threshold"):
        colA, colB = st.columns(2)
        with colA:
            st.number_input("Cos Phi Max TM", key="cos_phi_tm", value=0.4)
            st.number_input("Set Selisih Tegangan TM", key="low_v_diff_tm", value=2.0)
            st.number_input("Tegangan Maksimum TM", key="vmax_tm", value=62.0)
            st.number_input("Set Batas bawah Arus Maks pada TM", key="over_i_tm", value=5.0)
            st.number_input("Toleransi Unbalance TM", key="unbal_tol_tm", value=0.5)
        with colB:
            st.write("Pengaturan lain (singkat).")
        st.number_input("Banyak Data yang Ditampilkan", key="top_limit", value=50)

    df_raw = load_csv_safe(DATA_PATH_AMR)

    sub_threshold, sub_ml_auto, sub_upload = st.tabs(["🔎 Deteksi Threshold", "🤖 Deteksi ML – Otomatis", "➕ Upload Data"])

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
            indikator_agg['Skor'] = indikator_agg[boolean_cols].mul(pd.Series(INDIKATOR_BOBOT)).sum(axis=1)

            if 'TANGGAL' in df.columns and pd.api.types.is_datetime64_any_dtype(df['TANGGAL']):
                df_info = df.sort_values('TANGGAL').dropna(subset=['TANGGAL']).groupby('LOCATION_CODE').tail(1)
            else:
                df_info = df.drop_duplicates(subset='LOCATION_CODE', keep='last')
            use_cols = ['LOCATION_CODE','NAMA_PELANGGAN','POWER']
            if 'TARIF' in df_info.columns: use_cols.append('TARIF')
            elif 'TARIFF' in df_info.columns: use_cols.append('TARIFF')
            df_info = df_info[use_cols].rename(columns={'NAMA_PELANGGAN':'NAMA','TARIFF':'TARIF','POWER':'DAYA'})
            if 'TARIF' not in df_info.columns: df_info['TARIF'] = '-'
            if 'DAYA' not in df_info.columns: df_info['DAYA'] = '-'
            result = df_info.merge(indikator_agg, on='LOCATION_CODE', how='right')
            for c in ['NAMA','TARIF','DAYA']:
                if c in result.columns: result[c] = result[c].fillna('-')
            top_limit = st.session_state.get('top_limit',50)
            top50 = result.drop_duplicates('LOCATION_CODE').sort_values('Skor', ascending=False).head(top_limit)
            st.dataframe(top50[['LOCATION_CODE','NAMA','TARIF','DAYA','Jumlah Berulang','Jumlah Indikator','Skor']], use_container_width=True, height=520)

            # === Audit & Penjelasan Kenapa ID tidak muncul ===
            st.markdown("### 🔎 Audit IDPEL di Deteksi Threshold")
            qid = st.text_input("Masukkan IDPEL (LOCATION_CODE) untuk audit threshold", key="audit_idpel_thr")
            if st.button("Audit ID ini (Threshold)"):
                qid_s = str(qid).strip()
                if qid_s == "":
                    st.warning("Isi IDPEL dulu.")
                else:
                    in_raw = df_raw['LOCATION_CODE'].astype(str).str.strip().eq(qid_s).any() if not df_raw.empty else False
                    in_customer = _filter_customer_only(df_raw).assign(LOCATION_CODE=lambda x: x['LOCATION_CODE'].astype(str).str.strip())['LOCATION_CODE'].eq(qid_s).any() if not df_raw.empty else False
                    in_result = result['LOCATION_CODE'].astype(str).eq(qid_s).any()
                    reasons = []
                    if not in_raw:
                        reasons.append("IDPEL tidak ada di data AMR (data_harian.csv).")
                    elif not in_customer:
                        reasons.append("LOCATION_TYPE bukan Customer/COSTUMER → difilter.")
                    elif in_result:
                        row_id = result[result['LOCATION_CODE'].astype(str)==qid_s].iloc[0]
                        skor = row_id['Skor']
                        rank = int((result['Skor'] > skor).sum() + 1)
                        reasons.append(f"Skor={skor:.0f}, Peringkat global={rank}.")
                        if rank > st.session_state.get('top_limit',50):
                            reasons.append(f"Tertinggal di luar TOP-{st.session_state.get('top_limit',50)}.")
                        trig = [k for k in INDIKATOR_BOBOT.keys() if k in result.columns and bool(row_id.get(k, False))]
                        if len(trig)==0:
                            reasons.append("Tidak ada indikator threshold yang terpicu.")
                        else:
                            reasons.append("Indikator terpicu: " + ", ".join(trig))
                    else:
                        reasons.append("ID ada di data Customer, tapi tidak terhitung karena semua indikator=0 (Skor=0).")
                    st.info(" ; ".join(reasons))
                    hist_id = df[df['LOCATION_CODE'].astype(str)==qid_s].sort_values('TANGGAL').tail(30)
                    if not hist_id.empty:
                        st.dataframe(hist_id[['TANGGAL','LOCATION_CODE','NAMA','TARIF','DAYA'] + [c for c in FITUR_TEKNIS if c in hist_id.columns]].tail(10), use_container_width=True)

            # === Label Lapangan Overlay (opsional) ===
            store_df = load_labels_store()
            if not store_df.empty:
                st.markdown("### 🏷️ Overlay Label Lapangan")
                colL, colR = st.columns([1,2])
                with colL:
                    pin_label = st.checkbox("Tampilkan pelanggan berlabel pelanggaran teratas", value=True)
                if pin_label:
                    pos = store_df[store_df['LABEL_TO']==1].copy()
                    if not pos.empty:
                        pos_ids = pos['LOCATION_CODE'].astype(str).unique().tolist()
                        pinned = result[result['LOCATION_CODE'].astype(str).isin(pos_ids)].copy()
                        if not pinned.empty:
                            st.success(f"Menampilkan {len(pinned)} pelanggan berlabel pelanggaran (lapangan).")
                            st.dataframe(pinned[['LOCATION_CODE','NAMA','TARIF','DAYA','Jumlah Berulang','Jumlah Indikator','Skor']].head(200), use_container_width=True, height=320)
                        else:
                            st.info("Tidak ada ID berlabel yang cocok dengan data AMR saat ini.")

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
            X = df[fitur_pakai].values.astype(float)
            scaler = RobustScaler() if scaler_choice.startswith("Robust") else StandardScaler()
            Xs = scaler.fit_transform(X)

            labels_file = st.file_uploader("Label TO (CSV: LOCATION_CODE, TANGGAL_INSPEKSI, LABEL_TO)", type=["csv"], key="labels_upload_auto")
            if labels_file is not None:
                try:
                    labels_new = pd.read_csv(labels_file)
                    store_df = save_labels_to_store(labels_new)
                    st.success(f"Label disimpan. Total label tersimpan: {len(store_df)} baris unik.")
                except Exception as e:
                    st.error(f"Gagal memproses label: {e}")

            store_df = load_labels_store()
            has_model = os.path.exists(MODEL_STORE)
            cA, cB, cC = st.columns(3)
            cA.metric("Label tersimpan", len(store_df))
            cB.metric("Model tersimpan", "Ada ✅" if has_model else "Belum ada")
            if not store_df.empty:
                cC.download_button("📥 Unduh seluruh label", data=store_df.to_csv(index=False).encode('utf-8'),
                                   file_name="labels_store.csv", mime="text/csv")

            colU, colS1, colS2 = st.columns([1.1,1,1])
            run_unsup = colU.button("🚀 Deteksi Tanpa Label (Anomali)")
            train_supervised = colS1.button("🧠 Latih Ulang & Simpan Model Supervised")
            score_with_saved = colS2.button("🎯 Skoring dengan Model Tersimpan")

            if run_unsup:
                N = len(df)
                cont_from_k = max(0.001, min(0.5, topk / max(1, N)))
                contamination = cont_from_k if topk else contam_pct
                model = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
                model.fit(Xs)
                score = model.decision_function(Xs)
                label = model.predict(Xs)

                df_res = df.copy()
                df_res["skor_anomali"] = score
                df_res["is_anomali"] = (label == -1).astype(int)
                df_res["alasan"] = top_feature_reasons(df[fitur_pakai].values, fitur_pakai, k=3)
                df_res = df_res.sort_values("skor_anomali", ascending=True)
                outliers = df_res[df_res["is_anomali"] == 1]
                outliers_topk = outliers.head(min(topk, len(outliers))).copy()
                show_cols = [c for c in ["LOCATION_CODE","NAMA_PELANGGAN","TARIF","TARIFF","POWER"] if c in outliers_topk.columns] + ["skor_anomali","alasan"]
                st.dataframe(outliers_topk[show_cols], use_container_width=True, height=520)
                plan_cols = ["LOCATION_CODE","NAMA_PELANGGAN","TARIF","POWER","skor_anomali","alasan","TANGGAL_INSPEKSI","LABEL_TO","CATATAN","PETUGAS"]
                plan_df = outliers_topk.copy()
                for c in ["TANGGAL_INSPEKSI","LABEL_TO","CATATAN","PETUGAS"]: plan_df[c] = ""
                for c in plan_cols:
                    if c not in plan_df.columns: plan_df[c] = ""
                st.download_button("📥 Export Inspection Plan (Top-K)",
                                   plan_df[plan_cols].to_csv(index=False).encode("utf-8"),
                                   file_name="inspection_plan_topk.csv",
                                   mime="text/csv")

            def build_supervised_table(df_amr: pd.DataFrame, labels_df: pd.DataFrame, window_days: int) -> pd.DataFrame:
                feats = []
                df_amr = _ensure_date_col(df_amr)
                for _, row in labels_df.iterrows():
                    loc = str(row['LOCATION_CODE']).strip()
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

            if train_supervised:
                if store_df.empty:
                    st.warning("Belum ada label tersimpan. Unggah label dulu.")
                else:
                    st.info("Mempersiapkan fitur dari label tersimpan...")
                    sup_df = build_supervised_table(df, store_df, int(window_days))
                    if sup_df.empty:
                        st.error("Tidak ada fitur yang terbentuk dari label (cek kecocokan IDPEL/tanggal).")
                    else:
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
                            artefact = {"model": clf, "feat_cols": feat_cols, "window_days": int(window_days), "timestamp": datetime.now().isoformat(timespec='seconds')}
                            with open(MODEL_STORE, "wb") as f:
                                pickle.dump(artefact, f)
                            st.success("✅ Model tersimpan ke file: {}".format(MODEL_STORE))

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
                    topk_df = feat_now.head(int(topk))[["LOCATION_CODE","NAMA","TARIF","DAYA","proba_TO"]]
                    st.dataframe(topk_df, use_container_width=True, height=420)
                    st.download_button("📥 Unduh Rekomendasi Operasional (Top-K)",
                                       topk_df.to_csv(index=False).encode('utf-8'),
                                       file_name="rekomendasi_operasional_supervised_topk.csv",
                                       mime="text/csv")

            # ========== Analisis IDPEL (Penjelasan) ========== #
            st.markdown("---")
            st.markdown("## 🔍 Analisis IDPEL (Penjelasan)")
            q_idpel = st.text_input("Masukkan IDPEL (LOCATION_CODE) untuk melihat evidence", key="explain_idpel")
            if st.button("Analisis IDPEL"):
                if q_idpel.strip() == "":
                    st.warning("Isi IDPEL terlebih dahulu.")
                else:
                    ev = explain_idpel(df, q_idpel)
                    if not ev.get("exists", False):
                        st.error("IDPEL tidak ditemukan di data historis.")
                    else:
                        colI, colM = st.columns([1,2])
                        with colI:
                            st.markdown("**Identitas**")
                            st.write(ev["identity"])
                            if ev.get("sup_available", False):
                                st.metric("Probabilitas TO (model)", f"{ev['proba_TO']:.3f}")
                            if ev.get("unsup_score", None) is not None:
                                st.metric("Skor anomali (IForest)", f"{ev['unsup_score']:.3f}")
                        with colM:
                            st.markdown("**Ringkasan Indikator (terpicu berapa kali)**")
                            ind_sum = ev["indicator_summary"]
                            st.dataframe(ind_sum.rename("jumlah").to_frame(), use_container_width=True, height=240)
                        st.markdown("**Alasan (tanpa label, z-score robust)**")
                        if len(ev["unsup_reasons"]) == 0:
                            st.info("Tidak ada alasan signifikan (atau ID tidak ada di agregat).")
                        else:
                            df_r = pd.DataFrame(ev["unsup_reasons"])
                            st.dataframe(df_r, use_container_width=True, height=220)
                        if ev.get("sup_available", False) and len(ev.get("sup_top_features", []))>0:
                            st.markdown("**Fitur dominan (model supervised, perkiraan kontribusi)**")
                            st.dataframe(pd.DataFrame(ev["sup_top_features"]), use_container_width=True, height=240)

                        with st.expander("📈 Grafik Tren Teknis (30 hari terakhir)"):
                            df_id = _ensure_date_col(df[df['LOCATION_CODE'].astype(str)==q_idpel]).copy()
                            if df_id.empty:
                                st.info("Tidak ada histori untuk digrafikkan.")
                            else:
                                maxd = pd.to_datetime(df_id['TANGGAL']).max()
                                t0 = maxd - pd.Timedelta(days=30)
                                d30 = df_id[df_id['TANGGAL'].between(t0, maxd)].sort_values('TANGGAL')
                                for grp in [["ACTIVE_POWER_L1","ACTIVE_POWER_L2","ACTIVE_POWER_L3"],
                                            ["VOLTAGE_L1","VOLTAGE_L2","VOLTAGE_L3"],
                                            ["POWER_FACTOR_L1","POWER_FACTOR_L2","POWER_FACTOR_L3"]]:
                                    avail = [c for c in grp if c in d30.columns]
                                    if len(avail) >= 2:
                                        fig = px.line(d30, x='TANGGAL', y=avail, title=", ".join(avail))
                                        st.plotly_chart(fig, use_container_width=True)

                        with st.expander("📄 Daftar Indikator per Hari (untuk unduh)"):
                            daily = ev["indicator_daily"].copy()
                            st.dataframe(daily, use_container_width=True, height=220)
                            st.download_button("📥 Download Evidence (CSV)",
                                               daily.to_csv(index=False).encode('utf-8'),
                                               file_name=f"evidence_{q_idpel}.csv",
                                               mime="text/csv")

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

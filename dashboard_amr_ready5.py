
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
from sklearn.inspection import permutation_importance

# ===================== Config ===================== #
st.set_page_config(page_title="T-Energy", layout="wide", page_icon="⚡")
DATA_PATH_AMR = "data_harian.csv"

# ===================== Utils ===================== #
@st.cache_data(show_spinner=False)
def load_csv_safe(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def _filter_customer_only(df: pd.DataFrame) -> pd.DataFrame:
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

def _numericize(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
        else:
            df[c] = 0.0
    return df

FITUR_TEKNIS = [
    "CURRENT_L1","CURRENT_L2","CURRENT_L3",
    "VOLTAGE_L1","VOLTAGE_L2","VOLTAGE_L3",
    "ACTIVE_POWER_L1","ACTIVE_POWER_L2","ACTIVE_POWER_L3",
    "ACTIVE_POWER_SIANG","ACTIVE_POWER_MALAM",
    "POWER_FACTOR_L1","POWER_FACTOR_L2","POWER_FACTOR_L3",
    "CURRENT_LOOP","FREEZE"
]

# ---------- Indicator logic (reuse thresholds from session_state) ---------- #
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
    indikator['v_drop'] = (max(v) - min(v)) > session.get('low_v_diff_tm', 2.0)
    indikator['cos_phi_kecil'] = any([row.get(f'POWER_FACTOR_L{i}', 1) < session.get('cos_phi_tm', 0.4) for i in range(1,4)])
    indikator['active_power_negative'] = any([row.get(f'ACTIVE_POWER_L{i}', 0) < 0 for i in range(1,4)])
    indikator['v_lost'] = (row.get('VOLTAGE_L1', 0) == 0 or row.get('VOLTAGE_L2', 0) == 0 or row.get('VOLTAGE_L3', 0) == 0)
    indikator['current_loop'] = row.get('CURRENT_LOOP', 0) == 1
    indikator['freeze'] = row.get('FREEZE', 0) == 1
    return indikator

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
    pick_cols = ["LOCATION_CODE"]
    if "NAMA_PELANGGAN" in info.columns: pick_cols.append("NAMA_PELANGGAN")
    if "TARIF" in info.columns: pick_cols.append("TARIF")
    elif "TARIFF" in info.columns: pick_cols.append("TARIFF")
    if "POWER" in info.columns: pick_cols.append("POWER")

    info = info[pick_cols].rename(columns={"NAMA_PELANGGAN":"NAMA","TARIFF":"TARIF","POWER":"DAYA"})
    if "TARIF" not in info.columns: info["TARIF"] = "-"
    if "DAYA" not in info.columns: info["DAYA"] = "-"

    merged = feat_agg.merge(info, on="LOCATION_CODE", how="left")
    return merged

def run_iforest(X: np.ndarray, contamination: float = 0.1, random_state: int = 42):
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=random_state)
    model.fit(X)
    score = model.decision_function(X)  # higher = more normal
    label = model.predict(X)            # -1 outlier, 1 inlier
    return model, score, label

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
tab_amr, = st.tabs(["📥 AMR Harian"])

# ===================== AMR Harian ===================== #
with tab_amr:
    st.title("📊 AMR – Deteksi Berbasis ML")
    st.markdown("---")
    df_raw = load_csv_safe(DATA_PATH_AMR)

    sub_bootstrap, sub_supervised, sub_upload = st.tabs([
        "🤖 Bootstrap (Tanpa Label)",
        "🧠 Mode Supervised (Berlabel)",
        "➕ Upload Data"
    ])

    # -------- Bootstrap (tanpa label / Top-K) -------- #
    with sub_bootstrap:
        st.info("Deteksi anomali tanpa label untuk membuat daftar inspeksi awal dan mengumpulkan label TO.")
        if df_raw.empty:
            st.warning("Belum ada data historis. Silakan upload pada tab 'Upload Data'.")
        else:
            df = df_raw.copy()
            df = df.dropna(subset=['LOCATION_CODE']).copy()
            df['LOCATION_CODE'] = df['LOCATION_CODE'].astype(str).str.strip()
            df = _filter_customer_only(df)
            df = _ensure_customer_cols(df)
            df = _numericize(df, FITUR_TEKNIS)

            st.markdown("### Segmentasi (opsional)")
            colf1, colf2 = st.columns([1,1])
            with colf1:
                if "TARIF" not in df.columns and "TARIFF" in df.columns:
                    df["TARIF"] = df["TARIFF"]
                tarif_vals = sorted([t for t in df["TARIF"].astype(str).str.strip().str.upper().unique() if t != "-"])
                prefixes = sorted(list(set([t[:1] for t in tarif_vals if len(t)>0])))
                pilih_prefix = st.multiselect("Prefix Tarif", options=prefixes, default=[])
            with colf2:
                if "POWER" in df.columns:
                    daya_num = pd.to_numeric(df["POWER"], errors="coerce")
                    dmin = int(np.nanmin(daya_num)) if not np.isnan(daya_num.min()) else 0
                    dmax = int(np.nanmax(daya_num)) if not np.isnan(daya_num.max()) else 10000
                else:
                    dmin, dmax = 0, 10000
                range_daya = st.slider("Rentang Daya (VA)", min_value=int(dmin), max_value=int(max(dmax, dmin+1)), value=(int(dmin), int(max(dmin+1000, min(dmax, dmin+5000)))))

            seg_df = df.copy()
            if len(pilih_prefix) > 0 and "TARIF" in seg_df.columns:
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

            st.markdown("### Parameter ML")
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

            fitur_pakai = pilih_fitur if len(pilih_fitur) > 0 else FITUR_TEKNIS
            X = df_ml[fitur_pakai].values.astype(float)

            scaler = RobustScaler() if scaler_choice.startswith("Robust") else StandardScaler()
            Xs = scaler.fit_transform(X)

            st.markdown("### Jalankan & Hasil")
            if st.button("🚀 Jalankan Deteksi", key="run_bootstrap"):
                N = len(df_ml)
                if N == 0:
                    st.warning("Data kosong pada segmen/opsi yang dipilih.")
                else:
                    cont_from_k = max(0.001, min(0.5, topk / max(1, N)))
                    contamination = cont_from_k if topk else contam_pct

                    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
                    model.fit(Xs)
                    score = model.decision_function(Xs)
                    label = model.predict(Xs)

                    df_res = df_ml.copy()
                    df_res["skor_anomali"] = score
                    df_res["is_anomali"] = (label == -1).astype(int)
                    reasons = top_feature_reasons(X, fitur_pakai, k=3)
                    df_res["alasan"] = reasons
                    df_res = df_res.sort_values("skor_anomali", ascending=True)

                    outliers = df_res[df_res["is_anomali"] == 1]
                    outliers_topk = outliers.head(min(topk, len(outliers))).copy()

                    st.success(f"Kandidat investigasi: {len(outliers_topk):,} / outlier terdeteksi {len(outliers):,} / total {N:,}.")
                    show_cols = [c for c in ["LOCATION_CODE","NAMA","TARIF","DAYA"] if c in outliers_topk.columns] + ["skor_anomali","alasan"]
                    st.dataframe(outliers_topk[show_cols], use_container_width=True, height=520)

                    plan_cols = ["LOCATION_CODE","NAMA","TARIF","DAYA","skor_anomali","alasan","TANGGAL_INSPEKSI","LABEL_TO","CATATAN","PETUGAS"]
                    plan_df = outliers_topk.copy()
                    for c in ["TANGGAL_INSPEKSI","LABEL_TO","CATATAN","PETUGAS"]:
                        plan_df[c] = ""
                    for c in plan_cols:
                        if c not in plan_df.columns:
                            plan_df[c] = ""
                    st.download_button("📥 Export Inspection Plan (Top-K)",
                                       plan_df[plan_cols].to_csv(index=False).encode("utf-8"),
                                       file_name="inspection_plan_topk.csv",
                                       mime="text/csv")
                    st.caption("Gunakan file ini sebagai daftar inspeksi. Setelah lapangan selesai, isikan LABEL_TO (1/0) dan unggah sebagai label untuk mulai mode supervised.")

    # -------- Supervised (berlabel) -------- #
    with sub_supervised:
        st.info("Mode Supervised: latih model klasifikasi menggunakan hasil inspeksi (label) untuk rekomendasi yang lebih presisi.")
        if df_raw.empty:
            st.warning("Belum ada data historis. Silakan upload pada tab 'Upload Data'.")
        else:
            df = df_raw.copy()
            if 'TANGGAL' not in df.columns:
                st.error("Kolom 'TANGGAL' tidak ditemukan pada data AMR. Supervised membutuhkan tanggal untuk split dan pembuatan fitur.")
            else:
                df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], errors='coerce')
                df = df.dropna(subset=['LOCATION_CODE']).copy()
                df['LOCATION_CODE'] = df['LOCATION_CODE'].astype(str).str.strip()
                df = _filter_customer_only(df)
                df = _ensure_customer_cols(df)
                df = _numericize(df, FITUR_TEKNIS)

                labels_file = st.file_uploader("📥 Upload Label TO (CSV)", type=["csv"], key="labels_upload")
                meta_file = st.file_uploader("📥 Upload Metadata Pelanggan (opsional, CSV)", type=["csv"], key="meta_upload")

                if labels_file:
                    labels = pd.read_csv(labels_file)
                    # basic checks
                    need_cols = {"LOCATION_CODE","TANGGAL_INSPEKSI","LABEL_TO"}
                    if not need_cols.issubset(set(labels.columns)):
                        st.error(f"Label harus memuat kolom: {need_cols}.")
                    else:
                        labels['LOCATION_CODE'] = labels['LOCATION_CODE'].astype(str).str.strip()
                        labels['TANGGAL_INSPEKSI'] = pd.to_datetime(labels['TANGGAL_INSPEKSI'], errors='coerce')
                        labels = labels.dropna(subset=['LOCATION_CODE','TANGGAL_INSPEKSI','LABEL_TO']).copy()
                        labels['LABEL_TO'] = labels['LABEL_TO'].astype(int)

                        if meta_file:
                            meta = pd.read_csv(meta_file)
                            if 'LOCATION_CODE' in meta.columns:
                                meta['LOCATION_CODE'] = meta['LOCATION_CODE'].astype(str).str.strip()
                            else:
                                meta = None
                        else:
                            meta = None

                        st.success(f"Label terbaca: {len(labels):,} baris. Positif TO: {int(labels['LABEL_TO'].sum()):,}.")

                        # ---- Feature Builder ---- #
                        st.markdown("### Builder Fitur")
                        window_days = st.slider("Jendela waktu fitur (hari, sebelum tanggal inspeksi)", min_value=7, max_value=90, value=30, step=1)
                        add_indicator_counts = st.checkbox("Tambahkan fitur hitung hari indikator aktif (arus_hilang, over_voltage, dst.)", value=True)

                        # Precompute indicator flags for entire AMR (once)
                        @st.cache_data(show_spinner=False)
                        def precompute_indicators(df_in: pd.DataFrame):
                            ind_df = df_in.copy()
                            ind_list = ind_df.apply(lambda r: cek_indikator_row(r, st.session_state), axis=1)
                            tmp = pd.DataFrame(ind_list.tolist())
                            tmp['LOCATION_CODE'] = ind_df['LOCATION_CODE'].values
                            tmp['TANGGAL'] = ind_df['TANGGAL'].values
                            return tmp
                        ind_flags = precompute_indicators(df) if add_indicator_counts else None

                        # Build features per labeled row
                        @st.cache_data(show_spinner=True)
                        def build_supervised_table(df_amr, labels_df, window_days: int, add_ind_counts: bool):
                            feats = []
                            for _, row in labels_df.iterrows():
                                loc = row['LOCATION_CODE']
                                t_ins = row['TANGGAL_INSPEKSI']
                                t0 = t_ins - pd.Timedelta(days=window_days)
                                sub = df_amr[(df_amr['LOCATION_CODE'] == loc) & (df_amr['TANGGAL'] >= t0) & (df_amr['TANGGAL'] <= t_ins)]
                                if len(sub) == 0:
                                    continue
                                # numeric stats
                                med = sub[FITUR_TEKNIS].median()
                                std = sub[FITUR_TEKNIS].std().fillna(0.0)
                                p95 = sub[FITUR_TEKNIS].quantile(0.95)

                                feat = {f"MED_{c}": med[c] for c in FITUR_TEKNIS}
                                feat.update({f"STD_{c}": std[c] for c in FITUR_TEKNIS})
                                feat.update({f"P95_{c}": p95[c] for c in FITUR_TEKNIS})
                                feat['LOCATION_CODE'] = loc
                                feat['LABEL_TO'] = int(row['LABEL_TO'])
                                feat['TANGGAL_INSPEKSI'] = t_ins

                                if add_ind_counts:
                                    sub_ind = ind_flags[(ind_flags['LOCATION_CODE'] == loc) & (ind_flags['TANGGAL'] >= t0) & (ind_flags['TANGGAL'] <= t_ins)]
                                    if not sub_ind.empty:
                                        for col in ['arus_hilang','over_current','over_voltage','v_drop','cos_phi_kecil','active_power_negative','v_lost','current_loop','freeze']:
                                            if col in sub_ind.columns:
                                                feat[f"COUNT_{col}"] = int(sub_ind[col].sum())
                                feats.append(feat)
                            return pd.DataFrame(feats)

                        sup_df = build_supervised_table(df, labels, window_days, add_indicator_counts)
                        if sup_df.empty:
                            st.warning("Tidak ada fitur yang terbentuk. Pastikan LOCATION_CODE & tanggal inspeksi cocok dengan data AMR.")
                        else:
                            st.success(f"Fitur terbentuk: {len(sup_df):,} baris, {sup_df.shape[1]-3:,} kolom fitur.")
                            if meta is not None:
                                sup_df = sup_df.merge(meta, on='LOCATION_CODE', how='left')

                            # ---- Split waktu ---- #
                            st.markdown("### Split Waktu (Train/Valid)")
                            min_d, max_d = sup_df['TANGGAL_INSPEKSI'].min(), sup_df['TANGGAL_INSPEKSI'].max()
                            q80 = min_d + (max_d - min_d) * 0.8
                            split_date = st.date_input("Pilih tanggal cutoff validasi (>= data ini ke VALID)", value=q80.date())
                            split_ts = pd.Timestamp(split_date)

                            train_df = sup_df[sup_df['TANGGAL_INSPEKSI'] < split_ts].copy()
                            valid_df = sup_df[sup_df['TANGGAL_INSPEKSI'] >= split_ts].copy()
                            if train_df.empty or valid_df.empty:
                                st.error("Split menghasilkan set kosong. Geser tanggal cutoff.")
                            else:
                                st.write(f"Train: {len(train_df):,}  •  Valid: {len(valid_df):,}")

                                # prepare X,y
                                feat_cols = [c for c in sup_df.columns if c not in ['LOCATION_CODE','LABEL_TO','TANGGAL_INSPEKSI'] and not sup_df[c].dtype.name in ('object','category')]
                                X_tr = train_df[feat_cols].fillna(0.0).values
                                y_tr = train_df['LABEL_TO'].values
                                X_va = valid_df[feat_cols].fillna(0.0).values
                                y_va = valid_df['LABEL_TO'].values

                                # Model
                                k_cap = st.number_input("Kapasitas inspeksi (K) untuk evaluasi Precision@K/Recall@K", min_value=10, max_value=10000, value=200, step=10)
                                if st.button("🏋️ Latih Model & Evaluasi"):
                                    clf = RandomForestClassifier(
                                        n_estimators=300,
                                        max_depth=None,
                                        n_jobs=-1,
                                        class_weight="balanced",
                                        random_state=42
                                    )
                                    clf.fit(X_tr, y_tr)
                                    proba = clf.predict_proba(X_va)[:,1]
                                    roc = roc_auc_score(y_va, proba) if len(np.unique(y_va)) > 1 else float('nan')
                                    pr  = average_precision_score(y_va, proba) if len(np.unique(y_va)) > 1 else float('nan')

                                    # Precision@K & Recall@K
                                    order = np.argsort(-proba)
                                    topK_idx = order[:min(int(k_cap), len(order))]
                                    y_topk = y_va[topK_idx]
                                    prec_at_k = y_topk.mean() if len(y_topk)>0 else 0.0
                                    recall_at_k = y_topk.sum() / y_va.sum() if y_va.sum() > 0 else float('nan')

                                    colm1, colm2, colm3 = st.columns(3)
                                    colm1.metric("ROC-AUC", f"{roc:.3f}" if not np.isnan(roc) else "NA")
                                    colm2.metric("PR-AUC", f"{pr:.3f}" if not np.isnan(pr) else "NA")
                                    colm3.metric(f"Precision@K={int(k_cap)}", f"{prec_at_k:.3f}")

                                    if not np.isnan(recall_at_k):
                                        st.metric(f"Recall@K={int(k_cap)}", f"{recall_at_k:.3f}")

                                    # Permutation importance (top 12)
                                    with st.spinner("Menghitung important fitur (permutation)..."):
                                        try:
                                            imp = permutation_importance(clf, X_va, y_va, n_repeats=5, random_state=1, n_jobs=-1)
                                            imp_df = pd.DataFrame({"feature": feat_cols, "importance": imp.importances_mean})
                                            imp_df = imp_df.sort_values("importance", ascending=False).head(12)
                                            fig_imp = px.bar(imp_df, x="feature", y="importance", title="Top Fitur (Permutation Importance)")
                                            st.plotly_chart(fig_imp, use_container_width=True)
                                        except Exception as e:
                                            st.warning(f"Gagal menghitung permutation importance: {e}")

                                    # Tabel prediksi validasi (terurut prob. TO)
                                    valid_pred = valid_df[['LOCATION_CODE']].copy()
                                    valid_pred['proba_TO'] = proba
                                    valid_pred['LABEL_TO'] = y_va
                                    valid_pred = valid_pred.sort_values('proba_TO', ascending=False)
                                    st.subheader("📄 Prediksi pada Set Validasi (diurutkan probabilitas TO)")
                                    st.dataframe(valid_pred.head(500), use_container_width=True, height=400)
                                    st.download_button("📥 Unduh Prediksi Validasi", valid_pred.to_csv(index=False).encode('utf-8'),
                                                       file_name="prediksi_validasi_supervised.csv", mime="text/csv")

                                    # ===== Scoring Unlabeled (operasional) ===== #
                                    st.markdown("---")
                                    st.markdown("### 📌 Scoring Operasional (Data Tanpa Label)")
                                    target_days = st.slider("Jendela waktu scoring (hari terakhir)", min_value=7, max_value=90, value=30, step=1)
                                    topk_oper = st.number_input("Top-K kandidat untuk inspeksi (operasional)", min_value=10, max_value=10000, value=int(k_cap), step=10)

                                    # Build features for current data (last N days up to max date)
                                    max_date = df['TANGGAL'].max()
                                    cutoff = max_date
                                    t0 = cutoff - pd.Timedelta(days=target_days)
                                    df_recent = df[df['TANGGAL'].between(t0, cutoff)].copy()

                                    # Per-IDPEL features
                                    med = df_recent.groupby('LOCATION_CODE')[FITUR_TEKNIS].median().add_prefix('MED_')
                                    std = df_recent.groupby('LOCATION_CODE')[FITUR_TEKNIS].std().fillna(0.0).add_prefix('STD_')
                                    p95 = df_recent.groupby('LOCATION_CODE')[FITUR_TEKNIS].quantile(0.95).add_prefix('P95_')
                                    feat_now = pd.concat([med, std, p95], axis=1).reset_index()

                                    # Indicator counts (optional)
                                    if add_indicator_counts:
                                        ind_recent = ind_flags[ind_flags['TANGGAL'].between(t0, cutoff)].copy()
                                        count_cols = ['arus_hilang','over_current','over_voltage','v_drop','cos_phi_kecil','active_power_negative','v_lost','current_loop','freeze']
                                        for c in count_cols:
                                            if c not in ind_recent.columns:
                                                ind_recent[c] = False
                                        counts = ind_recent.groupby('LOCATION_CODE')[count_cols].sum()
                                        counts.columns = [f"COUNT_{c}" for c in count_cols]
                                        counts = counts.reset_index()
                                        feat_now = feat_now.merge(counts, on='LOCATION_CODE', how='left')

                                    # Merge identity info
                                    info_cols = []
                                    if "NAMA_PELANGGAN" in df.columns: info_cols.append("NAMA_PELANGGAN")
                                    if "TARIF" in df.columns: info_cols.append("TARIF")
                                    elif "TARIFF" in df.columns: info_cols.append("TARIFF")
                                    if "POWER" in df.columns: info_cols.append("POWER")
                                    info = df.sort_values('TANGGAL').groupby('LOCATION_CODE').tail(1)[['LOCATION_CODE']+info_cols].copy()
                                    info = info.rename(columns={"NAMA_PELANGGAN":"NAMA","TARIFF":"TARIF","POWER":"DAYA"})
                                    if "TARIF" not in info.columns: info["TARIF"] = "-"
                                    if "DAYA" not in info.columns: info["DAYA"] = "-"
                                    feat_now = feat_now.merge(info, on='LOCATION_CODE', how='left')

                                    # Align columns to training features
                                    for c in feat_cols:
                                        if c not in feat_now.columns:
                                            feat_now[c] = 0.0
                                    X_now = feat_now[feat_cols].fillna(0.0).values
                                    proba_now = clf.predict_proba(X_now)[:,1]
                                    feat_now['proba_TO'] = proba_now
                                    feat_now = feat_now.sort_values('proba_TO', ascending=False)

                                    topk_df = feat_now.head(int(topk_oper))[['LOCATION_CODE','NAMA','TARIF','DAYA','proba_TO']].copy()
                                    st.subheader("🎯 Rekomendasi Operasional (Top-K)")
                                    st.dataframe(topk_df, use_container_width=True, height=400)
                                    st.download_button("📥 Unduh Rekomendasi Operasional (Top-K)",
                                                       topk_df.to_csv(index=False).encode('utf-8'),
                                                       file_name="rekomendasi_operasional_supervised_topk.csv",
                                                       mime="text/csv")


    # -------- Upload Data -------- #
    with sub_upload:
        uploaded_file = st.file_uploader("📥 Upload File Excel AMR Harian", type=["xlsx"])
        if uploaded_file:
            df_up = pd.read_excel(uploaded_file, sheet_name=0)
            df_up = df_up.dropna(subset=['LOCATION_CODE'])
            df_up['LOCATION_CODE'] = df_up['LOCATION_CODE'].astype(str).str.strip()
            if 'TANGGAL' in df_up.columns:
                df_up['TANGGAL'] = pd.to_datetime(df_up['TANGGAL'], errors='coerce')
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

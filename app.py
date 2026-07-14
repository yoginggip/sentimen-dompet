"""
Analisis Sentimen Ulasan Dompet Digital
GoPay, OVO, Dana — Google Play Store
Algoritma: Naive Bayes & SVM
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import time
import warnings
import os
import requests
import joblib
from io import BytesIO
from collections import Counter

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis Sentimen Dompet Digital",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background: #F8F9FB; }
    .stApp { background: #F8F9FB; }
    .hero-card {
        background: linear-gradient(135deg, #1B2A4A 0%, #2d4a8a 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero-card h1 { color: white; font-size: 1.9rem; margin: 0 0 0.4rem 0; }
    .hero-card p  { color: #cdd8f5; margin: 0; font-size: 0.95rem; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid #1B2A4A;
        margin-bottom: 1rem;
    }
    .metric-card .label { font-size: 0.78rem; color: #888; text-transform: uppercase; letter-spacing: .05em; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #1B2A4A; }
    .step-badge {
        display: inline-block;
        background: #1B2A4A;
        color: white;
        border-radius: 20px;
        padding: 0.15rem 0.8rem;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background: #e8f4fd;
        border-left: 4px solid #378ADD;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.8rem 0;
        font-size: 0.9rem;
    }
    .success-box {
        background: #e8f8f2;
        border-left: 4px solid #1D9E75;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.8rem 0;
        font-size: 0.9rem;
    }
    .model-loaded {
        background: #e8f8f2;
        border: 2px solid #1D9E75;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    div[data-testid="stSidebarContent"] { background: #1B2A4A; color: white; }
    div[data-testid="stSidebarContent"] .stMarkdown { color: white; }
    div[data-testid="stSidebarContent"] h3 { color: #EF9F27; }
    div[data-testid="stSidebarContent"] p  { color: #cdd8f5; }
    div[data-testid="stSidebarContent"] label { color: #cdd8f5 !important; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; color: #1B2A4A; }
    .stTabs [aria-selected="true"] { color: #1D9E75 !important; border-bottom-color: #1D9E75 !important; }
    .stButton>button {
        background: linear-gradient(90deg, #1B2A4A, #2d4a8a);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 1.5rem; font-weight: 600;
    }
    .stDownloadButton>button { background: #1D9E75; color: white; border: none; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# AUTO-LOAD MODEL .pkl
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models_from_pkl():
    """Load model dari file .pkl jika tersedia di folder yang sama."""
    result = {
        'nb_model':    None,
        'svm_model':   None,
        'tfidf':       None,
        'df_balanced': None,
        'loaded':      False,
        'missing':     []
    }
    files = {
        'nb_model':    'nb_model.pkl',
        'svm_model':   'svm_model.pkl',
        'tfidf':       'tfidf.pkl',
        'df_balanced': 'df_balanced.pkl',
    }
    missing = []
    for key, fname in files.items():
        if os.path.exists(fname):
            result[key] = joblib.load(fname)
        else:
            missing.append(fname)

    result['missing'] = missing
    result['loaded']  = len(missing) == 0
    return result


@st.cache_resource
def load_heavy_libs():
    import nltk
    try:
        from nltk.corpus import stopwords
        stopwords.words('indonesian')
    except Exception:
        nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords as sw
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, classification_report, confusion_matrix)
    from wordcloud import WordCloud
    return (sw, TfidfVectorizer, MultinomialNB, LinearSVC,
            train_test_split, StratifiedKFold, cross_val_score, GridSearchCV,
            accuracy_score, precision_score, recall_score, f1_score,
            classification_report, confusion_matrix, WordCloud)


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
def init_state():
    keys = ['df_raw', 'df_preprocessed', 'df_balanced',
            'tfidf', 'X_tfidf', 'y',
            'X_train', 'X_test', 'y_train', 'y_test',
            'nb_model', 'svm_model',
            'y_pred_nb', 'y_pred_svm',
            'hasil_evaluasi', 'cv_results',
            'pos_dict', 'neg_dict',
            'kamus_dict', 'stop_words',
            'pkl_loaded']
    for k in keys:
        if k not in st.session_state:
            st.session_state[k] = None

init_state()

# ── Auto-load pkl ke session state (sekali saja)
if st.session_state.pkl_loaded is None:
    pkl = load_models_from_pkl()
    if pkl['loaded']:
        st.session_state.nb_model    = pkl['nb_model']
        st.session_state.svm_model   = pkl['svm_model']
        st.session_state.tfidf       = pkl['tfidf']
        st.session_state.df_balanced = pkl['df_balanced']
    st.session_state.pkl_loaded = pkl


# ─────────────────────────────────────────────────────────────
# PREPROCESSING HELPERS
# ─────────────────────────────────────────────────────────────
def get_kamus():
    if st.session_state.kamus_dict is not None:
        return st.session_state.kamus_dict
    try:
        url = 'https://github.com/analysisdatasentiment/kamus_kata_baku/raw/main/kamuskatabaku.xlsx'
        resp = requests.get(url, timeout=20)
        kamus_df = pd.read_excel(BytesIO(resp.content))
        kamus_dict = dict(zip(
            kamus_df.iloc[:, 0].astype(str).str.lower(),
            kamus_df.iloc[:, 1].astype(str).str.lower()
        ))
    except Exception:
        kamus_dict = {}
    kamus_tambahan = {
        'apk': 'aplikasi', 'app': 'aplikasi', 'hp': 'handphone',
        'tf': 'transfer', 'wd': 'tarik tunai', 'topup': 'isi saldo',
        'trx': 'transaksi', 'eror': 'error', 'lemot': 'lambat',
        'gg': 'gagal', 'gk': 'tidak', 'ga': 'tidak', 'gak': 'tidak',
        'tdk': 'tidak', 'bgt': 'banget', 'yg': 'yang', 'dgn': 'dengan',
        'utk': 'untuk', 'krn': 'karena', 'udh': 'sudah', 'udah': 'sudah',
        'msh': 'masih', 'blm': 'belum', 'sdh': 'sudah', 'mnt': 'menit',
        'sgt': 'sangat', 'bkn': 'bukan', 'emg': 'memang', 'bngt': 'banget',
        'mantap': 'bagus', 'mantul': 'bagus', 'gacor': 'bagus',
        'parah': 'buruk', 'zonk': 'buruk', 'nyebelin': 'menjengkelkan',
    }
    kamus_dict.update(kamus_tambahan)
    st.session_state.kamus_dict = kamus_dict
    return kamus_dict

def get_stopwords():
    if st.session_state.stop_words is not None:
        return st.session_state.stop_words
    sw, *_ = load_heavy_libs()
    stop_words = set(sw.words('indonesian'))
    stop_words.update({
        'ya','iya','sih','deh','nih','loh','kok','dong',
        'ok','oke','cs','nya','si','bang','kak','mas','mbak',
        'gopay','ovo','dana','aplikasi','app','apk'
    })
    st.session_state.stop_words = stop_words
    return stop_words

def preprocess_text(text, kamus_dict, stop_words):
    if not isinstance(text, str): return ''
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    text = ' '.join([kamus_dict.get(w, w) for w in text.split()])
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💳 Dompet Digital")
    st.markdown("Analisis Sentimen ulasan **GoPay, OVO, Dana** menggunakan **Naive Bayes & SVM**.")
    st.markdown("---")

    # Status model
    pkl = st.session_state.pkl_loaded
    if pkl and pkl['loaded']:
        st.markdown("### ✅ Model Status")
        st.markdown("<p style='color:#1D9E75;font-weight:700;'>Model berhasil di-load otomatis!</p>", unsafe_allow_html=True)
        st.markdown("<p>🟢 Naive Bayes<br>🟢 SVM<br>🟢 TF-IDF<br>🟢 Data Balanced</p>", unsafe_allow_html=True)
    elif pkl and pkl['missing']:
        st.markdown("### ⚠️ Model Status")
        st.markdown(f"<p style='color:#EF9F27;'>File belum ditemukan:<br>{'<br>'.join(pkl['missing'])}</p>", unsafe_allow_html=True)
        st.markdown("<p style='color:#cdd8f5;font-size:0.8rem;'>Upload CSV di Tab Data lalu jalankan pipeline.</p>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Alur Pipeline")
    steps = ["1. Upload / Scraping Data", "2. Preprocessing Teks",
             "3. Labeling Sentimen (InSet)", "4. Undersampling",
             "5. WordCloud & EDA", "6. TF-IDF & Split",
             "7. Pemodelan (NB & SVM)", "8. Evaluasi & Perbandingan"]
    for s in steps:
        st.markdown(f"<p style='color:#cdd8f5;margin:0.2rem 0;font-size:0.85rem;'>▸ {s}</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='font-size:0.8rem;color:#cdd8f5;'>Universitas Gunadarma · Sistem Informasi</p>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
  <h1>💳 Analisis Sentimen Dompet Digital</h1>
  <p>GoPay · OVO · Dana &nbsp;|&nbsp; Google Play Store &nbsp;|&nbsp; Naive Bayes &amp; SVM &nbsp;|&nbsp; Lexicon-Based Labeling (InSet)</p>
</div>
""", unsafe_allow_html=True)

# Banner model loaded
if st.session_state.pkl_loaded and st.session_state.pkl_loaded['loaded']:
    st.markdown("""
    <div class="model-loaded">
        <h3 style="color:#1D9E75;margin:0;">✅ Model Siap Digunakan!</h3>
        <p style="color:#555;margin:0.3rem 0 0 0;">Naive Bayes & SVM berhasil di-load otomatis dari file .pkl — langsung ke Tab <b>🔍 Prediksi Teks</b></p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📥 Data",
    "🧹 Preprocessing & Labeling",
    "📊 EDA & WordCloud",
    "🤖 Pemodelan",
    "🔍 Prediksi Teks",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — DATA
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<span class="step-badge">TAHAP 1</span>', unsafe_allow_html=True)
    st.subheader("Upload Data atau Scraping dari Google Play")

    mode = st.radio("Pilih sumber data:", ["📁 Upload CSV", "🕷️ Scraping Google Play"], horizontal=True)

    if mode == "📁 Upload CSV":
        st.markdown("""
        <div class="info-box">
        📌 Upload file CSV dengan kolom: <code>aplikasi</code>, <code>username</code>, <code>score</code>, <code>ulasan</code>, <code>tanggal</code>
        </div>""", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload file CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            if 'content' in df.columns and 'ulasan' not in df.columns:
                df.rename(columns={'content': 'ulasan'}, inplace=True)
            if 'at' in df.columns and 'tanggal' not in df.columns:
                df.rename(columns={'at': 'tanggal'}, inplace=True)
            if 'userName' in df.columns and 'username' not in df.columns:
                df.rename(columns={'userName': 'username'}, inplace=True)
            st.session_state.df_raw = df
            st.success(f"✅ {len(df):,} baris data berhasil dimuat!")

    else:
        st.markdown("""
        <div class="info-box">
        ⚠️ Scraping memerlukan koneksi internet. Proses ~5–10 menit.
        </div>""", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            target = st.number_input("Target ulasan per aplikasi", 100, 5000, 500, 100)
        with col_b:
            apps_selected = st.multiselect("Pilih aplikasi", ["GoPay", "OVO", "Dana"], default=["GoPay", "OVO", "Dana"])

        APPS_MAP = {'GoPay': 'com.gojek.gopay', 'OVO': 'ovo.id', 'Dana': 'id.dana'}

        if st.button("🕷️ Mulai Scraping"):
            try:
                from google_play_scraper import reviews, Sort
            except ImportError:
                st.error("Install dulu: `pip install google-play-scraper`")
                st.stop()

            progress = st.progress(0)
            status   = st.empty()
            all_dfs  = []
            for i, app_name in enumerate(apps_selected):
                app_id = APPS_MAP[app_name]
                status.info(f"🔍 Scraping {app_name}...")
                semua, token = [], None
                while len(semua) < target:
                    ambil = min(200, target - len(semua))
                    try:
                        hasil, token = reviews(app_id, lang='id', country='id',
                                               sort=Sort.NEWEST, count=ambil,
                                               continuation_token=token)
                        if not hasil: break
                        semua.extend(hasil)
                        if token is None: break
                        time.sleep(0.5)
                    except Exception as e:
                        st.warning(f"⚠️ {e}")
                        break
                df_tmp = pd.DataFrame(semua)[['userName','score','content','at']]
                df_tmp.columns = ['username','score','ulasan','tanggal']
                df_tmp['aplikasi'] = app_name
                all_dfs.append(df_tmp[['aplikasi','username','score','ulasan','tanggal']])
                progress.progress((i + 1) / len(apps_selected))

            if all_dfs:
                st.session_state.df_raw = pd.concat(all_dfs, ignore_index=True)
                status.success(f"✅ Selesai: {len(st.session_state.df_raw):,} ulasan")

    if st.session_state.df_raw is not None:
        df_raw = st.session_state.df_raw
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="metric-card"><div class="label">Total Ulasan</div><div class="value">{len(df_raw):,}</div></div>', unsafe_allow_html=True)
        if 'aplikasi' in df_raw.columns:
            c2.markdown(f'<div class="metric-card"><div class="label">Aplikasi</div><div class="value">{df_raw["aplikasi"].nunique()}</div></div>', unsafe_allow_html=True)
        if 'score' in df_raw.columns:
            c3.markdown(f'<div class="metric-card"><div class="label">Rata-rata Bintang</div><div class="value">{df_raw["score"].mean():.2f}⭐</div></div>', unsafe_allow_html=True)
        st.dataframe(df_raw.head(20), use_container_width=True)
        st.download_button("⬇️ Download data_ulasan_raw.csv",
                           df_raw.to_csv(index=False).encode('utf-8'),
                           "data_ulasan_raw.csv", "text/csv")


# ══════════════════════════════════════════════════════════════
# TAB 2 — PREPROCESSING & LABELING
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<span class="step-badge">TAHAP 2 · 3 · 4</span>', unsafe_allow_html=True)
    st.subheader("Preprocessing → Labeling → Undersampling")

    # Cek apakah df_balanced sudah ada dari pkl
    if st.session_state.df_balanced is not None and st.session_state.pkl_loaded and st.session_state.pkl_loaded['loaded']:
        st.markdown("""
        <div class="success-box">
        ✅ <b>Data balanced sudah ter-load dari file .pkl</b> — tidak perlu preprocessing ulang.<br>
        Langsung lanjut ke Tab <b>📊 EDA</b> atau <b>🤖 Pemodelan</b>.
        </div>""", unsafe_allow_html=True)
        df_bal = st.session_state.df_balanced
        st.subheader("Preview Data Balanced")
        c1, c2 = st.columns(2)
        c1.metric("Total Data", f"{len(df_bal):,}")
        if 'Sentimen' in df_bal.columns:
            dist = df_bal['Sentimen'].value_counts()
            c2.metric("Kelas", f"{dist.to_dict()}")
            st.bar_chart(dist)
        st.dataframe(df_bal.head(10), use_container_width=True)

    elif st.session_state.df_raw is None:
        st.warning("⚠️ Belum ada data. Upload CSV di Tab **📥 Data** terlebih dahulu.")
    else:
        if st.button("🚀 Jalankan Preprocessing + Labeling + Undersampling"):
            sw, *_ = load_heavy_libs()
            kamus_dict = get_kamus()
            stop_words = get_stopwords()

            df = st.session_state.df_raw.copy()
            prog = st.progress(0, text="Menghapus duplikat...")
            df = df.dropna(subset=['ulasan'])
            df = df.drop_duplicates(subset=['ulasan'], keep='first')
            df = df[df['ulasan'].apply(lambda x: len(str(x).split()) >= 3)]

            prog.progress(25, text="Preprocessing teks...")
            df['hasil_preprocessing'] = df['ulasan'].apply(
                lambda x: preprocess_text(x, kamus_dict, stop_words)
            )
            df = df[df['hasil_preprocessing'].str.strip().str.len() > 0].copy()

            prog.progress(50, text="Memuat kamus InSet...")
            try:
                pos_df = pd.read_csv('https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv', sep='\t', header=None)
                neg_df = pd.read_csv('https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv', sep='\t', header=None)
                pos_dict = dict(zip(pos_df[0].str.lower().str.strip(), pd.to_numeric(pos_df[1], errors='coerce').fillna(0)))
                neg_dict = dict(zip(neg_df[0].str.lower().str.strip(), pd.to_numeric(neg_df[1], errors='coerce').fillna(0)))
                st.session_state.pos_dict = pos_dict
                st.session_state.neg_dict = neg_dict
            except Exception as e:
                st.error(f"❌ Gagal memuat InSet: {e}")
                st.stop()

            prog.progress(70, text="Labeling sentimen...")
            def hitung_skor(text):
                if not isinstance(text, str): return 0
                return sum(float(pos_dict.get(w, neg_dict.get(w, 0))) for w in text.split())

            df['skor_leksikon'] = df['hasil_preprocessing'].apply(hitung_skor)
            df['Sentimen'] = df['skor_leksikon'].apply(
                lambda s: 'Positif' if s > 0 else ('Negatif' if s < 0 else None)
            )
            df = df[df['Sentimen'].notna()].copy()

            prog.progress(85, text="Undersampling...")
            min_count = df['Sentimen'].value_counts().min()
            df_balanced = df.groupby('Sentimen', group_keys=False).apply(
                lambda x: x.sample(n=min_count, random_state=42)
            ).reset_index(drop=True).sample(frac=1, random_state=42).reset_index(drop=True)

            st.session_state.df_balanced = df_balanced
            prog.progress(100, text="Selesai!")
            st.success(f"✅ Selesai! {len(df_balanced):,} data balanced ({min_count} per kelas)")

        if st.session_state.df_balanced is not None and not (st.session_state.pkl_loaded and st.session_state.pkl_loaded['loaded']):
            df_bal = st.session_state.df_balanced
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Distribusi Sentimen")
                st.bar_chart(df_bal['Sentimen'].value_counts())
            with col2:
                if 'aplikasi' in df_bal.columns:
                    st.subheader("Per Aplikasi")
                    st.dataframe(df_bal.groupby(['aplikasi','Sentimen']).size().unstack(fill_value=0), use_container_width=True)
            st.download_button("⬇️ Download Hasil_Labelling_Balanced.csv",
                               df_bal.to_csv(index=False).encode('utf-8'),
                               "Hasil_Labelling_Balanced.csv", "text/csv")


# ══════════════════════════════════════════════════════════════
# TAB 3 — EDA & WORDCLOUD
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<span class="step-badge">TAHAP 5</span>', unsafe_allow_html=True)
    st.subheader("Analisis Eksplorasi & WordCloud")

    if st.session_state.df_balanced is None:
        st.warning("⚠️ Data belum tersedia. Jalankan Preprocessing atau pastikan df_balanced.pkl ada.")
    else:
        import matplotlib.pyplot as plt
        *_, WordCloud = load_heavy_libs()

        df = st.session_state.df_balanced
        warna = {'Positif': 'Greens', 'Negatif': 'Reds'}

        # WordCloud
        st.markdown("### ☁️ WordCloud")
        apps_avail = df['aplikasi'].unique().tolist() if 'aplikasi' in df.columns else ['Semua']
        app_wc = st.selectbox("Pilih aplikasi:", apps_avail)

        col1, col2 = st.columns(2)
        for col, sent in zip([col1, col2], ['Positif', 'Negatif']):
            with col:
                subset = df[(df['aplikasi'] == app_wc) & (df['Sentimen'] == sent)] if 'aplikasi' in df.columns else df[df['Sentimen'] == sent]
                teks = ' '.join(subset['hasil_preprocessing'].astype(str).tolist())
                if teks.strip():
                    wc = WordCloud(width=600, height=300, background_color='white',
                                   colormap=warna[sent], max_words=80).generate(teks)
                    fig, ax = plt.subplots(figsize=(7, 3.5))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title(f"{app_wc} — {sent}", fontsize=12, fontweight='bold')
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info(f"Tidak ada data {sent} untuk {app_wc}")

        # Top kata
        st.markdown("---")
        st.markdown("### 📊 Top 20 Kata")
        sent_filter = st.radio("Sentimen:", ['Positif', 'Negatif'], horizontal=True)
        subset2 = df[df['Sentimen'] == sent_filter]
        top_words = Counter(' '.join(subset2['hasil_preprocessing'].astype(str)).split()).most_common(20)
        df_words  = pd.DataFrame(top_words, columns=['Kata', 'Frekuensi'])

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.barh(df_words['Kata'][::-1], df_words['Frekuensi'][::-1],
                 color='#1D9E75' if sent_filter == 'Positif' else '#E24B4A', alpha=0.85)
        ax2.set_xlabel("Frekuensi")
        ax2.set_title(f"Top 20 Kata — {sent_filter}", fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # Distribusi per aplikasi
        if 'aplikasi' in df.columns:
            st.markdown("---")
            st.markdown("### 📊 Distribusi per Aplikasi")
            tbl = df.groupby(['aplikasi','Sentimen']).size().unstack(fill_value=0)
            st.dataframe(tbl, use_container_width=True)
            st.bar_chart(tbl)


# ══════════════════════════════════════════════════════════════
# TAB 4 — PEMODELAN
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<span class="step-badge">TAHAP 6 · 7 · 8 · 9 · 10</span>', unsafe_allow_html=True)
    st.subheader("TF-IDF · Pemodelan · Evaluasi")

    # Jika model sudah ada dari pkl — langsung tampilkan opsi evaluasi ulang
    if st.session_state.pkl_loaded and st.session_state.pkl_loaded['loaded'] and st.session_state.df_balanced is not None:
        st.markdown("""
        <div class="success-box">
        ✅ Model NB & SVM sudah ter-load dari <b>.pkl</b>. Kamu bisa langsung evaluasi atau latih ulang jika mau.
        </div>""", unsafe_allow_html=True)

    if st.session_state.df_balanced is None:
        st.warning("⚠️ Data belum tersedia.")
    else:
        import matplotlib.pyplot as plt
        (sw, TfidfVectorizer, MultinomialNB, LinearSVC,
         train_test_split, StratifiedKFold, cross_val_score, GridSearchCV,
         accuracy_score, precision_score, recall_score, f1_score,
         classification_report, confusion_matrix, WordCloud) = load_heavy_libs()
        from sklearn.metrics import ConfusionMatrixDisplay

        st.markdown("#### ⚙️ Konfigurasi")
        c1, c2, c3 = st.columns(3)
        max_features = c1.selectbox("max_features", [5000, 8000, 10000], index=2)
        ngram = c2.selectbox("ngram_range", ["(1,1)", "(1,2)", "(1,3)"], index=2)
        ngram_val = tuple(int(x) for x in ngram.strip("()").split(","))
        test_size = c3.slider("Test size", 0.1, 0.3, 0.2, 0.05)

        col_nb, col_svm = st.columns(2)
        with col_nb:
            run_nb = st.checkbox("Naive Bayes", value=True)
            alpha_values = st.multiselect("Nilai alpha NB:", [0.05, 0.1, 0.3, 0.5, 1.0], default=[0.05, 0.1, 0.3, 0.5, 1.0])
        with col_svm:
            run_svm = st.checkbox("SVM (LinearSVC)", value=True)
            c_values = st.multiselect("Nilai C SVM:", [0.1, 0.5, 1.0, 2.0, 5.0], default=[0.1, 0.5, 1.0, 2.0, 5.0])

        run_cv = st.checkbox("Cross Validation (5-Fold)", value=True)

        if st.button("🚀 Latih & Evaluasi Model"):
            df   = st.session_state.df_balanced
            X    = df['hasil_preprocessing'].astype(str)
            y    = df['Sentimen']

            with st.spinner("TF-IDF..."):
                tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_val,
                                        sublinear_tf=True, min_df=2, max_df=0.95)
                X_tfidf = tfidf.fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_tfidf, y, test_size=test_size, random_state=42, stratify=y)
                st.session_state.update({'tfidf': tfidf, 'X_tfidf': X_tfidf, 'y': y,
                                         'X_train': X_train, 'X_test': X_test,
                                         'y_train': y_train, 'y_test': y_test})

            hasil_rows = []

            if run_nb and alpha_values:
                with st.spinner("Training Naive Bayes..."):
                    best_alpha, best_f1 = 0.1, 0
                    for alpha in alpha_values:
                        s = cross_val_score(MultinomialNB(alpha=alpha), X_train, y_train, cv=5, scoring='f1_weighted')
                        if s.mean() > best_f1:
                            best_f1, best_alpha = s.mean(), alpha
                    nb_model = MultinomialNB(alpha=best_alpha)
                    nb_model.fit(X_train, y_train)
                    y_pred_nb = nb_model.predict(X_test)
                    st.session_state.nb_model  = nb_model
                    st.session_state.y_pred_nb = y_pred_nb
                    hasil_rows.append({
                        'Model'    : f'Naive Bayes (α={best_alpha})',
                        'Akurasi'  : round(accuracy_score(y_test, y_pred_nb) * 100, 2),
                        'Precision': round(precision_score(y_test, y_pred_nb, average='weighted', zero_division=0) * 100, 2),
                        'Recall'   : round(recall_score(y_test, y_pred_nb, average='weighted', zero_division=0) * 100, 2),
                        'F1-Score' : round(f1_score(y_test, y_pred_nb, average='weighted', zero_division=0) * 100, 2),
                    })

            if run_svm and c_values:
                with st.spinner("Training SVM (GridSearchCV)..."):
                    grid = GridSearchCV(LinearSVC(class_weight='balanced', max_iter=3000, random_state=42),
                                        {'C': c_values}, cv=5, scoring='f1_weighted', n_jobs=-1)
                    grid.fit(X_train, y_train)
                    svm_model  = grid.best_estimator_
                    y_pred_svm = svm_model.predict(X_test)
                    st.session_state.svm_model  = svm_model
                    st.session_state.y_pred_svm = y_pred_svm
                    hasil_rows.append({
                        'Model'    : f'SVM (C={grid.best_params_["C"]})',
                        'Akurasi'  : round(accuracy_score(y_test, y_pred_svm) * 100, 2),
                        'Precision': round(precision_score(y_test, y_pred_svm, average='weighted', zero_division=0) * 100, 2),
                        'Recall'   : round(recall_score(y_test, y_pred_svm, average='weighted', zero_division=0) * 100, 2),
                        'F1-Score' : round(f1_score(y_test, y_pred_svm, average='weighted', zero_division=0) * 100, 2),
                    })

            if hasil_rows:
                st.session_state.hasil_evaluasi = pd.DataFrame(hasil_rows).set_index('Model')

            if run_cv:
                with st.spinner("Cross Validation..."):
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    cv_results = {}
                    if run_nb:
                        cv_results['Naive Bayes'] = cross_val_score(MultinomialNB(), X_tfidf, y, cv=skf, scoring='f1_weighted', n_jobs=-1)
                    if run_svm:
                        cv_results['SVM'] = cross_val_score(LinearSVC(C=1.0, class_weight='balanced', max_iter=2000, random_state=42),
                                                            X_tfidf, y, cv=skf, scoring='f1_weighted', n_jobs=-1)
                    st.session_state.cv_results = cv_results

            st.success("✅ Training selesai!")

        # ── Tampilkan hasil evaluasi
        if st.session_state.hasil_evaluasi is not None:
            hasil_df = st.session_state.hasil_evaluasi
            y_test   = st.session_state.y_test

            st.markdown("---")
            st.markdown("### 🏆 Perbandingan Model")
            cols = st.columns(len(hasil_df))
            for idx, (model_name, row) in enumerate(hasil_df.iterrows()):
                with cols[idx]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="label">{model_name}</div>
                        <div class="value">{row['F1-Score']}%</div>
                        <div style="font-size:0.8rem;color:#555">
                        Akurasi: {row['Akurasi']}%<br>
                        Precision: {row['Precision']}%<br>
                        Recall: {row['Recall']}%
                        </div>
                    </div>""", unsafe_allow_html=True)

            st.dataframe(hasil_df.style.highlight_max(axis=0, color='#d4f0e4'), use_container_width=True)

            # Bar chart
            fig_cmp, ax_cmp = plt.subplots(figsize=(10, 5))
            metrik_cols   = ['Akurasi', 'Precision', 'Recall', 'F1-Score']
            x             = np.arange(len(metrik_cols))
            w             = 0.35
            colors_model  = ['#378ADD', '#EF9F27']
            for i, (model_name, row) in enumerate(hasil_df.iterrows()):
                offset = (i - (len(hasil_df) - 1) / 2) * w
                bars = ax_cmp.bar(x + offset, row[metrik_cols].values, w,
                                   label=model_name, color=colors_model[i % 2], alpha=0.87)
                for bar in bars:
                    h = bar.get_height()
                    ax_cmp.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                                f'{h:.1f}', ha='center', va='bottom', fontsize=8)
            ax_cmp.set_xticks(x)
            ax_cmp.set_xticklabels(metrik_cols)
            ax_cmp.set_ylabel("Nilai (%)")
            ax_cmp.set_ylim(0, 115)
            ax_cmp.legend()
            ax_cmp.set_title("Perbandingan Performa Model (%)", fontweight='bold')
            ax_cmp.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_cmp)
            plt.close()

            # Confusion matrix
            if y_test is not None:
                st.markdown("### 📊 Confusion Matrix")
                preds = []
                if st.session_state.y_pred_nb  is not None: preds.append(('Naive Bayes', st.session_state.y_pred_nb,  'Blues'))
                if st.session_state.y_pred_svm is not None: preds.append(('SVM',         st.session_state.y_pred_svm, 'Oranges'))
                if preds:
                    fig_cm, axes_cm = plt.subplots(1, len(preds), figsize=(6 * len(preds), 5))
                    if len(preds) == 1: axes_cm = [axes_cm]
                    for ax, (nama, pred, cmap) in zip(axes_cm, preds):
                        cm = confusion_matrix(y_test, pred, labels=['Positif','Negatif'])
                        ConfusionMatrixDisplay(cm, display_labels=['Positif','Negatif']).plot(ax=ax, colorbar=False, cmap=cmap)
                        ax.set_title(f'Confusion Matrix\n{nama}', fontsize=12, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig_cm)
                    plt.close()

            # CV Results
            if st.session_state.cv_results:
                st.markdown("### 🔄 Cross Validation (5-Fold)")
                cv_rows = []
                for model_name, scores in st.session_state.cv_results.items():
                    cv_rows.append({'Model': model_name,
                                    **{f'Fold {i+1}': round(s, 4) for i, s in enumerate(scores)},
                                    'Mean': round(scores.mean(), 4), 'Std': round(scores.std(), 4)})
                st.dataframe(pd.DataFrame(cv_rows).set_index('Model'), use_container_width=True)

            st.download_button("⬇️ Download hasil_evaluasi_model.csv",
                               st.session_state.hasil_evaluasi.to_csv().encode('utf-8'),
                               "hasil_evaluasi_model.csv", "text/csv")


# ══════════════════════════════════════════════════════════════
# TAB 5 — PREDIKSI TEKS
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<span class="step-badge">PREDIKSI REAL-TIME</span>', unsafe_allow_html=True)
    st.subheader("Prediksi Sentimen Teks Baru")

    nb_ready    = st.session_state.nb_model  is not None
    svm_ready   = st.session_state.svm_model is not None
    tfidf_ready = st.session_state.tfidf     is not None

    if not tfidf_ready:
        st.warning("⚠️ Model belum tersedia. Pastikan file **.pkl** ada di folder project, atau jalankan Tab **🤖 Pemodelan** terlebih dahulu.")
    else:
        st.markdown("""
        <div class="success-box">
        ✅ Model siap — masukkan teks ulasan di bawah untuk prediksi sentimen secara real-time.
        </div>""", unsafe_allow_html=True)

        contoh = [
            "Aplikasi ini sangat bagus dan mudah digunakan untuk transfer uang",
            "Error terus, saldo tidak masuk padahal sudah transfer berkali-kali",
            "Fitur cashback mantap banget, sering dapat promo menarik",
            "Lemot banget aplikasinya, loading lama dan sering gagal",
            "Pelayanan customer service cepat dan ramah sekali",
        ]

        teks_input = st.text_area("Masukkan ulasan (satu per baris untuk batch):",
                                   placeholder="Contoh: Aplikasi sangat membantu untuk transaksi sehari-hari",
                                   height=120)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📝 Coba Contoh Ulasan"):
                st.session_state['_sample'] = '\n'.join(contoh)
                st.rerun()
        with col_b:
            options = []
            if nb_ready:  options.append("Naive Bayes")
            if svm_ready: options.append("SVM")
            if nb_ready and svm_ready: options.append("Keduanya")
            pilih_model = st.selectbox("Pilih model:", options, index=len(options)-1 if options else 0)

        if '_sample' in st.session_state:
            teks_input = st.session_state.pop('_sample')

        if st.button("🔮 Prediksi Sentimen"):
            teks_to_use = teks_input.strip()
            if not teks_to_use:
                st.warning("Masukkan teks terlebih dahulu.")
            else:
                kamus_dict = get_kamus()
                stop_words = get_stopwords()
                lines       = [l.strip() for l in teks_to_use.split('\n') if l.strip()]
                preprocessed = [preprocess_text(l, kamus_dict, stop_words) for l in lines]
                X_new = st.session_state.tfidf.transform(preprocessed)

                results = []
                for i, (original, prep) in enumerate(zip(lines, preprocessed)):
                    row = {'Teks Asli': original}
                    if nb_ready  and pilih_model in ['Naive Bayes', 'Keduanya']:
                        row['NB Prediksi']  = st.session_state.nb_model.predict(X_new[i])[0]
                    if svm_ready and pilih_model in ['SVM', 'Keduanya']:
                        row['SVM Prediksi'] = st.session_state.svm_model.predict(X_new[i])[0]
                    results.append(row)

                df_result = pd.DataFrame(results)

                st.markdown("---")
                st.subheader("📊 Hasil Prediksi")

                def color_sentiment(val):
                    if val == 'Positif': return 'background-color:#d4f0e4;color:#1D9E75;font-weight:600'
                    if val == 'Negatif': return 'background-color:#fde8e8;color:#E24B4A;font-weight:600'
                    return ''

                styled = df_result.style
                for col in ['NB Prediksi', 'SVM Prediksi']:
                    if col in df_result.columns:
                        styled = styled.applymap(color_sentiment, subset=[col])
                st.dataframe(styled, use_container_width=True)

                pred_cols = [c for c in ['NB Prediksi', 'SVM Prediksi'] if c in df_result.columns]
                if pred_cols:
                    counts = df_result[pred_cols[0]].value_counts()
                    cc1, cc2 = st.columns(2)
                    cc1.metric("😊 Positif", counts.get('Positif', 0))
                    cc2.metric("😞 Negatif", counts.get('Negatif', 0))

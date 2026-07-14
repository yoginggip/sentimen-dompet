import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
import time
import warnings
import os
from io import BytesIO
from collections import Counter

warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis Sentimen Dompet Digital",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1a237e;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .section-header {
        background: linear-gradient(90deg, #1a237e, #283593);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 1rem 0 0.5rem 0;
        font-weight: 600;
    }
    .stProgress .st-bo { background-color: #1D9E75; }
    .success-badge {
        background: #e8f5e9;
        color: #2e7d32;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #1565c0;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ─────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">💳 Analisis Sentimen Dompet Digital</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">GoPay · OVO · Dana | Google Play Store | Naive Bayes & SVM</p>', unsafe_allow_html=True)
st.divider()

# ─── SESSION STATE ───────────────────────────────────────────────────────────
for key in ['df_raw','df_preprocessed','df_labeled','df_balanced',
            'tfidf','X_tfidf','y',
            'nb_model','svm_model',
            'y_test','y_pred_nb','y_pred_svm',
            'hasil_evaluasi','cv_results']:
    if key not in st.session_state:
        st.session_state[key] = None

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/wallet.png", width=80)
    st.title("⚙️ Navigasi")
    tab_choice = st.radio("Pilih Tahap", [
        "🏠 Beranda",
        "1️⃣ Scraping Data",
        "2️⃣ Preprocessing",
        "3️⃣ Labeling Sentimen",
        "4️⃣ Balancing Data",
        "5️⃣ WordCloud",
        "6️⃣ TF-IDF & Split",
        "7️⃣ Pemodelan",
        "8️⃣ Evaluasi & Hasil",
        "🔍 Prediksi Teks Baru"
    ])

    st.divider()
    st.markdown("**Status Data:**")
    status_items = [
        ("Data Raw", st.session_state['df_raw']),
        ("Preprocessing", st.session_state['df_preprocessed']),
        ("Labeling", st.session_state['df_labeled']),
        ("Balanced", st.session_state['df_balanced']),
        ("Model Trained", st.session_state['nb_model']),
    ]
    for label, val in status_items:
        icon = "✅" if val is not None else "⭕"
        st.markdown(f"{icon} {label}")

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_nltk():
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    return set(stopwords.words('indonesian'))

@st.cache_resource(show_spinner=False)
def load_sastrawi():
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    factory = StemmerFactory()
    return factory.create_stemmer()

@st.cache_data(show_spinner=False)
def load_kamus_normalisasi():
    url = 'https://github.com/analysisdatasentiment/kamus_kata_baku/raw/main/kamuskatabaku.xlsx'
    try:
        resp = requests.get(url, timeout=30)
        kamus_df = pd.read_excel(BytesIO(resp.content))
        kamus_dict = dict(zip(
            kamus_df.iloc[:,0].astype(str).str.lower(),
            kamus_df.iloc[:,1].astype(str).str.lower()
        ))
    except:
        kamus_dict = {}
    kamus_tambahan = {
        'apk':'aplikasi','app':'aplikasi','hp':'handphone',
        'tf':'transfer','wd':'tarik tunai','topup':'isi saldo',
        'trx':'transaksi','eror':'error','lemot':'lambat',
        'gg':'gagal','gk':'tidak','ga':'tidak','gak':'tidak',
        'tdk':'tidak','bgt':'banget','yg':'yang','dgn':'dengan',
        'utk':'untuk','krn':'karena','udh':'sudah','udah':'sudah',
        'msh':'masih','blm':'belum','sdh':'sudah','mnt':'menit',
        'sgt':'sangat','bkn':'bukan','emg':'memang','bngt':'banget',
        'mantap':'bagus','mantul':'bagus','gacor':'bagus',
        'parah':'buruk','zonk':'buruk','nyebelin':'menjengkelkan',
    }
    kamus_dict.update(kamus_tambahan)
    return kamus_dict

@st.cache_data(show_spinner=False)
def load_inset():
    pos_url = 'https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv'
    neg_url = 'https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv'
    pos_df = pd.read_csv(pos_url, sep='\t', header=None)
    neg_df = pd.read_csv(neg_url, sep='\t', header=None)
    pos_dict = dict(zip(pos_df[0].str.lower().str.strip(),
                        pd.to_numeric(pos_df[1], errors='coerce').fillna(0)))
    neg_dict = dict(zip(neg_df[0].str.lower().str.strip(),
                        pd.to_numeric(neg_df[1], errors='coerce').fillna(0)))
    return pos_dict, neg_dict

def cleaning(text):
    if not isinstance(text, str): return ''
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalisasi(text, kamus):
    if not isinstance(text, str): return ''
    return ' '.join([kamus.get(w, w) for w in text.split()])

def hapus_stopword(text, stop_words):
    if not isinstance(text, str): return ''
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

def hitung_skor(text, pos_dict, neg_dict):
    if not isinstance(text, str): return 0
    skor = 0
    for w in text.split():
        if w in pos_dict:
            skor += float(pos_dict[w])
        elif w in neg_dict:
            skor += float(neg_dict[w])
    return skor

def labeling_leksikon(skor):
    if skor > 0: return 'Positif'
    elif skor < 0: return 'Negatif'
    return None

# ═══════════════════════════════════════════════════════════════════════════
# HALAMAN: BERANDA
# ═══════════════════════════════════════════════════════════════════════════
if tab_choice == "🏠 Beranda":
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**📱 Aplikasi**\n\nGoPay · OVO · Dana")
    with col2:
        st.info("**🧠 Model ML**\n\nNaive Bayes + SVM")
    with col3:
        st.info("**📊 Metode Label**\n\nLexicon-Based InSet")

    st.markdown('<div class="section-header">📋 Alur Analisis</div>', unsafe_allow_html=True)

    steps = [
        ("1️⃣", "Scraping Data", "Ambil ulasan dari Google Play Store (GoPay, OVO, Dana)"),
        ("2️⃣", "Preprocessing", "Cleaning → Case Folding → Normalisasi → Stopword Removal"),
        ("3️⃣", "Labeling", "Kamus InSet (lexicon-based) → Positif / Negatif"),
        ("4️⃣", "Balancing", "Undersampling untuk menyeimbangkan distribusi kelas"),
        ("5️⃣", "WordCloud", "Visualisasi kata dominan per aplikasi & sentimen"),
        ("6️⃣", "TF-IDF", "Ekstraksi fitur teks (unigram + bigram + trigram)"),
        ("7️⃣", "Pemodelan", "Training Naive Bayes & SVM dengan GridSearchCV"),
        ("8️⃣", "Evaluasi", "Confusion Matrix, Cross Validation, Perbandingan Model"),
    ]
    for icon, title, desc in steps:
        st.markdown(f"{icon} **{title}** — {desc}")

    st.markdown('<div class="section-header">📤 Upload Data (Opsional)</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Jika sudah punya file CSV hasil scraping, upload di sini untuk skip tahap scraping.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV (data_ulasan_raw.csv)", type=['csv'])
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.session_state['df_raw'] = df_raw
        st.success(f"✅ Data berhasil diupload: **{len(df_raw):,} baris**, kolom: {list(df_raw.columns)}")
        st.dataframe(df_raw.head(), use_container_width=True)

    st.divider()
    uploaded_pre = st.file_uploader("Upload CSV Hasil Preprocessing (Hasil_Labelling_Balanced.csv)", type=['csv'], key="upload_balanced")
    if uploaded_pre:
        df_bal = pd.read_csv(uploaded_pre)
        st.session_state['df_balanced'] = df_bal
        st.session_state['df_labeled']  = df_bal
        st.success(f"✅ Data balanced diupload: **{len(df_bal):,} baris**")
        st.dataframe(df_bal.head(), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAHAP 1: SCRAPING
# ═══════════════════════════════════════════════════════════════════════════
elif tab_choice == "1️⃣ Scraping Data":
    st.markdown('<div class="section-header">🔍 Tahap 1 — Scraping Google Play Store</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2,1])
    with col1:
        target = st.slider("Target ulasan per aplikasi", 100, 2000, 500, step=100)
        apps_selected = st.multiselect(
            "Pilih aplikasi", ['GoPay','OVO','Dana'], default=['GoPay','OVO','Dana']
        )
    with col2:
        st.markdown('<div class="info-box">Scraping lebih banyak = waktu lebih lama. Disarankan 500 untuk demo.</div>', unsafe_allow_html=True)

    APPS = {'GoPay':'com.gojek.gopay','OVO':'ovo.id','Dana':'id.dana'}

    if st.button("🚀 Mulai Scraping", type="primary", use_container_width=True):
        try:
            from google_play_scraper import reviews, Sort
        except ImportError:
            st.error("❌ Package google-play-scraper tidak tersedia.")
            st.stop()

        df_list = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, app_name in enumerate(apps_selected):
            app_id = APPS[app_name]
            status_text.markdown(f"🔍 Scraping **{app_name}**...")
            semua_ulasan = []
            token = None

            while len(semua_ulasan) < target:
                ambil = min(200, target - len(semua_ulasan))
                try:
                    hasil, token = reviews(
                        app_id, lang='id', country='id',
                        sort=Sort.NEWEST, count=ambil,
                        continuation_token=token
                    )
                    if not hasil: break
                    semua_ulasan.extend(hasil)
                    if token is None: break
                    time.sleep(0.5)
                except Exception as e:
                    st.warning(f"⚠️ {app_name}: {e}")
                    break

            if semua_ulasan:
                df_app = pd.DataFrame(semua_ulasan)[['userName','score','content','at']]
                df_app.columns = ['username','score','ulasan','tanggal']
                df_app['aplikasi'] = app_name
                df_list.append(df_app[['aplikasi','username','score','ulasan','tanggal']])

            progress_bar.progress((idx+1)/len(apps_selected))

        if df_list:
            df_raw = pd.concat(df_list, ignore_index=True)
            st.session_state['df_raw'] = df_raw
            status_text.success(f"✅ Scraping selesai! Total: **{len(df_raw):,} ulasan**")

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(df_raw.head(20), use_container_width=True)
            with col2:
                st.markdown("**Distribusi per Aplikasi:**")
                dist = df_raw['aplikasi'].value_counts().reset_index()
                dist.columns = ['Aplikasi','Jumlah']
                st.dataframe(dist, use_container_width=True)
                st.bar_chart(df_raw['aplikasi'].value_counts())

            csv = df_raw.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download CSV", csv, "data_ulasan_raw.csv", "text/csv", use_container_width=True)
        else:
            st.error("❌ Tidak ada data yang berhasil di-scrape.")

    if st.session_state['df_raw'] is not None and not st.button("🚀 Mulai Scraping", key="btn2"):
        st.info(f"ℹ️ Data raw sudah tersedia: **{len(st.session_state['df_raw']):,} baris**. Lanjut ke Preprocessing.")

# ═══════════════════════════════════════════════════════════════════════════
# TAHAP 2: PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════
elif tab_choice == "2️⃣ Preprocessing":
    st.markdown('<div class="section-header">⚙️ Tahap 2 — Preprocessing Teks</div>', unsafe_allow_html=True)

    if st.session_state['df_raw'] is None:
        st.warning("⚠️ Silakan scraping data terlebih dahulu (Tahap 1) atau upload CSV di Beranda.")
        st.stop()

    df = st.session_state['df_raw'].copy()
    st.info(f"📊 Data input: **{len(df):,} baris**")

    st.markdown("**Pipeline Preprocessing:**")
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown("🧹 **Cleaning**\nURL, HTML, emoji, tanda baca, angka")
    col2.markdown("🔡 **Case Folding**\nSemua huruf kecil")
    col3.markdown("📖 **Normalisasi**\nKamus kata baku + domain dompet digital")
    col4.markdown("🚫 **Stopword**\nHapus kata tidak bermakna")

    if st.button("▶️ Jalankan Preprocessing", type="primary", use_container_width=True):
        with st.spinner("Memuat kamus normalisasi..."):
            kamus = load_kamus_normalisasi()
            stop_words = load_nltk()
            stop_tambahan = {
                'ya','iya','sih','deh','nih','loh','kok','dong',
                'ok','oke','cs','nya','si','bang','kak','mas','mbak',
                'gopay','ovo','dana','aplikasi','app','apk'
            }
            stop_words.update(stop_tambahan)

        progress = st.progress(0)
        status = st.empty()

        status.text("🧹 Cleaning & case folding...")
        df = df.dropna(subset=['ulasan'])
        df = df.drop_duplicates(subset=['aplikasi','ulasan'], keep='first')
        df = df[df['ulasan'].apply(lambda x: len(str(x).split()) >= 3)]
        df['cleaning']     = df['ulasan'].apply(cleaning)
        df['case_folding'] = df['cleaning'].str.lower()
        progress.progress(33)

        status.text("📖 Normalisasi kata...")
        df['normalisasi'] = df['case_folding'].apply(lambda t: normalisasi(t, kamus))
        progress.progress(66)

        status.text("🚫 Hapus stopword...")
        df['hasil_preprocessing'] = df['normalisasi'].apply(lambda t: hapus_stopword(t, stop_words))
        df = df[df['hasil_preprocessing'].str.strip().str.len() > 0].copy()
        progress.progress(100)

        st.session_state['df_preprocessed'] = df
        status.success(f"✅ Preprocessing selesai! **{len(df):,} data** tersisa.")

        # Preview
        st.dataframe(
            df[['aplikasi','ulasan','case_folding','normalisasi','hasil_preprocessing']].head(10),
            use_container_width=True
        )

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Hasil Preprocessing", csv, "Hasil_Preprocessing_Data.csv", "text/csv", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAHAP 3: LABELING
# ═══════════════════════════════════════════════════════════════════════════
elif tab_choice == "3️⃣ Labeling Sentimen":
    st.markdown('<div class="section-header">🏷️ Tahap 3 — Labeling Sentimen (InSet Lexicon)</div>', unsafe_allow_html=True)

    import matplotlib.pyplot as plt

    if st.session_state['df_preprocessed'] is None:
        st.warning("⚠️ Silakan jalankan preprocessing terlebih dahulu.")
        st.stop()

    df = st.session_state['df_preprocessed'].copy()

    with st.expander("ℹ️ Tentang InSet Lexicon"):
        st.markdown("""
        **InSet** (Indonesian Sentiment Lexicon) dikembangkan oleh Fajri Koto & Gemala Vania (2017).
        
        | Kondisi | Label |
        |---------|-------|
        | Skor leksikon **> 0** | ✅ Positif |
        | Skor leksikon **< 0** | ❌ Negatif |
        | Skor leksikon **= 0** | 🗑️ Dibuang |
        """)

    if st.button("▶️ Jalankan Labeling", type="primary", use_container_width=True):
        with st.spinner("Memuat kamus InSet..."):
            pos_dict, neg_dict = load_inset()
            st.info(f"✅ Kamus positif: **{len(pos_dict):,}** kata | Kamus negatif: **{len(neg_dict):,}** kata")

        with st.spinner("Menghitung skor leksikon..."):
            df['skor_leksikon'] = df['hasil_preprocessing'].apply(
                lambda t: hitung_skor(t, pos_dict, neg_dict)
            )
            df['Sentimen'] = df['skor_leksikon'].apply(labeling_leksikon)

            sebelum = len(df)
            df = df[df['Sentimen'].notna()].copy()
            dibuang = sebelum - len(df)

        st.session_state['df_labeled'] = df
        st.success(f"✅ Labeling selesai! Data dibuang (skor=0): **{dibuang:,}** | Tersisa: **{len(df):,}**")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Distribusi Sentimen:**")
            dist = df['Sentimen'].value_counts()
            st.dataframe(dist.reset_index(), use_container_width=True)

        with col2:
            fig, ax = plt.subplots(figsize=(5,4))
            colors = {'Positif':'#1D9E75','Negatif':'#E24B4A'}
            vals = df['Sentimen'].value_counts()
            ax.bar(vals.index, vals.values, color=[colors.get(s,'gray') for s in vals.index])
            for i, v in enumerate(vals.values):
                ax.text(i, v + 5, f'{v:,}\n({v/len(df)*100:.1f}%)', ha='center', fontsize=9)
            ax.set_title('Distribusi Sentimen', fontweight='bold')
            ax.set_ylabel('Jumlah')
            st.pyplot(fig)
            plt.close()

        st.markdown("**Distribusi per Aplikasi:**")
        pivot = df.groupby(['aplikasi','Sentimen']).size().unstack(fill_value=0)
        st.dataframe(pivot, use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Hasil Labeling", csv, "Hasil_Labelling_Data.csv", "text/csv", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAHAP 4: BALANCING
# ═══════════════════════════════════════════════════════════════════════════
elif tab_choice == "4️⃣ Balancing Data":
    st.markdown('<div class="section-header">⚖️ Tahap 4 — Undersampling (Penyeimbangan Data)</div>', unsafe_allow_html=True)

    import matplotlib.pyplot as plt

    if st.session_state['df_labeled'] is None:
        st.warning("⚠️ Silakan jalankan labeling terlebih dahulu.")
        st.stop()

    df = st.session_state['df_labeled'].copy()
    df = df.dropna(subset=['hasil_preprocessing','Sentimen'])

    st.markdown("**Distribusi Sebelum Undersampling:**")
    dist_before = df['Sentimen'].value_counts()
    st.dataframe(dist_before.reset_index(), use_container_width=True)

    min_count = dist_before.min()
    st.info(f"🎯 Target per kelas: **{min_count:,}** | Total setelah balancing: **{min_count * len(dist_before):,}**")

    if st.button("▶️ Jalankan Undersampling", type="primary", use_container_width=True):
        df_balanced = df.groupby('Sentimen', group_keys=False).apply(
            lambda x: x.sample(n=min_count, random_state=42)
        ).reset_index(drop=True)
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

        st.session_state['df_balanced'] = df_balanced
        st.success(f"✅ Undersampling selesai! Total: **{len(df_balanced):,}** ({min_count:,} per kelas)")

        col1, col2 = st.columns(2)
        with col1:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            colors = ['#1D9E75', '#E24B4A']
            sebelum = df['Sentimen'].value_counts()
            sesudah = df_balanced['Sentimen'].value_counts()

            axes[0].bar(sebelum.index, sebelum.values, color=colors[:len(sebelum)])
            axes[0].set_title('Sebelum', fontweight='bold')
            for i, v in enumerate(sebelum.values):
                axes[0].text(i, v + 5, str(v), ha='center', fontsize=9)

            axes[1].bar(sesudah.index, sesudah.values, color=colors[:len(sesudah)])
            axes[1].set_title('Setelah', fontweight='bold')
            for i, v in enumerate(sesudah.values):
                axes[1].text(i, v + 5, str(v), ha='center', fontsize=9)

            plt.suptitle('Perbandingan Distribusi Data', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.dataframe(df_balanced['Sentimen'].value_counts().reset_index(), use_container_width=True)
            csv = df_balanced.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Balanced CSV", csv, "Hasil_Labelling_Balanced.csv", "text/csv", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAHAP 5: WORDCLOUD
# ═══════════════════════════════════════════════════════════════════════════
elif tab_choice == "5️⃣ WordCloud":
    st.markdown('<div class="section-header">☁️ Tahap 5 — WordCloud per Aplikasi & Sentimen</div>', unsafe_allow_html=True)

    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    if st.session_state['df_balanced'] is None:
        st.warning("⚠️ Silakan jalankan balancing data terlebih dahulu.")
        st.stop()

    df = st.session_state['df_balanced']
    df = df.dropna(subset=['hasil_preprocessing','Sentimen'])

    apps = ['GoPay','OVO','Dana']
    warna = {'Positif':'Greens','Negatif':'Reds'}
    sentimen_list = ['Positif','Negatif']

    filter_app = st.selectbox("Filter Aplikasi", ["Semua"] + apps)
    filter_sent = st.selectbox("Filter Sentimen", ["Semua"] + sentimen_list)

    if st.button("☁️ Generate WordCloud", type="primary", use_container_width=True):
        df_filter = df.copy()
        if filter_app != "Semua":
            df_filter = df_filter[df_filter['aplikasi'] == filter_app]
        if filter_sent != "Semua":
            df_filter = df_filter[df_filter['Sentimen'] == filter_sent]

        apps_to_show = [filter_app] if filter_app != "Semua" else apps
        sent_to_show = [filter_sent] if filter_sent != "Semua" else sentimen_list

        fig, axes = plt.subplots(len(apps_to_show), len(sent_to_show),
                                  figsize=(7*len(sent_to_show), 4*len(apps_to_show)))
        if len(apps_to_show) == 1 and len(sent_to_show) == 1:
            axes = np.array([[axes]])
        elif len(apps_to_show) == 1:
            axes = np.array([axes])
        elif len(sent_to_show) == 1:
            axes = np.array([[ax] for ax in axes])

        for row, app in enumerate(apps_to_show):
            for col, sent in enumerate(sent_to_show):
                subset = df_filter[(df_filter['aplikasi']==app) & (df_filter['Sentimen']==sent)]
                teks = ' '.join(subset['hasil_preprocessing'].astype(str).tolist())
                ax = axes[row][col]
                if len(teks.strip()) < 10:
                    ax.text(0.5, 0.5, 'Tidak ada data', ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
                else:
                    wc = WordCloud(width=600, height=300, background_color='white',
                                   colormap=warna[sent], max_words=80).generate(teks)
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                ax.set_title(f'{app} — {sent}', fontsize=11, fontweight='bold')

        plt.suptitle('WordCloud Ulasan per Aplikasi', fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Top kata
        st.markdown("**Top 15 Kata Terbanyak:**")
        teks_all = ' '.join(df_filter['hasil_preprocessing'].astype(str).tolist())
        counter = Counter(teks_all.split())
        top_words = pd.DataFrame(counter.most_common(15), columns=['Kata','Frekuensi'])
        st.dataframe(top_words, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAHAP 6: TF-IDF & SPLIT
# ═══════════════════════════════════════════════════════════════════════════
elif tab_choice == "6️⃣ TF-IDF & Split":
    st.markdown('<div class="section-header">📐 Tahap 6 — TF-IDF & Split Data</div>', unsafe_allow_html=True)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split

    if st.session_state['df_balanced'] is None:
        st.warning("⚠️ Silakan jalankan balancing data terlebih dahulu.")
        st.stop()

    df = st.session_state['df_balanced'].dropna(subset=['hasil_preprocessing','Sentimen'])

    col1, col2, col3 = st.columns(3)
    max_features = col1.number_input("Max Features TF-IDF", 1000, 20000, 10000, step=1000)
    test_size = col2.slider("Test Size (%)", 10, 40, 20) / 100
    ngram_max = col3.selectbox("N-gram max", [1,2,3], index=2)

    if st.button("▶️ Jalankan TF-IDF & Split", type="primary", use_container_width=True):
        with st.spinner("Menghitung TF-IDF..."):
            X = df['hasil_preprocessing'].astype(str)
            y = df['Sentimen']

            tfidf = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, ngram_max),
                sublinear_tf=True,
                min_df=2, max_df=0.95,
                analyzer='word'
            )
            X_tfidf = tfidf.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_tfidf, y, test_size=test_size, random_state=42, stratify=y
            )

        st.session_state['tfidf']   = tfidf
        st.session_state['X_tfidf'] = X_tfidf
        st.session_state['y']        = y
        st.session_state['X_train']  = X_train
        st.session_state['X_test']   = X_test
        st.session_state['y_train']  = y_train
        st.session_state['y_test']   = y_test

        st.success(f"✅ TF-IDF selesai: **{X_tfidf.shape[0]:,} dokumen × {X_tfidf.shape[1]:,} fitur**")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Data", f"{len(df):,}")
        col2.metric("Data Train", f"{X_train.shape[0]:,}")
        col3.metric("Data Test",  f"{X_test.shape[0]:,}")

        st.markdown("**Distribusi kelas data train:**")
        st.dataframe(y_train.value_counts().reset_index(), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAHAP 7: PEMODELAN
# ═══════════════════════════════════════════════════════════════════════════
elif tab_choice == "7️⃣ Pemodelan":
    st.markdown('<div class="section-header">🤖 Tahap 7 — Training Model ML</div>', unsafe_allow_html=True)

    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import cross_val_score, GridSearchCV
    from sklearn.metrics import accuracy_score, classification_report

    if st.session_state.get('X_train') is None:
        st.warning("⚠️ Silakan jalankan TF-IDF & Split terlebih dahulu.")
        st.stop()

    X_train = st.session_state['X_train']
    X_test  = st.session_state['X_test']
    y_train = st.session_state['y_train']
    y_test  = st.session_state['y_test']

    tab1, tab2 = st.tabs(["🔵 Naive Bayes", "🟠 SVM"])

    with tab1:
        st.markdown("**Naive Bayes — Alpha Tuning (Cross Validation)**")
        alphas = st.multiselect("Nilai Alpha yang diuji", [0.05, 0.1, 0.3, 0.5, 1.0],
                                default=[0.05, 0.1, 0.3, 0.5, 1.0])

        if st.button("▶️ Training Naive Bayes", type="primary", use_container_width=True):
            with st.spinner("Mencari alpha terbaik via Cross Validation..."):
                best_alpha, best_f1 = 0.1, 0
                tuning_results = []
                for alpha in alphas:
                    nb_tmp = MultinomialNB(alpha=alpha)
                    scores = cross_val_score(nb_tmp, X_train, y_train, cv=5, scoring='f1_weighted')
                    tuning_results.append({'Alpha': alpha, 'F1 CV Mean': f"{scores.mean()*100:.2f}%", 'Std': f"±{scores.std()*100:.2f}%"})
                    if scores.mean() > best_f1:
                        best_f1, best_alpha = scores.mean(), alpha

                nb_model = MultinomialNB(alpha=best_alpha)
                nb_model.fit(X_train, y_train)
                y_pred_nb = nb_model.predict(X_test)

            st.session_state['nb_model']   = nb_model
            st.session_state['y_pred_nb']  = y_pred_nb
            st.session_state['y_test']     = y_test

            st.success(f"✅ Alpha terbaik: **{best_alpha}** | F1 CV: **{best_f1*100:.2f}%**")
            st.dataframe(pd.DataFrame(tuning_results), use_container_width=True)

            acc = accuracy_score(y_test, y_pred_nb)
            st.metric("Akurasi Test Set", f"{acc*100:.2f}%")
            st.code(classification_report(y_test, y_pred_nb))

    with tab2:
        st.markdown("**SVM (LinearSVC) — GridSearchCV untuk nilai C**")
        c_values = st.multiselect("Nilai C yang diuji", [0.1, 0.5, 1.0, 2.0, 5.0],
                                  default=[0.1, 0.5, 1.0, 2.0, 5.0])
        cv_fold = st.slider("Jumlah Fold CV", 3, 10, 5)

        if st.button("▶️ Training SVM", type="primary", use_container_width=True):
            with st.spinner("GridSearchCV sedang berjalan (mungkin beberapa menit)..."):
                param_grid = {'C': c_values}
                svm_base   = LinearSVC(class_weight='balanced', max_iter=3000, random_state=42)
                grid_search = GridSearchCV(svm_base, param_grid, cv=cv_fold,
                                           scoring='f1_weighted', n_jobs=-1)
                grid_search.fit(X_train, y_train)

                svm_model  = grid_search.best_estimator_
                y_pred_svm = svm_model.predict(X_test)

            st.session_state['svm_model']  = svm_model
            st.session_state['y_pred_svm'] = y_pred_svm

            st.success(f"✅ C terbaik: **{grid_search.best_params_['C']}** | F1 CV: **{grid_search.best_score_*100:.2f}%**")

            # CV results table
            cv_df = pd.DataFrame(grid_search.cv_results_)[['param_C','mean_test_score','std_test_score']]
            cv_df.columns = ['C','F1 Mean','F1 Std']
            cv_df['F1 Mean'] = cv_df['F1 Mean'].apply(lambda x: f"{x*100:.2f}%")
            st.dataframe(cv_df, use_container_width=True)

            acc = accuracy_score(y_test, y_pred_svm)
            st.metric("Akurasi Test Set", f"{acc*100:.2f}%")
            st.code(classification_report(y_test, y_pred_svm))

# ═══════════════════════════════════════════════════════════════════════════
# TAHAP 8: EVALUASI
# ═══════════════════════════════════════════════════════════════════════════
elif tab_choice == "8️⃣ Evaluasi & Hasil":
    st.markdown('<div class="section-header">📊 Tahap 8 — Evaluasi & Perbandingan Model</div>', unsafe_allow_html=True)

    import matplotlib.pyplot as plt
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, confusion_matrix, ConfusionMatrixDisplay)
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC

    y_test    = st.session_state.get('y_test')
    y_pred_nb  = st.session_state.get('y_pred_nb')
    y_pred_svm = st.session_state.get('y_pred_svm')

    if y_test is None or y_pred_nb is None:
        st.warning("⚠️ Silakan training model terlebih dahulu (Tahap 7).")
        st.stop()

    def hitung_metrik(y_true, y_pred, label):
        return {
            'Model':     label,
            'Akurasi':   round(accuracy_score(y_true, y_pred)*100, 2),
            'Precision': round(precision_score(y_true, y_pred, average='weighted', zero_division=0)*100, 2),
            'Recall':    round(recall_score(y_true, y_pred, average='weighted', zero_division=0)*100, 2),
            'F1-Score':  round(f1_score(y_true, y_pred, average='weighted', zero_division=0)*100, 2),
        }

    rows = [hitung_metrik(y_test, y_pred_nb, 'Naive Bayes')]
    if y_pred_svm is not None:
        rows.append(hitung_metrik(y_test, y_pred_svm, 'SVM'))

    hasil = pd.DataFrame(rows).set_index('Model')
    st.session_state['hasil_evaluasi'] = hasil

    # ── Metrik ringkasan
    st.markdown("**Perbandingan Metrik Model:**")
    st.dataframe(hasil.style.highlight_max(axis=0, color='#d4edda'), use_container_width=True)

    best = hasil['F1-Score'].idxmax()
    st.success(f"🏆 Model terbaik: **{best}** (F1-Score: {hasil.loc[best,'F1-Score']}%)")

    # ── Bar chart perbandingan
    metrik_cols = ['Akurasi','Precision','Recall','F1-Score']
    x  = np.arange(len(metrik_cols))
    w  = 0.35
    colors_model = ['#378ADD','#EF9F27']

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (model_name, color) in enumerate(zip(hasil.index, colors_model)):
        vals = hasil.loc[model_name, metrik_cols].values
        offset = (i - (len(hasil.index)-1)/2) * w
        bars = ax.bar(x + offset, vals, w, label=model_name, color=color, alpha=0.87)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.5, f'{h:.1f}%',
                    ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrik_cols, fontsize=11)
    ax.set_ylabel('Nilai (%)')
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.title('Perbandingan Performa Model', fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Confusion Matrix
    st.markdown("**Confusion Matrix:**")
    label_order = ['Positif','Negatif']
    models_pred = [('Naive Bayes', y_pred_nb, 'Blues')]
    if y_pred_svm is not None:
        models_pred.append(('SVM', y_pred_svm, 'Oranges'))

    fig, axes = plt.subplots(1, len(models_pred), figsize=(6*len(models_pred), 5))
    if len(models_pred) == 1:
        axes = [axes]
    for ax, (nama, pred, cmap) in zip(axes, models_pred):
        cm   = confusion_matrix(y_test, pred, labels=label_order)
        disp = ConfusionMatrixDisplay(cm, display_labels=label_order)
        disp.plot(ax=ax, colorbar=False, cmap=cmap)
        ax.set_title(f'Confusion Matrix\n{nama}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Cross Validation
    st.markdown("**Cross Validation (5-Fold Stratified):**")
    if st.button("▶️ Jalankan Cross Validation", use_container_width=True):
        X_tfidf = st.session_state.get('X_tfidf')
        y       = st.session_state.get('y')

        if X_tfidf is None:
            st.warning("Jalankan TF-IDF terlebih dahulu.")
        else:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            models_cv = {
                'Naive Bayes': MultinomialNB(),
                'SVM': LinearSVC(C=1.0, class_weight='balanced', max_iter=2000, random_state=42),
            }
            cv_rows = []
            for nama, model in models_cv.items():
                with st.spinner(f"CV untuk {nama}..."):
                    scores = cross_val_score(model, X_tfidf, y, cv=skf, scoring='f1_weighted', n_jobs=-1)
                cv_rows.append({
                    'Model': nama,
                    'F1 Fold 1': f"{scores[0]:.4f}",
                    'F1 Fold 2': f"{scores[1]:.4f}",
                    'F1 Fold 3': f"{scores[2]:.4f}",
                    'F1 Fold 4': f"{scores[3]:.4f}",
                    'F1 Fold 5': f"{scores[4]:.4f}",
                    'Rata-rata': f"{scores.mean():.4f}",
                    'Std':       f"±{scores.std():.4f}",
                })
            st.dataframe(pd.DataFrame(cv_rows), use_container_width=True)

    csv = hasil.to_csv().encode('utf-8')
    st.download_button("⬇️ Download Hasil Evaluasi", csv, "hasil_evaluasi_model.csv", "text/csv", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# PREDIKSI TEKS BARU
# ═══════════════════════════════════════════════════════════════════════════
elif tab_choice == "🔍 Prediksi Teks Baru":
    st.markdown('<div class="section-header">🔍 Prediksi Sentimen Teks Baru</div>', unsafe_allow_html=True)

    nb_model  = st.session_state.get('nb_model')
    svm_model = st.session_state.get('svm_model')
    tfidf     = st.session_state.get('tfidf')

    if nb_model is None or tfidf is None:
        st.warning("⚠️ Silakan training model terlebih dahulu (Tahap 7).")
        st.stop()

    stop_words = load_nltk()
    stop_tambahan = {
        'ya','iya','sih','deh','nih','loh','kok','dong',
        'ok','oke','cs','nya','si','bang','kak','mas','mbak',
        'gopay','ovo','dana','aplikasi','app','apk'
    }
    stop_words.update(stop_tambahan)

    with st.spinner("Memuat kamus..."):
        kamus = load_kamus_normalisasi()
        pos_dict, neg_dict = load_inset()

    input_text = st.text_area(
        "Masukkan teks ulasan:",
        placeholder="Contoh: Aplikasi ini sangat lambat dan sering error saat transfer...",
        height=120
    )

    col1, col2 = st.columns(2)
    model_choice = col1.radio("Pilih Model", ["Naive Bayes", "SVM", "Keduanya"], horizontal=True)
    app_label    = col2.selectbox("Aplikasi yang dimaksud", ["GoPay","OVO","Dana","Tidak diketahui"])

    contoh = [
        "Aplikasi ini sangat membantu transaksi sehari-hari, cepat dan mudah digunakan",
        "Sering error, transfer gagal terus, sangat mengecewakan",
        "Fitur cashback mantap banget, recommended buat semua orang",
        "Sudah tiga kali top up tidak masuk, customer service tidak responsif",
    ]
    selected_contoh = st.selectbox("Atau pilih contoh ulasan:", ["-- Pilih --"] + contoh)
    if selected_contoh != "-- Pilih --":
        input_text = selected_contoh

    if st.button("🔮 Prediksi Sentimen", type="primary", use_container_width=True) and input_text.strip():
        # Preprocessing
        text_clean   = cleaning(input_text)
        text_fold    = text_clean.lower()
        text_norm    = normalisasi(text_fold, kamus)
        text_proc    = hapus_stopword(text_norm, stop_words)
        skor_leksikon = hitung_skor(text_proc, pos_dict, neg_dict)

        # Vectorize
        text_vec = tfidf.transform([text_proc])

        st.markdown("---")
        st.markdown("### 📋 Hasil Preprocessing:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Original:** {input_text}")
            st.markdown(f"**After clean:** {text_clean}")
        with col2:
            st.markdown(f"**After normalisasi:** {text_norm}")
            st.markdown(f"**After stopword:** {text_proc}")

        skor_color = "🟢" if skor_leksikon > 0 else ("🔴" if skor_leksikon < 0 else "⚪")
        st.markdown(f"**Skor Leksikon InSet:** {skor_color} {skor_leksikon:.2f}")

        st.markdown("### 🎯 Prediksi Model:")
        results = {}

        if model_choice in ["Naive Bayes","Keduanya"]:
            pred_nb = nb_model.predict(text_vec)[0]
            results['Naive Bayes'] = pred_nb

        if model_choice in ["SVM","Keduanya"] and svm_model is not None:
            pred_svm = svm_model.predict(text_vec)[0]
            results['SVM'] = pred_svm
        elif model_choice == "SVM" and svm_model is None:
            st.warning("Model SVM belum di-training.")

        cols = st.columns(len(results))
        for col, (model_name, pred) in zip(cols, results.items()):
            with col:
                color = "#1D9E75" if pred == "Positif" else "#E24B4A"
                emoji = "😊" if pred == "Positif" else "😞"
                st.markdown(f"""
                <div style="background:{color};color:white;padding:1rem;border-radius:10px;text-align:center;">
                    <h3>{emoji}</h3>
                    <b>{model_name}</b><br>
                    <h2>{pred}</h2>
                </div>
                """, unsafe_allow_html=True)

        # Batch prediction
        st.markdown("---")
        st.markdown("### 📤 Prediksi Batch (Upload CSV)")
        batch_file = st.file_uploader("Upload CSV dengan kolom 'ulasan'", type=['csv'], key="batch_pred")
        if batch_file:
            df_batch = pd.read_csv(batch_file)
            if 'ulasan' not in df_batch.columns:
                st.error("❌ Kolom 'ulasan' tidak ditemukan.")
            else:
                with st.spinner("Memproses batch..."):
                    df_batch['clean']   = df_batch['ulasan'].apply(cleaning)
                    df_batch['fold']    = df_batch['clean'].str.lower()
                    df_batch['norm']    = df_batch['fold'].apply(lambda t: normalisasi(t, kamus))
                    df_batch['proc']    = df_batch['norm'].apply(lambda t: hapus_stopword(t, stop_words))
                    vec_batch           = tfidf.transform(df_batch['proc'].fillna(''))
                    df_batch['Pred_NB'] = nb_model.predict(vec_batch)
                    if svm_model:
                        df_batch['Pred_SVM'] = svm_model.predict(vec_batch)

                st.success(f"✅ Selesai: {len(df_batch):,} ulasan diprediksi")
                st.dataframe(df_batch[['ulasan','Pred_NB'] + (['Pred_SVM'] if svm_model else [])].head(20), use_container_width=True)

                csv = df_batch.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Hasil Prediksi", csv, "hasil_prediksi_batch.csv", "text/csv", use_container_width=True)

# ─── FOOTER ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.8rem;'>"
    "Analisis Sentimen Dompet Digital | Naive Bayes & SVM | InSet Lexicon"
    "</div>",
    unsafe_allow_html=True
)

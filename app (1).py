import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import joblib
import warnings
from collections import Counter

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Sentimen Dompet Digital",
    page_icon="💳",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    files = ['nb_model.pkl', 'svm_model.pkl', 'tfidf.pkl', 'df_balanced.pkl']
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        return None, None, None, None, missing
    nb    = joblib.load('nb_model.pkl')
    svm   = joblib.load('svm_model.pkl')
    tfidf = joblib.load('tfidf.pkl')
    df    = joblib.load('df_balanced.pkl')
    return nb, svm, tfidf, df, []

nb_model, svm_model, tfidf, df_balanced, missing = load_models()

# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_stopwords():
    import nltk
    try:
        from nltk.corpus import stopwords
        stopwords.words('indonesian')
    except Exception:
        nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    stop = set(stopwords.words('indonesian'))
    stop.update({'ya','iya','sih','deh','nih','loh','kok','dong',
                 'ok','oke','cs','nya','si','bang','kak','mas','mbak',
                 'gopay','ovo','dana','aplikasi','app','apk'})
    return stop

KAMUS = {
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

def preprocess(text):
    if not isinstance(text, str): return ''
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    text = ' '.join([KAMUS.get(w, w) for w in text.split()])
    stop = get_stopwords()
    return ' '.join([w for w in text.split() if w not in stop and len(w) > 2])


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
st.title("💳 Analisis Sentimen Dompet Digital")
st.caption("GoPay · OVO · Dana | Naive Bayes & SVM | Lexicon-Based (InSet)")
st.divider()

if missing:
    st.error(f"File model tidak ditemukan: {', '.join(missing)}")
    st.stop()

# ── TABS
tab1, tab2, tab3 = st.tabs(["🔍 Prediksi Teks", "📊 Evaluasi Model", "☁️ WordCloud & EDA"])


# ══════════════════════════════════════════════════════════════
# TAB 1 — PREDIKSI
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Prediksi Sentimen")
    st.write("Masukkan ulasan aplikasi dompet digital untuk diprediksi sentimennya.")

    model_pilihan = st.radio("Pilih model:", ["Naive Bayes", "SVM", "Keduanya"], horizontal=True)

    teks = st.text_area("Teks ulasan:", placeholder="Contoh: Aplikasi sangat mudah digunakan dan transfer cepat", height=120)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Prediksi", type="primary", use_container_width=True):
            if not teks.strip():
                st.warning("Masukkan teks terlebih dahulu.")
            else:
                lines = [l.strip() for l in teks.split('\n') if l.strip()]
                preprocessed = [preprocess(l) for l in lines]
                X = tfidf.transform(preprocessed)

                results = []
                for i, original in enumerate(lines):
                    row = {"Teks": original}
                    if model_pilihan in ["Naive Bayes", "Keduanya"]:
                        row["Naive Bayes"] = nb_model.predict(X[i])[0]
                    if model_pilihan in ["SVM", "Keduanya"]:
                        row["SVM"] = svm_model.predict(X[i])[0]
                    results.append(row)

                df_result = pd.DataFrame(results)
                st.markdown("**Hasil:**")

                def warna(val):
                    if val == 'Positif': return 'color: green; font-weight: bold'
                    if val == 'Negatif': return 'color: red; font-weight: bold'
                    return ''

                styled = df_result.style
                for col in ["Naive Bayes", "SVM"]:
                    if col in df_result.columns:
                        styled = styled.applymap(warna, subset=[col])
                st.dataframe(styled, use_container_width=True, hide_index=True)

                # Ringkasan
                pred_col = "Naive Bayes" if "Naive Bayes" in df_result.columns else "SVM"
                counts = df_result[pred_col].value_counts()
                c1, c2 = st.columns(2)
                c1.metric("😊 Positif", counts.get("Positif", 0))
                c2.metric("😞 Negatif", counts.get("Negatif", 0))

    with col2:
        if st.button("Coba Contoh", use_container_width=True):
            st.session_state['contoh'] = (
                "Aplikasi sangat bagus dan mudah digunakan\n"
                "Error terus saldo tidak masuk padahal sudah transfer\n"
                "Fitur cashback mantap sering dapat promo\n"
                "Lemot banget loading lama dan sering gagal"
            )
            st.rerun()

    if 'contoh' in st.session_state:
        st.info("Contoh ulasan sudah diisi — klik **Prediksi**")
        teks = st.session_state.pop('contoh')


# ══════════════════════════════════════════════════════════════
# TAB 2 — EVALUASI MODEL
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Evaluasi Model")

    if df_balanced is not None and 'Sentimen' in df_balanced.columns:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (accuracy_score, precision_score,
                                     recall_score, f1_score,
                                     confusion_matrix, ConfusionMatrixDisplay,
                                     classification_report)
        import matplotlib.pyplot as plt

        X_all = tfidf.transform(df_balanced['hasil_preprocessing'].astype(str))
        y_all = df_balanced['Sentimen']
        _, X_test, _, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

        y_pred_nb  = nb_model.predict(X_test)
        y_pred_svm = svm_model.predict(X_test)

        # Tabel perbandingan
        hasil = pd.DataFrame([
            {
                'Model'    : 'Naive Bayes',
                'Akurasi'  : round(accuracy_score(y_test, y_pred_nb) * 100, 2),
                'Precision': round(precision_score(y_test, y_pred_nb, average='weighted', zero_division=0) * 100, 2),
                'Recall'   : round(recall_score(y_test, y_pred_nb, average='weighted', zero_division=0) * 100, 2),
                'F1-Score' : round(f1_score(y_test, y_pred_nb, average='weighted', zero_division=0) * 100, 2),
            },
            {
                'Model'    : 'SVM',
                'Akurasi'  : round(accuracy_score(y_test, y_pred_svm) * 100, 2),
                'Precision': round(precision_score(y_test, y_pred_svm, average='weighted', zero_division=0) * 100, 2),
                'Recall'   : round(recall_score(y_test, y_pred_svm, average='weighted', zero_division=0) * 100, 2),
                'F1-Score' : round(f1_score(y_test, y_pred_svm, average='weighted', zero_division=0) * 100, 2),
            },
        ]).set_index('Model')

        st.dataframe(hasil.style.highlight_max(axis=0, color='#d4f0e4'), use_container_width=True)

        best = hasil['F1-Score'].idxmax()
        st.success(f"✅ Model terbaik: **{best}** (F1-Score: {hasil.loc[best, 'F1-Score']}%)")

        # Confusion matrix
        st.markdown("**Confusion Matrix:**")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, (nama, pred, cmap) in zip(axes, [
            ('Naive Bayes', y_pred_nb,  'Blues'),
            ('SVM',         y_pred_svm, 'Oranges'),
        ]):
            cm = confusion_matrix(y_test, pred, labels=['Positif', 'Negatif'])
            ConfusionMatrixDisplay(cm, display_labels=['Positif', 'Negatif']).plot(ax=ax, colorbar=False, cmap=cmap)
            ax.set_title(nama, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Classification report
        with st.expander("Lihat Classification Report"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Naive Bayes**")
                st.dataframe(pd.DataFrame(classification_report(y_test, y_pred_nb, output_dict=True)).T.round(3))
            with col2:
                st.write("**SVM**")
                st.dataframe(pd.DataFrame(classification_report(y_test, y_pred_svm, output_dict=True)).T.round(3))

        st.download_button("⬇️ Download hasil evaluasi",
                           hasil.to_csv().encode('utf-8'),
                           "hasil_evaluasi.csv", "text/csv")
    else:
        st.info("Data balanced tidak tersedia untuk evaluasi.")


# ══════════════════════════════════════════════════════════════
# TAB 3 — WORDCLOUD & EDA
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("WordCloud & Distribusi Data")

    if df_balanced is None or 'Sentimen' not in df_balanced.columns:
        st.info("Data tidak tersedia.")
    else:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        # Distribusi sentimen
        st.markdown("**Distribusi Sentimen:**")
        dist = df_balanced['Sentimen'].value_counts()
        st.bar_chart(dist)

        # WordCloud
        st.markdown("**WordCloud:**")
        sent = st.radio("Pilih sentimen:", ['Positif', 'Negatif'], horizontal=True)
        subset = df_balanced[df_balanced['Sentimen'] == sent]
        teks_wc = ' '.join(subset['hasil_preprocessing'].astype(str))

        if teks_wc.strip():
            wc = WordCloud(width=700, height=350, background_color='white',
                           colormap='Greens' if sent == 'Positif' else 'Reds',
                           max_words=100).generate(teks_wc)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            plt.close()

        # Top kata
        top = Counter(teks_wc.split()).most_common(10)
        st.markdown(f"**Top 10 Kata — {sent}:**")
        df_top = pd.DataFrame(top, columns=['Kata', 'Frekuensi'])
        st.dataframe(df_top, use_container_width=True, hide_index=True)

        # Per aplikasi
        if 'aplikasi' in df_balanced.columns:
            st.markdown("**Distribusi per Aplikasi:**")
            tbl = df_balanced.groupby(['aplikasi', 'Sentimen']).size().unstack(fill_value=0)
            st.dataframe(tbl, use_container_width=True)

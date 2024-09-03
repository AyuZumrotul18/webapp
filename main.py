import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def display_csv(file_path, title, columns_to_display=None):
    """
    Menampilkan dataset dari file CSV dengan pilihan kolom tertentu.
    """
    try:
        data = pd.read_csv(file_path, delimiter=',', encoding='utf-8', on_bad_lines='skip')
        if columns_to_display:
            data = data[columns_to_display]
        st.subheader(title)
        st.dataframe(data)
    except pd.errors.ParserError as e:
        st.error(f"Error saat membaca file CSV: {e}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

def plot_confusion_matrix(file_path):
    try:
        data = pd.read_csv(file_path, delimiter=',', encoding='utf-8', on_bad_lines='skip')
        if 'Actual Negative' not in data.columns or 'Predicted Negative' not in data.columns:
            st.error("Kolom yang diperlukan tidak ditemukan dalam file CSV.")
            return
        cm = data.values
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted Negative', 'Predicted Positive'], 
                    yticklabels=['Actual Negative', 'Actual Positive'])
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('Actual Label')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
    except pd.errors.ParserError as e:
        st.error(f"Error saat membaca file CSV: {e}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

def plot_metrics(file_path):
    """
    Membuat grafik bar dari metrik K-Fold Cross Validation.
    """
    try:
        metrics_df = pd.read_csv(file_path, delimiter=',', encoding='utf-8', on_bad_lines='skip')
        metrics_df['Score'] = metrics_df['Score'] * 100

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Metric', y='Score', data=metrics_df, palette='viridis')

        for index, value in enumerate(metrics_df['Score']):
            plt.text(index, value + 0.3, f'{value:.2f}%', ha='center', fontsize=8)

        plt.xticks(rotation=45, ha='right')
        plt.title('Rata-rata Metrik Evaluasi dari K-Fold Cross-Validation (Weighted Avg) dalam Persen')
        plt.xlabel('Metrik')
        plt.ylabel('Nilai Rata-rata (%)')
        plt.ylim(70, 80)  # Sesuaikan skala hingga 80
        plt.yticks(range(70, 81, 1))  # Menggunakan kelipatan 1 untuk y-ticks
        plt.tight_layout()
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memplot grafik: {e}")

def main():
    st.set_page_config(page_title="Analisis Sentimen", layout="wide")

    st.title("Analisis Sentimen Pengguna Sosial Media X Terhadap Sengketa Pemilihan Presiden Indonesia 2024 Menggunakan Random Forest")

    st.sidebar.header("Menu")
    page = st.sidebar.radio("Pilih Halaman", ["Random Forest", "Metrics", "K-Fold Cross Validation"])

    if page == "Random Forest":
        st.subheader("Random Forest")
        st.write("""
        **Random Forest** adalah teknik pembelajaran mesin yang digunakan untuk klasifikasi dan regresi. 
        Teknik ini terdiri dari sejumlah besar **pohon keputusan** (decision trees) yang digabungkan untuk menghasilkan model yang lebih kuat dan akurat. 
        Setiap pohon keputusan dalam hutan (forest) memberikan suara untuk kelas (dalam hal klasifikasi) atau nilai (dalam hal regresi), dan 
        kelas atau nilai dengan suara terbanyak atau rata-rata dihitung untuk memberikan hasil akhir.
        """)

        st.write("### Raw Data Dan Clean Data")
        columns_to_display_crawling = ['full_text', 'cleasing','case_folding','tokenize','Filtering/stopwords removal','stemming_data']
        display_csv('Hasil_Preprocessing_Data.csv', "Data Sentimen", columns_to_display_crawling)
        
        st.write("### Dataset Hasil Labeling")
        columns_to_display_labeling = ['stemming_data', 'sentiment']
        display_csv('HasilLabeling.csv', "Data Sentimen (Hasil Labeling)", columns_to_display_labeling)

    elif page == "Metrics":
        st.subheader("Confusion Matrix")
        st.image('confusion_matrix.png', caption='Confusion Matrix')

        st.write("Classification Report")
        classification_report_df = pd.read_csv('classification_report.txt', delimiter=',', encoding='utf-8', on_bad_lines='skip')
        st.dataframe(classification_report_df)

    elif page == "K-Fold Cross Validation":
        st.subheader("Hasil K-Fold Cross Validation")
        average_metrics_weighted_df = pd.read_csv('average_scores.csv', delimiter=',', encoding='utf-8', on_bad_lines='skip')
        st.dataframe(average_metrics_weighted_df)

        st.write("### Grafik Rata-Rata Metrik Evaluasi")
        plot_metrics('average_scores.csv')

if __name__ == "__main__":
    main()

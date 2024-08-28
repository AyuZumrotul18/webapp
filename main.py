import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_pie_chart(data):
    sentiment_counts = data['sentiment'].value_counts()
    labels = sentiment_counts.index
    sizes = sentiment_counts.values
    colors = ['#98fb98', '#dfff4f']
    
    fig, ax = plt.subplots(figsize=(9, 5), dpi=100)
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=10, colors=colors)
    ax.axis('equal')
    plt.tight_layout()
    st.pyplot(fig)

def plot_barplot(data):
    sentiment_counts = data['sentiment'].value_counts()
    colors = ['#98fb98', '#dfff4f']
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax, palette=colors)
    for i in ax.containers:
        ax.bar_label(i,)
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    ax.set_title('Distribusi Sentimen')
    plt.tight_layout()
    st.pyplot(fig)

def display_csv(file_path, title, as_percentage=False):
    data = pd.read_csv(file_path)
    if as_percentage and 'Value' in data.columns:
        data['Value (%)'] = data['Value'] * 100
        data = data.drop('Value', axis=1)
    st.subheader(title)
    st.dataframe(data)

def plot_confusion_matrix(file_path):
    data = pd.read_csv(file_path)
    
    if 'Actual Negative' not in data.columns or 'Predicted Negative' not in data.columns:
        st.error("Kolom yang diperlukan tidak ditemukan dalam file CSV.")
        return

    cm = data.values

    fig, ax = plt.subplots(figsize=(10, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('Actual Label')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

def main():
    st.set_page_config(page_title="Analisis Sentimen", layout="wide")

    st.title("Analisis Sentimen Pengguna Sosial Media X Terhadap Sengketa Pemilihan Presiden Indonesia 2024 Menggunakan Random Forest")

    st.sidebar.header("Menu")
    page = st.sidebar.radio("Pilih Halaman", ["Dashboard", "Data", "Metrics"])

    if page == "Dashboard":
        sentiment_file_path = 'Hasil_Labeling_Data_nltk.csv'
        sentiment_data = pd.read_csv(sentiment_file_path)

        st.subheader("Diagram Lingkaran Sentimen")
        plot_pie_chart(sentiment_data)
        
        st.subheader("Barplot Distribusi Sentimen")
        plot_barplot(sentiment_data)
        
        total_count = sentiment_data.shape[0]
        st.write(f"Jumlah total data: {total_count}")

    elif page == "Data":
        sentiment_file_path = 'Hasil_Labeling_Data_nltk.csv'
        sentiment_data = pd.read_csv(sentiment_file_path)
        
        st.subheader("Hasil Preprocessing Data dan Labelling Data")
        st.dataframe(sentiment_data[['stemming_data', 'sentiment']].head(10))
        
    elif page == "Metrics":
        st.subheader("Confusion Matrix")
        st.image('confusion_matrix_plot.png', caption='Confusion Matrix')

        st.write("Classification Report")
        classification_report_df = pd.read_csv('classification_report_rf.csv')
        st.dataframe(classification_report_df)

        st.write("Rata-Rata K-Fold Cross Validation")
        average_metrics_weighted_df = pd.read_csv('average_metrics_weighted_fold.csv')
        st.dataframe(average_metrics_weighted_df)

if __name__ == "__main__":
    main()

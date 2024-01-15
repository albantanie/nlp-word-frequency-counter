from flask import Flask, render_template, request
from bs4 import BeautifulSoup
import requests
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import seaborn as sns
import matplotlib.pyplot as plt
import os

nltk.download('punkt')
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        url = request.form['url']
        news_text = get_news_text(url)

        # Preprocessing
        words = preprocess(news_text)

        # Menghitung frekuensi kata dan mengurutkannya
        freq_dist = calculate_word_frequency(words)

        # Menampilkan chart menggunakan Seaborn
        plot_charts(freq_dist)

        return render_template('result.html', freq_dist=freq_dist, news_text=news_text)


def get_news_text(url):
    # Mengambil teks berita dari URL menggunakan BeautifulSoup
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    news_text = ' '.join([paragraph.get_text() for paragraph in paragraphs])
    return news_text


def preprocess(text):
    # Tokenisasi dan stemming menggunakan Sastrawi dan NLTK
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    words = word_tokenize(text)
    words = [stemmer.stem(word) for word in words if word.isalnum()]
    return words


def calculate_word_frequency(words):
    # Menghitung frekuensi kata menggunakan NLTK FreqDist
    freq_dist = FreqDist(words)

    # Mengurutkan frekuensi kata dari yang terbesar
    sorted_freq_dist = {k: v for k, v in sorted(
        freq_dist.items(), key=lambda item: item[1], reverse=True)}

    return sorted_freq_dist


def plot_charts(freq_dist):
    # Mengambil 30 kata dengan frekuensi terbanyak
    top_words = list(freq_dist.items())[:30]
    words, frequencies = zip(*top_words)

    # Line chart
    plt.figure(figsize=(12, 6))
    # Menggunakan kata sebagai sumbu x
    sns.lineplot(x=words, y=frequencies, marker='o')
    plt.title('Frekuensi Kata Terbanyak')
    plt.xlabel('Kata')
    plt.ylabel('Frekuensi')
    line_chart_path = 'static/line_chart.png'
    plt.xticks(rotation=45, ha='right')  # Rotasi label kata agar lebih terbaca
    plt.tight_layout()
    plt.savefig(line_chart_path)  # Simpan gambar chart

    # Bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(x=words, y=frequencies)
    plt.title('Frekuensi Kata Terbanyak')
    plt.xlabel('Kata')
    plt.ylabel('Frekuensi')
    plt.xticks(rotation=45, ha='right')  # Rotasi label kata agar lebih terbaca
    plt.tight_layout()
    bar_chart_path = 'static/bar_chart.png'
    plt.savefig(bar_chart_path)  # Simpan gambar chart


if __name__ == '__main__':
    app.run(debug=True)

# 🎤 TED Talks Recommendation System

A Streamlit-based machine learning app that recommends TED Talks based on content similarity using transcript data. It also offers powerful data visualizations from TED talk metadata.

## 📂 Project Structure

```
TEDTalksPro/
├── data/
│   ├── ted_main.csv          # TED Talks metadata
│   └── transcripts.csv       # Full transcripts of TED Talks
├── app.py                    # Streamlit frontend interface
├── main.py                   # Backend logic (used in development)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Features

* 🔮 **TED Talk Recommendations**
  Get top similar talks using transcript-based text similarity.

* 📊 **Data Visualizations**
  Explore TED Talks by views, comments, duration, languages, and speakers.

* 🧠 **Content-Based Filtering**
  Uses TF-IDF vectorization + Cosine Similarity on talk transcripts.

* 💡 **Interactive UI**
  Built with Streamlit for simplicity and ease of use.

---

## 🧠 How It Works

### 🔍 Machine Learning Logic

* **TF-IDF**: Converts transcript text into numerical vectors by importance of words.
* **Cosine Similarity**: Finds the most textually similar talks to a selected one.

### 📈 Visualizations

Plots include:

* Top 10 talks by Views, Comments, Duration, Languages
* Top 10 Frequent Speakers

---

## 📦 Installation & Usage

### ✅ Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/TEDTalksRecommender.git
cd TEDTalksRecommender
```

### ✅ Step 2: Install Requirements

```bash
pip install -r requirements.txt
```

### ✅ Step 3: Run the App

```bash
streamlit run app.py
```

> If `streamlit` is not recognized, try:

```bash
python -m streamlit run app.py
```

---

## 📝 Dataset Information

* `ted_main.csv` — Metadata like title, views, comments, speaker, duration.
* `transcripts.csv` — Full transcripts of TED Talks.
* 📦 Source: [Kaggle - TED Talks Dataset](https://www.kaggle.com/datasets/rounakbanik/ted-talks)


---

## 👨‍💻 Technologies Used

* Python 3.x
* Streamlit
* Scikit-learn
* Pandas
* Seaborn & Matplotlib

---

## 💡 Future Improvements

* Add BERT embeddings for semantic similarity
* Add filters (by duration, topic, year)
* Save user history and feedback
* Deploy on Streamlit Cloud or HuggingFace Spaces

---

## 📄 License

MIT License © 2025 Srivatsa
Feel free to use, modify, and share.

---

## 🙌 Acknowledgements

* [Kaggle: TED Talks Dataset by Rounak Banik](https://www.kaggle.com/datasets/rounakbanik/ted-talks)
* Streamlit and Scikit-learn teams

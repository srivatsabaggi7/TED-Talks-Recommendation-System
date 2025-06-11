import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
df = pd.read_csv("data/ted_main.csv")
transcripts = pd.read_csv("data/transcripts.csv")

# Preprocess the transcript URLs
def url_text_extraction(text):
    text = text.split("/")[-1]
    text = text.replace("_", " ")
    text = re.sub(r"\n", "", text)
    text = text.upper()
    return text

transcripts["url"] = transcripts["url"].apply(url_text_extraction)
transcript_texts = transcripts["transcript"].tolist()

# Vectorize and calculate similarity
tfidf = TfidfVectorizer(stop_words="english")
transcript_vectors = tfidf.fit_transform(transcript_texts)
similarities = cosine_similarity(transcript_vectors)

# Recommendation function
def recommend_talks(url, n=5):
    try:
        url_index = transcripts[transcripts["url"] == url.upper()].index[0]
    except IndexError:
        return []
    sim_scores = similarities[url_index]
    top_indices = sim_scores.argsort()[::-1][1:n+1]
    return [transcripts["url"].iloc[i] for i in top_indices]

# Streamlit UI
st.set_page_config(page_title="TED Talk Recommender", layout="wide")
st.title("🎤 TED Talks Recommendation System")

# Tabs for different sections
tab1, tab2 = st.tabs(["🔮 Recommendations", "📊 Data Exploration"])

# ---------- TAB 1: Recommendations ----------
with tab1:
    st.subheader("Get Recommendations")
    talk_titles = transcripts["url"].tolist()
    selected_title = st.selectbox("Choose a TED Talk:", sorted(talk_titles))

    if st.button("Recommend"):
        recommendations = recommend_talks(selected_title, 5)
        if recommendations:
            st.success("Top 5 Recommended Talks:")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec.title()}")
        else:
            st.error("No recommendations found.")

# ---------- TAB 2: Visualizations ----------
with tab2:
    st.subheader("Explore TED Talk Data")

    st.markdown("Select a plot to visualize from the dropdown below:")
    plot_choice = st.selectbox("Choose a plot", [
        "Top 10 by Comments",
        "Top 10 by Views",
        "Top 10 by Duration",
        "Top 10 by Languages",
        "Top 10 by Number of Speakers",
        "Top 10 Frequent Speakers"
    ])

    fig, ax = plt.subplots(figsize=(10, 6))

    if plot_choice == "Top 10 by Comments":
        sns.barplot(data=df, x="comments", y="title", order=df.sort_values("comments", ascending=False).title.head(10), ax=ax)
        plt.xticks(rotation=90)
        st.pyplot(fig)

    elif plot_choice == "Top 10 by Views":
        sns.barplot(data=df, x="views", y="title", order=df.sort_values("views", ascending=False).title.head(10), ax=ax)
        plt.xticks(rotation=90)
        st.pyplot(fig)

    elif plot_choice == "Top 10 by Duration":
        sns.barplot(data=df, x="duration", y="title", order=df.sort_values("duration", ascending=False).title.head(10), ax=ax)
        plt.xticks(rotation=90)
        st.pyplot(fig)

    elif plot_choice == "Top 10 by Languages":
        sns.barplot(data=df, x="languages", y="title", order=df.sort_values("languages", ascending=False).title.head(10), ax=ax)
        plt.xticks(rotation=90)
        st.pyplot(fig)

    elif plot_choice == "Top 10 by Number of Speakers":
        sns.barplot(data=df, x="num_speaker", y="title", order=df.sort_values("num_speaker", ascending=False).title.head(10), ax=ax)
        plt.xticks(rotation=90)
        st.pyplot(fig)

    elif plot_choice == "Top 10 Frequent Speakers":
        sns.countplot(data=df, y="main_speaker", order=df["main_speaker"].value_counts().head(10).index, ax=ax)
        st.pyplot(fig)

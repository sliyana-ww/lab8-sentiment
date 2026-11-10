import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS

# Page configuration
st.set_page_config(page_title="Lab 8 | Feedback Assignment", layout="wide")
st.title("Product Feedback: Enhanced Text Summary & Sentiment Audit")
st.markdown("**Student Name:** Liyana Siah | **Student ID:** 22006725")
st.caption("Practice data with topic categorization, negation testing, and qualitative review.")

# 1. Updated reviews including the negation test ("The bottle is not easy to clean.")
reviews = [
    "The bottle keeps my water cold all day.",
    "Great size and easy to carry to class.",
    "I love the simple design and sturdy lid.",
    "It leaks in my bag. Very disappointing.",
    "The cap is hard to open and the bottle feels heavy.",
    "The bottle is not easy to clean.",  # Negation test comment
    "Keeps drinks cold, though the paint scratches easily.",
    "The bottle is fine. Nothing special either way.",
    "I expected better insulation for this price.",
    "Lightweight, useful, and easy to wash.",
    "Not bad, but I would prefer a larger capacity.",
    "Excellent bottle. I use it every day.",
    "The lid broke after one week. Poor quality.",
    "It works as described and arrived on time.",
    "Nice color, but it does not keep drinks cold.",
    "Affordable and practical for carrying water.",
    "The bottle is okay; the handle is uncomfortable.",
    "Very happy with the purchase and the quality.",
]

# Main topics assigned for each comment (Task 4 requirement)
topics = [
    "Insulation",
    "Portability",
    "Design/Lid",
    "Durability",
    "Usability",
    "Cleanability", # Testing negation here
    "Insulation",
    "General",
    "Insulation",
    "Usability",
    "Capacity",
    "General",
    "Durability",
    "General",
    "Insulation",
    "Value",
    "Design/Lid",
    "Quality"
]

df = pd.DataFrame({
    "id": [f"R{i:02d}" for i in range(1, 19)],
    "text": reviews,
    "topic": topics
})

# Sentiment analysis using VADER
analyzer = SentimentIntensityAnalyzer()

def label_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive", score
    elif score <= -0.05:
        return "Negative", score
    return "Neutral", score

df[["sentiment", "compound"]] = df["text"].apply(
    lambda text: pd.Series(label_sentiment(text))
)

# Layout: Two columns for charts
left, right = st.columns(2)

with left:
    st.subheader("Comment counts by sentiment")
    order = ["Positive", "Neutral", "Negative"]
    counts = df["sentiment"].value_counts().reindex(order, fill_value=0)
    st.bar_chart(counts, horizontal=True)

with right:
    st.subheader("Words used in the comments")
    # Task 2: Removed "bottle" from stop words to observe its frequency in the word cloud,
    # while retaining filler stopwords and "drinks"/"drink".
    stop_words = set(STOPWORDS) | {"drinks", "drink"}
    cloud = WordCloud(width=900, height=500, background_color="white",
                      colormap="viridis", stopwords=stop_words,
                      collocations=False).generate(" ".join(df["text"]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)
    st.caption("Note: Word size reflects term frequency. It does not indicate sentiment or importance; common filler words are filtered out.")

# Sentiment by Topic Comparison
st.subheader("Sentiment Breakdown by Topic")
topic_sentiment = pd.crosstab(df["topic"], df["sentiment"])
st.dataframe(topic_sentiment, use_container_width=True)

# Review original comments with topics and audited sentiment
st.subheader("Review the original comments & audited labels")
st.dataframe(df[["id", "topic", "text", "sentiment", "compound"]],
             use_container_width=True, hide_index=True)
st.caption("A compound score is an automated model output, not a definitive confidence rating.")

# Task 4: Product Team Summary (150-200 words)
st.subheader("Product Team Summary & Recommendations")
summary_text = """
**Executive Summary for Product Team**

**Finding:** Insulation and portability are the primary drivers of positive customer sentiment, with users frequently praising temperature retention and lightweight design. Conversely, durability issues regarding lids and scratch-resistant coatings generate negative sentiment. 

**Limitation:** The current dataset relies on a small sample size of 18 synthetic comments and automated VADER sentiment scoring. Automated tools struggle with nuance, sarcasm, and negation (e.g., handling "not easy to clean"), leading to occasional misclassification where human context is required. Furthermore, cross-topic comparisons are limited due to sparse data distribution across categories.

**Recommendation:** To gather more actionable feedback, the product team should implement structured rating fields alongside open text boxes in customer surveys, tag incoming feedback automatically by feature category at ingestion, and deploy human-in-the-loop auditing for borderline sentiment scores. Expanding the sample size across diverse user segments will improve reliability.
"""
st.markdown(summary_text)
import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure NLTK data is downloaded (run this once)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


# --- Function for Sentiment Analysis ---
def get_vader_sentiment(text):
    """
    Analyzes the sentiment of the input text using VADER.
    Returns the compound score and a description of the sentiment.
    """

    # Get the sentiment scores
    # polarity_scores() from NLTK VADER always returns a Python dictionary with four keys
    # Example for vs = {'neg': 0.1, 'neu': 0.3, 'pos': 0.6, 'compound': 0.75}
    vs = analyzer.polarity_scores(text)
    compound_score = vs['compound']

    # Determine the sentiment category and color
    if compound_score >= 0.05:
        sentiment_label = "🟢 Positive"
        color_hex = "#4CAF50"  # Green
    elif compound_score <= -0.05:
        sentiment_label = "🔴 Negative"
        color_hex = "#F44336"  # Red
    else:
        sentiment_label = "🟡 Neutral"
        color_hex = "#FFEB3B"  # Yellow

    return compound_score, sentiment_label, color_hex, vs


# --- Streamlit Frontend ---
def main():
    st.set_page_config(page_title="VADER Sentiment Analysis App", layout="centered")

    # 📝 Header Section
    st.title("Text Sentiment Analysis with VADER")
    st.markdown(
        "Enter a piece of text below to perform a simple **VADER** (Valence Aware Dictionary and Sentiment Reasoner) sentiment analysis.")

    # 📥 User Input
    user_input = st.text_area(
        "**Enter your text here:**",
        "This is a fantastic application, and I absolutely love the simplicity!",
        height=150
    )

    if user_input:
        # 🧠 Analyze the sentiment
        compound, label, color, scores = get_vader_sentiment(user_input)

        # 📊 Display Results
        st.subheader("Analysis Results:")

        # Display the overall sentiment with color-coding
        st.markdown(
            f"""
            <div style="padding: 10px; border-radius: 5px; background-color: {color}; text-align: center; color: black; font-size: 20px;">
                **Overall Sentiment:** {label}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(f"**Compound Score:** `{compound:.4f}`")

        # 🔬 Detailed Scores Section
        st.subheader("Detailed Polarity Scores (VADER):")

        # Create a table/dictionary of the detailed scores
        score_data = {
            "Positive Score (pos)": f"{scores['pos']:.4f}",
            "Neutral Score (neu)": f"{scores['neu']:.4f}",
            "Negative Score (neg)": f"{scores['neg']:.4f}",
            "Compound Score (compound)": f"{scores['compound']:.4f}"
        }

        st.json(score_data)

        # Add a note explaining the compound score
        st.info(
            "The **Compound Score** is a normalized, weighted composite score. It's the primary metric for determining overall sentiment."
        )
    else:
        st.warning("Please enter some text to perform the sentiment analysis.")


if __name__ == "__main__":
    main()

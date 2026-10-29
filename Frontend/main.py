import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# ---------------------------------------------
# Page Config
# ---------------------------------------------
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    layout="wide"
)

st.title("💬 Sentiment Analysis Dashboard")
st.write("Interactive NLP dashboard using **Pretrained Transformer Models**")

# ---------------------------------------------
# Load Model (cached)
# ---------------------------------------------
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

classifier = load_model()

# ---------------------------------------------
# Text Truncation
# ---------------------------------------------
MAX_WORDS = 400
def truncate_text(text):
    words = str(text).split()
    if len(words) > MAX_WORDS:
        return " ".join(words[:MAX_WORDS])
    return text

# =============================================
# 🔹 SECTION 1: LIVE TEXT ANALYSIS
# =============================================
st.subheader("✍️ Analyze Single Text")

user_text = st.text_area(
    "Enter a comment or review",
    height=120,
    placeholder="This product exceeded my expectations..."
)

if st.button("Analyze Text"):
    if user_text.strip():
        result = classifier(truncate_text(user_text))[0]
        col1, col2 = st.columns(2)
        col1.metric("Prediction", result["label"])
        col2.metric("Confidence", round(result["score"], 3))
    else:
        st.warning("Please enter text")

# =============================================
# 🔹 SECTION 2: DEMO DATA
# =============================================
st.divider()
st.subheader("📌 Demo Data (One-Click)")

if st.button("Load Demo Reviews"):
    
    demo_data = {
    "review": [
        # Positive (14)
        "This movie was absolutely fantastic!",
        "Amazing performance by the lead actor.",
        "I loved every moment of this film, brilliant storyline.",
        "Exceeded my expectations, highly recommended!",
        "A masterpiece! Cinematography and acting were top notch.",
        "Pretty good, but could have been better.",
        "A delightful experience, I laughed and cried.",
        "Great value for money, satisfied with the purchase.",
        "Outstanding performance, I will watch it again.",
        "The product works as expected, nothing extraordinary.",
        "I am really happy with this purchase.",
        "Best experience ever, highly satisfied!",
        "Excellent quality, very pleased.",
        "Totally loved it, amazing!",
        # Negative (6)
        "Worst experience ever, totally disappointed.",
        "I would not recommend this to anyone.",
        "Absolutely horrible! I regret buying this.",
        "Not worth the hype, very disappointing.",
        "The story was boring and too predictable.",
        "Horrible taste and texture, won't buy again."
    ]
}


    df = pd.DataFrame(demo_data)
    st.session_state["data"] = df
    st.success("Demo data loaded!")

# =============================================
# 🔹 SECTION 3: CSV UPLOAD
# =============================================
st.divider()
st.subheader("📂 Upload CSV File")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state["data"] = df

# =============================================
# 🔹 SECTION 4: COLUMN SELECTION
# =============================================
if "data" in st.session_state:
    df = st.session_state["data"]

    st.subheader("🧠 Select Text Column")
    text_column = st.selectbox(
        "Which column contains text?",
        df.columns
    )

    df = df[[text_column]].dropna()
    df[text_column] = df[text_column].astype(str).apply(truncate_text)

    rows = st.slider(
        "Number of rows to analyze",
        min_value=1,
        max_value=len(df),
        value=min(1000, len(df))
    )

    df = df.head(rows)

    # =============================================
    # 🔹 SECTION 5: SENTIMENT ANALYSIS
    # =============================================
    with st.spinner("Analyzing sentiments..."):
        results = classifier(list(df[text_column]), batch_size=16)
        df["prediction"] = [r["label"] for r in results]
        df["score"] = [r["score"] for r in results]

    st.success("Sentiment analysis completed!")

    # =============================================
    # 🔹 SECTION 6: DATA PREVIEW
    # =============================================
    st.subheader("📄 Data Preview")
    st.dataframe(df.head(10))

    # =============================================
    # 🔹 SECTION 7: VISUALIZATIONS (3+2 Layout)
    # =============================================
    st.subheader("📊 Sentiment Insights")

    # -------- First Row (3 plots) --------
    with st.container():
        col1, col2, col3 = st.columns(3)

        # 1️⃣ Bar Chart
        with col1:
            st.write("### Sentiment Count")
            st.bar_chart(df["prediction"].value_counts())

        # 2️⃣ Pie Chart
        with col2:
            st.write("### Sentiment Distribution")
            fig1, ax1 = plt.subplots()
            df["prediction"].value_counts().plot(
                kind="pie",
                autopct="%1.1f%%",
                ax=ax1
            )
            ax1.set_ylabel("")
            st.pyplot(fig1)

        # 3️⃣ Histogram
        with col3:
            st.write("### Confidence Score Histogram")
            fig2, ax2 = plt.subplots()
            ax2.hist(df["score"], bins=20)
            ax2.set_xlabel("Confidence")
            ax2.set_ylabel("Frequency")
            st.pyplot(fig2)

    # -------- Second Row (2 plots centered) --------
    with st.container():
        col4, col5, col6 = st.columns([1,1,1])

        # 4️⃣ Box Plot
        with col4:
            st.write("### Score by Sentiment")
            fig3, ax3 = plt.subplots()
            df.boxplot(column="score", by="prediction", ax=ax3)
            plt.suptitle("")
            st.pyplot(fig3)

        # 5️⃣ Trend View
        with col5:
            st.write("### Sentiment Trend")
            trend_df = df["prediction"].map({"POSITIVE": 1, "NEGATIVE": 0})
            st.line_chart(trend_df)

    # =============================================
    # 🔹 SECTION 8: FILTER REVIEWS
    # =============================================
    st.subheader("🔍 Filter Reviews")
    sentiment_filter = st.selectbox(
        "Select sentiment",
        ["ALL", "POSITIVE", "NEGATIVE"]
    )

    if sentiment_filter == "ALL":
        st.dataframe(df)
    else:
        st.dataframe(df[df["prediction"] == sentiment_filter])

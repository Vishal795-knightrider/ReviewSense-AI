import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# ---------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------
st.set_page_config(
    page_title="ReviewSense AI — Product Review Sentiment Analyzer",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Subtle custom styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .stMetric {
        background-color: #F8FAFC;
        padding: 0.75rem;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🛒 ReviewSense AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Transformer-Based Product Review Sentiment Analyzer — College CA2 Project</div>', unsafe_allow_html=True)

# ---------------------------------------------
# Model Configuration & Caching
# ---------------------------------------------
# Pretrained 3-class sentiment model: positive, neutral, negative
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_TOKENS = 512

@st.cache_resource(show_spinner="Loading ReviewSense Transformer Model...")
def load_model():
    """
    Loads and caches the pretrained 3-class sentiment Transformer pipeline.
    Uses RoBERTa architecture with automatic truncation.
    """
    return pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        truncation=True,
        max_length=MAX_TOKENS
    )

classifier = load_model()

def normalize_label(raw_label: str) -> str:
    """Standardizes model output labels to clean uppercase names."""
    lbl = str(raw_label).strip().lower()
    if "pos" in lbl:
        return "POSITIVE"
    elif "neg" in lbl:
        return "NEGATIVE"
    elif "neu" in lbl:
        return "NEUTRAL"
    return str(raw_label).upper()

# ---------------------------------------------
# Sidebar: Project & Model Metadata
# ---------------------------------------------
with st.sidebar:
    st.header("ℹ️ Project Information")
    st.markdown("**ReviewSense AI**")
    st.caption("College CA2 Project: Natural Language Processing")
    st.markdown("---")
    
    st.markdown("### 🤖 Model Details")
    st.markdown(f"- **Architecture:** RoBERTa-base")
    st.markdown(f"- **Pretrained Model:** `{MODEL_NAME}`")
    st.markdown("- **Output Classes:** `POSITIVE`, `NEUTRAL`, `NEGATIVE`")
    st.markdown(f"- **Max Sequence Length:** {MAX_TOKENS} tokens")
    st.markdown("- **Task:** Inference-only (zero training required)")
    
    st.markdown("---")
    st.markdown("### ⚠️ Model Characteristics & Limitations")
    st.info(
        "**Corpus Origin:** This model was pretrained on large-scale conversational text "
        "and fine-tuned on the TweetEval benchmark.\n\n"
        "**Strengths:** Effectively handles informal customer phrasing, ratings, emojis, "
        "and mixed sentiments (positive/neutral/negative).\n\n"
        "**Limitations:** It was not trained exclusively on specialized product catalogs, "
        "so deeply technical hardware jargon or long corporate documentation may yield "
        "lower confidence. Input texts longer than 512 tokens are automatically truncated."
    )

# =============================================
# 🔹 SECTION 1: LIVE TEXT ANALYSIS
# =============================================
st.subheader("✍️ Analyze Single Product Review")

user_text = st.text_area(
    "Enter a customer review or feedback",
    height=120,
    placeholder="e.g., The battery easily lasts two full days, but the speaker volume is mediocre..."
)

if st.button("Analyze Review", type="primary"):
    if user_text.strip():
        with st.spinner("Analyzing sentiment..."):
            result = classifier(user_text, truncation=True, max_length=MAX_TOKENS)[0]
            label = normalize_label(result["label"])
            score = round(float(result["score"]), 4)

        col1, col2 = st.columns(2)
        col1.metric("Predicted Sentiment", label)
        col2.metric("Confidence Score", f"{score * 100:.1f}%")
    else:
        st.warning("Please enter a review before analyzing.")

# =============================================
# 🔹 SECTION 2: DEMO DATA (SMARTPHONE & E-COMMERCE)
# =============================================
st.divider()
st.subheader("📌 Demo Product Reviews (One-Click)")

if st.button("Load Demo Smartphone & E-Commerce Reviews"):
    demo_data = {
        "review": [
            # Positive (8)
            "The battery easily lasts 2 full days of heavy use and the AMOLED display is stunning in bright sunlight.",
            "Exceptional active noise cancellation and crystal clear audio quality. Best headphones for travel.",
            "Blazing fast performance, completely silent operation, and the keyboard feel is top notch.",
            "Ultra-fast 100W charging gets me from 10% to 100% in 25 minutes. Smooth 120Hz display.",
            "The build quality feels premium and sturdy in hand. Definitely worth every penny.",
            "The glare-free screen is very comfortable for night reading and battery lasts weeks on a single charge.",
            "The computational photography and voice dictation features are the best on any Android device.",
            "Fast shipping! The product arrived well-packaged in pristine condition and worked right out of the box.",
            # Neutral (8)
            "The phone performs decent for daily social media and browsing, but the speaker volume is mediocre.",
            "Adequate processing power for office work, though the battery life is average at roughly four hours.",
            "Sound is acceptable for the cheap price, but the ear cushions start hurting after an hour of use.",
            "Gaming performance is decent at 1080p, but the cooling fans get fairly loud under load.",
            "Tracks steps and heart rate reasonably well, but having to charge it every single night is annoying.",
            "Volume output is loud enough for small rooms, but bass is somewhat flat at maximum levels.",
            "Clean stock Android interface, but low-light camera photos have noticeable digital noise.",
            "Delivery was delayed by two days, but the item itself matches the seller description.",
            # Negative (8)
            "Device overheats severely within ten minutes of recording video or gaming and throttles down.",
            "Stopped syncing steps to the smartphone app after two weeks. Customer support was completely unresponsive.",
            "Screen developed a visible green vertical line after a routine software update. Outrageous repair estimate.",
            "Hinge cracked near the left corner during normal opening and closing after just three months.",
            "Left earbud stopped charging completely and the battery drains to zero in ten minutes.",
            "Extremely slow transfer speeds below 5MB/s and device disconnects constantly during file copy.",
            "Frequently drops 5GHz Wi-Fi connection requiring a manual power cycle twice everyday.",
            "Cheap flimsy plastic casing that creaks constantly and trackpad stopped registering double clicks."
        ]
    }
    df = pd.DataFrame(demo_data)
    st.session_state["data"] = df
    st.success(f"Loaded {len(df)} realistic product reviews across Positive, Neutral, and Negative sentiments!")

# =============================================
# 🔹 SECTION 3: CSV UPLOAD
# =============================================
st.divider()
st.subheader("📂 Upload Custom CSV Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing reviews (e.g., from Data/Clean/product_reviews.csv)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state["data"] = df
    st.success(f"Uploaded dataset successfully with {len(df)} rows!")

# =============================================
# 🔹 SECTION 4: COLUMN SELECTION & ROW LIMITER
# =============================================
if "data" in st.session_state:
    df = st.session_state["data"]

    st.subheader("🧠 Select Text Column")
    
    # Try to auto-select common review column names if present
    default_idx = 0
    candidate_cols = ["review", "comment", "text", "review_text", "Review", "Comment"]
    for i, col in enumerate(df.columns):
        if col in candidate_cols:
            default_idx = i
            break

    text_column = st.selectbox(
        "Which column contains the customer reviews?",
        df.columns,
        index=default_idx
    )

    # Clean text column
    working_df = df.copy()
    working_df = working_df.dropna(subset=[text_column])
    working_df[text_column] = working_df[text_column].astype(str)

    rows = st.slider(
        "Number of rows to analyze",
        min_value=1,
        max_value=len(working_df),
        value=min(500, len(working_df))
    )

    working_df = working_df.head(rows).copy()

    # =============================================
    # 🔹 SECTION 5: BATCH SENTIMENT INFERENCE
    # =============================================
    with st.spinner(f"Running transformer inference on {len(working_df)} reviews..."):
        # Batch inference with tokenizer truncation and max_length
        results = classifier(
            list(working_df[text_column]),
            batch_size=16,
            truncation=True,
            max_length=MAX_TOKENS
        )
        working_df["prediction"] = [normalize_label(r["label"]) for r in results]
        working_df["score"] = [round(float(r["score"]), 4) for r in results]

    st.success("Sentiment analysis completed successfully!")

    # =============================================
    # 🔹 SECTION 6: DATA PREVIEW & CSV DOWNLOAD
    # =============================================
    st.subheader("📄 Analyzed Reviews Preview")
    st.dataframe(working_df.head(10), use_container_width=True)

    # CSV Download Button
    csv_bytes = working_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Analyzed CSV (with Predictions & Confidence)",
        data=csv_bytes,
        file_name="reviewsense_analyzed_reviews.csv",
        mime="text/csv",
        help="Download the complete dataset containing original reviews, predicted sentiment, and confidence scores."
    )

    # =============================================
    # 🔹 SECTION 7: VISUALIZATIONS (3 + 2 LAYOUT)
    # =============================================
    st.subheader("📊 Sentiment Insights Dashboard")

    color_palette = {
        "POSITIVE": "#10B981", # Emerald green
        "NEUTRAL": "#64748B",  # Slate gray
        "NEGATIVE": "#EF4444"  # Coral red
    }

    # -------- First Row (3 plots) --------
    with st.container():
        col1, col2, col3 = st.columns(3)

        # 1️⃣ Bar Chart: Sentiment Count
        with col1:
            st.write("### Sentiment Count")
            counts = working_df["prediction"].value_counts()
            st.bar_chart(counts)

        # 2️⃣ Pie Chart: Sentiment Distribution
        with col2:
            st.write("### Sentiment Distribution")
            fig1, ax1 = plt.subplots(figsize=(4, 3.5))
            pred_counts = working_df["prediction"].value_counts()
            pie_colors = [color_palette.get(label, "#94A3B8") for label in pred_counts.index]
            ax1.pie(
                pred_counts,
                labels=pred_counts.index,
                autopct="%1.1f%%",
                startangle=140,
                colors=pie_colors
            )
            ax1.axis("equal")
            st.pyplot(fig1)
            plt.close(fig1)

        # 3️⃣ Histogram: Confidence Scores
        with col3:
            st.write("### Confidence Score Histogram")
            fig2, ax2 = plt.subplots(figsize=(4, 3.5))
            ax2.hist(working_df["score"], bins=20, color="#3B82F6", edgecolor="white")
            ax2.set_xlabel("Confidence")
            ax2.set_ylabel("Frequency")
            st.pyplot(fig2)
            plt.close(fig2)

    # -------- Second Row (2 plots centered) --------
    with st.container():
        col4, col5 = st.columns(2)

        # 4️⃣ Box Plot: Score by Sentiment
        with col4:
            st.write("### Score by Sentiment")
            fig3, ax3 = plt.subplots(figsize=(5, 3.5))
            # Pandas boxplot with explicit grouping
            working_df.boxplot(column="score", by="prediction", ax=ax3, grid=False)
            ax3.set_title("")
            ax3.set_xlabel("Predicted Sentiment")
            ax3.set_ylabel("Confidence Score")
            plt.suptitle("")
            st.pyplot(fig3)
            plt.close(fig3)

        # 5️⃣ Line Chart: Sentiment Polarity Trend
        with col5:
            st.write("### Sentiment Polarity Trend")
            st.caption("Polarity scale: Positive (+1), Neutral (0), Negative (-1)")
            trend_series = working_df["prediction"].map({"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1})
            st.line_chart(trend_series)

    # =============================================
    # 🔹 SECTION 8: FILTER REVIEWS
    # =============================================
    st.subheader("🔍 Filter Analyzed Reviews")
    sentiment_filter = st.selectbox(
        "Filter by predicted sentiment:",
        ["ALL", "POSITIVE", "NEUTRAL", "NEGATIVE"]
    )

    if sentiment_filter == "ALL":
        st.dataframe(working_df, use_container_width=True)
    else:
        filtered_df = working_df[working_df["prediction"] == sentiment_filter]
        st.dataframe(filtered_df, use_container_width=True)

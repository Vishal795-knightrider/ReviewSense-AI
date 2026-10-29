# ReviewSense AI — Transformer-Based Product Review Sentiment Analyzer

An interactive **Product Review Sentiment Analysis Dashboard** built using **pretrained Transformer models** (`cardiffnlp/twitter-roberta-base-sentiment-latest`) and **Streamlit**. Built for high-performance Natural Language Processing and applied Deep Learning.

The project demonstrates an end-to-end NLP workflow: **e-commerce review input → safe token truncation → batch transformer inference → interactive visual analytics & CSV export**.

---

## Project Overview

This dashboard allows users to:

* **Live Single Review Analysis:** Input any customer review or feedback and get predicted 3-class sentiment (`POSITIVE`, `NEUTRAL`, `NEGATIVE`) with model confidence.
* **What Customers Like & Dislike:** Aspect-level feedback extraction identifying specific hardware and feature opinions (battery, display, camera, performance, heating, charging, build quality, audio, price) with quoted snippets and an executive opinion summary.
* **One-Click Demo Data:** Instantly test with realistic smartphone and e-commerce product reviews.
* **Batch CSV Processing:** Upload custom CSV files, select text columns, and analyze sentiment with row limiters.
* **Safe Token Truncation:** Handles long reviews up to 512 tokens using subword truncation without runtime crashes.
* **Visual Analytics (5-Graph Layout):**
  1. Sentiment Count (Bar Chart)
  2. Sentiment Distribution (Pie Chart)
  3. Confidence Score Distribution (Histogram)
  4. Confidence Score by Sentiment (Box Plot)
  5. Sentiment Polarity Trend (Line Chart: Positive=+1, Neutral=0, Negative=-1)
* **Analyzed CSV Export:** One-click download button for processed datasets enriched with predictions and confidence scores.
* **Interactive Sentiment Filter:** Filter reviews by `ALL`, `POSITIVE`, `NEUTRAL`, or `NEGATIVE`.

---

## Model Details & Limitations

* **Model Used:** `cardiffnlp/twitter-roberta-base-sentiment-latest` via Hugging Face Transformers.
* **Architecture:** RoBERTa-base (12-layer, 768-hidden, 12-heads, ~125M parameters).
* **Classes Supported:** 3-class sentiment (`POSITIVE`, `NEUTRAL`, `NEGATIVE`).
* **Training Corpus:** Pretrained on large-scale conversational text and fine-tuned on the TweetEval sentiment benchmark.
* **Limitations (Documented Honestly):**
  * The model is **not trained specifically on a proprietary product catalog**; it is a general-domain sentiment classifier fine-tuned on conversational and social opinion data.
  * Because informal product reviews share linguistic patterns with social feedback (slang, colloquialisms, ratings, emojis), it performs effectively on consumer reviews. However, deeply technical hardware specifications or enterprise documentation may yield ambiguous predictions.
  * Inputs are bounded to the Transformer's maximum context window (512 tokens) with truncation enabled.
* **Zero Custom Training Required:** The project runs inference using cached pretrained weights.

---

## Tech Stack

* **Python 3.10+**
* **Streamlit** (Interactive Web Dashboard)
* **Hugging Face Transformers** (Pretrained Transformer pipeline)
* **PyTorch** (Deep Learning backend)
* **Pandas** (Data manipulation and CSV handling)
* **Matplotlib** (Statistical visualization)

---

## Folder Structure

```
Sentiment-Dashboard/
├── Data/
│   ├── Clean/
│   │   ├── product_reviews.csv  <-- Realistic smartphone & product reviews dataset
│   │   └── comments.csv         <-- Pre-existing sample comments
│   └── Raw/
│       └── imdb_dataset.csv
├── Frontend/
│   ├── main.py                  <-- Streamlit web application
│   └── requirements.txt         <-- Application dependencies
├── Notebook/
│   ├── Clean_Demo_Data_For_Sentimental_Dashboard_Project.ipynb
│   └── sentiment_analysis.ipynb
├── requirements.txt             <-- Cleaned project dependencies
└── Readme.md                    <-- Project documentation
```

---

## How to Run Locally on Windows (PowerShell)

### Step 1: Open PowerShell in Project Directory
```powershell
cd e:\Sentiment-Dashboard-main
```

### Step 2: (Optional but Recommended) Virtual Environment
```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application
```powershell
streamlit run Frontend/main.py
```
Open your web browser at `http://localhost:8501`.

# Sentiment Analysis Dashboard

A production-style **Sentiment Analysis Dashboard** built using **pretrained Transformer models** and **Streamlit**.
The project demonstrates how to go from **raw text data → cleaned dataset → model inference → interactive visual analytics**.

This repository is suitable for:

* Machine Learning / NLP internships
* Data Science portfolios
* Demonstrating real-world use of pretrained models

---

## Live Demo

* Streamlit App: [https://sentimental-dashboard.streamlit.app/](https://sentimental-dashboard.streamlit.app/)
* GitHub Repository: [https://github.com/maroofiums/Sentiment-Dashboard](https://github.com/maroofiums/Sentiment-Dashboard)
* Kaggle Notebook: [https://www.kaggle.com/code/maroofiums/sentiment-analysis/](https://www.kaggle.com/code/maroofiums/sentiment-analysis/)

---

## Project Overview

This dashboard allows users to:

* Analyze sentiment of a **single text input** in real time
* Upload a **CSV file** and run batch sentiment analysis
* Use **demo data** for instant visualization
* Explore results through multiple interactive graphs

The model is not trained from scratch. Instead, it uses **pretrained Transformer models** from Hugging Face, which reflects how NLP systems are commonly built in industry.

---

## Tech Stack

* Python
* Streamlit
* Hugging Face Transformers
* PyTorch (backend for transformers)
* Pandas
* Matplotlib

---

## Folder Structure

```
Sentiment-Dashboard/
├── Data
│   ├── Clean
│   │   └── comments.csv
│   └── Raw
│       └── imdb_dataset.csv
├── Frontend
│   ├── main.py
│   └── requirements.txt
├── Notebook
│   ├── Clean_Demo_Data_For_Sentimental_Dashboard_Project.ipynb
│   └── sentiment_analysis.ipynb
└── requirements.txt
```

---

## Features

### 1. Live Text Sentiment Analysis

* Enter any text
* Instantly see:

  * Predicted sentiment (Positive / Negative)
  * Confidence score

### 2. Demo Data Mode

* One-click loading of demo reviews
* Useful for quick testing and visualization

### 3. CSV Upload & Column Selection

* Upload your own dataset
* Select the column that contains text
* Limit number of rows for faster analysis

### 4. Batch Sentiment Analysis

* Efficient batch inference using pretrained models
* Cached model loading for performance

### 5. Visual Analytics (5 Graphs)

* Sentiment Count (Bar Chart)
* Sentiment Distribution (Pie Chart)
* Confidence Score Histogram
* Score by Sentiment (Box Plot)
* Sentiment Trend (Line Chart)

Layout is optimized as:

* First row: 3 plots
* Second row: 2 centered plots

---

## Dataset Details

* Raw data sourced from IMDB-style reviews
* Cleaned using Pandas in Jupyter notebooks
* Dataset is intentionally **noisy and slightly imbalanced** to simulate real-world user reviews

This helps make trends, histograms, and distributions more realistic.

---

## How to Run Locally

### 1. Create Conda Environment

```bash
conda create -n sentiment-dashboard python=3.10 -y
conda activate sentiment-dashboard
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit App

```bash
cd Frontend
streamlit run main.py
```

---

## Model Used

* Default Hugging Face `sentiment-analysis` pipeline
* Transformer-based pretrained model
* No custom training required

This reflects real-world usage where pretrained models are fine-tuned or directly deployed.

---

## Limitations

* Binary sentiment only (Positive / Negative)
* Very long texts are truncated for performance
* Confidence score reflects model probability, not human certainty

---

## Future Improvements

* Neutral sentiment support
* Time-based sentiment aggregation
* Export analyzed results as CSV
* Replace Matplotlib with Plotly for richer interactivity
* Authentication and user-specific dashboards

---

This project is designed to demonstrate practical NLP skills, clean code structure, and real-world ML workflows rather than just model accuracy.

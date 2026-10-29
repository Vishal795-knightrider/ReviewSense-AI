import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import re

import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# ---------------------------------------------
# 1. Page Configuration & HopUp SaaS Styling
# ---------------------------------------------
st.set_page_config(
    page_title="ReviewSense AI — Product Review Sentiment Analyzer",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium SaaS Aesthetic: Warm off-white background, charcoal text, vibrant orange accents
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Global App Canvas */
    html, body, [class*="css"], .stApp {
        background-color: #FAF8F5 !important;
        color: #18181B !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    }

    /* Force all Streamlit markdown and labels to dark charcoal */
    .stApp p, .stApp span, .stApp label, .stApp div[data-testid="stMarkdownContainer"] {
        color: #18181B;
    }
    .stApp .stCaption, .stApp [data-testid="stCaptionContainer"] {
        color: #71717A !important;
    }

    /* Top Navigation / Header */
    .header-container {
        display: flex;
        flex-direction: column;
        gap: 0.45rem;
        padding: 0.75rem 0 1.25rem 0;
        border-bottom: 1px solid #ECE7DF;
        margin-bottom: 1.5rem;
    }
    .header-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .brand-title {
        font-size: 2.15rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #18181B !important;
        display: flex;
        align-items: center;
        gap: 0.55rem;
        line-height: 1.1;
    }
    .brand-title span.accent {
        color: #FF5A1F !important;
    }
    .brand-subtitle {
        font-size: 1.02rem;
        color: #71717A !important;
        font-weight: 500;
        line-height: 1.4;
    }
    .header-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background-color: #ECFDF5;
        color: #065F46 !important;
        border: 1px solid #A7F3D0;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.01em;
        box-shadow: 0 1px 2px rgba(6, 95, 70, 0.05);
    }
    .header-status-pill .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
    }
    .meta-ticker {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.75rem;
        margin-top: 0.25rem;
    }
    .ticker-item {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.8rem;
        font-weight: 600;
        color: #52525B;
        background: #F4EFEA;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        border: 1px solid #EBE4D8;
    }
    .ticker-item .dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #FF5A1F;
    }

    /* Cards & Containers */
    .hopup-card {
        background: #FFFFFF;
        border: 1px solid #ECE7DF;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02), 0 6px 16px rgba(0,0,0,0.015);
        margin-bottom: 1.5rem;
    }
    .hopup-card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #18181B !important;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        letter-spacing: -0.015em;
    }
    .hopup-card-subtitle {
        font-size: 0.88rem;
        color: #71717A !important;
        margin-bottom: 1rem;
        line-height: 1.45;
    }

    /* Metric Stat Box (HopUp Minimal Grid) */
    .stat-card {
        background: #FFFFFF;
        border: 1px solid #ECE7DF;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stat-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    }
    .stat-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #71717A !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stat-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #18181B !important;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    .stat-desc {
        font-size: 0.78rem;
        color: #A1A1AA !important;
        font-weight: 500;
    }

    /* Single Review Result Cards (Light & Dark Theme Safe) */
    .result-card {
        background: #FFFFFF !important;
        border: 1px solid #ECE7DF !important;
        border-radius: 12px !important;
        padding: 1.25rem 1.45rem !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03) !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 0.35rem !important;
    }
    .result-card.positive {
        border-left: 5px solid #10B981 !important;
    }
    .result-card.neutral {
        border-left: 5px solid #64748B !important;
    }
    .result-card.negative {
        border-left: 5px solid #EF4444 !important;
    }
    .result-card.score {
        border-left: 5px solid #FF5A1F !important;
    }
    .result-title {
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: #71717A !important;
    }
    .result-val {
        font-size: 1.75rem !important;
        font-weight: 800 !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
        line-height: 1.2 !important;
        letter-spacing: -0.02em !important;
    }
    .result-sub {
        font-size: 0.82rem !important;
        color: #71717A !important;
        margin-top: 0.2rem !important;
        font-weight: 500 !important;
    }

    /* Sentiment Badge Pills */
    .sentiment-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.7rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .pill-positive {
        background-color: #ECFDF5;
        color: #065F46 !important;
        border: 1px solid #A7F3D0;
    }
    .pill-neutral {
        background-color: #F1F5F9;
        color: #334155 !important;
        border: 1px solid #CBD5E1;
    }
    .pill-negative {
        background-color: #FEF2F2;
        color: #991B1B !important;
        border: 1px solid #FECACA;
    }
    .pill-all {
        background-color: #FFF4ED;
        color: #C2410C !important;
        border: 1px solid #FED7AA;
    }

    /* Aspect Card Styling (What Customers Like & Dislike) */
    .aspect-card {
        background: #FFFFFF !important;
        border-radius: 10px !important;
        padding: 0.85rem 1rem !important;
        margin-bottom: 0.65rem !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .aspect-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 8px rgba(0,0,0,0.04) !important;
    }
    .aspect-card.like {
        border: 1px solid #D1FAE5 !important;
        border-left: 4px solid #10B981 !important;
    }
    .aspect-card.dislike {
        border: 1px solid #FEE2E2 !important;
        border-left: 4px solid #EF4444 !important;
    }
    .aspect-name-like {
        font-weight: 700;
        font-size: 0.88rem;
        color: #065F46 !important;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }
    .aspect-name-dislike {
        font-weight: 700;
        font-size: 0.88rem;
        color: #991B1B !important;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }
    .aspect-snippet {
        font-size: 0.8rem;
        color: #52525B !important;
        margin-top: 0.25rem;
        font-style: italic;
        line-height: 1.4;
    }
    .summary-card {
        background: #FFFFFF !important;
        border: 1px solid #ECE7DF !important;
        border-left: 4px solid #FF5A1F !important;
        border-radius: 10px !important;
        padding: 0.95rem 1.25rem !important;
        margin-top: 0.9rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
    }
    .summary-card-title {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #FF5A1F !important;
        margin-bottom: 0.25rem;
    }
    .summary-card-body {
        font-size: 0.9rem;
        color: #18181B !important;
        line-height: 1.45;
        font-weight: 500;
    }

    /* Orange Primary Buttons */
    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="baseButton-primary"] {
        background-color: #FF5A1F !important;
        color: #FFFFFF !important;
        border: 1px solid #FF5A1F !important;
        border-radius: 9px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.5rem !important;
        box-shadow: 0 2px 6px rgba(255, 90, 31, 0.25) !important;
        transition: all 0.18s ease !important;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[data-testid="baseButton-primary"]:hover {
        background-color: #EA4C12 !important;
        border-color: #EA4C12 !important;
        box-shadow: 0 4px 12px rgba(255, 90, 31, 0.35) !important;
        transform: translateY(-1px) !important;
    }

    /* Secondary Buttons */
    div.stButton > button[kind="secondary"],
    div.stButton > button[data-testid="baseButton-secondary"] {
        background-color: #FFFFFF !important;
        color: #18181B !important;
        border: 1px solid #E5E0D8 !important;
        border-radius: 9px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.25rem !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
        transition: all 0.15s ease !important;
    }
    div.stButton > button[kind="secondary"]:hover,
    div.stButton > button[data-testid="baseButton-secondary"]:hover {
        background-color: #F8F5EE !important;
        border-color: #D6CEC2 !important;
    }

    /* Download Buttons */
    div[data-testid="stDownloadButton"] > button {
        background-color: #FF5A1F !important;
        color: #FFFFFF !important;
        border: 1px solid #FF5A1F !important;
        border-radius: 9px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.4rem !important;
        box-shadow: 0 2px 6px rgba(255, 90, 31, 0.25) !important;
        transition: all 0.18s ease !important;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        background-color: #EA4C12 !important;
        border-color: #EA4C12 !important;
        box-shadow: 0 4px 12px rgba(255, 90, 31, 0.35) !important;
        transform: translateY(-1px) !important;
    }

    /* Form Controls & Inputs */
    .stTextArea textarea {
        border-radius: 9px !important;
        border: 1px solid #E5E0D8 !important;
        background-color: #FFFFFF !important;
        color: #18181B !important;
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
    }
    .stTextArea textarea:focus {
        border-color: #FF5A1F !important;
        box-shadow: 0 0 0 3px rgba(255, 90, 31, 0.15) !important;
    }

    /* Clean Code Styling (eliminates dark block in light theme) */
    code, pre code {
        background-color: #FFF4ED !important;
        color: #C2410C !important;
        border: 1px solid #FED7AA !important;
        border-radius: 5px !important;
        padding: 0.18rem 0.45rem !important;
        font-size: 0.85em !important;
        font-weight: 600 !important;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
    }

    /* File Uploader Complete Light Theme Styling */
    div[data-testid="stFileUploader"] {
        background-color: #FFFFFF !important;
        border-radius: 14px !important;
        padding: 0.2rem !important;
    }
    div[data-testid="stFileUploader"] section,
    div[data-testid="stFileUploader"] [data-testid*="Dropzone"],
    div[data-testid="stFileUploader"] [data-testid*="dropzone"],
    section[data-testid="stFileUploadDropzone"],
    div[data-testid="stFileUploadDropzone"] {
        background-color: #FFFFFF !important;
        border: 2px dashed #E5E0D8 !important;
        border-radius: 12px !important;
        padding: 1.5rem 1rem !important;
        text-align: center !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stFileUploader"] section:hover,
    section[data-testid="stFileUploadDropzone"]:hover {
        border-color: #FF5A1F !important;
        background-color: #FFFDFB !important;
    }
    /* Dropzone text and icon */
    div[data-testid="stFileUploader"] section span,
    div[data-testid="stFileUploader"] section small,
    div[data-testid="stFileUploader"] section p,
    div[data-testid="stFileUploader"] section div,
    section[data-testid="stFileUploadDropzone"] * {
        color: #52525B !important;
        background-color: transparent !important;
    }
    div[data-testid="stFileUploader"] section svg,
    section[data-testid="stFileUploadDropzone"] svg {
        fill: #FF5A1F !important;
        color: #FF5A1F !important;
    }
    /* "Browse files" button inside File Uploader */
    div[data-testid="stFileUploader"] section button,
    section[data-testid="stFileUploadDropzone"] button,
    div[data-testid="stFileUploader"] button[data-testid="baseButton-secondary"] {
        background-color: #FAF8F5 !important;
        color: #18181B !important;
        border: 1px solid #D6CEC2 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.45rem 1.25rem !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
        transition: all 0.15s ease !important;
    }
    div[data-testid="stFileUploader"] section button:hover,
    section[data-testid="stFileUploadDropzone"] button:hover,
    div[data-testid="stFileUploader"] button[data-testid="baseButton-secondary"]:hover {
        background-color: #FFFFFF !important;
        border-color: #FF5A1F !important;
        color: #FF5A1F !important;
        box-shadow: 0 2px 6px rgba(255, 90, 31, 0.15) !important;
    }
    /* Uploaded file badge / item */
    div[data-testid="stFileUploader"] ul,
    div[data-testid="stFileUploader"] li {
        background-color: #FFFFFF !important;
        border: 1px solid #ECE7DF !important;
        border-radius: 8px !important;
        color: #18181B !important;
    }
    div[data-testid="stFileUploader"] li * {
        color: #18181B !important;
    }

    /* Selectbox & Inputs */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border-color: #E5E0D8 !important;
        border-radius: 8px !important;
        color: #18181B !important;
    }

    /* Sidebar Clean HopUp Styling */
    section[data-testid="stSidebar"] {
        background-color: #F5F1EB !important;
        border-right: 1px solid #E8E2D7 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span {
        color: #27272A !important;
    }

    /* Fix Streamlit metric styling inside containers */
    div[data-testid="stMetricValue"] * {
        color: #18181B !important;
    }
    div[data-testid="stMetricLabel"] * {
        color: #71717A !important;
    }

    /* Hide default footer */
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------
# 2. Top Header Banner
# ---------------------------------------------
st.markdown("""
<div class="header-container">
    <div class="header-top">
        <div class="brand-title">
            🛍️ Review<span class="accent">Sense</span> AI
        </div>
        <div class="header-status-pill">
            <span class="status-dot"></span> Live RoBERTa Inference
        </div>
    </div>
    <div class="brand-subtitle">
        Transformer-Based Product Review Sentiment Analyzer with Pretrained Deep Learning
    </div>
    <div class="meta-ticker">
        <div class="ticker-item"><span class="dot"></span> RoBERTa-base (125M params)</div>
        <div class="ticker-item"><span class="dot"></span> 3 Output Classes (Pos/Neu/Neg)</div>
        <div class="ticker-item"><span class="dot"></span> 512 Max Token Subword Truncation</div>
        <div class="ticker-item"><span class="dot"></span> Zero-Shot / Pretrained Inference</div>
        <div class="ticker-item"><span class="dot"></span> Real-Time Analytics Dashboard</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------
# 3. Model Configuration & Caching
# ---------------------------------------------
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_TOKENS = 512

@st.cache_resource(show_spinner="Loading ReviewSense RoBERTa Transformer Model...")
def load_model():
    """
    Loads and caches the pretrained 3-class sentiment Transformer pipeline.
    Uses RoBERTa architecture with automatic truncation at 512 tokens.
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
# Aspect Extraction & Opinion Mining Engine
# ---------------------------------------------
ASPECT_TAXONOMY = {
    "Battery Life": {
        "keywords": ["battery", "battery life", "backup", "standby", "screen-on time", "sot", "drain", "draining", "power backup"],
        "inherent_pos": ["lasts long", "long-lasting", "lasts two days", "lasts 2 days", "lasts 2 full days", "lasts full day", "lasts weeks"],
        "inherent_neg": ["drains fast", "drains quickly", "rapid drain", "poor backup", "dies quickly", "battery drains", "drain to zero", "drains to zero", "battery life is average"]
    },
    "Fast Charging": {
        "keywords": ["charge", "charging", "charger", "fast charging", "quick charge", "100w", "65w", "adapter", "cable", "type-c", "wireless charging"],
        "inherent_pos": ["fast charging", "ultra-fast charging", "quick charge", "charges fast", "ultra-fast 100w", "100w charging", "65w charging", "fast charge"],
        "inherent_neg": ["slow charging", "charges slowly", "stopped charging", "overheating while charging", "does not charge", "stopped charging completely", "charge it every single night"]
    },
    "Display & Screen": {
        "keywords": ["display", "screen", "amoled", "oled", "lcd", "panel", "refresh rate", "120hz", "90hz", "brightness", "glare", "resolution"],
        "inherent_pos": ["glare-free", "glare-free screen", "stunning display", "vibrant colors", "crisp screen", "amoled display is stunning", "smooth 120hz", "stunning in bright sunlight", "bright display", "display is bright"],
        "inherent_neg": ["green line", "green vertical line", "screen flicker", "flickering", "dead pixel", "dim display", "washed out", "digital noise", "vertical line"]
    },
    "Camera & Photography": {
        "keywords": ["camera", "cameras", "photo", "photos", "picture", "pictures", "lens", "lenses", "sensor", "portrait", "night mode", "low-light", "video recording", "selfie", "photography"],
        "inherent_pos": ["stunning photos", "great detail", "computational photography", "crisp photos", "clear photos", "computational photography and voice dictation", "excellent camera"],
        "inherent_neg": ["blurry", "grainy", "digital noise", "poor camera", "camera is terrible", "washed out photos", "bad low-light", "blurry camera"]
    },
    "Performance & Speed": {
        "keywords": ["performance", "speed", "processor", "chipset", "cpu", "gpu", "ram", "multitasking", "gaming", "fps", "frame rate", "snappy", "lag", "latency", "performs", "processing power", "processing"],
        "inherent_pos": ["blazing fast", "blazing fast performance", "buttery smooth", "super fast", "no lag", "zero lag", "top notch performance", "gaming performance is decent", "performs decent", "adequate processing power"],
        "inherent_neg": ["laggy", "stutters", "stuttering", "slow down", "throttles", "throttles down", "throttling", "freezes", "unresponsive", "slow transfer speeds"]
    },
    "Heating & Thermals": {
        "keywords": ["heating", "heats", "overheating", "overheats", "thermal", "thermals", "temperature", "warm", "cooling", "fan", "fans", "heat"],
        "inherent_pos": ["runs cool", "stays cool", "no heating", "good cooling", "efficient cooling", "silent operation", "completely silent", "without heating"],
        "inherent_neg": ["overheats", "overheating", "heats up", "runs hot", "gets hot", "thermal throttling", "loud fan", "loud fans", "device overheats", "fans get fairly loud", "cooling fans get fairly loud", "phone overheats"]
    },
    "Build Quality & Design": {
        "keywords": ["build quality", "build", "design", "hinge", "casing", "plastic", "metal", "chassis", "durability", "durable", "sturdy", "finish", "creak", "creaks", "body", "look and feel", "keyboard", "trackpad", "scroll wheel", "ear cushions", "cushions"],
        "inherent_pos": ["sturdy build", "premium build", "solid feel", "durable build", "premium materials", "keyboard feel is top notch", "feels premium", "premium and sturdy", "ergonomics are unmatched", "compact design"],
        "inherent_neg": ["flimsy", "creaking", "creaks", "cheap plastic", "cracked", "hinge cracked", "scratches easily", "casing that creaks", "flimsy plastic", "cheap flimsy plastic", "cushions start hurting", "stopped registering double clicks"]
    },
    "Audio & Sound": {
        "keywords": ["speaker", "speakers", "audio", "sound", "volume", "soundbar", "bass", "treble", "earbuds", "earbud", "headphones", "anc", "noise cancellation", "mic", "microphone"],
        "inherent_pos": ["crystal clear audio", "punchy bass", "great sound", "active noise cancellation", "loud and clear", "exceptional active noise cancellation", "volume output is loud enough", "sound is acceptable"],
        "inherent_neg": ["distorted", "crackling", "muffled", "tinny", "low volume", "mediocre volume", "speaker volume is mediocre", "flat bass", "left earbud stopped", "crackling noise", "bass is somewhat flat"]
    },
    "Price & Value": {
        "keywords": ["price", "pricing", "cost", "value", "worth", "money", "affordable", "budget", "expensive", "overpriced"],
        "inherent_pos": ["worth every penny", "great value", "value for money", "worth it", "budget friendly", "affordable price", "cheap price"],
        "inherent_neg": ["overpriced", "too expensive", "not worth it", "waste of money", "rip off", "outrageous repair estimate", "expensive repair"]
    },
    "Connectivity & Shipping": {
        "keywords": ["wifi", "wi-fi", "bluetooth", "connection", "connects", "disconnects", "sync", "syncing", "app", "software", "update", "shipping", "delivery", "packaged", "packaging", "package", "transfer speeds", "5ghz"],
        "inherent_pos": ["seamless connection", "quick pairing", "clean interface", "stock android", "fast shipping", "well-packaged", "pristine condition", "worked right out of the box", "matches the seller description"],
        "inherent_neg": ["drops connection", "disconnects constantly", "bluetooth drops", "failed to sync", "stopped syncing", "software bug", "drops 5ghz", "frequently drops", "delivery was delayed", "delayed by two days"]
    }
}

POSITIVE_WORDS = {
    "great", "good", "excellent", "awesome", "amazing", "superb", "fantastic", "outstanding",
    "impressive", "perfect", "stunning", "crisp", "clear", "punchy", "smooth", "snappy",
    "fast", "solid", "reliable", "durable", "sturdy", "premium", "top-notch", "top notch",
    "decent", "fine", "acceptable", "adequate", "satisfactory", "love", "loved", "best",
    "flawless", "comfortable", "pristine", "easy", "well-built", "worth", "silent", "cool",
    "unmatched", "ultra-fast", "super-fast", "crystal", "addictive", "accurately", "well", "enough",
    "bright", "brightness", "sharp", "vivid", "responsive", "rich", "long"
}

NEGATIVE_WORDS = {
    "bad", "terrible", "horrible", "awful", "poor", "worst", "disappointing", "mediocre",
    "subpar", "slow", "sluggish", "lag", "laggy", "stutter", "overheats", "overheating",
    "hot", "warm", "drain", "drains", "crack", "cracked", "flimsy", "cheap", "creaks",
    "broken", "damaged", "defect", "distorted", "crackling", "unresponsive", "fails",
    "stopped", "drops", "expensive", "overpriced", "outrageous", "annoying", "painful",
    "useless", "noisy", "loud", "flat", "severe", "throttles", "burn", "cluttered", "stiff",
    "hurts", "hurting", "delayed", "noise", "creak", "average"
}

NEGATION_TERMS = {
    "not", "no", "never", "hardly", "barely", "scarcely", "without",
    "isn't", "isnt", "aren't", "arent", "wasn't", "wasnt", "weren't", "werent",
    "don't", "dont", "doesn't", "doesnt", "didn't", "didnt", "won't", "wont",
    "can't", "cant", "couldn't", "couldnt", "lacks", "lacking", "neither", "nor"
}

def split_into_clauses(text: str):
    """
    Splits review into clauses using punctuation, contrastive conjunctions,
    and coordinated conjunctions (e.g., ', and', ', while', and standalone 'and' when connecting distinct aspect propositions).
    """
    text = text.strip()
    primary_pattern = r'[.!?;\n]+|,\s*(?:but|however|although|though|yet|whereas|while|except|and)\s*|\s+(?:but|however|although|though|yet|whereas|while)\s+'
    raw_clauses = [c.strip() for c in re.split(primary_pattern, text, flags=re.IGNORECASE) if c.strip()]

    final_clauses = []
    for clause in raw_clauses:
        # Check if ' and ' conjoins two distinct aspect-carrying clauses
        sub_parts = re.split(r'\s+and\s+', clause, flags=re.IGNORECASE)
        if len(sub_parts) > 1:
            can_split = True
            for sp in sub_parts:
                sp_tokens = set(re.findall(r'\b\w+\b', sp.lower()))
                has_aspect = any(any(kw in sp.lower() for kw in asp["keywords"]) for asp in ASPECT_TAXONOMY.values())
                has_sent = bool(sp_tokens & (POSITIVE_WORDS | NEGATIVE_WORDS))
                if not (has_aspect or has_sent):
                    can_split = False
                    break
            if can_split:
                for sp in sub_parts:
                    if sp.strip():
                        final_clauses.append(sp.strip())
                continue
        final_clauses.append(clause)

    return final_clauses

def get_aspect_token_indices(tokens, aspect_info):
    """Locates token positions of the matched aspect keywords in the clause."""
    indices = []
    for kw in aspect_info["keywords"]:
        kw_tokens = re.findall(r'\b[\w\'-]+\b', kw.lower())
        kw_len = len(kw_tokens)
        for i in range(len(tokens) - kw_len + 1):
            if tokens[i:i+kw_len] == kw_tokens:
                indices.extend(range(i, i+kw_len))
    return set(indices)

def analyze_aspect_in_clause(clause: str, aspect_name: str, aspect_info: dict):
    """Determines whether a clause expresses positive, negative, or neutral opinion on an aspect using local evidence."""
    clause_lower = clause.lower()

    # Match check
    has_match = False
    for kw in aspect_info["keywords"]:
        if kw == "speed" and ("charging" in clause_lower or "transfer" in clause_lower):
            continue
        if re.search(r'\b' + re.escape(kw) + r'\b', clause_lower):
            has_match = True
            break

    if not has_match:
        for ip in aspect_info.get("inherent_pos", []):
            if ip in clause_lower:
                has_match = True
                break
        if not has_match:
            for ineg in aspect_info.get("inherent_neg", []):
                if ineg in clause_lower:
                    has_match = True
                    break

    if not has_match:
        return None

    # Inherent positive phrase check with negation guard
    for ip in aspect_info.get("inherent_pos", []):
        if ip in clause_lower:
            negated = any(re.search(r'\b' + re.escape(neg) + r'\b\s+(?:\w+\s+){0,2}' + re.escape(ip.split()[0]), clause_lower) for neg in NEGATION_TERMS)
            return "neg" if negated else "pos"

    # Inherent negative phrase check with negation guard
    for ineg in aspect_info.get("inherent_neg", []):
        if ineg in clause_lower:
            negated = any(re.search(r'\b' + re.escape(neg) + r'\b\s+(?:\w+\s+){0,2}' + re.escape(ineg.split()[0]), clause_lower) for neg in NEGATION_TERMS)
            return "pos" if negated else "neg"

    # Contextual token scanning using LOCAL EVIDENCE proximity
    tokens = re.findall(r'\b[\w\'-]+\b', clause_lower)
    aspect_indices = get_aspect_token_indices(tokens, aspect_info)

    pos_score = 0.0
    neg_score = 0.0

    for i, token in enumerate(tokens):
        subwords = token.split('-')
        token_is_pos = token in POSITIVE_WORDS or any(sw in POSITIVE_WORDS for sw in subwords)
        token_is_neg = token in NEGATIVE_WORDS or any(sw in NEGATIVE_WORDS for sw in subwords)

        if not (token_is_pos or token_is_neg):
            continue

        # Proximity weight: words closer to the aspect keyword have stronger local influence
        if aspect_indices:
            min_dist = min(abs(i - idx) for idx in aspect_indices)
            if min_dist > 6:
                weight = 0.2
            elif min_dist > 3:
                weight = 0.6
            else:
                weight = 1.0
        else:
            weight = 1.0

        window_start = max(0, i - 3)
        window = tokens[window_start:i]
        is_negated = any(neg in NEGATION_TERMS for neg in window)

        if token_is_pos:
            if is_negated:
                neg_score += 1.5 * weight
            else:
                pos_score += 1.0 * weight
        elif token_is_neg:
            if is_negated:
                pos_score += 1.0 * weight
            else:
                neg_score += 1.5 * weight

    if pos_score > neg_score:
        return "pos"
    elif neg_score > pos_score:
        return "neg"
    return None

def extract_customer_likes_dislikes(text: str, overall_sentiment: str = "NEUTRAL"):
    """
    Extracts explicit customer likes and dislikes per product aspect,
    along with quoted evidence snippets and an executive opinion summary.
    """
    clauses = split_into_clauses(text)
    likes = []
    dislikes = []

    seen_likes = set()
    seen_dislikes = set()

    for clause in clauses:
        for asp_name, asp_info in ASPECT_TAXONOMY.items():
            sent = analyze_aspect_in_clause(clause, asp_name, asp_info)
            if sent == "pos" and asp_name not in seen_likes:
                seen_likes.add(asp_name)
                likes.append({"aspect": asp_name, "snippet": clause})
            elif sent == "neg" and asp_name not in seen_dislikes:
                seen_dislikes.add(asp_name)
                dislikes.append({"aspect": asp_name, "snippet": clause})

    like_names = [item["aspect"] for item in likes]
    dislike_names = [item["aspect"] for item in dislikes]

    if like_names and dislike_names:
        summary = f"Mixed sentiment: Praises **{', '.join(like_names)}**, but expresses dissatisfaction with **{', '.join(dislike_names)}**."
    elif like_names:
        summary = f"Favorable sentiment: The customer expresses satisfaction with **{', '.join(like_names)}**."
    elif dislike_names:
        summary = f"Critical sentiment: The customer highlights dissatisfaction regarding **{', '.join(dislike_names)}**."
    else:
        summary = "No clear product-specific feedback detected."

    return {
        "likes": likes,
        "dislikes": dislikes,
        "summary": summary
    }

# ---------------------------------------------
# 4. Sidebar: Project & Model Metadata
# ---------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.2rem;">
        <span style="font-size: 1.4rem;">🛍️</span>
        <strong style="font-size: 1.15rem; color: #18181B;">Review<span style="color: #FF5A1F;">Sense</span> AI</strong>
    </div>
    <div style="font-size: 0.8rem; color: #71717A; margin-bottom: 0.9rem;">
        Transformer-Based Sentiment Intelligence
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("#### 🤖 Model Architecture")
    st.markdown(f"- **Backbone:** RoBERTa-base (12 layers, 768 hidden, 12 heads)")
    st.markdown(f"- **Pretrained Model:** `{MODEL_NAME}`")
    st.markdown("- **Output Classes:** `POSITIVE`, `NEUTRAL`, `NEGATIVE`")
    st.markdown(f"- **Max Sequence Length:** {MAX_TOKENS} tokens (subword truncation)")
    st.markdown("- **Deployment Mode:** Inference-only (zero training required)")

    st.markdown("---")
    st.markdown("#### ⚠️ Model Details & Limitations")
    st.markdown("""
    <div style="background: #FFFFFF; border: 1px solid #E5E0D8; border-radius: 10px; padding: 0.95rem; font-size: 0.82rem; color: #52525B; line-height: 1.5; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
        <strong style="color: #18181B;">Corpus Origin:</strong> Pretrained on large-scale conversational text and fine-tuned on the TweetEval sentiment benchmark.<br><br>
        <strong style="color: #18181B;">Strengths:</strong> Highly effective for real customer opinions, slang, rating nuances, emojis, and mixed consumer feedback.<br><br>
        <strong style="color: #18181B;">Limitations:</strong> Not trained specifically on a closed e-commerce catalog. Rare technical hardware jargon or long corporate manuals may yield lower confidence. Sequences beyond 512 tokens are automatically truncated.
    </div>
    """, unsafe_allow_html=True)

# =============================================
# 🔹 SECTION 1: LIVE SINGLE REVIEW ANALYSIS
# =============================================
st.markdown("""
<div class="hopup-card-title">✍️ Analyze Single Product Review</div>
<div class="hopup-card-subtitle">Test any customer review, smartphone feedback, or e-commerce comment in real time with RoBERTa deep learning.</div>
""", unsafe_allow_html=True)

user_text = st.text_area(
    "Customer Review Text",
    height=105,
    placeholder="e.g., The battery easily lasts two full days of heavy use, but the speaker volume is mediocre and distorts at maximum volume...",
    label_visibility="collapsed"
)

col_btn, col_clear, _ = st.columns([1.5, 1, 3.5])
with col_btn:
    analyze_clicked = st.button("Analyze Review", type="primary", use_container_width=True)
with col_clear:
    if st.button("Clear", type="secondary", use_container_width=True):
        st.session_state["single_result"] = None
        st.rerun()

# Run inference when clicked
if analyze_clicked:
    if user_text.strip():
        with st.spinner("Analyzing review sentiment with RoBERTa..."):
            result = classifier(user_text, truncation=True, max_length=MAX_TOKENS)[0]
            label = normalize_label(result["label"])
            score = round(float(result["score"]), 4)
            st.session_state["single_result"] = {
                "label": label,
                "score": score,
                "text": user_text
            }
    else:
        st.warning("Please enter a customer review before clicking Analyze.")

# Display results only when a valid prediction exists (Fixes blank white box bug)
if st.session_state.get("single_result") is not None:
    res = st.session_state["single_result"]
    label = res["label"]
    score = res["score"]

    color_meta = {
        "POSITIVE": {"color": "#10B981", "icon": "🟢", "css_class": "positive", "badge": "Favorable Feedback", "meter_bg": "#D1FAE5"},
        "NEUTRAL":  {"color": "#64748B", "icon": "⚪", "css_class": "neutral",  "badge": "Mixed / Balanced Opinion", "meter_bg": "#E2E8F0"},
        "NEGATIVE": {"color": "#EF4444", "icon": "🔴", "css_class": "negative", "badge": "Critical Feedback", "meter_bg": "#FEE2E2"}
    }
    meta = color_meta.get(label, {"color": "#FF5A1F", "icon": "🔹", "css_class": "score", "badge": "Classified Sentiment", "meter_bg": "#FFEDD5"})

    score_pct = score * 100
    st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="result-card {meta['css_class']}">
            <div class="result-title">Predicted Sentiment</div>
            <div class="result-val" style="color: {meta['color']} !important;">
                <span>{meta['icon']}</span>
                <span>{label}</span>
            </div>
            <div class="result-sub">
                <span class="sentiment-pill pill-{label.lower()}">{meta['badge']}</span>
                <span style="color: #A1A1AA; font-size: 0.78rem; margin-left: 0.4rem;">• RoBERTa Transformer</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="result-card score">
            <div class="result-title">Confidence Score</div>
            <div class="result-val" style="color: #18181B !important;">
                <span>{score_pct:.1f}%</span>
            </div>
            <div style="background-color: #F4EFEA; border-radius: 9999px; height: 6px; width: 100%; overflow: hidden; margin-top: 0.2rem;">
                <div style="background-color: #FF5A1F; height: 100%; width: {min(100, max(0, score_pct))}%; border-radius: 9999px;"></div>
            </div>
            <div class="result-sub" style="color: #71717A !important;">
                Probability certainty for <strong>{label}</strong> classification
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------------------------
    # 🎯 Section: What Customers Like & Dislike
    # ---------------------------------------------
    aspect_results = extract_customer_likes_dislikes(res["text"], overall_sentiment=label)
    likes = aspect_results["likes"]
    dislikes = aspect_results["dislikes"]
    summary_text = aspect_results["summary"]

    st.markdown("<div style='height: 1.1rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 1.08rem; font-weight: 700; color: #18181B; margin-bottom: 0.25rem; display: flex; align-items: center; gap: 0.45rem;">
        🎯 What Customers Like & Dislike
    </div>
    <div style="font-size: 0.86rem; color: #71717A; margin-bottom: 0.95rem;">
        Fine-grained aspect extraction identifying specific hardware and product attributes praised or criticized in this review.
    </div>
    """, unsafe_allow_html=True)

    if not likes and not dislikes:
        st.markdown("""
        <div style="background: #FFFFFF; border: 1px dashed #D6CEC2; border-radius: 12px; padding: 1.25rem; text-align: center; margin-bottom: 0.5rem;">
            <div style="font-size: 1.3rem; margin-bottom: 0.25rem;">🔍</div>
            <div style="font-weight: 700; color: #18181B; font-size: 0.95rem;">No clear product-specific feedback detected.</div>
            <div style="font-size: 0.83rem; color: #71717A; margin-top: 0.25rem; max-width: 550px; margin-left: auto; margin-right: auto;">
                The review conveys overall sentiment, but does not explicitly praise or criticize specific hardware or feature attributes (such as battery, camera, display, performance, or heating).
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        col_like, col_dislike = st.columns(2)

        with col_like:
            like_count_label = f"{len(likes)} Feature" if len(likes) == 1 else f"{len(likes)} Features"
            st.markdown(f"""
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.65rem;">
                <span style="font-size: 0.88rem; font-weight: 700; color: #065F46; display: flex; align-items: center; gap: 0.35rem;">
                    👍 Customers Like
                </span>
                <span class="sentiment-pill pill-positive">{like_count_label}</span>
            </div>
            """, unsafe_allow_html=True)

            if likes:
                for item in likes:
                    clean_snippet = item['snippet'].strip(' ,.-')
                    st.markdown(f"""
                    <div class="aspect-card like">
                        <div class="aspect-name-like">
                            <span>✓</span> {item['aspect']}
                        </div>
                        <div class="aspect-snippet">
                            "{clean_snippet}"
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="aspect-card" style="border: 1px dashed #D6CEC2; text-align: center; color: #A1A1AA; font-style: italic; font-size: 0.84rem;">
                    No specific positive product features highlighted in this review.
                </div>
                """, unsafe_allow_html=True)

        with col_dislike:
            dislike_count_label = f"{len(dislikes)} Feature" if len(dislikes) == 1 else f"{len(dislikes)} Features"
            st.markdown(f"""
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.65rem;">
                <span style="font-size: 0.88rem; font-weight: 700; color: #991B1B; display: flex; align-items: center; gap: 0.35rem;">
                    👎 Customers Dislike
                </span>
                <span class="sentiment-pill pill-negative">{dislike_count_label}</span>
            </div>
            """, unsafe_allow_html=True)

            if dislikes:
                for item in dislikes:
                    clean_snippet = item['snippet'].strip(' ,.-')
                    st.markdown(f"""
                    <div class="aspect-card dislike">
                        <div class="aspect-name-dislike">
                            <span>✕</span> {item['aspect']}
                        </div>
                        <div class="aspect-snippet">
                            "{clean_snippet}"
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="aspect-card" style="border: 1px dashed #D6CEC2; text-align: center; color: #A1A1AA; font-style: italic; font-size: 0.84rem;">
                    No specific negative product features criticized in this review.
                </div>
                """, unsafe_allow_html=True)

        # Executive Summary card
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-card-title">
                📝 Customer Opinion Summary
            </div>
            <div class="summary-card-body">
                {summary_text}
            </div>
        </div>
        """, unsafe_allow_html=True)

# =============================================
# 🔹 SECTION 2: DEMO PRODUCT REVIEWS (ONE-CLICK)
# =============================================
st.markdown("<hr style='margin: 1.8rem 0; border: none; border-top: 1px solid #ECE7DF;'>", unsafe_allow_html=True)

st.markdown("""
<div class="hopup-card-title">📌 One-Click Demo Product Reviews</div>
<div class="hopup-card-subtitle">
    Load a pre-curated collection of 24 realistic smartphone, laptop, audio, and gadget reviews across balanced Positive, Neutral, and Negative sentiments.
</div>
""", unsafe_allow_html=True)

col_demo_btn, col_demo_badges = st.columns([1.5, 3])
with col_demo_btn:
    demo_btn_clicked = st.button("⚡ Load 24 Demo Reviews", type="secondary", use_container_width=True)
with col_demo_badges:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.5rem; height: 100%; padding-top: 0.2rem; flex-wrap: wrap;">
        <span class="sentiment-pill pill-positive">8 Positive</span>
        <span class="sentiment-pill pill-neutral">8 Neutral</span>
        <span class="sentiment-pill pill-negative">8 Negative</span>
        <span class="sentiment-pill pill-all">24 Total Gadget Reviews</span>
    </div>
    """, unsafe_allow_html=True)

if demo_btn_clicked:
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
    st.session_state.pop("batch_cache", None)
    st.success(f"Loaded {len(df)} realistic product reviews across Positive, Neutral, and Negative sentiments!")

# =============================================
# 🔹 SECTION 3: BATCH CSV UPLOAD & CONFIGURATION
# =============================================
st.markdown("<hr style='margin: 1.8rem 0; border: none; border-top: 1px solid #ECE7DF;'>", unsafe_allow_html=True)

st.markdown("""
<div class="hopup-card-title">📂 Batch Review Analysis (CSV Upload)</div>
<div class="hopup-card-subtitle">Upload your custom e-commerce review dataset in CSV format (or test with <span style="background-color: #FFF4ED; color: #EA580C; padding: 0.15rem 0.45rem; border-radius: 5px; font-weight: 600; border: 1px solid #FED7AA; font-size: 0.85rem;">Data/Clean/product_reviews.csv</span>).</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a CSV file containing reviews",
    type=["csv"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        if "data" not in st.session_state or not uploaded_df.equals(st.session_state.get("data")):
            st.session_state["data"] = uploaded_df
            st.session_state.pop("batch_cache", None)
            st.success(f"Uploaded dataset successfully with {len(uploaded_df):,} rows!")
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")

# =============================================
# 🔹 SECTION 4 & 5: COLUMN SELECTION & BATCH INFERENCE
# =============================================
if "data" in st.session_state and st.session_state["data"] is not None:
    raw_df = st.session_state["data"]

    st.markdown("""
    <div style="font-size: 0.98rem; font-weight: 700; color: #18181B; margin: 1.2rem 0 0.4rem 0;">
        ⚙️ Batch Analysis Configuration
    </div>
    """, unsafe_allow_html=True)

    col_select, col_slider = st.columns([1, 1])

    # Smart candidate column auto-selection
    default_idx = 0
    candidate_cols = ["review", "comment", "text", "review_text", "Review", "Comment", "Text", "feedback"]
    for i, col in enumerate(raw_df.columns):
        if col in candidate_cols:
            default_idx = i
            break

    with col_select:
        text_column = st.selectbox(
            "Select Review Column",
            raw_df.columns,
            index=default_idx,
            help="Choose the column containing the product review text"
        )

    # Clean text column
    working_df = raw_df.copy()
    working_df = working_df.dropna(subset=[text_column])
    working_df[text_column] = working_df[text_column].astype(str)

    with col_slider:
        max_rows = len(working_df)
        rows = st.slider(
            "Number of Reviews to Analyze",
            min_value=1,
            max_value=max_rows,
            value=min(500, max_rows),
            help="Limit rows to speed up inference"
        )

    # Smart inference caching to avoid re-running 500 predictions on filter change
    cache_key = f"{id(raw_df)}_{text_column}_{rows}"
    if "batch_cache" in st.session_state and st.session_state["batch_cache"].get("key") == cache_key:
        working_df = st.session_state["batch_cache"]["df"]
    else:
        with st.spinner(f"Running RoBERTa transformer inference on {rows} reviews..."):
            working_slice = working_df.head(rows).copy()
            results = classifier(
                list(working_slice[text_column]),
                batch_size=16,
                truncation=True,
                max_length=MAX_TOKENS
            )
            working_slice["prediction"] = [normalize_label(r["label"]) for r in results]
            working_slice["score"] = [round(float(r["score"]), 4) for r in results]
            working_df = working_slice
            st.session_state["batch_cache"] = {"key": cache_key, "df": working_df}

    st.success(f"Completed sentiment analysis on {len(working_df):,} product reviews with RoBERTa Transformer!")

    # =============================================
    # 🔹 SECTION 6: ANALYTICS DASHBOARD (4 METRICS + 5 CHARTS)
    # =============================================
    st.markdown("<hr style='margin: 1.8rem 0; border: none; border-top: 1px solid #ECE7DF;'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="hopup-card-title">📊 Sentiment Analytics Dashboard</div>
    <div class="hopup-card-subtitle">Key sentiment distribution metrics and statistical visualizations across the analyzed dataset.</div>
    """, unsafe_allow_html=True)

    # 4 Compact Metric Cards
    total_reviews = len(working_df)
    pos_count = int((working_df["prediction"] == "POSITIVE").sum())
    neu_count = int((working_df["prediction"] == "NEUTRAL").sum())
    neg_count = int((working_df["prediction"] == "NEGATIVE").sum())

    pos_pct = (pos_count / total_reviews * 100) if total_reviews > 0 else 0
    neu_pct = (neu_count / total_reviews * 100) if total_reviews > 0 else 0
    neg_pct = (neg_count / total_reviews * 100) if total_reviews > 0 else 0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Total Reviews</div>
            <div class="stat-value">{total_reviews:,}</div>
            <div class="stat-desc">Processed reviews</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="stat-card" style="border-left: 4px solid #10B981;">
            <div class="stat-label" style="color: #059669 !important;">Positive Share</div>
            <div class="stat-value" style="color: #059669 !important;">{pos_pct:.1f}%</div>
            <div class="stat-desc">{pos_count:,} positive reviews</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="stat-card" style="border-left: 4px solid #64748B;">
            <div class="stat-label" style="color: #475569 !important;">Neutral Share</div>
            <div class="stat-value" style="color: #334155 !important;">{neu_pct:.1f}%</div>
            <div class="stat-desc">{neu_count:,} neutral reviews</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="stat-card" style="border-left: 4px solid #EF4444;">
            <div class="stat-label" style="color: #DC2626 !important;">Negative Share</div>
            <div class="stat-value" style="color: #DC2626 !important;">{neg_pct:.1f}%</div>
            <div class="stat-desc">{neg_count:,} negative reviews</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # Color Palette for Visualizations
    chart_colors = {
        "POSITIVE": "#10B981",  # Emerald Green
        "NEUTRAL": "#64748B",   # Slate Gray
        "NEGATIVE": "#EF4444"   # Coral Rose Red
    }

    # -------- First Row (3 charts) --------
    c1, c2, c3 = st.columns(3)

    # 1️⃣ Bar Chart: Sentiment Count
    with c1:
        st.markdown("<div style='font-weight: 700; font-size: 0.95rem; color: #18181B; margin-bottom: 0.35rem;'>1. Sentiment Count</div>", unsafe_allow_html=True)
        counts = working_df["prediction"].value_counts().reindex(["POSITIVE", "NEUTRAL", "NEGATIVE"], fill_value=0)
        
        fig_bar, ax_bar = plt.subplots(figsize=(4.2, 3.2), facecolor="#FFFFFF")
        ax_bar.set_facecolor("#FFFFFF")
        bar_colors = [chart_colors.get(lbl, "#64748B") for lbl in counts.index]
        bars = ax_bar.bar(counts.index, counts.values, color=bar_colors, edgecolor="#FFFFFF", width=0.52, linewidth=1.5)
        
        # Add value numbers above bars
        for bar in bars:
            height = bar.get_height()
            ax_bar.annotate(f"{int(height)}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#27272A')
        
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.spines["left"].set_color("#E5E0D8")
        ax_bar.spines["bottom"].set_color("#E5E0D8")
        ax_bar.tick_params(colors="#52525B", labelsize=8.5)
        ax_bar.grid(axis="y", linestyle="--", alpha=0.3, color="#D4D4D8")
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    # 2️⃣ Pie Chart: Sentiment Distribution
    with c2:
        st.markdown("<div style='font-weight: 700; font-size: 0.95rem; color: #18181B; margin-bottom: 0.35rem;'>2. Sentiment Distribution</div>", unsafe_allow_html=True)
        pred_counts = working_df["prediction"].value_counts()
        pred_counts = pred_counts[pred_counts > 0]
        
        fig_pie, ax_pie = plt.subplots(figsize=(4.2, 3.2), facecolor="#FFFFFF")
        ax_pie.set_facecolor("#FFFFFF")
        pie_colors = [chart_colors.get(lbl, "#64748B") for lbl in pred_counts.index]
        
        wedges, texts, autotexts = ax_pie.pie(
            pred_counts,
            labels=pred_counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=pie_colors,
            pctdistance=0.72,
            wedgeprops={"edgecolor": "#FFFFFF", "linewidth": 2}
        )
        for text in texts:
            text.set_color("#27272A")
            text.set_fontsize(8.5)
            text.set_fontweight("500")
        for autotext in autotexts:
            autotext.set_color("#FFFFFF")
            autotext.set_fontweight("bold")
            autotext.set_fontsize(8.5)
        ax_pie.axis("equal")
        plt.tight_layout()
        st.pyplot(fig_pie)
        plt.close(fig_pie)

    # 3️⃣ Histogram: Confidence Scores
    with c3:
        st.markdown("<div style='font-weight: 700; font-size: 0.95rem; color: #18181B; margin-bottom: 0.35rem;'>3. Confidence Histogram</div>", unsafe_allow_html=True)
        fig_hist, ax_hist = plt.subplots(figsize=(4.2, 3.2), facecolor="#FFFFFF")
        ax_hist.set_facecolor("#FFFFFF")
        ax_hist.hist(working_df["score"], bins=16, color="#FF5A1F", edgecolor="#FFFFFF", linewidth=1.2, alpha=0.9)
        ax_hist.spines["top"].set_visible(False)
        ax_hist.spines["right"].set_visible(False)
        ax_hist.spines["left"].set_color("#E5E0D8")
        ax_hist.spines["bottom"].set_color("#E5E0D8")
        ax_hist.tick_params(colors="#52525B", labelsize=8.5)
        ax_hist.set_xlabel("Confidence Score", color="#71717A", fontsize=8.5, fontweight="500")
        ax_hist.set_ylabel("Frequency", color="#71717A", fontsize=8.5, fontweight="500")
        ax_hist.grid(axis="y", linestyle="--", alpha=0.3, color="#D4D4D8")
        plt.tight_layout()
        st.pyplot(fig_hist)
        plt.close(fig_hist)

    # -------- Second Row (2 charts) --------
    c4, c5 = st.columns(2)

    # 4️⃣ Box Plot: Score by Sentiment
    with c4:
        st.markdown("<div style='font-weight: 700; font-size: 0.95rem; color: #18181B; margin-bottom: 0.35rem;'>4. Confidence by Sentiment Category</div>", unsafe_allow_html=True)
        fig_box, ax_box = plt.subplots(figsize=(5.2, 3.2), facecolor="#FFFFFF")
        ax_box.set_facecolor("#FFFFFF")
        working_df.boxplot(
            column="score",
            by="prediction",
            ax=ax_box,
            grid=False,
            patch_artist=True,
            boxprops=dict(facecolor="#FFF4ED", color="#FF5A1F", linewidth=1.5),
            medianprops=dict(color="#EA580C", linewidth=2.2),
            whiskerprops=dict(color="#A1A1AA", linewidth=1.2),
            capprops=dict(color="#A1A1AA", linewidth=1.2),
            flierprops=dict(marker='o', markerfacecolor='#FF5A1F', markersize=4, linestyle='none', markeredgecolor='none')
        )
        ax_box.set_title("")
        ax_box.spines["top"].set_visible(False)
        ax_box.spines["right"].set_visible(False)
        ax_box.spines["left"].set_color("#E5E0D8")
        ax_box.spines["bottom"].set_color("#E5E0D8")
        ax_box.tick_params(colors="#52525B", labelsize=8.5)
        ax_box.set_xlabel("Sentiment Category", color="#71717A", fontsize=8.5, fontweight="500")
        ax_box.set_ylabel("Confidence Score", color="#71717A", fontsize=8.5, fontweight="500")
        ax_box.grid(axis="y", linestyle="--", alpha=0.3, color="#D4D4D8")
        plt.suptitle("")
        plt.tight_layout()
        st.pyplot(fig_box)
        plt.close(fig_box)

    # 5️⃣ Line Chart: Sentiment Polarity Trend
    with c5:
        st.markdown("<div style='font-weight: 700; font-size: 0.95rem; color: #18181B; margin-bottom: 0.15rem;'>5. Sentiment Polarity Trend</div>", unsafe_allow_html=True)
        st.caption("Polarity mapping: Positive (+1), Neutral (0), Negative (-1)")
        trend_series = working_df["prediction"].map({"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1})
        st.line_chart(trend_series, color="#FF5A1F", height=230)

    # =============================================
    # 🔹 SECTION 7: DATA PREVIEW & CSV DOWNLOAD
    # =============================================
    st.markdown("<hr style='margin: 1.8rem 0; border: none; border-top: 1px solid #ECE7DF;'>", unsafe_allow_html=True)

    col_prev_head, col_dl = st.columns([2.5, 1.5])
    with col_prev_head:
        st.markdown("""
        <div class="hopup-card-title">📄 Analyzed Reviews Dataset</div>
        <div class="hopup-card-subtitle">Showing preview with predicted sentiment labels and confidence scores.</div>
        """, unsafe_allow_html=True)
    with col_dl:
        csv_bytes = working_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Analyzed CSV",
            data=csv_bytes,
            file_name="reviewsense_analyzed_reviews.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download complete dataset containing original reviews, predicted sentiment, and confidence scores."
        )

    st.dataframe(working_df.head(10), use_container_width=True)

    # =============================================
    # 🔹 SECTION 8: FILTER REVIEWS
    # =============================================
    st.markdown("<hr style='margin: 1.6rem 0; border: none; border-top: 1px solid #ECE7DF;'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="hopup-card-title">🔍 Filter Analyzed Reviews</div>
    <div class="hopup-card-subtitle">Inspect subsets of reviews filtered by classification category.</div>
    """, unsafe_allow_html=True)

    col_filt, _ = st.columns([1, 2])
    with col_filt:
        sentiment_filter = st.selectbox(
            "Filter reviews by sentiment category:",
            ["ALL", "POSITIVE", "NEUTRAL", "NEGATIVE"],
            label_visibility="collapsed"
        )

    if sentiment_filter == "ALL":
        filtered_df = working_df
    else:
        filtered_df = working_df[working_df["prediction"] == sentiment_filter]

    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.6rem;">
        <span style="font-size: 0.88rem; color: #71717A; font-weight: 500;">
            Displaying <strong style="color: #18181B;">{len(filtered_df):,}</strong> of <strong style="color: #18181B;">{len(working_df):,}</strong> reviews
        </span>
        <span class="sentiment-pill pill-{sentiment_filter.lower()}">
            {sentiment_filter}
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(filtered_df, use_container_width=True)

# Footer
st.markdown("<div style='height: 2.5rem;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; font-size: 0.8rem; color: #A1A1AA; border-top: 1px solid #ECE7DF; padding-top: 1.25rem;">
    ReviewSense AI • Transformer-Based Sentiment Intelligence
</div>
""", unsafe_allow_html=True)

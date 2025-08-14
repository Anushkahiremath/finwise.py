# ai_audiostudio_neon_emotional.py
# -- coding: utf-8 --
"""
AI Audio Studio — Neon UI with emotional TTS + sentiment analysis
Enhanced version with emotion detection and emotional voice synthesis.
"""

import sys, subprocess, importlib

def ensure(pkg, spec=None):
    try:
        importlib.import_module(pkg)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", spec or pkg, "--upgrade"])

# Auto-install additional dependencies
ensure("streamlit")
ensure("requests")
ensure("gtts")
ensure("spacy")
ensure("textblob")
ensure("pydub")

# SpaCy needs a language model
try:
    import spacy
    spacy.load("en_core_web_sm")
except (ImportError, OSError):
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

# Imports
import os
import io
import time
import tempfile
import html
import difflib
import streamlit as st
import requests
from gtts import gTTS
from textblob import TextBlob
from pydub import AudioSegment
from pydub.playback import play
import numpy as np
from scipy.io import wavfile
# ai_audiostudio_neon.py
# -*- coding: utf-8 -*-
"""
AI Audio Studio — Enhanced with Dynamic Tone, Advanced TTS, and Fixed Reruns
"""

import sys, subprocess, importlib
import numpy as np
import soundfile as sf
import torch
from gtts import gTTS
from textblob import TextBlob
from diff_match_patch import diff_match_patch

def ensure(pkg, spec=None):
    try:
        importlib.import_module(pkg)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", spec or pkg, "--upgrade"])

# Auto-install light deps
ensure("streamlit")
ensure("requests")
ensure("gtts")
ensure("spacy")
ensure("textblob")
ensure("diff-match-patch")
ensure("numpy")
ensure("soundfile")
ensure("transformers")
ensure("accelerate")
ensure("torch")

# SpaCy needs a language model.
try:
    import spacy
    spacy.load("en_core_web_sm")
except (ImportError, OSError):
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm")

# Imports
import os
import io
import time
import html
import streamlit as st
import requests

# -------------------- CONFIG --------------------
HF_TOKEN = "hf_WXFItHOOaPsyRwHkCRDNoRLwPVE"
GRANITE_MODEL = "ibm-granite/granite-3.3-2b-instruct"
HF_TTS_MODELS = {
    "gTTS (free)": None,
    "VITS (espnet) - Default": "espnet/kan-bayashi_ljspeech_vits",
    "Microsoft SpeechT5 TTS": "microsoft/speecht5_tts",
    "OpenVoice (Experimental)": "suno/bark-small",
}

# Initialize session state variables at the beginning of the script
if 'main_text_area' not in st.session_state:
    st.session_state.main_text_area = ""
if 'tone_input' not in st.session_state:
    st.session_state.tone_input = "Neutral"
if 'rewritten_text' not in st.session_state:
    st.session_state.rewritten_text = ""
if 'tts_model_key' not in st.session_state:
    st.session_state.tts_model_key = "VITS (espnet) - Default"

# -------------------- PAGE STYLING --------------------
st.set_page_config(page_title="Finwise AI Audio Studio", page_icon="🎧", layout="wide")
st.markdown(
    """
    <style>
    body, .stApp {
      background: radial-gradient(800px 400px at 10% -10%, #0ea5e9 0%, transparent 30%),
                  radial-gradient(600px 300px at 120% 20%, #7c3aed 0%, transparent 35%),
                  linear-gradient(120deg,#020617 0%, #071029 100%);
      color: #e6eef8;
      min-height: 100vh;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .neon-title { font-size: 48px; font-weight: 800; color: #38bdf8; margin-bottom:0; letter-spacing:1px;
      text-shadow: 0 8px 40px rgba(124,58,237,0.4), 0 4px 20px rgba(14,165,233,0.3); }
    .neon-sub { color:#94a3b8; font-size: 16px; margin-top:8px; margin-bottom:20px; }
    .glass { background: rgba(255,255,255,0.06); border-radius:24px; padding:24px;
             border:1px solid rgba(255,255,255,0.1); box-shadow: 0 16px 60px rgba(2,6,23,0.7); }
    .panel-title { font-weight:800; color:#e6eef8; font-size:16px; margin-bottom:12px; }
    .orig { border-left: 4px solid #0ea5e9; padding-left:16px; }
    .rewr { border-left: 4px solid #a855f7; padding-left:16px; }
    .ins { background: rgba(52,211,153,0.2); color:#bbf7d0; padding:3px 6px; border-radius:6px; font-weight:600; }
    .del { background: rgba(248,113,113,0.15); color:#fecaca; text-decoration: line-through; padding:3px 6px; border-radius:6px; }
    @keyframes floatIn { from { transform: translateY(20px); opacity:0 } to { transform: translateY(0); opacity:1 } }
    .glass { animation: floatIn 800ms ease both; }
    .stButton>button, .stDownloadButton>button {
        border-radius:12px;
        padding:12px 20px;
        font-weight:700;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        color: #e2e8f0;
        transition: all 0.3s;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background: rgba(255,255,255,0.15);
        transform: translateY(-2px);
    }
    .hint { color:#94a3b8; font-size:14px; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="neon-title">🎧 Neon AI Audio Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="neon-sub">Multilingual TTS • Dynamic Tone Control • Advanced Models</div>', unsafe_allow_html=True)

# -------------------- SIDEBAR CONTROLS --------------------
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    
    # Text rewriting is now on by default
    st.markdown("### Text Rewriting with IBM Granite")
    do_rewrite = st.checkbox("Enable/Disable Rewrite", value=True, help="Uses IBM Granite via Hugging Face Inference.")
    st.markdown("---")
    
    st.markdown("### 🌍 Language for TTS")
    lang_options = [
        ("en", "English"), ("hi", "Hindi"), ("es", "Spanish"), ("fr", "French"),
        ("de", "German"), ("kn", "Kannada"), ("ja", "Japanese"), ("pt", "Portuguese"),
        ("zh", "Chinese"), ("ko", "Korean"), ("ru", "Russian")
    ]
    lang = st.selectbox("Language", lang_options, format_func=lambda x: x[1])[0]
    st.markdown("---")
    
    st.markdown("### 🎙️ TTS Engine")
    tts_backend_name = st.selectbox("Engine", list(HF_TTS_MODELS.keys()), key='tts_model_key')
    
    st.markdown('<div class="hint"></div>', unsafe_allow_html=True)

# -------------------- INPUT AREA --------------------
st.markdown('<div class="glass">', unsafe_allow_html=True)
colA, colB = st.columns([2,1])
with colA:
    uploaded = st.file_uploader("Upload .txt file (optional)", type=["txt"])
    txt = st.text_area("Or paste your text here", height=260, placeholder="Paste or write your narration or paragraph(s)...", key="main_text_area")
with colB:
    st.markdown("### Dynamic Tone")
    tone = st.text_input("Enter a tone for rewriting:", key="tone_input")
    st.markdown('<div class="hint">Try "sarcastic", "professional", or "melancholy".</div>', unsafe_allow_html=True)
    st.markdown("### Load Examples")
    
    def set_example_text(text_key, tone_key):
        st.session_state.main_text_area = text_key
        st.session_state.tone_input = tone_key

    st.button("🌟 Inspiring", on_click=set_example_text, args=("The future belongs to those who believe in the beauty of their dreams. Rise up and make a difference.", "Inspiring"))
    st.button("😄 Happy", on_click=set_example_text, args=("What a beautiful day! The sun is shining, and I feel so joyful and full of energy.", "Happy"))
    st.button("😔 Sad", on_click=set_example_text, args=("I am feeling a little down today, thinking about the past and how things have changed.", "Sad"))
    st.button("🗣️ Formal", on_click=set_example_text, args=("We are pleased to inform you that your application has been approved. Further details will be provided shortly.", "Formal"))
    
    st.markdown("")
st.markdown("</div>", unsafe_allow_html=True)

# grab content
original_text = txt.strip()
current_tone = tone.strip() if tone else "Neutral"

# -------------------- HELPERS --------------------
HF_API = "https://api-inference.huggingface.co/models/"
max_new_tokens = 300

@st.cache_data(show_spinner=False)
def hf_text_generate_cached(model, prompt, max_tokens, token_val):
    if not token_val or token_val.startswith("hf_xxxxxxxxx"):
        raise RuntimeError("HF_TOKEN not configured.")
    
    headers = {"Authorization": f"Bearer {token_val}", "Accept":"application/json"}
    payload = {"inputs": prompt, "parameters":{"max_new_tokens": max_tokens, "temperature":0.7, "top_p":0.9}, "options":{"wait_for_model":True}}
    resp = requests.post(HF_API + model, headers=headers, json=payload, timeout=180)
    
    if resp.status_code != 200:
        raise RuntimeError(f"HF text generation failed [{resp.status_code}]: {resp.text}")
    try:
        data = resp.json()
        if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        return resp.text
    except Exception:
        return resp.text

def hf_tts_inference(model, text, token_val):
    if not token_val or token_val.startswith("hf_xxxxxxxxx"):
        raise RuntimeError("HF_TOKEN not configured.")
    headers = {"Authorization": f"Bearer {token_val}", "Accept":"audio/wav,audio/mpeg"}
    payload = {"inputs": text, "options":{"wait_for_model":True}}
    resp = requests.post(HF_API + model, headers=headers, json=payload, timeout=180, stream=True)
    
    if resp.status_code != 200:
        raise RuntimeError(f"HF TTS failed [{resp.status_code}]: {resp.text}")
    return resp.content, resp.headers.get("content-type","audio/mpeg")

def gtts_bytes(text, lang):
    tts = gTTS(text=text, lang=lang, slow=False)
    bio = io.BytesIO(); tts.write_to_fp(bio); bio.seek(0)
    return bio.read(), "audio/mpeg"

def render_diff_html(a, b):
    dmp = diff_match_patch()
    diffs = dmp.diff_main(a, b)
    dmp.diff_cleanupSemantic(diffs)
    
    a_html, b_html = [], []
    for op, text in diffs:
        if op == 0:
            a_html.append(html.escape(text))
            b_html.append(html.escape(text))
        elif op == -1:
            a_html.append(f'<span class="del">{html.escape(text)}</span>')
        elif op == 1:
            b_html.append(f'<span class="ins">{html.escape(text)}</span>')
    return "".join(a_html), "".join(b_html)

# -------------------- Automatic Rewrite Logic --------------------
# We will only rewrite if the original text is not empty and rewrite is enabled
if do_rewrite and original_text:
    with st.spinner(f"Polishing text with a {current_tone} tone..."):
        prompt = (f"Rewrite the following text in a {current_tone} tone. Preserve the core meaning but improve clarity and flow.\n\nText:\n{original_text}\n\nRewritten:")
        try:
            rewritten_text = hf_text_generate_cached(GRANITE_MODEL, prompt, max_new_tokens=max_new_tokens, token_val=HF_TOKEN)
            if "Rewritten:" in rewritten_text:
                rewritten_text = rewritten_text.split("Rewritten:",1)[-1].strip()
            st.session_state['rewritten_text'] = rewritten_text
        except Exception as e:
            st.error(f"Rewrite failed: {e}")
            st.session_state['rewritten_text'] = original_text
else:
    st.session_state['rewritten_text'] = original_text

rewritten_text = st.session_state.get('rewritten_text', original_text)

# -------------------- UI Buttons: Rewrite & Generate --------------------
col1, col2 = st.columns([1,1])
generate_audio_btn = col1.button("🔊 Generate Audio (TTS)", help="Generates audio from the rewritten text.")
def clear_all():
    st.session_state.main_text_area = ""
    st.session_state.rewritten_text = ""
    st.session_state.tone_input = "Neutral"
clear_btn = col2.button("🧹 Clear", on_click=clear_all, help="Clears all text and selections.")

# -------------------- Side-by-side display --------------------
st.markdown("<br/>", unsafe_allow_html=True)
left_col, right_col = st.columns(2)

left_col.markdown(f'<div class="glass"><div class="panel-title orig">Original</div><div style="white-space:pre-wrap;">{html.escape(original_text)}</div></div>', unsafe_allow_html=True)

_a_html, _b_html = render_diff_html(original_text or "", rewritten_text or "")
right_col.markdown(f'<div class="glass"><div class="panel-title rewr">Rewritten ({current_tone})</div><div style="white-space:pre-wrap;">{_b_html}</div></div>', unsafe_allow_html=True)

st.markdown('<div style="display:flex;gap:16px;margin-top:16px;color:#cbd5e1;font-size:14px;align-items:center;">'
            '<b>Legend:</b> <span class="ins">Inserted</span> <span class="del">Deleted</span> '
            '<span><b>Tip:</b> You can edit the rewritten text below before generating audio.</span>'
            '</div>', unsafe_allow_html=True)

edited_text_for_tts = st.text_area("Edit rewritten text (final text used for TTS)", value=rewritten_text or original_text, height=200, key="edited_text_area")

# -------------------- Generate Audio --------------------
if generate_audio_btn:
    final_text = (edited_text_for_tts or original_text).strip()
    if not final_text:
        st.warning("No text to synthesize.")
        st.stop()
    
    with st.spinner("Generating audio..."):
        try:
            audio_bytes = None
            ctype = None
            ext = "mp3"
            model_name = HF_TTS_MODELS.get(st.session_state.tts_model_key)
            
            if st.session_state.tts_model_key == "gTTS (free)":
                audio_bytes, ctype = gtts_bytes(final_text, lang)
            else:
                audio_bytes, ctype = hf_tts_inference(model_name, final_text, HF_TOKEN)

            if audio_bytes:
                st.success("Audio generated ✅")
                st.audio(audio_bytes, format=ctype)
                st.download_button(
                    "⬇ Download Audio", 
                    data=audio_bytes, 
                    file_name=f"neon_audio_{int(time.time())}.{ext}", 
                    mime=ctype
                )
        except Exception as e:
            st.error(f"Audio generation failed: {e}")

# -------------------- Footer / Tips --------------------
st.markdown("<br/><div class='glass'><b>Tips:</b> For long books, split by paragraph. Try different HF models if one fails. Keep HF_TOKEN secret.</div>", unsafe_allow_html=True)
# -------------------- CONFIG --------------------
HF_TOKEN = "hf_WXFItHOOaPsyRwHkCRDNoRLwPVE"
GRANITE_MODEL = "ibm-granite/granite-3.3-2b-instruct"
HF_TTS_MODEL_DEFAULT = "microsoft/speecht5_tts"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# Emotion mapping for TTS parameters
EMOTION_PARAMS = {
    'anger': {'rate': 165, 'pitch': 75, 'volume': 1.2},
    'fear': {'rate': 145, 'pitch': 85, 'volume': 0.9},
    'joy': {'rate': 155, 'pitch': 110, 'volume': 1.1},
    'neutral': {'rate': 150, 'pitch': 100, 'volume': 1.0},
    'sadness': {'rate': 130, 'pitch': 85, 'volume': 0.8},
    'surprise': {'rate': 170, 'pitch': 105, 'volume': 1.3},
    'disgust': {'rate': 140, 'pitch': 80, 'volume': 1.0}
}

# -------------------- PAGE STYLING --------------------
st.set_page_config(page_title="FinWise AI Audio Studio", page_icon="🎧", layout="wide")
st.markdown(
    """
    <style>
    /* background gradient animated */
    body, .stApp {
      background: radial-gradient(800px 400px at 10% -10%, #0ea5e9 0%, transparent 30%),
                  radial-gradient(600px 300px at 120% 20%, #7c3aed 0%, transparent 35%),
                  linear-gradient(120deg,#020617 0%, #071029 100%);
      color: #e6eef8;
      min-height: 100vh;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* header */
    .neon-title { font-size: 48px; font-weight: 800; color: #38bdf8; margin-bottom:0; letter-spacing:1px;
      text-shadow: 0 8px 40px rgba(124,58,237,0.4), 0 4px 20px rgba(14,165,233,0.3); }
    .neon-sub { color:#94a3b8; font-size: 16px; margin-top:8px; margin-bottom:20px; }
    /* glass card */
    .glass { background: rgba(255,255,255,0.06); border-radius:24px; padding:24px;
             border:1px solid rgba(255,255,255,0.1); box-shadow: 0 16px 60px rgba(2,6,23,0.7); }
    /* side-by-side panels */
    .panel-title { font-weight:800; color:#e6eef8; font-size:16px; margin-bottom:12px; }
    .orig { border-left: 4px solid #0ea5e9; padding-left:16px; }
    .rewr { border-left: 4px solid #a855f7; padding-left:16px; }
    /* diff highlights */
    .ins { background: rgba(52,211,153,0.2); color:#bbf7d0; padding:3px 6px; border-radius:6px; font-weight:600; }
    .del { background: rgba(248,113,113,0.15); color:#fecaca; text-decoration: line-through; padding:3px 6px; border-radius:6px; }
    /* animated entrance */
    @keyframes floatIn { from { transform: translateY(20px); opacity:0 } to { transform: translateY(0); opacity:1 } }
    .glass { animation: floatIn 800ms ease both; }
    /* buttons */
    .stButton>button, .stDownloadButton>button {
        border-radius:12px;
        padding:12px 20px;
        font-weight:700;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        color: #e2e8f0;
        transition: all 0.3s;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background: rgba(255,255,255,0.15);
        transform: translateY(-2px);
    }
    .hint { color:#94a3b8; font-size:14px; }
    /* emotion tags */
    .emotion-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 14px;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .anger { background: rgba(239,68,68,0.2); color: #fca5a5; border: 1px solid rgba(239,68,68,0.3); }
    .fear { background: rgba(167,139,250,0.2); color: #c4b5fd; border: 1px solid rgba(167,139,250,0.3); }
    .joy { background: rgba(52,211,153,0.2); color: #6ee7b7; border: 1px solid rgba(52,211,153,0.3); }
    .neutral { background: rgba(148,163,184,0.2); color: #cbd5e1; border: 1px solid rgba(148,163,184,0.3); }
    .sadness { background: rgba(99,102,241,0.2); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.3); }
    .surprise { background: rgba(249,168,212,0.2); color: #fbcfe8; border: 1px solid rgba(249,168,212,0.3); }
    .disgust { background: rgba(217,119,6,0.2); color: #fdba74; border: 1px solid rgba(217,119,6,0.3); }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="neon-title">🎧 FinWise AI Audio Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="neon-sub">Emotional TTS • Sentiment Analysis • Tone Rewriting • Side-by-side Diff</div>', unsafe_allow_html=True)

# -------------------- SIDEBAR CONTROLS --------------------
with st.sidebar:
    st.markdown("## ⚙ Controls")
    
    do_rewrite = st.checkbox("Rewrite text with IBM Granite", value=False, help="Uses IBM Granite via Hugging Face Inference (backend token required).")
    st.markdown("---")
    
    st.markdown("### 🌍 Language for TTS")
    lang_options = [
        ("en", "English"), ("hi", "Hindi"), ("es", "Spanish"), ("fr", "French"),
        ("de", "German"), ("kn", "Kannada"), ("ja", "Japanese"), ("pt", "Portuguese"),
        ("zh", "Chinese"), ("ko", "Korean"), ("ru", "Russian")
    ]
    lang = st.selectbox("Language", lang_options, format_func=lambda x: x[1])[0]
    st.markdown("---")
    
    st.markdown("### 🎙 TTS Engine")
    tts_backend = st.selectbox("Engine", ["gTTS (free)", "Hugging Face Inference (realistic)"], index=0)
    st.markdown("HF TTS model (if selected):")
    hf_tts_model = st.selectbox("HF TTS model", [
        HF_TTS_MODEL_DEFAULT,
        "espnet/kan-bayashi_ljspeech_vits",
        "facebook/mms-tts-eng",
        "facebook/mms-tts-spa"
    ])
    
    st.markdown("### 😊 Emotion Settings")
    auto_detect_emotion = st.checkbox("Auto-detect emotion from text", value=True)
    manual_emotion = st.selectbox(
        "Or manually select emotion",
        ["anger", "fear", "joy", "neutral", "sadness", "surprise", "disgust"],
        index=3,
        disabled=auto_detect_emotion
    )
    
    st.markdown("---")
    st.markdown('<div class="hint">HF_TOKEN is stored in the backend source (not visible in the UI). If using HF features, make sure your token has model access.</div>', unsafe_allow_html=True)

# -------------------- INPUT AREA --------------------
st.markdown('<div class="glass">', unsafe_allow_html=True)
colA, colB = st.columns([2,1])
with colA:
    uploaded = st.file_uploader("Upload .txt file (optional)", type=["txt"])
    txt = st.text_area("Or paste your text here", height=260, placeholder="Paste or write your narration or paragraph(s)...")
with colB:
    st.markdown("### Load Example Tones")
    if st.button("🌟 Inspiring"):
        st.session_state['txt'] = "The future belongs to those who believe in the beauty of their dreams. Rise up and make a difference."
        st.session_state['tone'] = "Inspiring"
        st.rerun()
    if st.button("😄 Happy"):
        st.session_state['txt'] = "What a beautiful day! The sun is shining, and I feel so joyful and full of energy."
        st.session_state['tone'] = "Happy"
        st.rerun()
    if st.button("😔 Sad"):
        st.session_state['txt'] = "I am feeling a little down today, thinking about the past and how things have changed."
        st.session_state['tone'] = "Sad"
        st.rerun()
    if st.button("🗣 Formal"):
        st.session_state['txt'] = "We are pleased to inform you that your application has been approved. Further details will be provided shortly."
        st.session_state['tone'] = "Formal"
        st.rerun()
    if st.button("😠 Angry"):
        st.session_state['txt'] = "I can't believe this happened! This is completely unacceptable and I demand an immediate explanation!"
        st.session_state['tone'] = "Angry"
        st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

# grab content
if uploaded:
    try:
        original_text = uploaded.read().decode("utf-8", errors="ignore")
    except Exception:
        original_text = txt or st.session_state.get('txt', '')
else:
    original_text = txt or st.session_state.get('txt', '')

original_text = original_text.strip()
tone = st.session_state.get('tone', 'Neutral')

# -------------------- HELPERS --------------------
HF_API = "https://api-inference.huggingface.co/models/"

@st.cache_data(show_spinner=False)
def analyze_emotion(text):
    """Analyze text emotion using Hugging Face model."""
    if not text.strip():
        return {'neutral': 1.0}
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": text}
    
    try:
        response = requests.post(HF_API + EMOTION_MODEL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                # Convert list of dicts to single emotion dict
                emotions = {item['label']: item['score'] for item in result[0]}
                return emotions
    except Exception as e:
        st.error(f"Emotion analysis failed: {str(e)}")
    
    return {'neutral': 1.0}

@st.cache_data(show_spinner=False)
def hf_text_generate(model, prompt, max_tokens):
    if not HF_TOKEN or HF_TOKEN.startswith("hf_xxxxxxxxx"):
        raise RuntimeError("HF_TOKEN not configured in backend source; cannot call HF text generation.")
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Accept":"application/json"}
    payload = {"inputs": prompt, "parameters":{"max_new_tokens": max_tokens, "temperature":0.7, "top_p":0.9}, "options":{"wait_for_model":True}}
    resp = requests.post(HF_API + model, headers=headers, json=payload, timeout=180)
    if resp.status_code != 200:
        raise RuntimeError(f"HF text generation failed [{resp.status_code}]: {resp.text}")
    try:
        data = resp.json()
        if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        if isinstance(data, str):
            return data
        return resp.text
    except Exception:
        return resp.text

@st.cache_data(show_spinner=False)
def hf_tts_inference(model, text, emotion="neutral"):
    """Call HF TTS with emotion parameters."""
    if not HF_TOKEN or HF_TOKEN.startswith("hf_xxxxxxxxx"):
        raise RuntimeError("HF_TOKEN not configured in backend source; cannot call HF TTS.")
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Accept":"audio/wav,audio/mpeg"}
    
    # Apply emotion-specific parameters if available
    params = EMOTION_PARAMS.get(emotion, EMOTION_PARAMS['neutral'])
    
    # Special handling for different TTS models
    if "speecht5" in model.lower():
        # SpeechT5 supports emotional TTS natively
        payload = {
            "inputs": text,
            "parameters": {
                "speaker_embeddings": emotion,
                "speed": params['rate'] / 150,  # Normalize
                "pitch": params['pitch'] / 100,  # Normalize
            },
            "options": {"wait_for_model": True}
        }
    else:
        # Generic TTS with basic parameters
        payload = {"inputs": text, "options": {"wait_for_model": True}}
    
    resp = requests.post(HF_API + model, headers=headers, json=payload, timeout=180, stream=True)
    if resp.status_code != 200:
        raise RuntimeError(f"HF TTS failed [{resp.status_code}]: {resp.text}")
    return resp.content, resp.headers.get("content-type","audio/mpeg")

def gtts_bytes(text, lang, emotion="neutral"):
    """Generate gTTS audio with basic emotion adjustments."""
    tts = gTTS(text=text, lang=lang, slow=False)
    bio = io.BytesIO()
    tts.write_to_fp(bio)
    bio.seek(0)
    
    # Apply basic emotion effects using pydub
    audio = AudioSegment.from_file(bio, format="mp3")
    
    # Get emotion parameters
    params = EMOTION_PARAMS.get(emotion, EMOTION_PARAMS['neutral'])
    
    # Adjust speed (rate)
    speed_factor = params['rate'] / 150  # 150 is neutral rate
    if speed_factor != 1.0:
        audio = audio.speedup(playback_speed=speed_factor)
    
    # Adjust pitch (crude approximation)
    # Note: gTTS doesn't support true pitch control, so we use speedup/slowdown
    # which affects pitch. For better quality, use HF TTS models.
    
    # Adjust volume
    audio = audio + (20 * np.log10(params['volume']))
    
    # Export back to bytes
    output = io.BytesIO()
    audio.export(output, format="mp3")
    output.seek(0)
    
    return output.read(), "audio/mpeg"

def render_diff_html(a, b):
    """Return HTML where deletions from a are marked .del and insertions in b marked .ins."""
    sm = difflib.SequenceMatcher(a=a.split(), b=b.split())
    a_html = []
    b_html = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        a_chunk = " ".join(a.split()[i1:i2])
        b_chunk = " ".join(b.split()[j1:j2])
        if tag == "equal":
            a_html.append(html.escape(a_chunk))
            b_html.append(html.escape(b_chunk))
        elif tag == "replace":
            if a_chunk:
                a_html.append(f'<span class="del">{html.escape(a_chunk)}</span>')
            if b_chunk:
                b_html.append(f'<span class="ins">{html.escape(b_chunk)}</span>')
        elif tag == "delete":
            a_html.append(f'<span class="del">{html.escape(a_chunk)}</span>')
        elif tag == "insert":
            b_html.append(f'<span class="ins">{html.escape(b_chunk)}</span>')
    return (" ".join(a_html) or html.escape(a)), (" ".join(b_html) or html.escape(b))

def display_emotion_results(emotions):
    """Display emotion analysis results with visual tags."""
    st.markdown("### 😊 Detected Emotions")
    
    # Sort emotions by score (descending)
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    
    # Create columns for the top emotions
    cols = st.columns(len(sorted_emotions))
    for idx, (emotion, score) in enumerate(sorted_emotions):
        with cols[idx]:
            # Display emotion tag with score
            st.markdown(
                f'<div class="emotion-tag {emotion}">{emotion.capitalize()}: {score:.0%}</div>',
                unsafe_allow_html=True
            )
    
    # Return the dominant emotion
    return sorted_emotions[0][0]

# -------------------- UI Buttons: Rewrite & Generate --------------------
col1, col2, col3 = st.columns([1,1,1])
do_rewrite_btn = col1.button("🔁 Rewrite & Preview")
generate_audio_btn = col2.button("🔊 Generate Audio (TTS)")
clear_btn = col3.button("🧹 Clear")

if clear_btn:
    st.session_state['txt'] = ""
    st.rerun()

rewritten_text = ""
if do_rewrite_btn:
    if not original_text:
        st.warning("Paste or upload text first.")
    elif not do_rewrite:
        st.info("Rewrite option is disabled in sidebar; enabling it will use IBM Granite to rewrite.")
    else:
        with st.spinner("Polishing text with IBM Granite..."):
            prompt = (f"Rewrite the following text, maintaining its original tone and emotion but improving clarity and flow.\n\nText:\n{original_text}\n\nRewritten:")
            try:
                rewritten_text = hf_text_generate(GRANITE_MODEL, prompt, max_new_tokens=300)
                if "Rewritten:" in rewritten_text:
                    rewritten_text = rewritten_text.split("Rewritten:",1)[-1].strip()
                st.success("Rewrite completed.")
                st.session_state['rewritten_text'] = rewritten_text
            except Exception as e:
                st.error(f"Rewrite failed: {e}")
                rewritten_text = original_text
                st.session_state['rewritten_text'] = rewritten_text

rewritten_text = st.session_state.get('rewritten_text', rewritten_text or original_text)

# -------------------- Side-by-side display --------------------
st.markdown("<br/>", unsafe_allow_html=True)
left_col, right_col = st.columns(2)

left_col.markdown(f'<div class="glass"><div class="panel-title orig">Original</div><div style="white-space:pre-wrap;">{html.escape(original_text)}</div></div>', unsafe_allow_html=True)

_a_html, _b_html = render_diff_html(original_text or "", rewritten_text or "")
right_col.markdown(f'<div class="glass"><div class="panel-title rewr">Rewritten ({tone})</div><div style="white-space:pre-wrap;">{_b_html}</div></div>', unsafe_allow_html=True)

st.markdown('<div style="display:flex;gap:16px;margin-top:16px;color:#cbd5e1;font-size:14px;align-items:center;">'
            '<b>Legend:</b> <span class="ins">Inserted</span> <span class="del">Deleted</span> '
            '<span><b>Tip:</b> You can edit the rewritten text below before generating audio.</span>'
            '</div>', unsafe_allow_html=True)

edited = st.text_area("Edit rewritten text (final text used for TTS)", value=rewritten_text or original_text, height=200, key="edited_text_area")

# -------------------- Emotion Analysis --------------------
if edited and auto_detect_emotion:
    with st.spinner("Analyzing text emotion..."):
        emotions = analyze_emotion(edited)
        dominant_emotion = display_emotion_results(emotions)
else:
    dominant_emotion = manual_emotion

# -------------------- Generate Audio --------------------
if generate_audio_btn:
    final_text = (edited or original_text).strip()
    if not final_text:
        st.warning("No text to synthesize.")
        st.stop()
    
    out_choice = st.selectbox("Output Format", ["MP3 (recommended)", "WAV (raw)"])
    output_format = "mp3" if "MP3" in out_choice else "wav"
    
    with st.spinner(f"Generating {dominant_emotion} audio..."):
        try:
            audio_bytes = None
            ctype = None
            ext = output_format
            
            if tts_backend == "gTTS (free)":
                audio_bytes, ctype = gtts_bytes(final_text, lang, dominant_emotion)
            else:
                audio_bytes, ctype = hf_tts_inference(hf_tts_model, final_text, dominant_emotion)

            if audio_bytes:
                st.success(f"Audio generated with {dominant_emotion} emotion ✅")
                st.audio(audio_bytes, format=ctype)
                
                # Add emotion metadata to filename
                filename = f"neon_audio_{dominant_emotion}_{int(time.time())}.{ext}"
                
                st.download_button(
                    "⬇ Download Audio", 
                    data=audio_bytes, 
                    file_name=filename, 
                    mime=ctype
                )
        except Exception as e:
            st.error(f"Audio generation failed: {e}")

# -------------------- Footer / Tips --------------------
st.markdown("<br/><div class='glass'><b>Tips:</b> For emotional TTS, use Hugging Face models. Longer texts (>30s) may need chunking. Keep HF_TOKEN secret.</div>", unsafe_allow_html=True)
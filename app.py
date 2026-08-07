import streamlit as st
import tempfile
import os
from src.pipeline import run_pipeline
from src.classifier import extract_fields

st.set_page_config(
    page_title="OCR Document Classifier",
    page_icon="📄",
    layout="centered"
)

if 'page' not in st.session_state:
    st.session_state.page = 'upload'
if 'result' not in st.session_state:
    st.session_state.result = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None

st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .step-bar {
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0.4rem 0 0.6rem;
    }
    .step {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 4px;
    }
    .step-circle {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .step-circle.active   { background:#6c63ff; color:white; border:2px solid #6c63ff; }
    .step-circle.done     { background:#6c63ff; color:white; border:2px solid #6c63ff; }
    .step-circle.inactive { background:transparent; color:#888; border:2px solid #444; }
    .step-label { font-size: 0.68rem; font-weight: 500; }
    .step-label.active   { color: #6c63ff; }
    .step-label.done     { color: #6c63ff; }
    .step-label.inactive { color: #888; }
    .step-line {
        flex: 1;
        height: 2px;
        max-width: 70px;
        margin-bottom: 16px;
    }
    .step-line.done     { background: #6c63ff; }
    .step-line.inactive { background: #444; }
    .result-card {
        border: 1px solid #2a2a3e;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        display: flex;
        align-items: center;
        gap: 14px;
        background: #1a1a2e;
        margin-bottom: 1rem;
    }
    .confidence-bar-bg {
        height: 6px;
        background: #2a2a3e;
        border-radius: 3px;
        margin-top: 6px;
        width: 100%;
    }
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #4f8ef7, #a855f7);
        color: white;
        border: none;
        padding: 0.6rem;
        font-size: 0.95rem;
        font-weight: 600;
        border-radius: 10px;
        cursor: pointer;
    }
    div.stButton > button:hover {
        opacity: 0.9;
        color: white;
        border: none;
    }
    /* Hide the default streamlit uploader label gap */
    [data-testid="stFileUploader"] {
        margin-bottom: 0rem;
    }
    </style>
""", unsafe_allow_html=True)


def render_stepper(current_step):
    steps = [("📤", "Upload"), ("⚙️", "Processing"), ("✅", "Results")]
    html = '<div class="step-bar">'
    for i, (icon, label) in enumerate(steps):
        step_num = i + 1
        if step_num < current_step:
            cc, lc, ic = "done", "done", "✓"
        elif step_num == current_step:
            cc, lc, ic = "active", "active", icon
        else:
            cc, lc, ic = "inactive", "inactive", icon
        html += f'<div class="step"><div class="step-circle {cc}">{ic}</div><span class="step-label {lc}">{label}</span></div>'
        if i < len(steps) - 1:
            line = "done" if step_num < current_step else "inactive"
            html += f'<div class="step-line {line}"></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def get_confidence(scores, doc_type):
    total = sum(scores.values())
    if total == 0:
        return 0
    return round((scores.get(doc_type, 0) / total) * 100)


# ─────────────────────────────────────────
#  PAGE 1 — UPLOAD
# ─────────────────────────────────────────
if st.session_state.page == 'upload':

    st.markdown('<div class="title">📄 OCR Document Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload a document image to extract text and identify its type</div>', unsafe_allow_html=True)

    render_stepper(1)

    st.markdown("""
        <div style="text-align:center; margin-bottom:0.6rem;">
            <span style="background:#1e3a5f; color:#4f8ef7; padding:3px 10px; border-radius:20px; font-size:0.72rem; margin:2px; display:inline-block;">🛂 Passport</span>
            <span style="background:#1a3a2a; color:#4ade80; padding:3px 10px; border-radius:20px; font-size:0.72rem; margin:2px; display:inline-block;">📋 Citizenship</span>
            <span style="background:#3a2a1a; color:#fb923c; padding:3px 10px; border-radius:20px; font-size:0.72rem; margin:2px; display:inline-block;">🪪 National ID</span>
            <span style="background:#2a1a3a; color:#c084fc; padding:3px 10px; border-radius:20px; font-size:0.72rem; margin:2px; display:inline-block;">🧾 PAN Card</span>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload document",
        type=["jpg", "jpeg", "png", "jfif", "bmp", "tiff", "webp"],
        label_visibility="collapsed"
    )
    st.markdown('<p style="text-align:center; color:#555; font-size:0.75rem; margin-top:2px; margin-bottom:0.5rem;">Supported: jpg, jpeg, png, jfif, bmp, tiff, webp</p>', unsafe_allow_html=True)

    if uploaded_file is not None:
        if st.button("🔍 Classify Document"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            with st.spinner("🔧 Preprocessing → 🔍 Extracting → 🏷️ Classifying..."):
                result = run_pipeline(tmp_path)
            os.unlink(tmp_path)
            if result:
                st.session_state.result         = result
                st.session_state.uploaded_image = uploaded_file.getvalue()
                st.session_state.file_name      = uploaded_file.name
                st.session_state.page           = 'results'
                st.rerun()
            else:
                st.error("❌ Could not process the image. Please try a clearer image.")


# ─────────────────────────────────────────
#  PAGE 2 — RESULTS
# ─────────────────────────────────────────

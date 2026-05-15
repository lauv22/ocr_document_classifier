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
        margin-bottom: 0.8rem;
    }
    .step-bar {
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0.5rem 0 0.8rem;
    }
    .step {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 4px;
    }
    .step-circle {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .step-circle.active   { background:#6c63ff; color:white; border:2px solid #6c63ff; }
    .step-circle.done     { background:#6c63ff; color:white; border:2px solid #6c63ff; }
    .step-circle.inactive { background:transparent; color:#888; border:2px solid #444; }
    .step-label { font-size: 0.7rem; font-weight: 500; }
    .step-label.active   { color: #6c63ff; }
    .step-label.done     { color: #6c63ff; }
    .step-label.inactive { color: #888; }
    .step-line {
        flex: 1;
        height: 2px;
        max-width: 70px;
        margin-bottom: 18px;
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
    .result-icon {
        width: 48px;
        height: 48px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.6rem;
        flex-shrink: 0;
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
        <div style="text-align:center; margin-bottom:0.8rem;">
            <span style="background:#1e3a5f; color:#4f8ef7; padding:3px 10px; border-radius:20px; font-size:0.75rem; margin:2px; display:inline-block;">🛂 Passport</span>
            <span style="background:#1a3a2a; color:#4ade80; padding:3px 10px; border-radius:20px; font-size:0.75rem; margin:2px; display:inline-block;">📋 Citizenship</span>
            <span style="background:#3a2a1a; color:#fb923c; padding:3px 10px; border-radius:20px; font-size:0.75rem; margin:2px; display:inline-block;">🪪 National ID</span>
            <span style="background:#2a1a3a; color:#c084fc; padding:3px 10px; border-radius:20px; font-size:0.75rem; margin:2px; display:inline-block;">🧾 PAN Card</span>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload document",
        type=["jpg", "jpeg", "png", "jfif", "bmp", "tiff", "webp"],
        label_visibility="collapsed"
    )
    st.markdown('<p style="text-align:center; color:#666; font-size:0.8rem; margin-top:4px;">Supported: jpg, jpeg, png, jfif, bmp, tiff, webp</p>', unsafe_allow_html=True)

    if uploaded_file is not None:
        st.markdown(f"""
            <div style="background:#1a1a2e; border:1px solid #2a2a3e; border-radius:10px; padding:0.8rem 1rem; display:flex; align-items:center; gap:12px; margin:0.8rem 0;">
                <span style="font-size:1.5rem;">🖼️</span>
                <div style="flex:1;">
                    <p style="margin:0; font-size:0.9rem; font-weight:600; color:#fff;">{uploaded_file.name}</p>
                    <p style="margin:0; font-size:0.78rem; color:#888;">{round(uploaded_file.size / 1024, 1)} KB • {uploaded_file.type}</p>
                </div>
                <span style="font-size:1.2rem;">✅</span>
            </div>
        """, unsafe_allow_html=True)

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
elif st.session_state.page == 'results':

    result   = st.session_state.result
    doc_type = result['document_type']
    scores   = result['scores']

    icon_map  = {'Passport':'🛂','Citizenship':'📋','PAN':'🧾','National ID':'🪪','Unknown':'❓'}
    color_map = {'Passport':'#4f8ef7','Citizenship':'#4ade80','PAN':'#c084fc','National ID':'#fb923c','Unknown':'#ef4444'}
    bg_map    = {'Passport':'#1e3a5f','Citizenship':'#1a3a2a','PAN':'#2a1a3a','National ID':'#3a2a1a','Unknown':'#3a1a1a'}

    icon       = icon_map.get(doc_type, '❓')
    color      = color_map.get(doc_type, '#888')
    bg         = bg_map.get(doc_type, '#1a1a2e')
    confidence = get_confidence(scores, doc_type)

    st.markdown('<div class="title">📄 OCR Document Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Classification complete</div>', unsafe_allow_html=True)

    render_stepper(3)

    if st.button("← Classify Another Document"):
        st.session_state.page   = 'upload'
        st.session_state.result = None
        st.rerun()

    st.divider()

    left_col, right_col = st.columns([1, 1])

    with left_col:
        if st.session_state.uploaded_image:
            st.image(
                st.session_state.uploaded_image,
                caption=st.session_state.file_name,
                use_container_width=True
            )

    with right_col:
        st.markdown(f"""
            <div class="result-card">
                <div class="result-icon" style="background:{bg};">{icon}</div>
                <div style="flex:1;">
                    <p style="font-size:0.78rem; color:#888; margin:0;">Classified as</p>
                    <p style="font-size:1.4rem; font-weight:700; margin:2px 0; color:{color};">{doc_type}</p>
                    <div style="display:flex; align-items:center; gap:8px;">
                        <div class="confidence-bar-bg" style="flex:1;">
                            <div style="width:{confidence}%; height:100%; background:{color}; border-radius:3px;"></div>
                        </div>
                        <span style="font-size:0.78rem; color:{color}; font-weight:600; white-space:nowrap;">{confidence}% match</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("**📈 Keyword scores**")
        max_score = max(scores.values()) if max(scores.values()) > 0 else 1
        for doc, score in scores.items():
            ca, cb = st.columns([4, 1])
            with ca:
                st.progress(min(score / max(max_score, 1), 1.0))
            with cb:
                st.markdown(f"**{score}**")

    st.divider()

    st.markdown("**🗂️ Extracted Information**")
    fields = extract_fields(result['extracted_text'], doc_type)

    h1, h2 = st.columns([1, 2])
    with h1:
        st.markdown("<p style='font-size:0.75rem; color:#888; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; margin:0;'>Field</p>", unsafe_allow_html=True)
    with h2:
        st.markdown("<p style='font-size:0.75rem; color:#888; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; margin:0;'>Value</p>", unsafe_allow_html=True)
    st.markdown("<hr style='margin:6px 0; border-color:#2a2a3e;'>", unsafe_allow_html=True)

    for key, value in fields.items():
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"<p style='color:#888; font-size:0.88rem; margin:0; padding:4px 0;'>{key}</p>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<p style='color:#fff; font-size:0.88rem; font-weight:500; margin:0; padding:4px 0;'>{value}</p>", unsafe_allow_html=True)
        st.markdown("<hr style='margin:2px 0; border-color:#1a1a2e;'>", unsafe_allow_html=True)

    st.divider()

    with st.expander("🔍 View raw OCR text"):
        st.code(result['extracted_text'], language=None)
import streamlit as st
import tempfile
import os
from src.pipeline import run_pipeline
from src.classifier import extract_fields

# Page config
st.set_page_config(
    page_title="OCR Document Classifier",
    page_icon="📄",
    layout="centered"
)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'upload'
if 'result' not in st.session_state:
    st.session_state.result = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None

# Custom CSS
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4f8ef7, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-box {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border: 1px solid #4f8ef7;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .extracted-box {
        background-color: #1a1a2e;
        border-left: 4px solid #a855f7;
        border-radius: 8px;
        padding: 1rem;
        color: #ccc;
        font-family: monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .upload-hint {
        text-align: center;
        color: #666;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #4f8ef7, #a855f7);
        color: white;
        border: none;
        padding: 0.75rem;
        font-size: 1rem;
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


# ─────────────────────────────────────────
#  PAGE 1 — UPLOAD
# ─────────────────────────────────────────
if st.session_state.page == 'upload':

    st.markdown('<div class="title">📄 OCR Document Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload a document image to extract text and identify the document type</div>', unsafe_allow_html=True)

    # Supported badges
    st.markdown("""
        <div style="text-align:center; margin-bottom: 2rem;">
            <span style="background:#1e3a5f; color:#4f8ef7; padding:4px 12px; border-radius:20px; font-size:0.8rem; margin:3px;">🛂 Passport</span>
            <span style="background:#1a3a2a; color:#4ade80; padding:4px 12px; border-radius:20px; font-size:0.8rem; margin:3px;">📋 Citizenship</span>
            <span style="background:#3a2a1a; color:#fb923c; padding:4px 12px; border-radius:20px; font-size:0.8rem; margin:3px;">🪪 National ID</span>
            <span style="background:#2a1a3a; color:#c084fc; padding:4px 12px; border-radius:20px; font-size:0.8rem; margin:3px;">🧾 PAN Card</span>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop your document image here",
        type=["jpg", "jpeg", "png", "jfif", "bmp", "tiff", "webp"],
        label_visibility="collapsed"
    )
    st.markdown('<div class="upload-hint">Supported: jpg, jpeg, png, jfif, bmp, tiff, webp</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        st.divider()
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(uploaded_file, caption="Uploaded Document", use_container_width=True)

        with col2:
            st.markdown("**File Details**")
            st.markdown(f"- 📁 **Name:** {uploaded_file.name}")
            st.markdown(f"- 📦 **Size:** {round(uploaded_file.size / 1024, 1)} KB")
            st.markdown(f"- 🖼️ **Type:** {uploaded_file.type}")
            st.markdown("")

            if st.button("🔍 Classify Document"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                with st.spinner("🔧 Preprocessing → 🔍 Extracting text → 🏷️ Classifying..."):
                    result = run_pipeline(tmp_path)

                os.unlink(tmp_path)

                if result:
                    # Save to session and switch page
                    st.session_state.result = result
                    st.session_state.uploaded_image = uploaded_file.getvalue()
                    st.session_state.file_name = uploaded_file.name
                    st.session_state.page = 'results'
                    st.rerun()
                else:
                    st.error("❌ Could not process the image. Please try a clearer image.")


# ─────────────────────────────────────────
#  PAGE 2 — RESULTS
# ─────────────────────────────────────────
elif st.session_state.page == 'results':

    result   = st.session_state.result
    doc_type = result['document_type']

    icon_map = {
        'Passport'   : '🛂',
        'Citizenship': '📋',
        'PAN'        : '🧾',
        'National ID': '🪪',
        'Unknown'    : '❓'
    }
    color_map = {
        'Passport'   : '#4f8ef7',
        'Citizenship': '#4ade80',
        'PAN'        : '#c084fc',
        'National ID': '#fb923c',
        'Unknown'    : '#ef4444'
    }

    icon  = icon_map.get(doc_type, '❓')
    color = color_map.get(doc_type, '#888')

    # Back button
    if st.button("← Back"):
        st.session_state.page = 'upload'
        st.session_state.result = None
        st.rerun()

    st.divider()

    # Two columns — image on left, result on right
    left_col, right_col = st.columns([1, 1])

    with left_col:
        if st.session_state.uploaded_image:
            st.image(st.session_state.uploaded_image,
                     caption=st.session_state.file_name,
                     use_container_width=True)

    with right_col:
        # Result box
        st.markdown(f"""
            <div class="result-box">
                <div style="font-size:3rem;">{icon}</div>
                <div style="color:#aaa; font-size:0.9rem; margin-top:0.5rem;">Document Classified As</div>
                <div style="font-size:2rem; font-weight:800; color:{color}; margin-top:0.3rem;">{doc_type}</div>
            </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Extracted fields
    st.markdown("**🗂️ Extracted Information**")
    fields = extract_fields(result['extracted_text'], doc_type)

    for key, value in fields.items():
        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.markdown(f"<span style='color:#888; font-size:0.9rem; font-weight:600;'>{key}</span>", unsafe_allow_html=True)
        with col_right:
            st.markdown(f"<span style='color:#ffffff; font-size:0.9rem;'>{value}</span>", unsafe_allow_html=True)
        st.divider()

    st.divider()

    # Keyword scores
    st.markdown("**📈 Keyword Match Scores**")
    scores    = result['scores']
    max_score = max(scores.values()) if max(scores.values()) > 0 else 1
    for doc, score in scores.items():
        col_a, col_b = st.columns([4, 1])
        with col_a:
            st.progress(min(score / max(max_score, 1), 1.0))
        with col_b:
            st.markdown(f"**{doc}**: {score}")

    st.divider()

    # Raw OCR text
    st.markdown("**📝 Raw OCR Text**")
    st.markdown(f'<div class="extracted-box">{result["extracted_text"]}</div>', unsafe_allow_html=True)

    st.divider()

    # Classify another
    if st.button("🔄 Classify Another Document"):
        st.session_state.page = 'upload'
        st.session_state.result = None
        st.rerun()
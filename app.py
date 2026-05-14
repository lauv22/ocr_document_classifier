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
        font-size: 2.4rem;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .step-bar {
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 1.5rem 0 2rem;
        gap: 0;
    }
    .step {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 6px;
    }
    .step-circle {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        font-weight: 600;
    }
    .step-circle.active {
        background: #6c63ff;
        color: white;
        border: 2px solid #6c63ff;
    }
    .step-circle.done {
        background: #6c63ff;
        color: white;
        border: 2px solid #6c63ff;
    }
    .step-circle.inactive {
        background: transparent;
        color: #888;
        border: 2px solid #444;
    }
    .step-label {
        font-size: 0.75rem;
        font-weight: 500;
    }
    .step-label.active { color: #6c63ff; }
    .step-label.done   { color: #6c63ff; }
    .step-label.inactive { color: #888; }
    .step-line {
        flex: 1;
        height: 2px;
        max-width: 80px;
        margin-bottom: 22px;
    }
    .step-line.done     { background: #6c63ff; }
    .step-line.inactive { background: #444; }
    .result-card {
        border: 1px solid #2a2a3e;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        display: flex;
        align-items: center;
        gap: 16px;
        background: #1a1a2e;
        margin-bottom: 1.5rem;
    }
    .result-icon {
        width: 52px;
        height: 52px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;
        flex-shrink: 0;
    }
    .confidence-bar-bg {
        height: 6px;
        background: #2a2a3e;
        border-radius: 3px;
        margin-top: 6px;
        width: 100%;
    }
    .field-table {
        width: 100%;
        border-collapse: collapse;
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #2a2a3e;
        font-size: 0.9rem;
    }
    .field-table th {
        background: #1a1a2e;
        color: #888;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 10px 14px;
        text-align: left;
    }
    .field-table td {
        padding: 10px 14px;
        border-top: 1px solid #2a2a3e;
    }
    .field-table tr:nth-child(even) td {
        background: #111120;
    }
    .field-table tr:nth-child(odd) td {
        background: #16162a;
    }
    .field-key-cell { color: #888; }
    .field-val-cell { color: #fff; font-weight: 500; }
    .upload-zone {
        border: 2px dashed #3a3a5e;
        border-radius: 14px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 1rem 0;
        background: #111120;
    }
    .upload-zone-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
    .upload-zone-title { font-size: 1rem; font-weight: 600; margin-bottom: 0.3rem; }
    .upload-zone-sub   { font-size: 0.85rem; color: #888; }
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


def render_stepper(current_step):
    # current_step: 1 = upload, 2 = processing, 3 = results
    steps = [
        ("📤", "Upload"),
        ("⚙️", "Processing"),
        ("✅", "Results"),
    ]
    html = '<div class="step-bar">'
    for i, (icon, label) in enumerate(steps):
        step_num = i + 1
        if step_num < current_step:
            circle_class = "done"
            label_class  = "done"
            icon_display = "✓"
        elif step_num == current_step:
            circle_class = "active"
            label_class  = "active"
            icon_display = icon
        else:
            circle_class = "inactive"
            label_class  = "inactive"
            icon_display = icon

        html += f'''
            <div class="step">
                <div class="step-circle {circle_class}">{icon_display}</div>
                <span class="step-label {label_class}">{label}</span>
            </div>
        '''
        if i < len(steps) - 1:
            line_class = "done" if step_num < current_step else "inactive"
            html += f'<div class="step-line {line_class}"></div>'

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

    # Supported badges
    st.markdown("""
        <div style="text-align:center; margin-bottom: 1.5rem;">
            <span style="background:#1e3a5f; color:#4f8ef7; padding:4px 12px; border-radius:20px; font-size:0.8rem; margin:3px; display:inline-block;">🛂 Passport</span>
            <span style="background:#1a3a2a; color:#4ade80; padding:4px 12px; border-radius:20px; font-size:0.8rem; margin:3px; display:inline-block;">📋 Citizenship</span>
            <span style="background:#3a2a1a; color:#fb923c; padding:4px 12px; border-radius:20px; font-size:0.8rem; margin:3px; display:inline-block;">🪪 National ID</span>
            <span style="background:#2a1a3a; color:#c084fc; padding:4px 12px; border-radius:20px; font-size:0.8rem; margin:3px; display:inline-block;">🧾 PAN Card</span>
        </div>
    """, unsafe_allow_html=True)

    # Styled upload zone hint
    st.markdown("""
        <div class="upload-zone">
            <div class="upload-zone-icon">☁️</div>
            <div class="upload-zone-title">Drop your document image here</div>
            <div class="upload-zone-sub">Supported: JPG, JPEG, PNG, JFIF, BMP, TIFF, WEBP</div>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload document",
        type=["jpg", "jpeg", "png", "jfif", "bmp", "tiff", "webp"],
        label_visibility="collapsed"
    )

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
    bg_map = {
        'Passport'   : '#1e3a5f',
        'Citizenship': '#1a3a2a',
        'PAN'        : '#2a1a3a',
        'National ID': '#3a2a1a',
        'Unknown'    : '#3a1a1a'
    }

    icon       = icon_map.get(doc_type, '❓')
    color      = color_map.get(doc_type, '#888')
    bg         = bg_map.get(doc_type, '#1a1a2e')
    confidence = get_confidence(scores, doc_type)

    st.markdown('<div class="title">📄 OCR Document Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Classification complete</div>', unsafe_allow_html=True)

    render_stepper(3)

    # Back button
    if st.button("← Classify Another Document"):
        st.session_state.page   = 'upload'
        st.session_state.result = None
        st.rerun()

    st.divider()

    # Image + result card side by side
    left_col, right_col = st.columns([1, 1])

    with left_col:
        if st.session_state.uploaded_image:
            st.image(
                st.session_state.uploaded_image,
                caption=st.session_state.file_name,
                use_container_width=True
            )

    with right_col:
        # Result card with confidence bar
        st.markdown(f"""
            <div class="result-card">
                <div class="result-icon" style="background:{bg};">{icon}</div>
                <div style="flex:1;">
                    <p style="font-size:0.8rem; color:#888; margin:0;">Classified as</p>
                    <p style="font-size:1.5rem; font-weight:700; margin:2px 0; color:{color};">{doc_type}</p>
                    <div style="display:flex; align-items:center; gap:8px;">
                        <div class="confidence-bar-bg" style="flex:1;">
                            <div style="width:{confidence}%; height:100%; background:{color}; border-radius:3px;"></div>
                        </div>
                        <span style="font-size:0.8rem; color:{color}; font-weight:600; white-space:nowrap;">{confidence}% match</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Keyword scores compact
        st.markdown("**📈 Keyword scores**")
        max_score = max(scores.values()) if max(scores.values()) > 0 else 1
        for doc, score in scores.items():
            ca, cb = st.columns([4, 1])
            with ca:
                st.progress(min(score / max(max_score, 1), 1.0))
            with cb:
                st.markdown(f"**{score}**")

    st.divider()

    # Extracted fields as clean table
    st.markdown("**🗂️ Extracted Information**")
    fields = extract_fields(result['extracted_text'], doc_type)

    rows_html = ""
    for key, value in fields.items():
        rows_html += f"""
            <tr>
                <td class="field-key-cell">{key}</td>
                <td class="field-val-cell">{value}</td>
            </tr>
        """

    st.markdown(f"""
        <table class="field-table">
            <thead>
                <tr>
                    <th>Field</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    """, unsafe_allow_html=True)

    st.divider()

    # Collapsible raw OCR text
    with st.expander("🔍 View raw OCR text"):
        st.code(result['extracted_text'], language=None)
import streamlit as st
import tempfile
import os
from src.pipeline import run_pipeline

# Page config
st.set_page_config(
    page_title="OCR Document Classifier",
    page_icon="📄",
    layout="centered"
)

# Title
st.title("📄 OCR Document Classifier")
st.markdown("Upload a document image and the system will extract text and classify the document type.")
st.divider()

# File uploader
uploaded_file = st.file_uploader(
    "Choose a document image",
    type=["jpg", "jpeg", "png", "jfif", "bmp", "tiff", "webp"],
    help="Supported formats: jpg, jpeg, png, jfif, bmp, tiff, webp"
)

if uploaded_file is not None:

    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Document", use_column_width=True)
    st.divider()

    # Run pipeline on button click
    if st.button("🔍 Classify Document", type="primary", use_container_width=True):

        # Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("Processing document..."):
            result = run_pipeline(tmp_path)

        # Clean up temp file
        os.unlink(tmp_path)

        if result:
            # Document type result
            doc_type = result['document_type']

            # Color based on document type
            color_map = {
                'Passport': '🟦',
                'Citizenship': '🟩',
                'PAN': '🟨',
                'National ID': '🟧',
                'Unknown': '🟥'
            }
            icon = color_map.get(doc_type, '⬜')

            st.success(f"{icon}  Document Classified As: **{doc_type}**")
            st.divider()

            # Keyword scores
            st.subheader("📈 Keyword Scores")
            scores = result['scores']
            for doc, score in scores.items():
                st.progress(
                    min(score / 10, 1.0),
                    text=f"{doc}: {score} keyword(s) matched"
                )

            st.divider()

            # Extracted text
            st.subheader("📝 Extracted Text")
            st.text_area(
                label="Raw OCR Output",
                value=result['extracted_text'],
                height=200,
                disabled=True
            )

        else:
            st.error("❌ Could not process the image. Please try a clearer image.")
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import io
import os
import shutil
import cv2
import numpy as np
import torch
from transformers import pipeline, TrOCRProcessor, VisionEncoderDecoderModel
from diff_match_patch import diff_match_patch
import pytesseract

# ----------------------------------------------------
# CONFIG & INITIALIZATION
# ----------------------------------------------------

st.set_page_config(
    page_title="ScanWise | Document Processor",
    page_icon="🤖",
    layout="wide"
)

# --- CONFIGURE TESSERACT PATH DYNAMICALLY ---
def configure_tesseract():
    """Detects and configures the tesseract executable path cross-platform."""
    # 1. If tesseract is already available on PATH (Linux, Docker, or configured Windows), use it directly
    if shutil.which("tesseract"):
        pytesseract.pytesseract.tesseract_cmd = "tesseract"
        return True

    # 2. On Windows, check standard installation paths if not on PATH
    if os.name == "nt":
        common_windows_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        ]
        for path in common_windows_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return True

    return False

tesseract_available = configure_tesseract()

# ----------------------------------------------------
# MODEL LOADING (with caching)
# ----------------------------------------------------

@st.cache_resource
def load_ocr_model():
    """Loads the TrOCR model for handwriting recognition."""
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-handwritten')
    model.eval()
    return processor, model

@st.cache_resource
def load_correction_model():
    """Loads the text correction model."""
    return pipeline("text2text-generation", model="google/flan-t5-small", device=-1)

# ----------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------

def extract_text_with_trocr(image, ocr_processor, ocr_model):
    """Extracts text using the accurate (but slower) TrOCR model."""
    try:
        with torch.no_grad():
            pixel_values = ocr_processor(images=image, return_tensors="pt").pixel_values
            generated_ids = ocr_model.generate(pixel_values, max_new_tokens=256)
            text = ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text.strip()
    except Exception as e:
        return f"[Error during TrOCR processing: {e}]"

def extract_text_with_tesseract(image):
    """Extracts text using the fast Tesseract engine."""
    try:
        text = pytesseract.image_to_string(image)
        return text.strip()
    except pytesseract.TesseractNotFoundError:
        return "[Error: Tesseract Engine not found. Please ensure Tesseract is installed and available in PATH.]"
    except Exception as e:
        return f"[Error during Tesseract processing: {e}]"

def generate_diff_html(text1, text2):
    dmp = diff_match_patch()
    diffs = dmp.diff_main(text1, text2)
    dmp.diff_cleanupSemantic(diffs)
    html = dmp.diff_prettyHtml(diffs)
    html = html.replace('style="background:#e6ffe6;"', 'style="color: #2b823a; font-weight: 600; text-decoration: none;"')
    html = html.replace('style="background:#ffe6e6;"', 'style="color: #c93c3c; text-decoration: line-through;"')
    return f'<div style="font-family: monospace; white-space: pre-wrap; padding: 1rem; border-radius: 0.5rem; background-color: #fafafa">{html}</div>'

def correct_text(text, model):
    if not model or not text.strip() or text.startswith("[Error"):
        return text
    try:
        prompt = f"Correct the grammar and spelling in this text: {text}"
        corrected_list = model(prompt, max_length=512, truncation=True)
        return corrected_list[0]['generated_text'] if corrected_list else text
    except Exception:
        return text

def is_page_blank(page_data, text_threshold=10, ink_threshold=0.5):
    corrected = page_data.get("corrected_text", "")
    if corrected.startswith("[Error"):
        return False, "Could not determine (OCR error)"
        
    word_count = len(corrected.split())
    if word_count > text_threshold:
        return False, "Contains sufficient text"
        
    try:
        image = page_data["image"].convert('RGB')
        open_cv_image = np.array(image)
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        if thresh.size == 0:
            return True, "Page is visually empty"
        ink_percentage = (np.count_nonzero(thresh) / thresh.size) * 100
        if ink_percentage < ink_threshold:
            return True, "Page is visually empty"
        return False, "Contains markings but little text"
    except Exception as e:
        return False, f"Could not analyze ink ({e})"

# ----------------------------------------------------
# STREAMLIT USER INTERFACE
# ----------------------------------------------------

title_col, author_col = st.columns([5, 1])
with title_col:
    st.title("ScanWise")
    st.markdown("Digitize documents, correct text, and analyze content. Choose your engine for speed or accuracy.")
with author_col:
    st.link_button("ialksng", "https://ialksng.me", help="ScanWise is made by Alok Singh")

# --- ENGINE SELECTION ---
st.header("1. Choose Your OCR Engine")
engine_choice = st.radio(
    "Select an engine based on your document type:",
    ('🚀 Fast Engine (for Printed Text)', '✍️ Accurate Engine (for Handwriting)'),
    horizontal=True,
    help="Fast Engine uses Tesseract for quick results. Accurate Engine uses Microsoft TrOCR for better handwriting recognition, but is slower."
)

if 'Fast Engine' in engine_choice and not tesseract_available:
    st.info("ℹ️ Tesseract OCR is not detected in your system PATH. If you run into issues, please ensure Tesseract is installed or switch to the 'Accurate Engine'.")

# --- FILE UPLOAD ---
st.header("2. Upload Your Document")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    if not file_bytes:
        st.error("The uploaded file appears to be empty.")
        st.stop()

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        st.error(f"Could not read the PDF file. It might be corrupted: {e}")
        st.stop()

    if len(doc) == 0:
        st.warning("The uploaded PDF file contains 0 pages.")
        st.stop()

    # --- MODEL LOADING ---
    with st.spinner("Initializing models..."):
        corrector = load_correction_model()
        ocr_processor, ocr_model = (load_ocr_model() if 'Accurate' in engine_choice else (None, None))

    spinner_text = "Analyzing your document..."
    if 'Accurate' in engine_choice:
        spinner_text = "Reading handwriting with TrOCR... This advanced analysis may take some time."

    with st.spinner(spinner_text):
        pages_data = []
        progress_bar = st.progress(0, "Processing page 1...")
        total_pages = len(doc)

        for i, page in enumerate(doc):
            progress_bar.progress((i + 1) / total_pages, f"Processing page {i+1}/{total_pages}...")
            pix = page.get_pixmap(dpi=150)
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

            raw_text = ""
            if 'Accurate' in engine_choice:
                raw_text = extract_text_with_trocr(image, ocr_processor, ocr_model)
            else:
                raw_text = extract_text_with_tesseract(image)

            pages_data.append({"page_number": i + 1, "image": image, "raw_text": raw_text})

        # --- NLP CORRECTION AND ANALYSIS ---
        final_text_for_download = ""
        for page in pages_data:
            page["corrected_text"] = correct_text(page["raw_text"], corrector)
            is_blank, reason = is_page_blank(page)
            page["is_blank"] = is_blank
            page["reason"] = reason
            if not is_blank:
                final_text_for_download += f"--- Page {page['page_number']} ---\n\n{page['corrected_text']}\n\n"

        progress_bar.empty()

    st.success("Analysis Complete!")

    # --- DISPLAY RESULTS ---
    st.header("3. Review Your Results")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("Here is a summary of your document's content and paper usage.")
    with col2:
        st.download_button(
            label="📥 Download Corrected Text",
            data=final_text_for_download.encode("utf-8"),
            file_name=f"{uploaded_file.name.replace('.pdf', '')}_corrected.txt",
            mime="text/plain"
        )

    total_pages_count = len(pages_data)
    wasted_pages_count = sum(1 for p in pages_data if p["is_blank"])
    used_pages_count = total_pages_count - wasted_pages_count
    waste_percentage = (wasted_pages_count / total_pages_count) * 100 if total_pages_count > 0 else 0

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Total Pages", f"{total_pages_count}")
    m_col2.metric("Used Pages", f"{used_pages_count}")
    m_col3.metric("Wasted Pages", f"{wasted_pages_count}", help="Pages identified as blank or containing minimal content.")
    m_col4.metric("Paper Waste", f"{waste_percentage:.1f}%")

    st.subheader("Page-by-Page Breakdown")
    for page in pages_data:
        status_icon = "🔴 Wasted" if page["is_blank"] else "🟢 Used"
        with st.expander(f"**Page {page['page_number']}** — Status: {status_icon} (Reason: {page['reason']})"):
            col_img, col_text = st.columns(2)
            with col_img:
                st.image(page["image"], caption=f"Scanned view of Page {page['page_number']}")
            with col_text:
                st.subheader("Raw Scanned Text")
                st.text_area("The text as initially read from the page:", value=page["raw_text"], height=150, disabled=True, key=f"raw_{page['page_number']}")
                st.subheader("AI-Powered Corrections")
                st.markdown("Deletions are in <span style='color: #c93c3c'>red</span> and additions are in <span style='color: #2b823a'>green</span>.", unsafe_allow_html=True)
                diff_html = generate_diff_html(page["raw_text"], page["corrected_text"])
                st.markdown(diff_html, unsafe_allow_html=True)

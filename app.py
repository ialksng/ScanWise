import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import io
import cv2
import numpy as np
from transformers import pipeline, TrOCRProcessor, VisionEncoderDecoderModel
from diff_match_patch import diff_match_patch
import pytesseract

# ----------------------------------------------------
# CONFIG & INITIALIZATION
# ----------------------------------------------------

st.set_page_config(
    page_title="Document Processor",
    page_icon="🤖",
    layout="wide"
)

# --- IMPORTANT: CONFIGURE TESSERACT PATH ---
# Make sure this path is correct for your system if you use the Fast Engine.
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except Exception:
    # This will be handled gracefully if Tesseract is not found but TrOCR is used.
    pass

# ----------------------------------------------------
# MODEL LOADING (with caching)
# ----------------------------------------------------

@st.cache_resource
def load_ocr_model():
    """Loads the TrOCR model for handwriting recognition."""
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
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
    pixel_values = ocr_processor(images=image, return_tensors="pt").pixel_values
    generated_ids = ocr_model.generate(pixel_values)
    return ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def extract_text_with_tesseract(image):
    """Extracts text using the fast Tesseract engine."""
    try:
        return pytesseract.image_to_string(image)
    except pytesseract.TesseractNotFoundError:
        st.error("Tesseract Engine not found. Please install it and configure the path in the script to use the 'Fast Engine'.")
        return "Tesseract not found. Please configure it."

# ... (The other helper functions: generate_diff_html, correct_text, is_page_blank remain the same)
def generate_diff_html(text1, text2):
    dmp = diff_match_patch()
    diffs = dmp.diff_main(text1, text2)
    dmp.diff_cleanupSemantic(diffs)
    html = dmp.diff_prettyHtml(diffs)
    html = html.replace('style="background:#e6ffe6;"', 'style="color: #2b823a; font-weight: 600; text-decoration: none;"')
    html = html.replace('style="background:#ffe6e6;"', 'style="color: #c93c3c; text-decoration: line-through;"')
    return f'<div style="font-family: monospace; white-space: pre-wrap; padding: 1rem; border-radius: 0.5rem; background-color: #fafafa">{html}</div>'

def correct_text(text, model):
    if not model or not text.strip(): return text
    try:
        prompt = f"Correct the grammar and spelling in this text: {text}"
        corrected_list = model(prompt, max_length=1024)
        return corrected_list[0]['generated_text'] if corrected_list else text
    except Exception: return text

def is_page_blank(page_data, text_threshold=10, ink_threshold=0.5):
    word_count = len(page_data["corrected_text"].split())
    if word_count > text_threshold: return False, "Contains sufficient text"
    image = page_data["image"].convert('RGB')
    open_cv_image = np.array(image)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink_percentage = (np.count_nonzero(thresh) / thresh.size) * 100
    if ink_percentage < ink_threshold: return True, "Page is visually empty"
    return False, "Contains markings but little text"

# ----------------------------------------------------
# STREAMLIT USER INTERFACE
# ----------------------------------------------------

st.title("Document Processor")
st.markdown("Digitize documents, correct text, and analyze content. Choose your engine for speed or accuracy.")

# --- ENGINE SELECTION ---
st.header("1. Choose Your OCR Engine")
engine_choice = st.radio(
    "Select an engine based on your document type:",
    ('🚀 Fast Engine (for Printed Text)', '✍️ Accurate Engine (for Handwriting)'),
    horizontal=True,
    help="Fast Engine uses Tesseract for quick results. Accurate Engine uses Microsoft TrOCR for better handwriting recognition, but is slower."
)

# --- FILE UPLOAD ---
st.header("2. Upload Your Document")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # --- MODEL LOADING ---
    corrector = load_correction_model()
    
    spinner_text = "Analyzing your document..."
    if 'Accurate' in engine_choice:
        spinner_text = "Reading handwriting... This advanced analysis may take some time."
    
    with st.spinner(spinner_text):
        # --- PDF PROCESSING ---
        try:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        except Exception:
            st.error("Could not read the PDF file. It might be corrupted.")
            st.stop()
        
        pages_data = []
        
        # --- CONDITIONAL OCR BASED ON USER CHOICE ---
        ocr_processor, ocr_model = (load_ocr_model() if 'Accurate' in engine_choice else (None, None))
        
        progress_bar = st.progress(0, "Processing page 1...")
        for i, page in enumerate(doc):
            progress_bar.progress((i + 1) / len(doc), f"Processing page {i+1}/{len(doc)}...")
            image = Image.open(io.BytesIO(page.get_pixmap(dpi=150).tobytes("png"))).convert("RGB")
            
            raw_text = ""
            if 'Accurate' in engine_choice:
                raw_text = extract_text_with_trocr(image, ocr_processor, ocr_model)
            else:
                raw_text = extract_text_with_tesseract(image)

            pages_data.append({"page_number": i + 1, "image": image, "raw_text": raw_text})
        
        # --- NLP CORRECTION AND ANALYSIS (same for both engines) ---
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
    
    # --- DISPLAY RESULTS (same layout as before) ---
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

    total_pages = len(pages_data)
    wasted_pages_count = sum(1 for p in pages_data if p["is_blank"])
    used_pages_count = total_pages - wasted_pages_count
    waste_percentage = (wasted_pages_count / total_pages) * 100 if total_pages > 0 else 0
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Total Pages", f"{total_pages}")
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
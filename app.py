import hashlib
import io
import os
import shutil

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
import streamlit as st
import torch
from diff_match_patch import diff_match_patch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, pipeline


st.set_page_config(
    page_title="ScanWise | Document Processor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { color-scheme: light; }
      .stApp { background: #f4f7fb; }
      [data-testid="stHeader"] { background: rgba(244, 247, 251, 0.88); }
      .block-container { max-width: 1280px; padding-top: 2rem; padding-bottom: 4rem; }
      [data-testid="stSidebar"] { background: #eef3f9; border-right: 1px solid #e0e8f2; }
      [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.8rem; }
      .sw-kicker { color: #3867a6; font-size: .75rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; margin: 0 0 .65rem; }
      .sw-title { color: #14233b; font-size: clamp(2.2rem, 4vw, 3.25rem); font-weight: 800; letter-spacing: -.055em; line-height: 1.05; margin: 0; }
      .sw-subtitle { color: #53657c; font-size: 1.04rem; line-height: 1.65; max-width: 690px; margin: .75rem 0 0; }
      [data-testid="stVerticalBlockBorderWrapper"] { background: radial-gradient(circle at 88% 4%, rgba(105, 164, 255, .18), transparent 27rem), linear-gradient(135deg, #ffffff, #f8fbff); border: 1px solid #e3ebf5; border-radius: 24px; padding: 1rem 1.3rem; box-shadow: 0 14px 36px rgba(23, 48, 84, .06); margin-bottom: 1.6rem; }
      .sw-section-title { color: #172943; font-size: 1.25rem; font-weight: 750; letter-spacing: -.02em; margin-bottom: .2rem; }
      .sw-muted { color: #64748b; font-size: .92rem; }
      .sw-step { color: #60738c; font-size: .8rem; font-weight: 750; letter-spacing: .07em; text-transform: uppercase; }
      .sw-file-meta { background: #fff; border: 1px solid #e0e8f2; border-radius: 14px; color: #334155; padding: .8rem 1rem; }
      [data-testid="stFileUploader"] { background: rgba(255,255,255,.78); border: 1px dashed #aebfd4; border-radius: 16px; padding: .7rem; }
      [data-testid="stMetric"] { background: #fff; border: 1px solid #e2eaf4; border-radius: 16px; padding: 1rem 1.1rem; box-shadow: 0 5px 16px rgba(23, 48, 84, .035); }
      [data-testid="stMetricLabel"] { color: #63758d; }
      [data-testid="stMetricValue"] { color: #14233b; }
      .stButton > button, [data-testid="stLinkButton"] a { border-radius: 11px; font-weight: 700; }
      .stButton > button[kind="primary"] { box-shadow: 0 7px 16px rgba(31, 91, 161, .18); }
      [data-testid="stExpander"] { background: #fff; border: 1px solid #e3ebf5; border-radius: 14px; }
      .sw-diff { background: #fff; border: 1px solid #e3ebf5; border-radius: 12px; color: #24364d; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; line-height: 1.75; padding: 1rem; white-space: pre-wrap; overflow-wrap: anywhere; }
      .sw-footer { border-top: 1px solid #e1e9f2; color: #8190a4; font-size: .8rem; margin-top: 3rem; padding-top: 1rem; }
      @media (max-width: 700px) { .block-container { padding: 1rem 1rem 3rem; } [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 18px; padding: .7rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def configure_tesseract():
    """Locate Tesseract across Linux, Docker, and Windows installations."""
    if shutil.which("tesseract"):
        pytesseract.pytesseract.tesseract_cmd = "tesseract"
        return True

    if os.name == "nt":
        common_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        ]
        for path in common_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return True
    return False


tesseract_available = configure_tesseract()


@st.cache_resource(show_spinner=False)
def load_ocr_model():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
    model.eval()
    return processor, model


@st.cache_resource(show_spinner=False)
def load_correction_model():
    return pipeline("text2text-generation", model="google/flan-t5-small", device=-1)


def extract_text_with_trocr(image, processor, model):
    with torch.inference_mode():
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values, max_new_tokens=256)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def extract_text_with_tesseract(image):
    return pytesseract.image_to_string(image).strip()


def correct_text(text, model):
    if not text.strip() or text.startswith("[Error"):
        return text
    prompt = f"Correct the grammar and spelling in this text: {text}"
    generated = model(prompt, max_new_tokens=256, truncation=True)
    return generated[0]["generated_text"].strip() if generated else text


def page_usage(image, text):
    if text.startswith("[Error"):
        return False, "OCR could not read this page"
    if len(text.split()) > 10:
        return False, "Text detected"

    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink_percent = (np.count_nonzero(ink) / ink.size) * 100 if ink.size else 0
    if ink_percent < 0.5:
        return True, "Almost blank"
    return False, "Marks detected; little text found"


def make_diff_html(original, corrected):
    diff = diff_match_patch()
    parts = diff.diff_main(original, corrected)
    diff.diff_cleanupSemantic(parts)
    rendered = diff.diff_prettyHtml(parts)
    rendered = rendered.replace(
        'style="background:#e6ffe6;"',
        'style="background:#e7f7ed;color:#17643a;text-decoration:none;"',
    )
    rendered = rendered.replace(
        'style="background:#ffe6e6;"',
        'style="background:#fff0ee;color:#a33b34;text-decoration:line-through;"',
    )
    return f'<div class="sw-diff">{rendered}</div>'


def analyze_pdf(file_bytes, engine_choice, progress):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    if len(doc) == 0:
        raise ValueError("This PDF has no pages to analyze.")

    accurate = engine_choice == "Handwriting · TrOCR"
    ocr_processor, ocr_model = load_ocr_model() if accurate else (None, None)
    pages = []
    page_count = len(doc)

    for index, page in enumerate(doc):
        progress.progress((index + 1) / (page_count * 2), text=f"Reading page {index + 1} of {page_count}…")
        pixmap = page.get_pixmap(dpi=150)
        image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
        try:
            raw_text = (
                extract_text_with_trocr(image, ocr_processor, ocr_model)
                if accurate
                else extract_text_with_tesseract(image)
            )
        except pytesseract.TesseractNotFoundError:
            raise RuntimeError("Tesseract is missing on the server. Switch to TrOCR or install Tesseract.")
        except Exception as error:
            raw_text = f"[Error: {error}]"
        pages.append({"page_number": index + 1, "image": image, "raw_text": raw_text})

    progress.progress(0.52, text="Loading the text correction model…")
    corrector = load_correction_model()
    for index, page in enumerate(pages):
        progress.progress(0.52 + ((index + 1) / page_count) * 0.48, text=f"Reviewing page {index + 1} of {page_count}…")
        try:
            corrected_text = correct_text(page["raw_text"], corrector)
        except Exception as error:
            corrected_text = page["raw_text"]
            page["correction_error"] = str(error)
        page["corrected_text"] = corrected_text
        page["is_blank"], page["reason"] = page_usage(page["image"], corrected_text)

    return pages


def render_results(pages, uploaded_name):
    total_pages = len(pages)
    wasted_pages = sum(page["is_blank"] for page in pages)
    used_pages = total_pages - wasted_pages
    waste_percent = (wasted_pages / total_pages * 100) if total_pages else 0
    download_text = "\n\n".join(
        f"--- Page {page['page_number']} ---\n\n{page['corrected_text']}" for page in pages
    )

    st.markdown('<div class="sw-section-title">Your document at a glance</div>', unsafe_allow_html=True)
    st.caption("Review recognized text, inspect corrections, and check page usage.")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Pages scanned", total_pages)
    metric_cols[1].metric("Pages with content", used_pages)
    metric_cols[2].metric("Nearly blank", wasted_pages)
    metric_cols[3].metric("Estimated waste", f"{waste_percent:.1f}%")

    control_col, note_col = st.columns([1, 2])
    with control_col:
        st.download_button(
            "Download corrected text",
            data=download_text.encode("utf-8"),
            file_name=f"{os.path.splitext(uploaded_name)[0]}_corrected.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with note_col:
        if any(page.get("correction_error") for page in pages):
            st.info("Some pages could not be corrected, so their original OCR text is shown.")
        elif not any(page["corrected_text"].strip() for page in pages):
            st.warning("No text was recognized. Try the other OCR engine or a clearer scan.")
        else:
            st.success("Analysis complete. Your corrected text is ready to review.")

    st.markdown("### Page-by-page review")
    for page in pages:
        status = "Nearly blank" if page["is_blank"] else "Content found"
        with st.expander(f"Page {page['page_number']}  ·  {status}  ·  {page['reason']}", expanded=total_pages == 1):
            image_col, text_col = st.columns([0.9, 1.1], gap="large")
            with image_col:
                st.image(page["image"], caption=f"Page {page['page_number']} preview", use_container_width=True)
            with text_col:
                st.markdown("**Recognized text**")
                st.text_area(
                    "Recognized text",
                    value=page["raw_text"],
                    height=150,
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"raw_{page['page_number']}",
                )
                st.markdown("**Corrected text**")
                st.text_area(
                    "Corrected text",
                    value=page["corrected_text"],
                    height=150,
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"corrected_{page['page_number']}",
                )
                with st.expander("See what changed"):
                    st.caption("Removals are red; additions are green.")
                    st.markdown(make_diff_html(page["raw_text"], page["corrected_text"]), unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### 📄 ScanWise")
    st.caption("A calmer workspace for scanned documents")
    st.divider()
    st.markdown("**1 · Choose an engine**")
    engine_choice = st.radio(
        "OCR engine",
        ("Printed text · Tesseract", "Handwriting · TrOCR"),
        label_visibility="collapsed",
        help="Tesseract is faster for clean printed pages. TrOCR is designed for handwriting and takes longer.",
    )
    if engine_choice == "Printed text · Tesseract":
        if tesseract_available:
            st.success("Tesseract is ready")
        else:
            st.warning("Tesseract was not found. Try TrOCR or install Tesseract.")
    st.divider()
    st.markdown("**How it works**")
    st.caption("1. Add a PDF\n\n2. Extract and correct text\n\n3. Review page usage and download")
    st.markdown('<div class="sw-footer">Made by Alok Singh</div>', unsafe_allow_html=True)


with st.container(border=True):
    hero_col, link_col = st.columns([5, 1], vertical_alignment="center")
    with hero_col:
        st.markdown('<div class="sw-kicker">PDF document workspace</div>', unsafe_allow_html=True)
        st.markdown('<h1 class="sw-title">Make every page count.</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="sw-subtitle">Turn scanned PDFs into readable text, review AI-assisted corrections, and spot pages that may be going to waste.</p>',
            unsafe_allow_html=True,
        )
    with link_col:
        st.link_button("ialksng", "https://ialksng.me", help="ScanWise is made by Alok Singh", use_container_width=True)

upload_col, tips_col = st.columns([1.8, 1], gap="large")
with upload_col:
    st.markdown('<div class="sw-step">02 · Add your document</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-section-title">Start with a PDF</div>', unsafe_allow_html=True)
    st.caption("Choose a scanned PDF to extract text and review page usage.")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        st.markdown(
            f'<div class="sw-file-meta"><strong>{uploaded_file.name}</strong> &nbsp;·&nbsp; {len(file_bytes) / (1024 * 1024):.2f} MB</div>',
            unsafe_allow_html=True,
        )
        run_analysis = st.button("Analyze document", type="primary", use_container_width=True)
    else:
        file_bytes = b""
        run_analysis = False

with tips_col:
    st.markdown('<div class="sw-step">03 · Review with confidence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-section-title">What you will get</div>', unsafe_allow_html=True)
    st.markdown(
        "- Text extracted page by page\n- Original and corrected text side by side\n- A downloadable text file\n- An estimate of blank or underused pages"
    )
    st.caption("First use may take longer while the AI models are downloaded and cached.")

if file_bytes:
    analysis_key = f"{hashlib.sha256(file_bytes).hexdigest()}:{engine_choice}"
    if run_analysis:
        if not file_bytes:
            st.error("The selected PDF is empty.")
        else:
            try:
                with st.status("Preparing your document…", expanded=True) as status:
                    progress = st.progress(0, text="Starting…")
                    pages = analyze_pdf(file_bytes, engine_choice, progress)
                    progress.empty()
                    st.session_state["scanwise_analysis"] = {"key": analysis_key, "pages": pages}
                    status.update(label="Analysis complete", state="complete", expanded=False)
            except Exception as error:
                st.error("ScanWise could not complete this analysis.")
                st.info("Try a smaller PDF, switch OCR engines, or retry after the service finishes warming up.")
                with st.expander("Technical details"):
                    st.exception(error)

    analysis = st.session_state.get("scanwise_analysis")
    if analysis and analysis["key"] == analysis_key:
        st.divider()
        render_results(analysis["pages"], uploaded_file.name)
    elif analysis and analysis["key"] != analysis_key:
        st.info("The PDF or OCR engine changed. Select **Analyze document** to refresh the results.")
else:
    st.session_state.pop("scanwise_analysis", None)

st.markdown(
    '<div class="sw-footer">ScanWise · Printed OCR by Tesseract · Handwriting OCR by TrOCR · Text correction by FLAN-T5</div>',
    unsafe_allow_html=True,
)

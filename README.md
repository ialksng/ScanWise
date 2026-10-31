# ScanWise

ScanWise is a browser-based PDF document processing app built with Streamlit. It extracts text from scanned PDFs, optionally corrects spelling and grammar, compares the original and corrected text, and estimates how many pages are blank or underused. Choose Tesseract for quick printed-text OCR or TrOCR for handwriting recognition.

## Links

- **Live app / landing page:** [https://ialksng.me/work/projects/scanwise/app](https://ialksng.me/work/projects/scanwise/app)
- **Developer:** [Alok Singh — ialksing.me](https://ialksing.me)

The app includes an **ialksing** button linking to the developer website.

The Streamlit server is configured with `server.baseUrlPath` for this mounted path. The portfolio site's Vercel configuration forwards the path and all app subpaths to the Render service.

## Features

- Extract text from PDF pages with Tesseract OCR or Microsoft's TrOCR handwritten-text model.
- Correct OCR output with Google's FLAN-T5 text-to-text model.
- Review original and corrected text with additions and deletions highlighted.
- Estimate page usage from extracted word counts and image ink coverage.
- Download corrected text as a `.txt` file.
- Cache downloaded models for reuse during a running Streamlit session.

## Technology stack

- **Language:** Python
- **App and UI:** Streamlit
- **PDF rendering:** PyMuPDF (`fitz`)
- **Image handling and analysis:** Pillow, OpenCV, NumPy
- **Printed text OCR:** Tesseract via `pytesseract`
- **Handwriting OCR:** Hugging Face Transformers, Microsoft `microsoft/trocr-small-handwritten`, and PyTorch
- **Text correction:** Hugging Face Transformers pipeline with `google/flan-t5-small`
- **Text change highlighting:** `diff-match-patch`
- **Model tokenizer support:** SentencePiece and `timm`
- **Deployment/runtime support:** Docker and Streamlit Community Cloud configuration

## Run locally

### Requirements

- Python 3.10 or 3.11 is recommended.
- Tesseract OCR is needed for the Fast Engine. Install it separately and make sure the `tesseract` executable is on your PATH. On Windows, the app also checks common installation locations. If using only the Accurate Engine, Tesseract is not required.
- Internet access is needed the first time each Hugging Face model is loaded. The model files are cached afterward.
- Model inference can use substantial memory. The app currently runs the correction model on CPU; PyTorch may use a GPU for TrOCR if available and configured.

### Setup

```bash
git clone <repository-url>
cd ScanWise
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install dependencies and start the app:

```bash
python -m pip install --upgrade pip
# Install the matching CPU-only PyTorch pair before the rest of the requirements.
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501/work/projects/scanwise/app/](http://localhost:8501/work/projects/scanwise/app/). Upload a PDF, select an OCR engine, and review the results. The pinned PyTorch 2.7.1 and TorchVision 0.22.1 versions are a compatible pair; use the appropriate [PyTorch installation instructions](https://pytorch.org/get-started/locally/) if you need a GPU-specific build.

## Run with Docker

Build and start the container from the repository root:

```bash
docker build -t scanwise .
docker run --rm -p 8501:8501 scanwise
```

Then open [http://localhost:8501/work/projects/scanwise/app/](http://localhost:8501/work/projects/scanwise/app/). The first model download can take time and requires internet access.

## Deployment

### Streamlit Community Cloud

1. Push the repository to GitHub.
2. Create an app at [share.streamlit.io](https://share.streamlit.io/).
3. Select the repository and branch, and set `app.py` as the main file.
4. Deploy. The included `packages.txt` installs the system Tesseract package in the hosted environment.

### Docker host (including Render)

Connect the repository to a Docker-capable host, select the included `Dockerfile`, and deploy. Provide enough memory for PyTorch and the Hugging Face models; model downloads also require outbound internet access on first startup.

## Project files

- `app.py` — Streamlit UI and document-processing workflow.
- `requirements.txt` — Python dependencies.
- `packages.txt` — system packages for Streamlit Community Cloud.
- `Dockerfile` — container build and run configuration.
- `.streamlit/config.toml` — Streamlit appearance settings.
- `sample.pdf` — sample document for trying the app.
- `LICENSE` — MIT License.

## License

This project is licensed under the [MIT License](LICENSE). See the license file for the full terms.

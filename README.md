# ScanWise

ScanWise is a web app for turning scanned PDF documents into searchable, reviewable text. Upload a PDF in the workspace, choose printed-text or handwriting OCR, inspect the recognized and corrected text page by page, and download the result. With a Groq API key configured, **Ask with AI** creates a concise summary of the complete document.

## Links

- **Live app:** [https://ialksng.me/work/projects/scanwise/app](https://ialksng.me/work/projects/scanwise/app)
- **Render service:** [https://scanwise-u9km.onrender.com/](https://scanwise-u9km.onrender.com/)
- **Developer:** [Alok Singh — ialksng.me](https://ialksng.me)
- **Source code:** [github.com/ialksng/ScanWise](https://github.com/ialksng/ScanWise)

The app's top navigation contains **Home**, **Features**, **Workspace**, and **Account**, plus an **ialksng** link to the developer site. The portfolio route forwards the app path to the Render service so Streamlit's WebSocket connection is made directly to the app host.

## What it does

- Extracts printed text with Tesseract OCR or handwritten text with Microsoft's TrOCR model.
- Corrects spelling and grammar with Google's FLAN-T5 model.
- Shows page previews, recognized text, corrected text, and highlighted changes.
- Estimates blank or underused pages and lets you download corrected text.
- Summarizes all recognized document text with Groq when **Ask with AI** is selected.
- Provides MongoDB-backed account creation and sign-in. Passwords are stored as bcrypt hashes. Uploaded PDFs and OCR output are not saved in MongoDB; they remain in the active Streamlit session.

## Technology stack

- **Language/runtime:** Python 3.11
- **Web app and interface:** Streamlit
- **PDF rendering:** PyMuPDF (`fitz`)
- **Image processing:** Pillow, OpenCV, NumPy
- **Printed OCR:** Tesseract OCR through `pytesseract`
- **Handwriting OCR:** Hugging Face Transformers, PyTorch, and Microsoft's `microsoft/trocr-small-handwritten`
- **Text correction:** Hugging Face Transformers and Google's `google/flan-t5-small`
- **AI summaries:** Groq Python SDK and Groq Chat Completions API (default model: `openai/gpt-oss-20b`)
- **Accounts:** MongoDB through PyMongo; passwords hashed with bcrypt
- **Text comparison:** `diff-match-patch`
- **Deployment:** Docker, Render, and Streamlit Community Cloud support files

## Configuration

Set secrets as environment variables on the host. Do not commit real credentials to the repository.

| Variable | Required for | Description |
| --- | --- | --- |
| `MONGODB_URI` | Sign up and login | MongoDB connection string. For MongoDB Atlas, allow the deployed host in Network Access. |
| `MONGODB_DATABASE` | Optional | Database name; defaults to `scanwise`. |
| `GROQ_API_KEY` | Ask with AI | API key from the Groq console. |
| `GROQ_MODEL` | Optional | Groq model ID; defaults to `llama-3.3-70b-versatile`. |

For local development, create `.streamlit/secrets.toml` (this file should not be committed):

```toml
MONGODB_URI = "mongodb+srv://<user>:<password>@<cluster>/?retryWrites=true&w=majority"
MONGODB_DATABASE = "scanwise"
GROQ_API_KEY = "gsk_your_key_here"
GROQ_MODEL = "openai/gpt-oss-20b"
```

Add the same values under the Render service's environment variables/secrets to enable account access and AI summaries in production. The app still serves its public Home and Features pages when these integrations have not been configured; account creation and AI summaries remain unavailable until their respective secrets are set.

## Run locally

### Prerequisites

- Python 3.11 (recommended) and Git.
- Tesseract OCR for the printed-text engine. Install Tesseract for your OS and ensure `tesseract` is on `PATH`. Windows installations in the usual Program Files locations are detected automatically.
- Internet access for the first download of the Hugging Face OCR and correction models.
- MongoDB and Groq credentials only if you want to use account registration/login and **Ask with AI** locally.

### Install and start

```bash
git clone https://github.com/ialksng/ScanWise.git
cd ScanWise
python -m venv .venv
```

Activate the virtual environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Install the matched CPU-only PyTorch/TorchVision versions first, then the rest of the app requirements:

```bash
python -m pip install --upgrade pip
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501/work/projects/scanwise/app/](http://localhost:8501/work/projects/scanwise/app/). Visit **Account** to register/sign in, then open **Workspace** to upload and analyze a PDF. The first model load may take several minutes. Transformers is pinned to 4.57.6 because this app uses its `text2text-generation` pipeline for FLAN-T5.

## Run with Docker

Build and run from the repository root:

```bash
docker build -t scanwise .
docker run --rm -p 8501:8501 \
  -e MONGODB_URI="your-mongodb-uri" \
  -e GROQ_API_KEY="your-groq-key" \
  scanwise
```

Open [http://localhost:8501/work/projects/scanwise/app/](http://localhost:8501/work/projects/scanwise/app/). Omit either environment variable when you do not need that integration. Docker installs the system Tesseract package and uses the matched CPU PyTorch/TorchVision pair.

## Deployment

### Current recommendation

**Keep the existing Render deployment and use a 2 GB web-service instance (Render's current `1c-2g` plan, formerly Standard).** ScanWise loads PyTorch and OCR models into memory, and the app is already deployed on Render with its domain routing in place. Render currently lists 2 GB for `1c-2g`; confirm its current price in your dashboard before upgrading. The Render free instance has substantially less memory and may restart or fail while loading the models. See [Render compute plans](https://render.com/docs/compute-plans) and [Render pricing](https://render.com/pricing).

Streamlit Community Cloud can be a no-cost way to try the app, but its resource limits vary by app and can throttle or stop apps that exceed them; the published approximate range is 690 MB to 2.7 GB. Hugging Face Spaces lists 16 GB RAM for CPU Basic, but current docs say creating a compute-based Docker/Gradio Space requires a paid plan/subscription, so do not assume it is a free deployment for a new account. See [Streamlit app resources](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app) and [Hugging Face Spaces overview](https://huggingface.co/docs/hub/en/spaces-overview).

### Render setup

1. Connect the GitHub repository and deploy using the included `Dockerfile`.
2. Choose a 2 GB web-service instance (`1c-2g` / legacy Standard) for the OCR models.
3. Set `MONGODB_URI`, `GROQ_API_KEY`, and optionally `MONGODB_DATABASE` and `GROQ_MODEL` in the service's Environment settings.
4. Keep the app bound to the Render-provided `PORT`; the Docker command already does this.
5. Keep the portfolio redirect for `/work/projects/scanwise/app` pointed at the Render service so Streamlit WebSockets work correctly.

The summary feature sends extracted document text to Groq when the user selects **Ask with AI**. Do not send sensitive documents unless you are comfortable processing their text through the configured Groq account and its applicable data terms.

## Repository files

- `app.py` — Streamlit navigation, account forms, OCR workflow, results, and Groq summary.
- `requirements.txt` — Python dependencies and compatible model runtime pins.
- `packages.txt` — system package list for Streamlit Community Cloud.
- `Dockerfile` — Render and other Docker host setup.
- `.streamlit/config.toml` — theme and mounted URL path configuration.
- `sample.pdf` — sample document.
- `LICENSE` — MIT License.

## License

Released under the [MIT License](LICENSE).

# ScanWise

ScanWise is a browser-based app for extracting and reviewing text in scanned PDF documents. It uses Tesseract OCR for printed text, compares the extracted text with optional Groq proofreading, estimates blank or underused pages, and can summarize the complete document with Groq.

The app is designed to run on small instances such as Render Free: local OCR uses Tesseract rather than downloading and loading PyTorch/Transformers models. It currently targets printed text; handwriting OCR is not included in this lightweight version.

## Links

- **Live app:** [https://ialksng.me/work/projects/scanwise/app](https://ialksng.me/work/projects/scanwise/app)
- **Developer:** [Alok Singh — ialksng.me](https://ialksng.me)

The app's navbar contains **Home**, **Features**, **Workspace**, and **Account**. The footer contains the **ialksng** developer link. The portfolio route proxies requests to Render while keeping the public URL on `ialksng.me`, including the Streamlit WebSocket connection.

## Features

- Extract printed text from PDFs with Tesseract OCR.
- Optionally correct spelling and punctuation with Groq.
- Review page previews, extracted text, optional corrections, and highlighted changes.
- Estimate blank or underused pages and download text output.
- Summarize all extracted document text using Groq's chat API.
- Create MongoDB-backed accounts; passwords are hashed with bcrypt.
- Keep PDFs and OCR results in the active Streamlit session instead of saving them to MongoDB.

## Technology stack

- **Language/runtime:** Python 3.11
- **App and UI:** Streamlit
- **PDF rendering:** PyMuPDF (`fitz`)
- **OCR:** Tesseract OCR through `pytesseract`
- **Image processing and page analysis:** Pillow
- **Text comparison:** `diff-match-patch`
- **Optional proofreading and summaries:** Groq Python SDK, Groq Chat Completions API, default model `openai/gpt-oss-20b`
- **Accounts:** MongoDB through PyMongo; bcrypt password hashing
- **Deployment:** Docker and Render; `packages.txt` supports Streamlit Community Cloud

## Configuration

Store credentials as environment variables on the host. Never commit real credentials to Git.

| Variable | Used for | Description |
| --- | --- | --- |
| `MONGODB_URI` | Sign-up and login | MongoDB connection URI. For MongoDB Atlas, allow the app host in Network Access. |
| `MONGODB_DATABASE` | Optional | Database name; defaults to `scanwise`. |
| `GROQ_API_KEY` | Proofreading and summaries | API key from the Groq console. If omitted, text is extracted without AI proofreading and summaries are disabled. |
| `GROQ_MODEL` | Optional | Groq model ID; defaults to `openai/gpt-oss-20b`. |

For local development, create `.streamlit/secrets.toml` (do not commit it):

```toml
MONGODB_URI = "mongodb+srv://<user>:<password>@<cluster>/?retryWrites=true&w=majority"
MONGODB_DATABASE = "scanwise"
GROQ_API_KEY = "gsk_your_key_here"
GROQ_MODEL = "openai/gpt-oss-20b"
```

The Home and Features pages work without secrets. Account registration and login need `MONGODB_URI`; AI proofreading and summaries need `GROQ_API_KEY`.

## Run locally

### Prerequisites

- Python 3.11 and Git.
- Tesseract OCR installed and available on `PATH`. On Windows, the app also checks common installation locations.
- MongoDB and Groq credentials only for account features and AI proofreading/summaries.

### Install and start

```bash
git clone https://github.com/ialksng/ScanWise.git
cd ScanWise
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Install and run:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501/work/projects/scanwise/app/](http://localhost:8501/work/projects/scanwise/app/), sign in or create an account if MongoDB is configured, and process a PDF from **Workspace**.

## Run with Docker

```bash
docker build -t scanwise .
docker run --rm -p 8501:8501 \
  -e MONGODB_URI="your-mongodb-uri" \
  -e GROQ_API_KEY="your-groq-key" \
  scanwise
```

Open [http://localhost:8501/work/projects/scanwise/app/](http://localhost:8501/work/projects/scanwise/app/). Docker installs Tesseract and the app's Python dependencies; it does not download any PyTorch or Transformers models.

## Render Free deployment

1. Connect this GitHub repository to a Render Web Service and use the included `Dockerfile`.
2. Set `MONGODB_URI` and `GROQ_API_KEY` in the service's Environment settings if you want sign-in and AI features. Set `MONGODB_DATABASE` or `GROQ_MODEL` only to override their defaults.
3. Keep Render's dynamic `PORT` setting; the Docker start command already uses it.
4. Keep the portfolio Pages Function pointed at the Render origin to preserve the custom app URL and proxy WebSockets.

This version removes local PyTorch/Transformers model inference to reduce RAM use on Render Free. Tesseract still needs CPU and memory to render and OCR PDFs, so start with small PDFs. Free web services can spin down while idle and take time to wake up. Uploaded documents and page previews are kept in session memory only.

When **Ask with AI** or Groq proofreading is used, extracted text is sent to Groq. Do not submit sensitive documents unless you are comfortable with that processing under your Groq account's applicable terms.

## Repository files

- `app.py` — Streamlit navigation, account forms, Tesseract OCR, results, and Groq calls.
- `requirements.txt` — lightweight Python dependencies.
- `packages.txt` — Tesseract packages for Streamlit Community Cloud.
- `Dockerfile` — Render and Docker deployment setup.
- `.streamlit/config.toml` — app theme and mounted URL path.
- `sample.pdf` — sample document.
- `LICENSE` — MIT License.

## License

Released under the [MIT License](LICENSE).

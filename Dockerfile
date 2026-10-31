# Base image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy configuration and application source code
COPY .streamlit/ .streamlit/
COPY app.py .

# Expose default port
EXPOSE 8501

# Support dynamic PORT environment variable (required by Render, Cloud Run, etc.)
ENV PORT=8501
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false"]

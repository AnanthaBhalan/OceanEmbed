FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for spatial/oceanographic libraries
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose FastAPI (8000) and Streamlit (8501) ports
EXPOSE 8000 8501

HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; \
                 urllib.request.urlopen('http://localhost:8000/health',timeout=5); \
                 sys.exit(0)" 2>/dev/null || exit 1

CMD ["python", "src/app/api.py"]
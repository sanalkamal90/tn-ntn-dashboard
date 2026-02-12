FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (cacheable layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Copy model and data
COPY models/ models/
COPY data/ data/

# Expose port (Railway/Render use PORT env var)
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/api/status')" || exit 1

# Run
ENV PYTHONUNBUFFERED=1
CMD ["python", "app.py"]

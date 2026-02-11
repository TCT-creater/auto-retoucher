FROM python:3.11-slim

# システム依存ライブラリ (OpenCV headless + MediaPipe)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit 設定: Railway のリバースプロキシ対応
RUN mkdir -p ~/.streamlit && \
    echo '[server]\n\
headless = true\n\
address = "0.0.0.0"\n\
port = 8501\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n' > ~/.streamlit/config.toml

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501"]

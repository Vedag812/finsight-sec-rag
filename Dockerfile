FROM python:3.11-slim

WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source code
COPY src/ src/
COPY data/ data/
COPY scripts/ scripts/

# expose streamlit port
EXPOSE 8501

# health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# run the app
CMD ["streamlit", "run", "src/app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]

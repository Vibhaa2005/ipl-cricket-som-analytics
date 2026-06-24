FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer-cached separately from code)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and pre-trained artifacts (no CSV — training is offline)
COPY streamlit_app.py      .
COPY transformer_model.py  .
COPY artifacts/            ./artifacts/

# Non-interactive matplotlib backend
ENV MPLBACKEND=Agg

# Render injects $PORT; default 8501 for local docker run
ENV PORT=8501
EXPOSE $PORT

CMD streamlit run streamlit_app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false

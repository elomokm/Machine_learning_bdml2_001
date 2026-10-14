FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Étape 1 : calcul des features à partir des données brutes (90 Mo, non gardées dans l'image finale)
FROM base AS prepare
COPY app/features.py app/features.py
COPY prepare.py .
COPY data/ data/
RUN python prepare.py

# Étape 2 : image finale, avec seulement le code et les artifacts (quelques Mo)
FROM base
RUN useradd --create-home appuser
COPY app/ app/
COPY --from=prepare /app/artifacts/ artifacts/
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]

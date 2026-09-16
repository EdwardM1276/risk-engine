FROM python:3.11-slim-trixie

# Keep the base tag current at build time and apply the latest Debian security
# updates. The image should still be rebuilt regularly; this is not a substitute
# for scanning the resulting image.
ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

WORKDIR /app

RUN apt-get update \
    && apt-get dist-upgrade -y --no-install-recommends \
    && groupadd --system app \
    && useradd --system --gid app --home-dir /app --shell /usr/sbin/nologin app \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir --upgrade -r requirements.txt \
    && python -m pip uninstall -y pip setuptools wheel

COPY --chown=app:app . .
RUN mkdir -p /app/outputs /app/data/cache \
    && chown -R app:app /app/outputs /app/data/cache

USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"

ENTRYPOINT ["python", "-m", "streamlit", "run", "streamlit_app.py"]
CMD ["--server.address=0.0.0.0", "--server.port=8501"]
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf_cache

COPY pyproject.toml .
RUN pip install --no-cache-dir -e . 2>/dev/null || pip install --no-cache-dir .

COPY . .

EXPOSE 8000 8501

# default: API; override command for streamlit
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "asyncio"]

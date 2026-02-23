FROM nvcr.io/nvidia/tritonserver:24.08-py3

ENV PIP_NO_CACHE_DIR=1 \
    TRITON_PORT=8020 \
    TRITON_HTTP_URL=http://127.0.0.1:8000

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip && python3 -m pip install -r /app/requirements.txt
COPY wrapper.py /app/wrapper.py

EXPOSE 8020
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${TRITON_PORT}/health || exit 1

CMD ["uvicorn", "wrapper:app", "--host", "0.0.0.0", "--port", "8020"]

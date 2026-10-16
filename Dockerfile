# Backend for the web demo. Built for a free Hugging Face Docker Space (CPU, port 7860).
FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir uv

# CPU-only torch: the default Linux wheels bring ~3 GB of CUDA libraries a CPU Space can't use
COPY pyproject.toml uv.lock README.md ./
RUN uv export --frozen --no-dev --no-emit-project --no-hashes | grep -v -E '^(nvidia-|triton|cuda-)' > requirements.txt \
 && pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data/candidates ./data/candidates
RUN pip install --no-cache-dir --no-deps .

# download both models while building, so the container starts without fetching ~1 GB
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from bioclip_co.entrypoints.api import load_models; load_models()"

# the weights are in the image now, never go back to the hub at startup
ENV HF_HUB_OFFLINE=1
ENV ALLOWED_ORIGINS=https://gaova777.github.io,http://localhost:5173
EXPOSE 7860
CMD ["uvicorn", "bioclip_co.entrypoints.api:app", "--host", "0.0.0.0", "--port", "7860"]

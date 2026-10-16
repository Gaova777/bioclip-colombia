#!/bin/sh
# Everything runs on this machine. Each service in its own terminal:
#   ./run.sh ollama   ./run.sh api   ./run.sh web     the demo, with the field guide agent
#   ./run.sh mlflow   ./run.sh jenkins                 experiment tracking and CI
set -e
cd "$(dirname "$0")"

case "$1" in
  test)      uv run pytest -q ;;
  api)       uv run uvicorn bioclip_co.entrypoints.api:app --port 8000 ;;          # http://localhost:8000
  web)       cd web && npm install && npm run dev ;;                                # http://localhost:5173
  ollama)    ollama serve ;;                                                        # http://localhost:11434
  mlflow)    uv run --group mlops mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 ;;  # http://localhost:5000
  jenkins)   jenkins-bioclip ;;                                                     # http://localhost:8080
  evaluate)  uv run scripts/evaluate.py ;;
  calibrate) uv run scripts/calibrate.py --max-error 0.05 ;;
  gate)      uv run --group mlops scripts/quality_gate.py --register ;;
  onnx)      uv run --group onnx scripts/export_onnx.py bioclip && uv run --group onnx scripts/benchmark.py ;;
  lambda)
    mkdir -p events
    uv run scripts/lambda_event.py data/samples/uranomitra_franciae_cali.jpg > events/classify.json
    sam build && sam local invoke ClassifyFunction -e events/classify.json ;;
  docker)    docker build -t bioclip-colombia-api . ;;
  *) sed -n '2,4p' "$0"; echo "commands: test api web ollama mlflow jenkins evaluate calibrate gate onnx lambda docker"; exit 1 ;;
esac

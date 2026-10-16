# Everything runs on this machine. Each service in its own terminal:
#   make ollama   make api   make web          the demo, with the field guide agent
#   make mlflow   make jenkins                  experiment tracking and CI
.PHONY: test api web ollama mlflow jenkins evaluate calibrate gate onnx lambda docker

test:
	uv run pytest -q

api:  # http://localhost:8000
	uv run uvicorn bioclip_co.entrypoints.api:app --port 8000

web:  # http://localhost:5173
	cd web && npm install && npm run dev

ollama:  # local LLM for the agent, http://localhost:11434
	ollama serve

mlflow:  # http://localhost:5000
	uv run --group mlops mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

jenkins:  # http://localhost:8080
	jenkins-bioclip

evaluate:
	uv run scripts/evaluate.py

calibrate:
	uv run scripts/calibrate.py --max-error 0.05

gate:
	uv run --group mlops scripts/quality_gate.py --register

onnx:
	uv run --group onnx scripts/export_onnx.py bioclip
	uv run --group onnx scripts/benchmark.py

lambda: onnx
	uv run scripts/lambda_event.py data/samples/uranomitra_franciae_cali.jpg > events/classify.json
	sam build && sam local invoke ClassifyFunction -e events/classify.json

docker:
	docker build -t bioclip-colombia-api .

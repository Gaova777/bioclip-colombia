# Lambda container image for the classifier: ONNX Runtime on CPU, no torch (1.8 GB with both encoders, vs. 3.9 GB for the PyTorch API image).
# Needs the files from `uv run --group onnx scripts/export_onnx.py bioclip` in models/.
# python3.13 is on Amazon Linux 2023; the 3.11 image (Amazon Linux 2, glibc 2.26) is too old for current onnxruntime wheels
FROM public.ecr.aws/lambda/python:3.13

RUN pip install --no-cache-dir onnxruntime==1.31.0 numpy==2.4.6 pillow==12.3.0

COPY src/bioclip_co ${LAMBDA_TASK_ROOT}/bioclip_co
COPY data/candidates/aves_colombia.csv ${LAMBDA_TASK_ROOT}/data/candidates/
COPY data/calibration.json ${LAMBDA_TASK_ROOT}/data/
COPY models/bioclip_image.onnx models/bioclip_image_int8.onnx models/bioclip_logit_scale.txt models/bioclip_text_emb.npy ${LAMBDA_TASK_ROOT}/models/

CMD ["bioclip_co.entrypoints.lambda_handler.handler"]

# Serving BioCLIP: PyTorch vs. ONNX vs. INT8

How much does it cost to encode one photo, and what do the cheaper versions lose?
Image encoder only (the 200 candidate texts are encoded once at startup), batch of one photo, 50 runs after a warm-up.
Accuracy is zero-shot on the same 1,000-photo test set as the main results.

| variant | image encoder | latency p50 | latency p95 | species acc. | genus acc. |
|---|---:|---:|---:|---:|---:|
| PyTorch, GPU (RTX 4060) | 345 MB | 10.5 ms | 12.0 ms | 77.8% | 80.2% |
| PyTorch, CPU | 345 MB | 96.3 ms | 98.9 ms | 77.8% | 80.2% |
| ONNX Runtime, CPU | 345 MB | 50.6 ms | 52.8 ms | 77.5% | 80.1% |
| ONNX Runtime INT8, CPU | 88 MB | 32.0 ms | 34.1 ms | 73.3% | 76.5% |

([`benchmark.csv`](benchmark.csv); CPU: Intel Core i7-12700KF, 20 threads, default onnxruntime and torch settings)

```
uv run --group onnx scripts/export_onnx.py bioclip
uv run --group onnx scripts/benchmark.py
```

## Notes

- **ONNX on CPU is almost twice as fast as PyTorch on CPU** for the same accuracy. The 0.3-point difference comes from the
  preprocessing: the ONNX adapter resizes with Pillow instead of torchvision, so it doesn't need torch for images at all.
- **INT8 is 4× smaller and 1.6× faster than ONNX FP32, and costs 4.5 points of species accuracy.** Whether that's worth it
  depends on where it runs; on a small CPU server it probably is, on this GPU it isn't.
- **What didn't work: quantizing everything.** My first INT8 export quantized both encoders and species accuracy fell to
  35.6%. Trying the two halves separately showed the text encoder is the fragile one (37.5% with only the text in INT8,
  73.3% with only the image). The text side runs once at startup, so it has nothing to gain from INT8 anyway;
  now only the image encoder is quantized. Per-channel quantization didn't change anything (73.2%).
- To export, I had to turn off PyTorch's fused attention kernel (`torch.backends.mha.set_fastpath_enabled(False)`),
  which has no ONNX equivalent. Same math, slower path, only during the export.
- The ONNX version is just another adapter ([`onnx_encoder.py`](../../src/bioclip_co/adapters/onnx_encoder.py)) behind the
  same `ImageTextEncoder` port. The benchmark, the evaluation and the API don't know which one they're using.

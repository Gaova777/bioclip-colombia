"""Write an API Gateway (HTTP API) event with a photo in it, for `sam local invoke`.

usage: uv run scripts/lambda_event.py data/samples/uranomitra_franciae_cali.jpg > events/classify.json
"""
import base64
import json
import sys
from pathlib import Path

photo = Path(sys.argv[1])
print(json.dumps({
    "version": "2.0",
    "routeKey": "POST /classify",
    "rawPath": "/classify",
    "headers": {"content-type": "image/jpeg"},
    "isBase64Encoded": True,
    "body": base64.b64encode(photo.read_bytes()).decode(),
}))

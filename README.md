# Ranksmith models

Model files used by Ranksmith Studio's subject cutouts, kept here so the app's build can fetch them
by a fixed URL and check them byte for byte. Model files only: no app code, data or secrets.

Each file is attached to a release named after the method that uses it. The app pins each file's
URL, size and SHA-256 and refuses any other bytes.

## yolo26s-seg-v1

`yolo26s-seg.onnx`: Ultralytics YOLO26s-seg (COCO, 80 classes) exported to ONNX with a free batch
size and without NMS. Its outputs are `[batch, 116, anchors]` and `[batch, 32, 160, 160]` prototype
masks, the same layout as YOLO11n-seg. Ultralytics' own `yolo26s-seg.onnx` is the end-to-end export
(a fixed batch of one, `[1, 300, 38]`), so this one is exported separately.

| File               | Bytes      | SHA-256                                                            |
| ------------------ | ---------- | ------------------------------------------------------------------ |
| `yolo26s-seg.onnx` | 42,212,304 | `e85b3f2db46c3a80e7834f1b78d2695c7b9ce6fa85e2fb502279bd3bf1dddfd1` |

Exported on 10 November 2026 from Ultralytics' `yolo26s-seg.pt` (SHA-256
`3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4`) with ultralytics 8.4.208,
torch 2.15.0, onnx 1.23.2 and onnxslim 0.1.99:

```python
from ultralytics import YOLO
YOLO("yolo26s-seg.pt").export(format="onnx", imgsz=640, opset=18, simplify=True, dynamic=True)
```

The weights are Ultralytics' and keep their licence (AGPL-3.0, see
https://github.com/ultralytics/ultralytics).

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

## edgetam-track-v1

EdgeTAM's video tracker (weights `yonigozlan/EdgeTAM-hf`, revision
`c266ce53b3fc00f0f495b583f6a116c4e57f53bb`, `model.safetensors` SHA-256
`8858f8e4757b0b96dab8763f296ecffd845efbbbf698f64163cfa20a63d5fff4`) split into five fixed-shape graphs.
The app's driver keeps the memory bank between them (the clicked frame plus the last 6 followed
frames, and up to 16 object pointers, padded with a −1e4 attention bias).

| File                   | Bytes      | SHA-256                                                            | Graph                                                                                                  |
| ---------------------- | ---------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| `edgetam-encoder.onnx` | 19,754,857 | `743ec717604066915d17f7e653e3aeb6e9ba638dbccf1869131ec4836a3c51ed` | a frame at 1024² → three feature levels                                                                |
| `edgetam-click.onnx`   | 21,157,020 | `e269ef1b4b74f096c24f3e36d12c66564d7b7021bb08afe7f7636275b8b017c7` | features + a click → three masks and their predicted IoU                                               |
| `edgetam-cond.onnx`    | 28,141,614 | `f923b047e0a022a2cce23a2ac9c451662b4df88ed988969bb9202bc74bf7f421` | features + a click + `choice` (0–2) → the clicked frame's mask, the memory it seeds, the memory's position tables |
| `edgetam-memattn.onnx` | 31,936,067 | `27f6539961af2bd8e0942b6fbbb86c29ae0d553751b105f527a98ccccad83f42` | features + the memory bank → features conditioned on memory                                            |
| `edgetam-track.onnx`   | 32,646,205 | `026b9590e46c9fdd96cbd2421ef2cd039027bf5052e687719848d2d3e838eef4` | conditioned features → the frame's mask, object pointer, object score and memory                       |

Exported on 9 and 10 November 2026 with Python 3.12.12, torch 2.15.0, transformers 5.22.2,
onnx 1.23.2, onnxruntime 1.31.1, onnxscript 0.7.2 and timm 1.0.31, opset 18, the legacy TorchScript
exporter (`dynamo=False`). The scripts are in the app's repository under
`docs/channel-engine/research/`: `cutout-models-2026-11/edgetam/export.py` (encoder, memattn and
track; `et_wrappers.py` holds the wrappers), `hosted-gpu-2026-11/scripts/export_click.py` (click) and
`hosted-gpu-2026-11/scripts/export_cond.py` (cond). Each was checked against PyTorch in ONNX
Runtime on the CPU (largest difference 3e-4, on track's memory); `export_cond.py` gives these exact
bytes again, and with `choice` set to the click's best-ranked index it matches the earlier
best-choice-only cond graph.

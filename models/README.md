# Signature and stamp detector

This directory includes custom detector weights used by [`src/visual_detector.py`](../src/visual_detector.py).

| File | Format | Loading priority |
| --- | --- | --- |
| `signature_stamp_yolo.onnx` | ONNX export | First |
| `signature_stamp_yolo.pt` | PyTorch checkpoint | Second |

The files are included in the repository; no separate download is required. Both are loaded through Ultralytics. ONNX inference may require its runtime dependencies to be installed in the environment.

## Detection categories

The included ONNX export records two classes in its metadata:

| Class ID | Category | Output field |
| --- | --- | --- |
| 0 | Signature | `signature` |
| 1 | Stamp | `stamp` |

The current detector code assumes five class IDs, which does not match this two-class export. This is a known prototype limitation; detections may be mislabeled. The source code is preserved as submitted.

The highest-confidence detection in each output category is selected. Bounding boxes use `[x1, y1, x2, y2]` coordinates on the preprocessed image.

## Fallback behavior

If neither custom weight file exists, the code attempts to load `yolo26n.pt`. The visual detector does not use that general pretrained model for signature/stamp predictions. An OpenCV circular-stamp heuristic is also available and can produce false positives.

## Training artifacts

Saved curves and validation annotations are in [`yolo-tune-images/`](../yolo-tune-images/). A training notebook, annotation guide, public dataset provenance, and reproducible training configuration are not included. The ONNX metadata records an export date of January 22, 2026 and Ultralytics version 8.4.7; it does not establish the base-model architecture. Treat the plots as retained artifacts, rather than a reproducible benchmark or evidence of document-level accuracy.

## Runtime verification

The included ONNX file passes `onnx.checker.check_model` and executes with ONNX Runtime's CPU provider. Its input is a float32 tensor shaped `[1, 3, 1280, 1280]`; its output is `[1, 6, 33600]`. The smoke check used the included annotated validation montage, so it establishes executable weights, not detection accuracy. The preserved pipeline was also exercised on `samples/quotation.png` using the EasyOCR fallback; its imperfect result is saved in `sample_output/demo_result.json`.

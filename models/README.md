# Trained YOLO Models

Place your trained model file(s) here.

## Required Model

- **`signature_stamp_yolo.pt`** - Trained YOLOv8 model for signature/stamp detection

## How to Get the Model

1. **Annotate images** using Roboflow (see `docs/annotation_guide.md`)
2. **Run training** in Google Colab (see `yolo_signature_stamp_training.ipynb`)
3. **Download** `signature_stamp_yolo.pt` from Google Drive
4. **Place** the file in this directory

## Usage

Once the model is placed here, `visual_detector.py` will automatically load it:

```python
from src.visual_detector import VisualDetector

detector = VisualDetector(use_yolo=True)
# ✅ Loaded custom signature/stamp model: signature_stamp_yolo.pt
```

## Model Classes

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | signature | Handwritten signatures |
| 1 | stamp | Dealer stamps (circular/rectangular) |

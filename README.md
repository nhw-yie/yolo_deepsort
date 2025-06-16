# Vehicle Tracking and Speed Estimation using YOLOv11 and DeepSORT

This project performs vehicle detection, tracking, and speed estimation on a given video using YOLOv11, DeepSORT, and two speed estimation methods (region-based and reference line-based).

## Features

- Detect vehicles (`car`, `bus`, `truck`) using YOLOv11.
- Track vehicles using DeepSORT tracker.
- Estimate vehicle speed using:
  - Region-based calibration (Perspective-aware PPM).
  - Reference line timing method.
- Count number of vehicles and track movements (in/out).
- Draw bounding boxes, labels, speed, and counting on video.
- Export processed video with all visualizations.

## Dependencies

Install required packages with:

```bash
pip install -r requirements.txt
```

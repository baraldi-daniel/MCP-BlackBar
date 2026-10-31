from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def _model_path() -> Path:
    return Path(
        os.getenv(
            "BLACKBAR_YUNET_MODEL",
            "models/face_detection_yunet_2023mar.onnx",
        )
    )


def _detect_yunet(rgb: np.ndarray) -> list[tuple]:
    model = _model_path()

    if not model.is_file():
        raise RuntimeError(
            f"YuNet model not found: {model}"
        )

    height, width = rgb.shape[:2]

    detector = cv2.FaceDetectorYN.create(
        str(model),
        "",
        (width, height),
        score_threshold=0.40,
        nms_threshold=0.3,
        top_k=5000,
    )

    boxes = []

    for scale in (1.0, 1.5, 2.0):
        scaled = cv2.resize(
            rgb,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR,
        )

        sh, sw = scaled.shape[:2]

        detector.setInputSize((sw, sh))

        _, detections = detector.detect(
            cv2.cvtColor(
                scaled,
                cv2.COLOR_RGB2BGR,
            )
        )

        if detections is None:
            continue

        for face in detections:
            x, y, w, h = (
                float(value) / scale
                for value in face[:4]
            )

            if w >= 12 and h >= 12:
                boxes.append((x, y, x + w, y + h))

    return boxes


def _detect_haar(rgb: np.ndarray) -> list[tuple]:
    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    classifier = cv2.CascadeClassifier(
        cascade_path
    )

    if classifier.empty():
        raise RuntimeError("Haar cascade unavailable")

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )

    detections = classifier.detectMultiScale(
        gray,
        scaleFactor=1.04,
        minNeighbors=3,
        minSize=(12, 12),
    )

    return [
        (
            float(x),
            float(y),
            float(x + w),
            float(y + h),
        )
        for x, y, w, h in detections
    ]


def _iou(a: tuple, b: tuple) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])

    intersection = (
        max(0, x1 - x0)
        * max(0, y1 - y0)
    )

    area_a = (
        (a[2] - a[0])
        * (a[3] - a[1])
    )

    area_b = (
        (b[2] - b[0])
        * (b[3] - b[1])
    )

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0


def _deduplicate(boxes: list[tuple]) -> list[tuple]:
    result = []

    for box in sorted(
        boxes,
        key=lambda b: (
            (b[2] - b[0]) * (b[3] - b[1])
        ),
        reverse=True,
    ):
        if all(_iou(box, other) < 0.35 for other in result):
            result.append(box)

    return result


def mask_faces(
    image: Image.Image,
    return_diagnostics: bool = False,
):
    result = image.convert("RGB").copy()
    rgb = np.asarray(result)

    yunet_boxes = _detect_yunet(rgb)
    haar_boxes = _detect_haar(rgb)

    boxes = _deduplicate(
        yunet_boxes + haar_boxes
    )

    draw = ImageDraw.Draw(result)

    width, height = result.size

    for x0, y0, x1, y1 in boxes:
        face_width = x1 - x0
        face_height = y1 - y0

        margin_x = face_width * 0.10
        margin_y = face_height * 0.10

        draw.ellipse(
            (
                max(0, x0 - margin_x),
                max(0, y0 - margin_y),
                min(width, x1 + margin_x),
                min(height, y1 + margin_y),
            ),
            fill=(0, 0, 0),
        )

    diagnostics = {
        "yunet_available": True,
        "yunet_detections": len(yunet_boxes),
        "haar_detections": len(haar_boxes),
        "faces_masked": len(boxes),
    }

    if return_diagnostics:
        return result, boxes, diagnostics

    return result
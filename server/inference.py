import io
import cv2
import math
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Dict, Any
from ultralytics import YOLO
from torchvision import models, transforms


# --- Константы ---
CLASS_NAMES_RU = ["берёза", "дуб", "ель", "сосна", "тополь"]

TREE_MODEL_PATH = "models/tree_model.pt"
STICK_MODEL_PATH = "models/stick_model.pt"
CLASSIFIER_PATH = "models/classifier.pth"
REAL_STICK_LENGTH_M = 1.0


# --- Утилиты ---
def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    """Морфология + крупнейшая компонента"""
    if mask is None or mask.size == 0:
        return None
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    if mask.max() == 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8) * 255


def measure_tree(mask: np.ndarray, meters_per_px: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Измеряет высоту, ширину кроны и диаметр ствола"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None, None, None
    y_min, y_max = ys.min(), ys.max()
    height_px = max(1, y_max - y_min)
    height_m = height_px * meters_per_px

    # ширина кроны (верхние 70%)
    crown_top = int(y_min)
    crown_bot = int(y_min + 0.7 * height_px)
    crown_w = 0
    for y in range(crown_top, crown_bot):
        row = np.where(mask[y] > 0)[0]
        if len(row) > 0:
            crown_w = max(crown_w, row.max() - row.min())
    crown_m = crown_w * meters_per_px if crown_w else None

    # диаметр ствола (нижние 20%)
    trunk_top = int(y_max - 0.2 * height_px)
    trunk_w = []
    for y in range(trunk_top, y_max):
        row = np.where(mask[y] > 0)[0]
        if len(row) > 0:
            width = row.max() - row.min()
            if width > 10:
                trunk_w.append(width)
    trunk_m = (float(np.mean(trunk_w)) * meters_per_px) if trunk_w else None

    return round(height_m, 2), round(crown_m, 2) if crown_m else None, round(trunk_m, 2) if trunk_m else None


# --- Загрузка моделей ---
device = torch.device("cpu")

tree_model = YOLO(TREE_MODEL_PATH)
stick_model = YOLO(STICK_MODEL_PATH)

clf = models.resnet18(weights=None)
clf.fc = torch.nn.Linear(clf.fc.in_features, len(CLASS_NAMES_RU))
clf.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
clf.eval()

transform_clf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# --- Основные функции ---
def _np_from_file(file_bytes: bytes) -> np.ndarray:
    pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _classify(img_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[str]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = Image.fromarray(cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    tens = transform_clf(crop).unsqueeze(0)
    with torch.no_grad():
        logits = clf(tens)
        cls_id = int(torch.argmax(logits, dim=1).item())
    return CLASS_NAMES_RU[cls_id]


def analyze_image(image_bytes: bytes,
                  manual_points: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
                  ) -> Dict[str, Any]:
    """Главная функция для FastAPI"""
    img = _np_from_file(image_bytes)
    H, W = img.shape[:2]

    # --- дерево (YOLOv8-seg)
    r_tree = tree_model(img)[0]
    if r_tree.masks is None or len(r_tree.masks) == 0:
        return {"ok": False, "reason": "tree_not_found"}

    areas, valid_masks, valid_boxes = [], [], []
    for i, md in enumerate(r_tree.masks.data):
        mask = (md.cpu().numpy() > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = postprocess_mask(mask)
        if mask is not None and mask.max() > 0:
            area = int(cv2.countNonZero(mask))
            if area < 500:
                continue
            areas.append(area)
            valid_masks.append(mask)
            valid_boxes.append(r_tree.boxes.xyxy[i].cpu().numpy().astype(int))

    if not valid_masks:
        return {"ok": False, "reason": "tree_mask_invalid"}

    idx = int(np.argmax(areas))
    tree_mask = valid_masks[idx]
    xyxy = tuple(int(v) for v in valid_boxes[idx])

    # --- масштаб
    scale_m_per_px = None
    scale_source = None

    if manual_points is not None:
        (x1, y1), (x2, y2) = manual_points
        stick_px = math.hypot(x2 - x1, y2 - y1)
        if stick_px >= 5:
            scale_m_per_px = REAL_STICK_LENGTH_M / stick_px
            scale_source = "manual"
    if scale_m_per_px is None:
        r_stick = stick_model(img, conf=0.3)[0]
        if len(r_stick.boxes) > 0:
            best = max(r_stick.boxes, key=lambda b: (b.xyxy[0][3] - b.xyxy[0][1]))
            x1s, y1s, x2s, y2s = best.xyxy[0].cpu().numpy().astype(int)
            stick_height_px = max(1, (y2s - y1s))
            s = REAL_STICK_LENGTH_M / stick_height_px
            if 0.001 < s < 0.05:
                scale_m_per_px, scale_source = s, "auto"

    # --- измерения
    meters_per_px = scale_m_per_px if scale_m_per_px is not None else 1.0
    h_m, cw_m, dbh_m = measure_tree(tree_mask, meters_per_px)

    # --- классификация
    species = _classify(img, xyxy) or "не определён"

    return {
        "ok": True,
        "species": species,
        "height_m": h_m,
        "crown_width_m": cw_m,
        "trunk_diameter_m": dbh_m,
        "scale_m_per_px": round(scale_m_per_px, 6) if scale_m_per_px else None,
        "scale_source": scale_source,
        "needs_manual_scale": scale_m_per_px is None,
    }

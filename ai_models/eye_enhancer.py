"""
ai_models/eye_enhancer.py
=========================
目の輝き強調モジュール（OpenCV ベース — GPU 不要）。

機能:
  1. キャッチライト (catchlight) 強調 — 瞳の反射光を明るく
  2. 白目 (sclera) の白さ補正 — 血管の赤みを抑えて透明感を出す
  3. 瞳ディテール — 虹彩のコントラストを強調

すべて LAB/HSV 色空間で処理し、色飽和を防止。
"""

from __future__ import annotations

import cv2
import numpy as np
from core.types import ImageRGB


# ============================================================
# ユーティリティ: 目の領域マスク生成
# ============================================================

# MediaPipe FaceMesh の目周囲ランドマーク索引
_LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# 虹彩ランドマーク (MediaPipe refine_landmarks=True で利用可能)
_LEFT_IRIS_IDX = [474, 475, 476, 477]
_RIGHT_IRIS_IDX = [469, 470, 471, 472]


def _eye_mask(landmarks, indices: list[int], img_w: int, img_h: int,
              expand: float = 1.0) -> np.ndarray:
    """ランドマークから目の領域マスクを生成。"""
    pts = []
    for i in indices:
        x = int(landmarks.points[i].x * img_w)
        y = int(landmarks.points[i].y * img_h)
        pts.append((x, y))
    pts = np.array(pts, dtype=np.int32)

    if expand > 1.0:
        cx, cy = pts.mean(axis=0).astype(int)
        pts = ((pts - [cx, cy]) * expand + [cx, cy]).astype(np.int32)

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _iris_center_radius(landmarks, indices: list[int],
                        img_w: int, img_h: int) -> tuple[int, int, int]:
    """虹彩の中心と半径を計算。"""
    pts = []
    for i in indices:
        x = int(landmarks.points[i].x * img_w)
        y = int(landmarks.points[i].y * img_h)
        pts.append((x, y))
    pts = np.array(pts, dtype=np.float32)
    cx, cy = pts.mean(axis=0).astype(int)
    radius = int(np.linalg.norm(pts - [cx, cy], axis=1).max())
    return cx, cy, max(radius, 3)


# ============================================================
# 1. キャッチライト強調
# ============================================================

def enhance_catchlight(
    image: ImageRGB,
    landmarks,
    img_w: int,
    img_h: int,
    strength: float = 0.5,
) -> ImageRGB:
    """瞳の反射光（キャッチライト）を明るくして目に輝きを与える。

    仕組み:
      - 虹彩領域内のハイライト（上位 5% の輝度ピクセル）を検出
      - LAB L チャンネルのみを持ち上げ → 色濁りなし
      - ガウシアンボケでスムーズなグロー効果

    Args:
        image:    入力画像 (RGB, uint8)
        landmarks: MediaPipe ランドマーク
        img_w, img_h: 画像幅・高さ
        strength: 0.0(無効) → 1.0(最大輝き)
    """
    if strength <= 0.0:
        return image.copy()

    s = min(max(strength, 0.0), 1.0)
    result = image.copy()

    for iris_idx in [_LEFT_IRIS_IDX, _RIGHT_IRIS_IDX]:
        try:
            cx, cy, radius = _iris_center_radius(landmarks, iris_idx, img_w, img_h)
        except (IndexError, KeyError):
            continue

        # 虹彩マスク（円形）
        iris_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.circle(iris_mask, (cx, cy), radius, 255, -1)

        # LAB に変換して L チャンネルのハイライトを検出
        bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_ch = lab[:, :, 0].astype(np.float32)

        # 虹彩内の輝度統計
        iris_pixels = l_ch[iris_mask > 0]
        if len(iris_pixels) == 0:
            continue
        threshold = np.percentile(iris_pixels, 95)

        # ハイライトマスク（上位 5%）
        highlight_mask = (l_ch > threshold).astype(np.uint8) * 255
        highlight_mask = cv2.bitwise_and(highlight_mask, iris_mask)

        # グロー効果: ガウシアンぼかし
        glow = cv2.GaussianBlur(highlight_mask.astype(np.float32), (0, 0), radius * 0.4)
        glow = glow / (glow.max() + 1e-6) * 255

        # L チャンネルを持ち上げ
        boost = s * 60  # 最大 +60
        l_ch = np.clip(l_ch + glow / 255.0 * boost, 0, 255)
        lab[:, :, 0] = l_ch.astype(np.uint8)

        bgr_out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        result = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)

    return result


# ============================================================
# 2. 白目（強膜）の白さ補正
# ============================================================

def whiten_sclera(
    image: ImageRGB,
    landmarks,
    img_w: int,
    img_h: int,
    strength: float = 0.3,
) -> ImageRGB:
    """白目の赤みを抑えて透明感を出す。

    仕組み:
      - 目の領域マスクから虹彩マスクを引く → 白目マスク
      - HSV の S（彩度）を下げる → 血管の赤みが消える
      - LAB の L を微上げ → 白さが増す

    Args:
        strength: 0.0(無効) → 1.0(最大)
    """
    if strength <= 0.0:
        return image.copy()

    s = min(max(strength, 0.0), 1.0)
    result = image.copy()

    for eye_idx, iris_idx in [(_LEFT_EYE_IDX, _LEFT_IRIS_IDX),
                               (_RIGHT_EYE_IDX, _RIGHT_IRIS_IDX)]:
        try:
            eye_mask = _eye_mask(landmarks, eye_idx, img_w, img_h, expand=1.0)
            cx, cy, radius = _iris_center_radius(landmarks, iris_idx, img_w, img_h)
        except (IndexError, KeyError):
            continue

        # 虹彩マスク
        iris_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.circle(iris_mask, (cx, cy), int(radius * 1.3), 255, -1)

        # 白目マスク = 目マスク - 虹彩
        sclera_mask = cv2.subtract(eye_mask, iris_mask)
        sclera_mask = cv2.GaussianBlur(sclera_mask, (5, 5), 2)

        if np.sum(sclera_mask) == 0:
            continue

        # HSV: 彩度を下げる
        bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat_reduce = s * 0.7  # 最大 70% 彩度削減
        mask_f = sclera_mask.astype(np.float32) / 255.0

        hsv[:, :, 1] = hsv[:, :, 1] * (1.0 - sat_reduce * mask_f)

        # LAB: L を微上げ
        hsv_u8 = np.clip(hsv, 0, 255).astype(np.uint8)
        hsv_u8[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179).astype(np.uint8)
        bgr_out = cv2.cvtColor(hsv_u8, cv2.COLOR_HSV2BGR)
        lab = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2LAB).astype(np.float32)

        l_boost = s * 15  # 最大 +15
        lab[:, :, 0] = np.clip(lab[:, :, 0] + l_boost * mask_f, 0, 255)
        lab_u8 = lab.astype(np.uint8)
        bgr_final = cv2.cvtColor(lab_u8, cv2.COLOR_LAB2BGR)
        result = cv2.cvtColor(bgr_final, cv2.COLOR_BGR2RGB)

    return result


# ============================================================
# 3. 瞳ディテール強調
# ============================================================

def enhance_iris_detail(
    image: ImageRGB,
    landmarks,
    img_w: int,
    img_h: int,
    strength: float = 0.3,
) -> ImageRGB:
    """虹彩のコントラストとディテールを強調。

    仕組み:
      - 虹彩マスクを生成し、CLAHE を虹彩のみに適用
      - マスクブレンドで周囲との境界を滑らかに

    Args:
        strength: 0.0(無効) → 1.0(最大)
    """
    if strength <= 0.0:
        return image.copy()

    s = min(max(strength, 0.0), 1.0)
    result = image.copy()

    for iris_idx in [_LEFT_IRIS_IDX, _RIGHT_IRIS_IDX]:
        try:
            cx, cy, radius = _iris_center_radius(landmarks, iris_idx, img_w, img_h)
        except (IndexError, KeyError):
            continue

        # 虹彩マスク（ソフトエッジ）
        iris_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.circle(iris_mask, (cx, cy), radius, 255, -1)
        iris_mask = cv2.GaussianBlur(iris_mask, (5, 5), 2)

        if np.sum(iris_mask) == 0:
            continue

        # CLAHE を L チャンネルに適用
        bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_ch = lab[:, :, 0]

        clip_limit = 2.0 + s * 3.0  # [2.0, 5.0]
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(4, 4))
        l_enhanced = clahe.apply(l_ch)

        # マスクブレンド
        mask_f = iris_mask.astype(np.float32) / 255.0
        alpha = s * 0.7
        l_blended = l_ch.astype(np.float32) * (1.0 - alpha * mask_f) + \
                     l_enhanced.astype(np.float32) * alpha * mask_f
        lab[:, :, 0] = np.clip(l_blended, 0, 255).astype(np.uint8)

        bgr_out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        result = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)

    return result

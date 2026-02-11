"""
ai_models/blemish_detector.py
=============================
シミ・ホクロ自動検出 & OpenCV Inpainting 除去（GPU 不要）。

処理フロー:
  1. 肌マスク内のダークスポットを自動検出
  2. 検出結果のマスクを生成
  3. OpenCV Inpainting (Telea or Navier-Stokes) で除去

すべて LAB 色空間ベースで処理。
"""

from __future__ import annotations

import cv2
import numpy as np
from core.types import ImageRGB


# ============================================================
# 1. シミ・ホクロ検出
# ============================================================

def detect_blemishes(
    image: ImageRGB,
    skin_mask: np.ndarray,
    sensitivity: float = 0.5,
) -> np.ndarray:
    """肌マスク内のシミ・ホクロを自動検出してマスクを返す。

    検出アルゴリズム:
      1. LAB の L チャンネルで輝度マップを取得
      2. 肌領域の平均輝度を基準に、局所的に暗いスポットを検出
      3. 面積フィルタ: 1px〜肌面積の 0.5% のスポットのみ抽出
         (巨大な影や、ノイズ的な 1px スポットを除外)

    Args:
        image:       入力画像 (RGB, uint8)
        skin_mask:   肌領域マスク (H, W) uint8, 0 or 255
        sensitivity: 0.0(ほぼ検出しない) → 1.0(積極的に検出)

    Returns:
        blemish_mask: 検出されたシミ・ホクロのマスク (H, W) uint8
    """
    if sensitivity <= 0.0:
        return np.zeros(image.shape[:2], dtype=np.uint8)

    s = min(max(sensitivity, 0.0), 1.0)

    # LAB に変換して L チャンネルを取得
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0].astype(np.float32)

    # 肌領域のみを対象に
    skin_f = skin_mask.astype(np.float32) / 255.0
    skin_pixels = l_ch[skin_mask > 0]
    if len(skin_pixels) == 0:
        return np.zeros(image.shape[:2], dtype=np.uint8)

    # 局所平均との差分で暗いスポットを検出
    # ガウシアンぼかしで局所平均を近似
    blur_size = max(31, int(min(image.shape[:2]) * 0.05) | 1)  # 奇数に
    local_mean = cv2.GaussianBlur(l_ch, (blur_size, blur_size), 0)

    # 暗いスポット = 局所平均よりも一定以上暗い領域
    threshold = 8.0 + (1.0 - s) * 15.0  # 感度高い→閾値低い [8, 23]
    dark_spots = (local_mean - l_ch) > threshold

    # 肌マスク内のみ
    blemish_raw = (dark_spots * (skin_mask > 0)).astype(np.uint8) * 255

    # モルフォロジーで細かいノイズを除去
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    blemish_clean = cv2.morphologyEx(blemish_raw, cv2.MORPH_OPEN, kernel)
    blemish_clean = cv2.morphologyEx(blemish_clean, cv2.MORPH_CLOSE, kernel)

    # 面積フィルタ: 小さすぎ / 大きすぎるスポットを除外
    contours, _ = cv2.findContours(
        blemish_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    skin_area = np.sum(skin_mask > 0)
    min_area = max(4, int(skin_area * 0.0001))  # 肌面積の 0.01%
    max_area = int(skin_area * 0.005)            # 肌面積の 0.5%

    blemish_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            cv2.drawContours(blemish_mask, [cnt], -1, 255, -1)

    # エッジを少し膨張させてインペイント境界を滑らかに
    blemish_mask = cv2.dilate(blemish_mask, kernel, iterations=1)

    return blemish_mask


# ============================================================
# 2. シミ・ホクロ除去 (Inpainting)
# ============================================================

def remove_blemishes(
    image: ImageRGB,
    blemish_mask: np.ndarray,
    strength: float = 0.7,
    method: str = "telea",
) -> ImageRGB:
    """検出されたシミ・ホクロを OpenCV Inpainting で除去。

    Args:
        image:        入力画像 (RGB, uint8)
        blemish_mask: シミ・ホクロマスク (H, W) uint8
        strength:     0.0(元画像) → 1.0(完全除去)
        method:       "telea" (高速) or "ns" (Navier-Stokes, 高品質)

    Returns:
        シミ除去済み画像 (RGB, uint8)
    """
    if strength <= 0.0 or np.sum(blemish_mask) == 0:
        return image.copy()

    s = min(max(strength, 0.0), 1.0)

    # BGR に変換してインペイント
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    inpaint_radius = 5  # インペイント半径（ピクセル）
    if method == "ns":
        flag = cv2.INPAINT_NS
    else:
        flag = cv2.INPAINT_TELEA

    inpainted = cv2.inpaint(bgr, blemish_mask, inpaint_radius, flag)
    inpainted_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)

    # strength でブレンド
    blended = cv2.addWeighted(
        image, 1.0 - s,
        inpainted_rgb, s,
        0,
    )
    return blended


# ============================================================
# 3. ワンショット: 検出 + 除去
# ============================================================

def auto_remove_blemishes(
    image: ImageRGB,
    skin_mask: np.ndarray,
    sensitivity: float = 0.5,
    strength: float = 0.7,
) -> tuple[ImageRGB, np.ndarray, int]:
    """シミ・ホクロの検出と除去をワンショットで実行。

    Args:
        image:       入力画像 (RGB, uint8)
        skin_mask:   肌領域マスク
        sensitivity: 検出感度 (0.0-1.0)
        strength:    除去強度 (0.0-1.0)

    Returns:
        (除去済み画像, 検出マスク, 検出スポット数)
    """
    blemish_mask = detect_blemishes(image, skin_mask, sensitivity)

    # スポット数カウント
    contours, _ = cv2.findContours(
        blemish_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    n_spots = len(contours)

    if n_spots == 0:
        return image.copy(), blemish_mask, 0

    result = remove_blemishes(image, blemish_mask, strength)
    return result, blemish_mask, n_spots

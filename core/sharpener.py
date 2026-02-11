"""
core/sharpener.py
=================
アンシャープマスク (USM) による画像のシャープネス強調。
最終仕上げとして、レタッチ後にディテールの鮮明さを回復する。
"""

from __future__ import annotations

import cv2
import numpy as np
from core.types import ImageRGB


def sharpen(
    image: ImageRGB,
    strength: float = 0.3,
) -> ImageRGB:
    """アンシャープマスクでシャープネスを強調する。

    原理: 元画像 - ぼかし画像 = ディテール成分
         元画像 + ディテール成分 × 強度 = シャープ画像

    Args:
        image: 入力画像 (RGB, uint8)
        strength: 0.0(変化なし) → 1.0(最大シャープ)

    Returns:
        シャープネス強調済み画像
    """
    if strength <= 0.0:
        return image.copy()

    s = min(max(strength, 0.0), 1.0)

    # ぼかし半径 (強度に応じて 3〜7)
    ksize = int(3 + s * 4)
    if ksize % 2 == 0:
        ksize += 1

    # ガウシアンぼかし
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)

    # ディテール成分を抽出してブレンド
    # amount: 1.0〜3.0
    amount = 1.0 + s * 2.0

    sharpened = cv2.addWeighted(
        image, amount,
        blurred, -(amount - 1.0),
        0,
    )

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def selective_sharpen(
    image: ImageRGB,
    mask: np.ndarray,
    strength: float = 0.3,
) -> ImageRGB:
    """マスク領域のみにシャープネスを適用。

    目や唇など特定パーツのみを鮮明にしたい場合に使用。

    Args:
        image: 入力画像 (RGB, uint8)
        mask: 適用領域マスク (0=適用しない, 255=適用)
        strength: 0.0(変化なし) → 1.0(最大シャープ)

    Returns:
        部分シャープネス済み画像
    """
    if strength <= 0.0:
        return image.copy()

    sharpened = sharpen(image, strength)

    # ソフトマスクブレンド
    soft_mask = cv2.GaussianBlur(mask, (7, 7), 0)
    alpha = soft_mask.astype(np.float32) / 255.0
    alpha = np.stack([alpha] * 3, axis=-1)

    result = (
        image.astype(np.float32) * (1.0 - alpha)
        + sharpened.astype(np.float32) * alpha
    ).astype(np.uint8)

    return result

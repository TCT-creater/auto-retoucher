"""
core/skin_smoother.py
=====================
バイラテラルフィルタ＋肌マスクによる自然な肌スムージング。
目・眉・唇の鮮明さを保ちながら、肌のシワ・毛穴のみを滑らかにする。
"""

from __future__ import annotations

import cv2
import numpy as np
from core.types import ImageRGB, MaskImage


def smooth_skin(
    image: ImageRGB,
    skin_mask: MaskImage,
    strength: float = 0.5,
) -> ImageRGB:
    """肌領域のみにバイラテラルフィルタを適用する。

    Args:
        image: 入力画像 (RGB, uint8)
        skin_mask: 肌領域マスク (0=非肌, 255=肌)
                   FaceDetector.get_skin_mask() で生成されたもの
        strength: 補正強度 (0.0=無補正, 1.0=最大スムージング)

    Returns:
        スムージング済み画像 (RGB, uint8)
    """
    if strength <= 0.0:
        return image.copy()

    # 強度を0-1からパラメータに変換
    # d: ピクセル近傍の直径 (5〜15)
    # sigmaColor: 色空間フィルタリングの標準偏差 (20〜80)
    # sigmaSpace: 座標空間フィルタリングの標準偏差 (20〜80)
    s = min(max(strength, 0.0), 1.0)
    d = int(5 + s * 10)             # 5 → 15
    sigma_color = 20 + s * 60       # 20 → 80
    sigma_space = 20 + s * 60       # 20 → 80

    # パスの回数（強度に応じて1〜3回）
    passes = 1 + int(s * 2)         # 1 → 3

    # --- スムージング処理 ---
    smoothed = image.copy()
    for _ in range(passes):
        smoothed = cv2.bilateralFilter(
            smoothed, d, sigma_color, sigma_space
        )

    # --- マスクによるブレンド ---
    # マスクの境界をぼかして自然な遷移にする
    blur_radius = max(int(15 * s), 3)
    if blur_radius % 2 == 0:
        blur_radius += 1
    soft_mask = cv2.GaussianBlur(
        skin_mask, (blur_radius, blur_radius), 0
    )

    # float32 に変換してブレンド
    alpha = soft_mask.astype(np.float32) / 255.0
    alpha = np.stack([alpha] * 3, axis=-1)  # (H, W, 3)

    # ブレンド元の強度も調整
    blend_strength = min(s * 1.2, 1.0)
    alpha *= blend_strength

    result = (
        image.astype(np.float32) * (1.0 - alpha)
        + smoothed.astype(np.float32) * alpha
    ).astype(np.uint8)

    return result


def enhance_skin_texture(
    image: ImageRGB,
    skin_mask: MaskImage,
    strength: float = 0.3,
) -> ImageRGB:
    """肌のテクスチャを保ちつつ色ムラを均一化する。

    バイラテラルフィルタとは別のアプローチ:
    ガウシアンぼかし + 元画像の高周波成分で
    「肌のキメを残しつつ色合いを均一に」する効果。

    Args:
        image: 入力画像 (RGB, uint8)
        skin_mask: 肌領域マスク
        strength: 補正強度 (0.0-1.0)

    Returns:
        テクスチャ補正済み画像 (RGB, uint8)
    """
    if strength <= 0.0:
        return image.copy()

    s = min(max(strength, 0.0), 1.0)

    # 大きなぼかし（低周波成分 = 全体の色合い）
    ksize = int(21 + s * 40)
    if ksize % 2 == 0:
        ksize += 1
    low_freq = cv2.GaussianBlur(image, (ksize, ksize), 0)

    # 高周波成分（元画像 - 低周波 = テクスチャのディテール）
    img_f = image.astype(np.float32)
    low_f = low_freq.astype(np.float32)
    high_freq = img_f - low_f + 128.0  # 128をオフセットに

    # 均一化された色合い + テクスチャを合成
    unified = low_f + (high_freq - 128.0) * (1.0 - s * 0.5)
    unified = np.clip(unified, 0, 255).astype(np.uint8)

    # マスクでブレンド
    soft_mask = cv2.GaussianBlur(skin_mask, (15, 15), 0)
    alpha = soft_mask.astype(np.float32) / 255.0 * s
    alpha = np.stack([alpha] * 3, axis=-1)

    result = (
        img_f * (1.0 - alpha) + unified.astype(np.float32) * alpha
    ).astype(np.uint8)

    return result


def reduce_shine(
    image: ImageRGB,
    skin_mask: MaskImage,
    strength: float = 0.3,
) -> ImageRGB:
    """肌のテカリ（ハイライト過多）を抑制する。

    HSV色空間のV（明度）チャンネルで過度に明るい部分を検出し、
    トーンダウンする。

    Args:
        image: 入力画像 (RGB, uint8)
        skin_mask: 肌領域マスク
        strength: 補正強度 (0.0-1.0)

    Returns:
        テカリ抑制済み画像 (RGB, uint8)
    """
    if strength <= 0.0:
        return image.copy()

    s = min(max(strength, 0.0), 1.0)
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    # V チャンネルのハイライトを検出
    v_channel = hsv[:, :, 2]
    threshold = 200 - s * 40  # 200 → 160
    shine_mask = (v_channel > threshold).astype(np.float32)

    # 肌マスクとの AND
    skin_f = skin_mask.astype(np.float32) / 255.0
    combined_mask = shine_mask * skin_f

    # ガウシアンぼかしでマスク境界を滑らかに
    combined_mask = cv2.GaussianBlur(combined_mask, (11, 11), 0)

    # テカリを抑制（V値を下げる）
    reduction = s * 30  # 最大 30 ポイント減少
    hsv[:, :, 2] -= combined_mask * reduction
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

    result_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    return result_rgb

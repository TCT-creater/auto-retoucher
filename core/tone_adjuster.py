"""
core/tone_adjuster.py
=====================
精密な色彩管理に基づくトーン調整モジュール。

設計原則:
  1. 輝度と色情報を分離: LAB / HSV 色空間を活用
  2. float32 パイプライン: 全演算を float32 で実行し、最後にのみ uint8 へ変換
  3. クリッピング防止: ソフトクリップ関数で色飽和を防ぐ
  4. 色濁り防止: 明るさは L、彩度は S、色温度は b* 軸を操作
"""

from __future__ import annotations

import cv2
import numpy as np
from core.types import ImageRGB


# ============================================================
# ユーティリティ: ソフトクリップ & 色空間変換
# ============================================================

def _soft_clip(x: np.ndarray, low: float = 0.0, high: float = 255.0,
               margin: float = 20.0) -> np.ndarray:
    """ハイライトが不自然に飛ばないソフトクリッピング。

    通常の np.clip は即座に上限/下限に張り付くが、
    ソフトクリップは境界付近で滑らかに減速する。

    計算式:
        high 付近 (x > high - margin):
            x_new = high - margin + margin * tanh((x - (high - margin)) / margin)
        low 付近 (x < low + margin):
            x_new = low + margin - margin * tanh(((low + margin) - x) / margin)

    これにより:
        - x = high のとき → x_new ≈ high - margin + margin * tanh(1) ≈ high - 4.8
        - x = high + 100 のとき → x_new ≈ high - margin + margin * tanh(6) ≈ high - 0.05
        → 真っ白(255)に飛ばず、255近辺で滑らかに収束
    """
    result = x.copy()

    # ハイライト側のソフトクリップ
    hi_mask = result > (high - margin)
    if np.any(hi_mask):
        over = result[hi_mask] - (high - margin)
        result[hi_mask] = (high - margin) + margin * np.tanh(over / margin)

    # シャドウ側のソフトクリップ
    lo_mask = result < (low + margin)
    if np.any(lo_mask):
        under = (low + margin) - result[lo_mask]
        result[lo_mask] = (low + margin) - margin * np.tanh(under / margin)

    return result


def _rgb_to_lab(image: ImageRGB) -> np.ndarray:
    """RGB → LAB (float32)。OpenCV は BGR 入力を要求する。"""
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def _lab_to_rgb(lab: np.ndarray) -> ImageRGB:
    """LAB (float32) → RGB (uint8)。"""
    lab_u8 = np.clip(lab, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(lab_u8, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _rgb_to_hsv(image: ImageRGB) -> np.ndarray:
    """RGB → HSV (float32)。"""
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)


def _hsv_to_rgb(hsv: np.ndarray) -> ImageRGB:
    """HSV (float32) → RGB (uint8)。"""
    hsv_u8 = np.clip(hsv, 0, 255).astype(np.uint8)
    hsv_u8[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179).astype(np.uint8)
    bgr = cv2.cvtColor(hsv_u8, cv2.COLOR_HSV2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ============================================================
# 1. CLAHE (適応的ヒストグラム均等化)
# ============================================================

def apply_clahe(
    image: ImageRGB,
    strength: float = 0.5,
) -> ImageRGB:
    """LAB の L チャンネルに CLAHE を適用。

    局所コントラストを改善し、平坦な画像に奥行きを与える。
    clipLimit と tileGridSize をスライダー強度に連動。

    技術仕様:
        - clipLimit:    1.0 + strength × 3.0  → [1.0, 4.0]
        - tileGridSize: (8, 8) 固定 (標準的なポートレートに最適)
        - LAB の L チャンネルのみ操作 → 色情報 (a*, b*) は不変
        - 最終ブレンド: α = strength × 0.8 (最大 80% で過剰補正を防止)

    Args:
        image: 入力画像 (RGB, uint8)
        strength: 0.0(無効) → 1.0(最大)

    Returns:
        CLAHE 適用済み画像
    """
    if strength <= 0.0:
        return image.copy()

    s = min(max(strength, 0.0), 1.0)

    # パラメータ計算
    clip_limit = 1.0 + s * 3.0  # [1.0, 4.0]

    # LAB 変換 → L チャンネル抽出
    lab = _rgb_to_lab(image)
    l_channel = lab[:, :, 0].astype(np.uint8)

    # CLAHE 適用
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    # L チャンネルを差し替え
    lab_enhanced = lab.copy()
    lab_enhanced[:, :, 0] = l_enhanced.astype(np.float32)

    result_rgb = _lab_to_rgb(lab_enhanced)

    # 元画像とブレンド（過補正防止）
    alpha = s * 0.8
    blended = cv2.addWeighted(image, 1.0 - alpha, result_rgb, alpha, 0)
    return blended


# ============================================================
# 2. 明るさ調整 (LAB L チャンネル)
# ============================================================

def adjust_brightness(
    image: ImageRGB,
    strength: float = 0.5,
) -> ImageRGB:
    """LAB の L チャンネル (輝度) のみを操作する明るさ調整。

    RGB 空間での単純な加算は色濁りを起こすため、
    輝度と色を分離した LAB 空間で L のみを変更する。

    計算式:
        delta = (strength - 0.5) × 100.0   → [-50, +50]
        L_new = soft_clip(L + delta, 0, 255, margin=20)

    ソフトクリップにより:
        - strength=1.0, L=240 のとき:
          L + 50 = 290 → soft_clip(290) ≈ 255 - 20 + 20*tanh(3.5) ≈ 254.7
          → 真っ白(255)に到達せず、ハイライトの階調が維持される

    Args:
        image: 入力画像 (RGB, uint8)
        strength: 0.0(暗く) ← 0.5(中立) → 1.0(明るく)

    Returns:
        明るさ調整済み画像
    """
    if abs(strength - 0.5) < 0.01:
        return image.copy()

    delta = (strength - 0.5) * 100.0  # [-50, +50]

    lab = _rgb_to_lab(image)
    lab[:, :, 0] = _soft_clip(lab[:, :, 0] + delta, 0, 255, margin=20)

    return _lab_to_rgb(lab)


# ============================================================
# 3. コントラスト調整 (LAB L チャンネル)
# ============================================================

def adjust_contrast(
    image: ImageRGB,
    strength: float = 0.5,
) -> ImageRGB:
    """LAB の L チャンネル (輝度) のコントラストを調整。

    L チャンネルの平均値を中心にスケーリングすることで、
    色情報 (a*, b*) に影響を与えずにコントラストを変更。

    計算式:
        factor = 0.5 + strength × 1.0   → [0.5, 1.5]
        L_mean = mean(L)
        L_new = soft_clip(L_mean + (L - L_mean) × factor, 0, 255)

    Args:
        image: 入力画像 (RGB, uint8)
        strength: 0.0(低) ← 0.5(中立) → 1.0(高)

    Returns:
        コントラスト調整済み画像
    """
    if abs(strength - 0.5) < 0.01:
        return image.copy()

    factor = 0.5 + strength * 1.0  # [0.5, 1.5]

    lab = _rgb_to_lab(image)
    l_ch = lab[:, :, 0]
    l_mean = l_ch.mean()
    lab[:, :, 0] = _soft_clip(l_mean + (l_ch - l_mean) * factor, 0, 255)

    return _lab_to_rgb(lab)


# ============================================================
# 4. 彩度調整 (HSV S チャンネル)
# ============================================================

def adjust_saturation(
    image: ImageRGB,
    strength: float = 0.5,
) -> ImageRGB:
    """HSV の S チャンネル (彩度) のみを操作する彩度調整。

    RGB 空間での彩度変更は色相が移動してしまうため、
    HSV 空間の S チャンネルのみをスケーリングする。

    計算式:
        scale = strength × 2.0   → [0.0, 2.0]
        S_new = clip(S × scale, 0, 255)

    Args:
        image: 入力画像 (RGB, uint8)
        strength: 0.0(白黒) ← 0.5(中立) → 1.0(鮮やか)

    Returns:
        彩度調整済み画像
    """
    if abs(strength - 0.5) < 0.01:
        return image.copy()

    scale = strength * 2.0  # [0.0, 2.0]

    hsv = _rgb_to_hsv(image)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)

    return _hsv_to_rgb(hsv)


# ============================================================
# 5. ホワイトバランス (Gray World 仮説)
# ============================================================

def adjust_white_balance(
    image: ImageRGB,
    strength: float = 0.3,
) -> ImageRGB:
    """Gray World 仮説に基づく自動ホワイトバランス補正。

    仮説: 自然画像の各チャンネル平均は灰色 (等しい値) になるべき。

    計算式:
        avg_R, avg_G, avg_B = 各チャンネルの平均値
        avg_all = (avg_R + avg_G + avg_B) / 3
        scale_c = avg_all / avg_c   (各チャンネル c について)
        scale_c_blended = 1.0 + (scale_c - 1.0) × strength

    LAB 空間で実行: a* (green-red), b* (blue-yellow) の平均を 0 に近づける。

    Args:
        image: 入力画像 (RGB, uint8)
        strength: 0.0(無効) → 1.0(完全補正)

    Returns:
        ホワイトバランス補正済み画像
    """
    if strength <= 0.0:
        return image.copy()

    s = min(max(strength, 0.0), 1.0)

    lab = _rgb_to_lab(image)

    # LAB の a*, b* の平均を 0 に近づける（128 が中性点）
    a_mean = lab[:, :, 1].mean()
    b_mean = lab[:, :, 2].mean()

    # 中性点 (128) からのオフセットを補正
    a_correction = (128.0 - a_mean) * s
    b_correction = (128.0 - b_mean) * s

    lab[:, :, 1] = np.clip(lab[:, :, 1] + a_correction, 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2] + b_correction, 0, 255)

    return _lab_to_rgb(lab)


# ============================================================
# 6. 色温度調整 (LAB b* 軸)
# ============================================================

def adjust_warmth(
    image: ImageRGB,
    strength: float = 0.5,
) -> ImageRGB:
    """LAB の b* 軸 (blue ↔ yellow) で色温度を調整。

    RGB 空間での R+/B- は色飽和を引き起こすため、
    LAB の b* 軸を操作することで色相を正確に制御する。

    計算式:
        offset = (strength - 0.5) × 40.0   → [-20, +20]
        b*_new = clip(b* + offset, 0, 255)

    b* 軸の意味:
        - b* < 128 → 青み (寒色)
        - b* = 128 → 中性
        - b* > 128 → 黄み (暖色)

    Args:
        image: 入力画像 (RGB, uint8)
        strength: 0.0(寒色/青み) ← 0.5(中立) → 1.0(暖色/黄み)

    Returns:
        色温度調整済み画像
    """
    if abs(strength - 0.5) < 0.01:
        return image.copy()

    offset = (strength - 0.5) * 40.0  # [-20, +20]

    lab = _rgb_to_lab(image)
    lab[:, :, 2] = np.clip(lab[:, :, 2] + offset, 0, 255)

    return _lab_to_rgb(lab)


# ============================================================
# 7. 自動露出解析 (画像から最適なパラメータを算出)
# ============================================================

def auto_adjust_exposure(image: ImageRGB) -> dict[str, float]:
    """画像の輝度・コントラストを解析し、最適なスライダー値を返す。

    解析アルゴリズム:
      1. LAB L チャンネルの平均輝度を算出
      2. 目標輝度 (130) との差分から brightness 値を計算
      3. L のヒストグラム幅からコントラスト不足を判定
      4. 彩度の平均からサチュレーション補正量を判定

    Returns:
        dict:
            brightness:  0.0-1.0 (0.5=中立)
            contrast:    0.0-1.0 (0.5=中立)
            saturation:  0.0-1.0 (0.5=中立)
            clahe:       0.0-1.0
            white_bal:   0.0-1.0
            warmth:      0.0-1.0 (0.5=中立)
    """
    lab = _rgb_to_lab(image)
    l_ch = lab[:, :, 0]
    a_ch = lab[:, :, 1]
    b_ch = lab[:, :, 2]

    # --- 明るさ: 平均輝度 → 目標 130 に補正 ---
    l_mean = float(l_ch.mean())
    target_l = 130.0
    l_diff = target_l - l_mean  # 正=暗すぎ→明るくする, 負=明るすぎ→暗くする
    # diff → strength: diff=0 → 0.5, diff=+50 → 1.0, diff=-50 → 0.0
    brightness = np.clip(0.5 + l_diff / 100.0, 0.0, 1.0)

    # --- コントラスト: L の標準偏差で判定 ---
    l_std = float(l_ch.std())
    # 目標 std ≈ 45 (標準的なポートレート)
    target_std = 45.0
    std_ratio = l_std / target_std
    # std_ratio < 1 → コントラスト不足 → 0.5 より上に
    # std_ratio > 1 → コントラスト過多 → 0.5 より下に
    contrast = np.clip(0.5 + (1.0 - std_ratio) * 0.3, 0.35, 0.65)

    # --- 彩度: HSV S の平均で判定 ---
    hsv = _rgb_to_hsv(image)
    s_mean = float(hsv[:, :, 1].mean())
    # 低彩度 (< 60) → 少し上げる, 高彩度 (> 120) → 少し下げる
    if s_mean < 60:
        saturation = 0.55 + (60 - s_mean) / 200.0
    elif s_mean > 120:
        saturation = 0.45 - (s_mean - 120) / 300.0
    else:
        saturation = 0.50
    saturation = float(np.clip(saturation, 0.35, 0.65))

    # --- CLAHE: 低コントラスト画像ほど強く ---
    clahe = float(np.clip(0.35 - (l_std - 40) * 0.008, 0.1, 0.5))

    # --- ホワイトバランス: a*, b* の偏りで判定 ---
    a_offset = abs(float(a_ch.mean()) - 128.0)
    b_offset = abs(float(b_ch.mean()) - 128.0)
    wb_need = (a_offset + b_offset) / 30.0  # 偏り大 → 補正大
    white_bal = float(np.clip(wb_need, 0.1, 0.5))

    # --- 色温度: b* の偏りで判定 ---
    b_mean = float(b_ch.mean())
    # b* < 128 → 青寄り → 少し暖色に, b* > 128 → 黄寄り → 少し寒色に
    warmth_correction = (128.0 - b_mean) / 80.0
    warmth = float(np.clip(0.5 + warmth_correction * 0.15, 0.4, 0.6))

    return {
        "brightness": round(float(brightness), 2),
        "contrast": round(float(contrast), 2),
        "saturation": round(saturation, 2),
        "clahe": round(clahe, 2),
        "white_bal": round(white_bal, 2),
        "warmth": round(warmth, 2),
        # デバッグ情報
        "_l_mean": round(l_mean, 1),
        "_l_std": round(l_std, 1),
        "_s_mean": round(s_mean, 1),
        "_a_offset": round(a_offset, 1),
        "_b_offset": round(b_offset, 1),
    }


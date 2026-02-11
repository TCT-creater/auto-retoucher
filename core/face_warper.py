"""
core/face_warper.py
===================
ランドマーク座標を基準にした顔パーツの幾何学変形。
Moving Least Squares (MLS) アフィン変形で自然な変形を実現。
目の拡大、鼻の縮小、顎ラインのシェイプ等。
"""

from __future__ import annotations

import cv2
import numpy as np
from core.types import ImageRGB, FaceLandmarks


# ============================================================
# MLS (Moving Least Squares) ワーピング
# ============================================================

def _mls_affine_deformation(
    image: ImageRGB,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    alpha: float = 1.0,
    density: int = 1,
) -> ImageRGB:
    """Moving Least Squares アフィン変形。

    制御点を移動させ、周囲のピクセルを自然に追従させる。
    顔パーツの微調整に最適なワーピング手法。

    Args:
        image: 入力画像 (RGB, uint8)
        src_points: 変形前の制御点 shape=(N, 2) ピクセル座標
        dst_points: 変形後の制御点 shape=(N, 2) ピクセル座標
        alpha: 重み関数の指数 (大きいほど局所的に変形)
        density: マッピング解像度 (1=全ピクセル)

    Returns:
        変形済み画像
    """
    h, w = image.shape[:2]
    src = src_points.astype(np.float64)
    dst = dst_points.astype(np.float64)
    n_ctrl = len(src)

    if n_ctrl < 2:
        return image.copy()

    # 出力座標グリッドを生成
    grid_x, grid_y = np.meshgrid(
        np.arange(0, w, density), np.arange(0, h, density)
    )
    gh, gw = grid_x.shape

    # フラット化して計算
    pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)  # (M, 2)
    n_pts = len(pts)

    # 各制御点からの重み (距離の逆数)
    # w_i = 1 / |p - p_i|^(2*alpha)
    result_pts = np.zeros_like(pts, dtype=np.float64)

    for i in range(n_pts):
        p = pts[i]  # (2,)
        diff = src - p  # (N, 2)
        dist_sq = np.sum(diff ** 2, axis=1)  # (N,)

        # 制御点に一致する場合
        on_ctrl = dist_sq < 1.0
        if np.any(on_ctrl):
            idx = np.argmin(dist_sq)
            result_pts[i] = dst[idx]
            continue

        weights = 1.0 / np.power(dist_sq, alpha)  # (N,)
        w_sum = weights.sum()

        # 重心
        p_star = np.sum(weights[:, None] * src, axis=0) / w_sum  # (2,)
        q_star = np.sum(weights[:, None] * dst, axis=0) / w_sum  # (2,)

        # p_hat, q_hat (重心からの差分)
        p_hat = src - p_star  # (N, 2)
        q_hat = dst - q_star  # (N, 2)

        # MLS アフィン行列の計算
        # M = sum(w_i * p_hat_i^T * p_hat_i)^(-1) * sum(w_i * p_hat_i^T * q_hat_i)
        A = np.zeros((2, 2))
        B = np.zeros((2, 2))
        for j in range(n_ctrl):
            wj = weights[j]
            ph = p_hat[j].reshape(2, 1)
            qh = q_hat[j].reshape(2, 1)
            A += wj * (ph @ ph.T)
            B += wj * (ph @ qh.T)

        det_A = np.linalg.det(A)
        if abs(det_A) < 1e-10:
            result_pts[i] = p + (q_star - p_star)
        else:
            M = np.linalg.inv(A) @ B
            result_pts[i] = (p - p_star) @ M + q_star

    # リマッピング用のマップを生成
    map_x = result_pts[:, 0].reshape(gh, gw).astype(np.float32)
    map_y = result_pts[:, 1].reshape(gh, gw).astype(np.float32)

    if density > 1:
        map_x = cv2.resize(map_x, (w, h), interpolation=cv2.INTER_LINEAR)
        map_y = cv2.resize(map_y, (w, h), interpolation=cv2.INTER_LINEAR)

    result = cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return result


# ============================================================
# 局所的な膨張/収縮ワーピング（高速版）
# ============================================================

def _local_warp(
    image: ImageRGB,
    center: tuple[int, int],
    radius: int,
    strength: float,
) -> ImageRGB:
    """中心点からの放射状の膨張/収縮を行う。

    正のstrengthで膨張（拡大）、負で収縮（縮小）。
    MLS より高速で、目の拡大・鼻の縮小に適する。

    Args:
        image: 入力画像
        center: 変形の中心 (x, y)
        radius: 影響範囲のピクセル半径
        strength: -1.0(最大収縮) ← 0(変化なし) → 1.0(最大膨張)

    Returns:
        変形済み画像
    """
    if abs(strength) < 0.01:
        return image.copy()

    h, w = image.shape[:2]
    cx, cy = center
    result = image.copy()

    # 影響範囲の矩形を計算（全画像を処理しない）
    x1 = max(cx - radius, 0)
    y1 = max(cy - radius, 0)
    x2 = min(cx + radius, w)
    y2 = min(cy + radius, h)

    # 局所座標グリッド
    gy, gx = np.mgrid[y1:y2, x1:x2]
    dx = gx - cx
    dy = gy - cy
    dist = np.sqrt(dx * dx + dy * dy)

    # 影響範囲内のマスク
    mask = dist < radius

    # 放射状の変形量を計算
    # r_new = r * (1 - strength * (1 - r/radius)^2)
    r_norm = dist[mask] / radius  # 0-1 に正規化
    factor = 1.0 - strength * (1.0 - r_norm) ** 2
    factor = np.clip(factor, 0.1, 3.0)

    # 元の座標を逆算
    src_x = cx + dx[mask] / factor
    src_y = cy + dy[mask] / factor

    # バイリニア補間でサンプリング
    src_x = np.clip(src_x, 0, w - 1)
    src_y = np.clip(src_y, 0, h - 1)

    # 整数部と小数部に分離
    x0 = src_x.astype(np.int32)
    y0 = src_y.astype(np.int32)
    x1_s = np.minimum(x0 + 1, w - 1)
    y1_s = np.minimum(y0 + 1, h - 1)
    fx = (src_x - x0).astype(np.float32)
    fy = (src_y - y0).astype(np.float32)

    # バイリニア補間
    for c in range(3):
        val = (
            image[y0, x0, c] * (1 - fx) * (1 - fy)
            + image[y0, x1_s, c] * fx * (1 - fy)
            + image[y1_s, x0, c] * (1 - fx) * fy
            + image[y1_s, x1_s, c] * fx * fy
        )
        result[gy[mask], gx[mask], c] = val.astype(np.uint8)

    return result


# ============================================================
# 顔パーツ変形 関数群
# ============================================================

def enlarge_eyes(
    image: ImageRGB,
    landmarks: FaceLandmarks,
    img_w: int,
    img_h: int,
    strength: float = 0.3,
) -> ImageRGB:
    """目を自然に拡大する。

    Args:
        strength: 0.0(変化なし) → 1.0(最大拡大)
    """
    if strength <= 0.0:
        return image.copy()

    s = min(max(strength, 0.0), 1.0) * 0.35  # 最大 35% の膨張に制限

    result = image.copy()

    for eye_indices in [(159, 145, 33, 133), (386, 374, 362, 263)]:
        # 目の中心と半径を計算
        pts = [landmarks.points[i] for i in eye_indices]
        cx = int(np.mean([p.x for p in pts]) * img_w)
        cy = int(np.mean([p.y for p in pts]) * img_h)

        # 目の幅から半径を推定
        x_coords = [int(p.x * img_w) for p in pts]
        eye_width = max(x_coords) - min(x_coords)
        radius = int(eye_width * 0.9)

        result = _local_warp(result, (cx, cy), radius, s)

    return result


def slim_nose(
    image: ImageRGB,
    landmarks: FaceLandmarks,
    img_w: int,
    img_h: int,
    strength: float = 0.2,
) -> ImageRGB:
    """鼻を細くする。

    鼻翼の左右を内側に押す変形を行う。

    Args:
        strength: 0.0(変化なし) → 1.0(最大縮小)
    """
    if strength <= 0.0:
        return image.copy()

    s = min(max(strength, 0.0), 1.0) * 0.25  # 最大 25% に制限

    # 鼻翼の座標（左右）
    # idx 48: 鼻翼右, idx 278: 鼻翼左
    nose_right = landmarks.points[48] if 48 < len(landmarks.points) else None
    nose_left = landmarks.points[278] if 278 < len(landmarks.points) else None
    nose_tip = landmarks.points[1]

    if not nose_right or not nose_left:
        return image.copy()

    # 鼻の幅を取得
    nr_px = int(nose_right.x * img_w)
    nl_px = int(nose_left.x * img_w)
    nose_width = abs(nl_px - nr_px)
    radius = int(nose_width * 1.2)

    # 鼻翼を中心に収縮
    cx_r = nr_px
    cy_r = int(nose_right.y * img_h)
    cx_l = nl_px
    cy_l = int(nose_left.y * img_h)

    result = image.copy()
    result = _local_warp(result, (cx_r, cy_r), radius, -s)
    result = _local_warp(result, (cx_l, cy_l), radius, -s)

    return result


def slim_jaw(
    image: ImageRGB,
    landmarks: FaceLandmarks,
    img_w: int,
    img_h: int,
    strength: float = 0.3,
) -> ImageRGB:
    """顎ライン（フェイスライン）をシャープにする。

    顎の左右ポイントを内側＋上方向に移動させる。
    MLS アフィン変形を使用してなめらかに変形。

    Args:
        strength: 0.0(変化なし) → 1.0(最大シェイプ)
    """
    if strength <= 0.0:
        return image.copy()

    s = min(max(strength, 0.0), 1.0)

    # 顎の左右ライン (下半分のみ)
    jaw_indices = [
        397, 365, 379, 378, 400, 377,  # 右顎
        152,                            # 顎先
        148, 176, 149, 150, 136, 172,  # 左顎
    ]

    # アンカーポイント（動かさない基準点）
    anchor_indices = [
        10,    # 額（上）
        234,   # 右耳
        454,   # 左耳
        168,   # 鼻根
    ]

    src_points = []
    dst_points = []

    # アンカーポイントを追加（変形しない）
    for idx in anchor_indices:
        if idx < len(landmarks.points):
            pt = landmarks.points[idx]
            px = int(pt.x * img_w)
            py = int(pt.y * img_h)
            src_points.append([px, py])
            dst_points.append([px, py])

    # 顎先の座標
    chin = landmarks.points[152]
    chin_x = int(chin.x * img_w)
    chin_y = int(chin.y * img_h)

    # 顔の中心x
    center_x = chin_x

    # 顎ポイントを内側に移動
    max_shift = s * 12  # 最大 12px のシフト
    for idx in jaw_indices:
        if idx < len(landmarks.points):
            pt = landmarks.points[idx]
            px = int(pt.x * img_w)
            py = int(pt.y * img_h)
            src_points.append([px, py])

            # 中心方向への移動量
            dx = center_x - px
            dy = -abs(py - chin_y) * 0.05  # わずかに上方向

            dist_from_center = abs(px - center_x) / (img_w * 0.5)
            shift_factor = dist_from_center * max_shift

            # 内側に移動
            new_px = px + int(np.sign(dx) * shift_factor)
            new_py = py + int(dy * s)
            dst_points.append([new_px, new_py])

    src_arr = np.array(src_points)
    dst_arr = np.array(dst_points)

    # 高速化のためdensityを上げる
    return _mls_affine_deformation(image, src_arr, dst_arr, alpha=1.0, density=4)


def plump_lips(
    image: ImageRGB,
    landmarks: FaceLandmarks,
    img_w: int,
    img_h: int,
    strength: float = 0.3,
) -> ImageRGB:
    """唇をふっくらさせる。

    上唇と下唇それぞれの中心から放射状に膨張させる。

    Args:
        strength: 0.0(変化なし) → 1.0(最大ボリュームアップ)
    """
    if strength <= 0.0:
        return image.copy()

    s = min(max(strength, 0.0), 1.0) * 0.30  # 最大 30% の膨張に制限

    # MediaPipe 唇ランドマーク
    # 上唇: 0, 267, 269, 270, 13, 37, 39, 40, 185
    # 下唇: 17, 314, 315, 316, 14, 84, 85, 86, 409
    upper_lip_idx = [0, 267, 269, 270, 13, 37, 39, 40, 185]
    lower_lip_idx = [17, 314, 315, 316, 14, 84, 85, 86, 409]

    result = image.copy()

    for lip_indices in [upper_lip_idx, lower_lip_idx]:
        valid_pts = [landmarks.points[i] for i in lip_indices
                     if i < len(landmarks.points)]
        if not valid_pts:
            continue

        # 唇の中心座標
        cx = int(np.mean([p.x for p in valid_pts]) * img_w)
        cy = int(np.mean([p.y for p in valid_pts]) * img_h)

        # 唇の幅から半径を推定
        x_coords = [int(p.x * img_w) for p in valid_pts]
        y_coords = [int(p.y * img_h) for p in valid_pts]
        lip_width = max(x_coords) - min(x_coords)
        lip_height = max(y_coords) - min(y_coords)
        radius = int(max(lip_width * 0.5, lip_height * 1.5))

        result = _local_warp(result, (cx, cy), radius, s)

    return result

"""
core/body_warper.py
===================
MediaPipe Pose Landmarker を使ったボディライン補正。
ウエスト・ヒップの曲線を微調整する。
"""

from __future__ import annotations

import os
import ssl
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

from core.types import ImageRGB


# モデルファイル
_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
_MODEL_FILENAME = "pose_landmarker_lite.task"
_MODEL_PATH = os.path.join(_MODEL_DIR, _MODEL_FILENAME)
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


def _ensure_pose_model() -> str:
    """ポーズモデルファイルが存在しなければダウンロード。"""
    if os.path.exists(_MODEL_PATH):
        return _MODEL_PATH

    os.makedirs(_MODEL_DIR, exist_ok=True)
    print(f"  [BodyWarper] ポーズモデルをダウンロード中...")

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(_MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx) as resp:
        data = resp.read()
        with open(_MODEL_PATH, "wb") as f:
            f.write(data)

    size_mb = len(data) / (1024 * 1024)
    print(f"  [BodyWarper] ダウンロード完了: {size_mb:.1f} MB")
    return _MODEL_PATH


# ============================================================
# MediaPipe Pose キーポイント インデックス
# ============================================================
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24
POSE_LEFT_KNEE = 25
POSE_RIGHT_KNEE = 26


# ============================================================
# ポーズ検出クラス
# ============================================================

class PoseDetector:
    """MediaPipe Pose Landmarker によるボディキーポイント検出。"""

    def __init__(self):
        model_path = _ensure_pose_model()

        with open(model_path, "rb") as f:
            model_data = f.read()

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_data),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)

    def detect(self, image: ImageRGB) -> list[dict] | None:
        """ポーズを検出してキーポイント辞書のリストを返す。

        Returns:
            [{"left_shoulder": (x, y), "right_shoulder": (x, y),
              "left_hip": (x, y), "right_hip": (x, y),
              "left_knee": (x, y), "right_knee": (x, y)}]
            検出できなかった場合は None
        """
        if self._landmarker is None:
            return None

        h, w = image.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None

        poses = []
        for pose_lm_list in result.pose_landmarks:
            kp = {}
            names = {
                POSE_LEFT_SHOULDER: "left_shoulder",
                POSE_RIGHT_SHOULDER: "right_shoulder",
                POSE_LEFT_HIP: "left_hip",
                POSE_RIGHT_HIP: "right_hip",
                POSE_LEFT_KNEE: "left_knee",
                POSE_RIGHT_KNEE: "right_knee",
            }
            for idx, name in names.items():
                if idx < len(pose_lm_list):
                    lm = pose_lm_list[idx]
                    kp[name] = (int(lm.x * w), int(lm.y * h))
            poses.append(kp)

        return poses

    def close(self):
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ============================================================
# 局所ワーピング（face_warper の _local_warp と同じ原理）
# ============================================================

def _body_local_warp(
    image: ImageRGB,
    center: tuple[int, int],
    radius: int,
    shift_x: float,
    shift_y: float,
) -> ImageRGB:
    """指定方向への局所的な平行移動ワープ。

    膨張/収縮ではなく、特定方向に押す/引く動きをする。
    ウエストの内側への収縮に使用。

    Args:
        image: 入力画像
        center: 変形の中心 (x, y)
        radius: 影響範囲のピクセル半径
        shift_x: x方向のシフト量 (px)
        shift_y: y方向のシフト量 (px)
    """
    if abs(shift_x) < 0.5 and abs(shift_y) < 0.5:
        return image.copy()

    h, w = image.shape[:2]
    cx, cy = center
    result = image.copy()

    x1 = max(cx - radius, 0)
    y1 = max(cy - radius, 0)
    x2 = min(cx + radius, w)
    y2 = min(cy + radius, h)

    gy, gx = np.mgrid[y1:y2, x1:x2]
    dx = gx - cx
    dy = gy - cy
    dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)

    mask = dist < radius

    # 距離に応じた減衰（中心が最大、端で0）
    falloff = np.zeros_like(dist)
    falloff[mask] = (1.0 - dist[mask] / radius) ** 2

    # ソース座標
    src_x = (gx - shift_x * falloff).astype(np.float32)
    src_y = (gy - shift_y * falloff).astype(np.float32)

    src_x = np.clip(src_x, 0, w - 1)
    src_y = np.clip(src_y, 0, h - 1)

    # バイリニア補間
    x0 = src_x.astype(np.int32)
    y0 = src_y.astype(np.int32)
    x1_s = np.minimum(x0 + 1, w - 1)
    y1_s = np.minimum(y0 + 1, h - 1)
    fx = src_x - x0
    fy = src_y - y0

    for c in range(3):
        val = (
            image[y0, x0, c] * (1 - fx) * (1 - fy)
            + image[y0, x1_s, c] * fx * (1 - fy)
            + image[y1_s, x0, c] * (1 - fx) * fy
            + image[y1_s, x1_s, c] * fx * fy
        )
        out = result[y1:y2, x1:x2, c].astype(np.float32)
        out[mask] = val[mask]
        result[y1:y2, x1:x2, c] = out.astype(np.uint8)

    return result


# ============================================================
# ボディライン補正 関数群
# ============================================================

def slim_waist(
    image: ImageRGB,
    pose_keypoints: dict,
    strength: float = 0.2,
) -> ImageRGB:
    """ウエストラインを細くする。

    ヒップと肩の中間点（ウエスト位置）を推定し、
    左右から内側に向けて変形する。

    Args:
        image: 入力画像 (RGB, uint8)
        pose_keypoints: PoseDetector.detect() の結果
        strength: 0.0(変化なし) → 1.0(最大補正)

    Returns:
        ウエスト補正済み画像
    """
    if strength <= 0.0:
        return image.copy()

    s = min(max(strength, 0.0), 1.0)

    ls = pose_keypoints.get("left_shoulder")
    rs = pose_keypoints.get("right_shoulder")
    lh = pose_keypoints.get("left_hip")
    rh = pose_keypoints.get("right_hip")

    if not all([ls, rs, lh, rh]):
        return image.copy()

    h, w = image.shape[:2]

    # ウエスト位置 = 肩とヒップの中間より少し下
    waist_y_left = int(ls[1] * 0.35 + lh[1] * 0.65)
    waist_y_right = int(rs[1] * 0.35 + rh[1] * 0.65)

    # 体幅の推定
    body_width = abs(lh[0] - rh[0])
    radius = int(body_width * 0.6)

    # シフト量（最大 body_width の 8%）
    max_shift = body_width * 0.08 * s

    result = image.copy()

    # 左ウエスト（内側 = 右方向にシフト）
    cx_left = lh[0]
    result = _body_local_warp(
        result, (cx_left, waist_y_left), radius,
        shift_x=max_shift, shift_y=0
    )

    # 右ウエスト（内側 = 左方向にシフト）
    cx_right = rh[0]
    result = _body_local_warp(
        result, (cx_right, waist_y_right), radius,
        shift_x=-max_shift, shift_y=0
    )

    return result


def enhance_curves(
    image: ImageRGB,
    pose_keypoints: dict,
    strength: float = 0.2,
) -> ImageRGB:
    """ヒップラインのカーブを強調する。

    ヒップ位置で外側に微妙に膨らませ、
    ウエスト→ヒップのメリハリを出す。

    Args:
        image: 入力画像 (RGB, uint8)
        pose_keypoints: PoseDetector.detect() の結果
        strength: 0.0(変化なし) → 1.0(最大補正)

    Returns:
        ヒップ補正済み画像
    """
    if strength <= 0.0:
        return image.copy()

    s = min(max(strength, 0.0), 1.0)

    lh = pose_keypoints.get("left_hip")
    rh = pose_keypoints.get("right_hip")
    lk = pose_keypoints.get("left_knee")
    rk = pose_keypoints.get("right_knee")

    if not all([lh, rh]):
        return image.copy()

    body_width = abs(lh[0] - rh[0])
    radius = int(body_width * 0.5)

    # ヒップの外側に少し膨らませる（最大 body_width の 5%）
    max_shift = body_width * 0.05 * s

    result = image.copy()

    # 左ヒップ（外側 = 左方向）
    result = _body_local_warp(
        result, (lh[0], lh[1]), radius,
        shift_x=-max_shift, shift_y=0
    )

    # 右ヒップ（外側 = 右方向）
    result = _body_local_warp(
        result, (rh[0], rh[1]), radius,
        shift_x=max_shift, shift_y=0
    )

    return result


def slim_legs(
    image: ImageRGB,
    pose_keypoints: dict,
    strength: float = 0.15,
) -> ImageRGB:
    """太もものラインを細くする。

    ヒップ〜膝の中間に収縮ワープを適用。

    Args:
        image: 入力画像 (RGB, uint8)
        pose_keypoints: PoseDetector.detect() の結果
        strength: 0.0(変化なし) → 1.0(最大補正)

    Returns:
        脚補正済み画像
    """
    if strength <= 0.0:
        return image.copy()

    s = min(max(strength, 0.0), 1.0)

    lh = pose_keypoints.get("left_hip")
    rh = pose_keypoints.get("right_hip")
    lk = pose_keypoints.get("left_knee")
    rk = pose_keypoints.get("right_knee")

    if not all([lh, rh, lk, rk]):
        return image.copy()

    body_width = abs(lh[0] - rh[0])
    radius = int(body_width * 0.4)
    max_shift = body_width * 0.04 * s

    result = image.copy()

    # 左太もも（中間点で内側にシフト）
    mid_left = (
        (lh[0] + lk[0]) // 2,
        (lh[1] + lk[1]) // 2,
    )
    result = _body_local_warp(
        result, mid_left, radius,
        shift_x=max_shift, shift_y=0
    )

    # 右太もも
    mid_right = (
        (rh[0] + rk[0]) // 2,
        (rh[1] + rk[1]) // 2,
    )
    result = _body_local_warp(
        result, mid_right, radius,
        shift_x=-max_shift, shift_y=0
    )

    return result


def enhance_bust(
    image: ImageRGB,
    pose_keypoints: dict,
    strength: float = 0.2,
) -> ImageRGB:
    """バストラインを強調する。

    肩とヒップの間のバスト位置を推定し、
    左右に微妙に膨らませてボリュームアップ。

    Args:
        image: 入力画像 (RGB, uint8)
        pose_keypoints: PoseDetector.detect() の結果
        strength: 0.0(変化なし) → 1.0(最大補正)

    Returns:
        バスト補正済み画像
    """
    if strength <= 0.0:
        return image.copy()

    s = min(max(strength, 0.0), 1.0)

    ls = pose_keypoints.get("left_shoulder")
    rs = pose_keypoints.get("right_shoulder")
    lh = pose_keypoints.get("left_hip")
    rh = pose_keypoints.get("right_hip")

    if not all([ls, rs, lh, rh]):
        return image.copy()

    h, w = image.shape[:2]

    # バスト位置 = 肩の少し下 (肩とヒップの 25% 地点)
    bust_y_left = int(ls[1] * 0.75 + lh[1] * 0.25)
    bust_y_right = int(rs[1] * 0.75 + rh[1] * 0.25)

    # 体幅の推定
    shoulder_width = abs(ls[0] - rs[0])
    body_width = abs(lh[0] - rh[0])
    radius = int(shoulder_width * 0.35)

    # 外側にシフト（最大 body_width の 6%）
    max_shift = body_width * 0.06 * s

    result = image.copy()

    # 左バスト（外側 = 左方向）
    cx_left = int(ls[0] * 0.7 + rs[0] * 0.3)  # やや中央寄り
    result = _body_local_warp(
        result, (cx_left, bust_y_left), radius,
        shift_x=-max_shift, shift_y=0
    )

    # 右バスト（外側 = 右方向）
    cx_right = int(rs[0] * 0.7 + ls[0] * 0.3)
    result = _body_local_warp(
        result, (cx_right, bust_y_right), radius,
        shift_x=max_shift, shift_y=0
    )

    return result

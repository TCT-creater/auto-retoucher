"""
Auto Retoucher — 型定義モジュール
=================================
画像処理パイプライン全体で使用するデータ型を一元定義。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from numpy.typing import NDArray


# ============================================================
# 基本画像型
# ============================================================

# BGR画像 (OpenCV標準) — shape: (H, W, 3), dtype: uint8
ImageBGR = NDArray[np.uint8]

# RGB画像 (Pillow/Streamlit標準) — shape: (H, W, 3), dtype: uint8
ImageRGB = NDArray[np.uint8]

# グレースケール画像 — shape: (H, W), dtype: uint8
ImageGray = NDArray[np.uint8]

# マスク画像 — shape: (H, W), dtype: uint8, 値域: 0-255
MaskImage = NDArray[np.uint8]

# 浮動小数点画像 (0.0-1.0) — shape: (H, W, 3), dtype: float32
ImageFloat = NDArray[np.float32]


# ============================================================
# MediaPipe ランドマーク型
# ============================================================

@dataclass
class LandmarkPoint:
    """正規化された単一ランドマーク座標。

    Attributes:
        x: 水平位置 (0.0=左端, 1.0=右端)  — float32
        y: 垂直位置 (0.0=上端, 1.0=下端)  — float32
        z: 深度 (カメラからの距離、相対値) — float32
        visibility: 可視性スコア (0.0-1.0)  — float32
    """
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0

    def to_pixel(self, img_w: int, img_h: int) -> tuple[int, int]:
        """正規化座標をピクセル座標に変換する。"""
        return (int(self.x * img_w), int(self.y * img_h))


@dataclass
class FaceLandmarks:
    """1つの顔から検出された468点のランドマーク群。

    Attributes:
        points: 468個の LandmarkPoint のリスト
                インデックスは MediaPipe Face Mesh の定義に準拠
        confidence: 検出信頼度 (0.0-1.0)
    """
    points: list[LandmarkPoint] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def count(self) -> int:
        return len(self.points)

    def get_pixel_coords(
        self, img_w: int, img_h: int
    ) -> NDArray[np.int32]:
        """全ランドマークをピクセル座標の配列として返す。

        Returns:
            shape: (N, 2), dtype: int32  — [[x, y], ...]
        """
        return np.array(
            [[int(p.x * img_w), int(p.y * img_h)] for p in self.points],
            dtype=np.int32,
        )


# ============================================================
# 顔パーツ インデックス定義
# MediaPipe Face Mesh の 468 点から主要パーツを抽出
# ============================================================

# 右目の輪郭 (上瞼＋下瞼)
RIGHT_EYE_IDX = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    173, 157, 158, 159, 160, 161, 246,
]

# 左目の輪郭
LEFT_EYE_IDX = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398,
]

# 右眉
RIGHT_EYEBROW_IDX = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

# 左眉
LEFT_EYEBROW_IDX = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

# 唇の外周
LIPS_OUTER_IDX = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
]

# 唇の内周
LIPS_INNER_IDX = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
]

# 鼻の中心ライン
NOSE_RIDGE_IDX = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]

# フェイスライン (顎の輪郭)
FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109,
]

# 右瞳
RIGHT_IRIS_IDX = [468, 469, 470, 471, 472]

# 左瞳
LEFT_IRIS_IDX = [473, 474, 475, 476, 477]


@dataclass
class FaceRegion:
    """検出された顔の領域情報。

    Attributes:
        bbox: バウンディングボックス (x, y, w, h) ピクセル座標
        landmarks: ランドマーク情報
        face_id: 複数顔検出時の識別子
    """
    bbox: tuple[int, int, int, int]
    landmarks: FaceLandmarks
    face_id: int = 0


@dataclass
class RetouchSettings:
    """各補正ステップの強度設定。

    全ての値は 0.0（無効）〜 1.0（最大）の範囲。
    """
    skin_smooth: float = 0.5
    brightness: float = 0.5
    contrast: float = 0.5
    saturation: float = 0.5
    white_balance: float = 0.3
    sharpness: float = 0.3
    eye_size: float = 0.3
    nose_slim: float = 0.2
    jaw_slim: float = 0.3
    body_curve: float = 0.2
    face_restore: float = 0.5   # Fidelity: GFPGAN AI Restoration


@dataclass
class RetouchResult:
    """レタッチパイプラインの出力結果。

    Attributes:
        original: 元画像 (RGB)
        retouched: レタッチ済み画像 (RGB)
        settings: 適用された設定
        faces_detected: 検出された顔の数
        processing_time_ms: 処理時間 (ミリ秒)
    """
    original: ImageRGB
    retouched: ImageRGB
    settings: RetouchSettings
    faces_detected: int = 0
    processing_time_ms: float = 0.0

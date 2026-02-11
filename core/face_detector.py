"""
core/face_detector.py
=====================
MediaPipe Face Landmarker (Tasks API) を使った顔検出＆478点ランドマーク取得。
全ての補正モジュールの基盤となる顔座標データを提供する。

MediaPipe 0.10.x 以降は mp.solutions ではなく mp.tasks.vision を使用。
"""

from __future__ import annotations

import os
import urllib.request
import ssl

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

from core.types import (
    ImageBGR,
    ImageRGB,
    LandmarkPoint,
    FaceLandmarks,
    FaceRegion,
    RIGHT_EYE_IDX,
    LEFT_EYE_IDX,
    NOSE_RIDGE_IDX,
    LIPS_OUTER_IDX,
    FACE_OVAL_IDX,
)

# モデルファイルのパス＆ダウンロードURL
_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
_MODEL_FILENAME = "face_landmarker.task"
_MODEL_PATH = os.path.join(_MODEL_DIR, _MODEL_FILENAME)
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)


def _ensure_model() -> str:
    """モデルファイルが存在しなければダウンロードする。"""
    if os.path.exists(_MODEL_PATH):
        return _MODEL_PATH

    os.makedirs(_MODEL_DIR, exist_ok=True)
    print(f"  [FaceDetector] モデルをダウンロード中... ({_MODEL_URL})")

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(_MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx) as resp:
        data = resp.read()
        with open(_MODEL_PATH, "wb") as f:
            f.write(data)

    size_mb = len(data) / (1024 * 1024)
    print(f"  [FaceDetector] ダウンロード完了: {size_mb:.1f} MB")
    return _MODEL_PATH


class FaceDetector:
    """MediaPipe Face Landmarker による顔ランドマーク検出器。

    使用例:
        detector = FaceDetector(max_faces=1)
        faces = detector.detect(image_rgb)
        if faces:
            landmarks = faces[0].landmarks
            print(f"検出: {landmarks.count} 点, 信頼度: {landmarks.confidence:.2f}")
        detector.close()

    コンテキストマネージャ対応:
        with FaceDetector() as detector:
            faces = detector.detect(image_rgb)
    """

    def __init__(
        self,
        max_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine_landmarks: bool = True,
    ):
        """
        Args:
            max_faces: 検出する顔の最大数
            min_detection_confidence: 検出の最小信頼度 (0.0-1.0)
            min_tracking_confidence: トラッキングの最小信頼度 (0.0-1.0)
            refine_landmarks: True で瞳（虹彩）の 10 点を追加 → 478 点に
        """
        self._max_faces = max_faces
        self._refine = refine_landmarks

        model_path = _ensure_model()

        # MediaPipe の C++ バックエンドは日本語パスを処理できないため
        # model_asset_buffer (バイト列) で読み込む
        with open(model_path, "rb") as f:
            model_data = f.read()

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_data),
            running_mode=RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    # ------ コンテキストマネージャ ------
    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        """MediaPipe リソースを解放。"""
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None

    # ------ メイン検出 ------
    def detect(self, image: ImageRGB) -> list[FaceRegion]:
        """RGB画像から顔を検出し、ランドマーク付き FaceRegion リストを返す。

        Args:
            image: RGB画像 (numpy.ndarray, shape=(H,W,3), dtype=uint8)

        Returns:
            検出された顔のリスト。顔が見つからない場合は空リスト。
        """
        if self._landmarker is None:
            raise RuntimeError("FaceDetector は既に close() されています")

        h, w = image.shape[:2]

        # MediaPipe Image に変換
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image,
        )
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return []

        faces: list[FaceRegion] = []
        for face_id, face_lm_list in enumerate(result.face_landmarks):
            # --- ランドマーク変換 ---
            points: list[LandmarkPoint] = []
            xs, ys = [], []

            for lm in face_lm_list:
                pt = LandmarkPoint(
                    x=lm.x,
                    y=lm.y,
                    z=lm.z,
                    visibility=lm.visibility if hasattr(lm, "visibility") and lm.visibility else 1.0,
                )
                points.append(pt)
                xs.append(lm.x)
                ys.append(lm.y)

            # 検出信頼度
            expected = 478 if self._refine else 468
            confidence = min(len(points) / expected, 1.0)

            landmarks = FaceLandmarks(points=points, confidence=confidence)

            # --- バウンディングボックス ---
            x_min = max(int(min(xs) * w) - 10, 0)
            y_min = max(int(min(ys) * h) - 10, 0)
            x_max = min(int(max(xs) * w) + 10, w)
            y_max = min(int(max(ys) * h) + 10, h)
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

            faces.append(FaceRegion(
                bbox=bbox,
                landmarks=landmarks,
                face_id=face_id,
            ))

        return faces

    # ------ ユーティリティ ------
    def get_part_coords(
        self,
        landmarks: FaceLandmarks,
        part_indices: list[int],
        img_w: int,
        img_h: int,
    ) -> np.ndarray:
        """特定パーツのピクセル座標を取得。

        Args:
            landmarks: FaceLandmarks オブジェクト
            part_indices: パーツのインデックスリスト (例: RIGHT_EYE_IDX)
            img_w: 画像幅
            img_h: 画像高さ

        Returns:
            shape: (N, 2), dtype: int32 のピクセル座標配列
        """
        coords = []
        for idx in part_indices:
            if idx < len(landmarks.points):
                pt = landmarks.points[idx]
                coords.append([int(pt.x * img_w), int(pt.y * img_h)])
        return np.array(coords, dtype=np.int32)

    def get_face_mask(
        self,
        landmarks: FaceLandmarks,
        img_w: int,
        img_h: int,
    ) -> np.ndarray:
        """フェイスラインに基づく顔領域マスクを生成。

        Returns:
            shape: (H, W), dtype: uint8, 値: 0 or 255
        """
        oval_coords = self.get_part_coords(
            landmarks, FACE_OVAL_IDX, img_w, img_h
        )
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        if len(oval_coords) > 2:
            cv2.fillConvexPoly(mask, oval_coords, 255)
        return mask

    def get_skin_mask(
        self,
        landmarks: FaceLandmarks,
        img_w: int,
        img_h: int,
    ) -> np.ndarray:
        """肌領域マスクを生成（目・眉・唇を除外）。

        顔全体マスクから目・唇領域を差し引いた肌のみのマスク。

        Returns:
            shape: (H, W), dtype: uint8, 値: 0 or 255
        """
        # ベース: 顔全体
        skin_mask = self.get_face_mask(landmarks, img_w, img_h)

        # 除外領域を黒で塗りつぶす
        exclude_parts = [
            RIGHT_EYE_IDX,
            LEFT_EYE_IDX,
            LIPS_OUTER_IDX,
        ]
        for part_idx in exclude_parts:
            coords = self.get_part_coords(landmarks, part_idx, img_w, img_h)
            if len(coords) > 2:
                cv2.fillConvexPoly(skin_mask, coords, 0)

        return skin_mask

    def sample_key_points(
        self,
        landmarks: FaceLandmarks,
        img_w: int,
        img_h: int,
    ) -> dict[str, dict]:
        """主要ランドマークの座標をサンプリングして辞書で返す（検証用）。

        Returns:
            {
                "right_eye_center": {"norm": (x, y), "pixel": (px, py), "idx": N},
                ...
            }
        """
        key_indices = {
            "right_eye_center": 159,   # 右目の中心付近
            "left_eye_center": 386,    # 左目の中心付近
            "nose_tip": 1,             # 鼻先
            "mouth_center": 13,        # 上唇中央
            "chin": 152,               # 顎先
            "right_ear": 234,          # 右耳付近
            "left_ear": 454,           # 左耳付近
            "forehead": 10,            # 額の中央上部
        }

        result = {}
        for name, idx in key_indices.items():
            if idx < len(landmarks.points):
                pt = landmarks.points[idx]
                result[name] = {
                    "norm": (round(pt.x, 5), round(pt.y, 5)),
                    "pixel": pt.to_pixel(img_w, img_h),
                    "idx": idx,
                    "z": round(pt.z, 5),
                }
        return result

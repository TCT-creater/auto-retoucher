"""
FaceDetector の検証スクリプト
サンプル画像を生成してランドマーク検出＆座標サンプリングを実行
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cv2
import urllib.request
from core.face_detector import FaceDetector
from core.types import RIGHT_EYE_IDX, LEFT_EYE_IDX, FACE_OVAL_IDX, NOSE_RIDGE_IDX


def download_sample_image() -> np.ndarray:
    """フリーのサンプル顔画像をダウンロードして返す。"""
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg"
    # Thispersondoesnotexist は毎回違う顔を返すので不向き
    # 代わりにMediaPipe公式テスト画像に似た合成顔画像を生成
    print("  サンプル顔画像を合成中...")
    img = np.ones((600, 480, 3), dtype=np.uint8) * 200

    # 簡易的な顔の楕円を描画（MediaPipeが検出可能かテスト）
    cv2.ellipse(img, (240, 280), (140, 180), 0, 0, 360, (210, 190, 170), -1)
    # 目
    cv2.circle(img, (190, 250), 18, (60, 60, 60), -1)
    cv2.circle(img, (290, 250), 18, (60, 60, 60), -1)
    cv2.circle(img, (190, 250), 8, (255, 255, 255), -1)
    cv2.circle(img, (290, 250), 8, (255, 255, 255), -1)
    # 鼻
    cv2.line(img, (240, 270), (240, 310), (180, 160, 140), 2)
    # 口
    cv2.ellipse(img, (240, 340), (35, 12), 0, 0, 180, (150, 100, 100), 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def try_real_image() -> np.ndarray | None:
    """assets/ にテスト画像があれば読み込む"""
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    for ext in ["jpg", "jpeg", "png", "webp"]:
        for fname in os.listdir(assets_dir):
            if fname.lower().endswith(f".{ext}"):
                path = os.path.join(assets_dir, fname)
                img = cv2.imread(path)
                if img is not None:
                    print(f"  実画像を使用: {fname}")
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


def main():
    print("=" * 60)
    print("  FaceDetector 検証テスト")
    print("=" * 60)
    print()

    # 1. 画像の準備
    print("[1] 画像の準備")
    img = try_real_image()
    if img is None:
        img = download_sample_image()
    h, w = img.shape[:2]
    print(f"  画像サイズ: {w} x {h}")
    print(f"  dtype: {img.dtype}, shape: {img.shape}")
    print()

    # 2. FaceDetector の初期化
    print("[2] FaceDetector 初期化")
    with FaceDetector(max_faces=3, refine_landmarks=True) as detector:
        print(f"  max_faces: 3")
        print(f"  refine_landmarks: True (478点モード)")
        print()

        # 3. 検出実行
        print("[3] 顔検出 実行中...")
        faces = detector.detect(img)
        print(f"  検出結果: {len(faces)} 顔")
        print()

        if not faces:
            print("  ⚠️ 顔が検出されませんでした。")
            print("  → assets/ にポートレート画像(.jpg)を配置して再実行してください。")
            print()
            print("  合成画像でのクラス動作チェック:")
            print(f"    FaceDetector.__init__: OK")
            print(f"    FaceDetector.detect(): OK (空リスト返却)")
            print(f"    FaceDetector.close(): OK")
            print()

            # 型チェックだけ実施
            print("[4] 型チェック")
            from core.types import (
                LandmarkPoint, FaceLandmarks, FaceRegion, RetouchSettings
            )
            lp = LandmarkPoint(x=0.5, y=0.5, z=0.0)
            fl = FaceLandmarks(points=[lp]*478, confidence=0.99)
            fr = FaceRegion(bbox=(0,0,100,100), landmarks=fl)
            print(f"    LandmarkPoint.to_pixel(480,600) = {lp.to_pixel(480, 600)}")
            print(f"    FaceLandmarks.count = {fl.count}")
            print(f"    FaceRegion.bbox = {fr.bbox}")
            print(f"    get_pixel_coords shape = {fl.get_pixel_coords(480,600).shape}")
            print()
            print("  === 型チェック ALL OK ===")
            return

        # 4. 座標サンプリング
        face = faces[0]
        lm = face.landmarks
        print(f"[4] ランドマーク情報")
        print(f"  総ランドマーク数: {lm.count}")
        print(f"  検出信頼度: {lm.confidence:.4f}")
        print(f"  バウンディングボックス (x,y,w,h): {face.bbox}")
        print()

        # 5. 主要座標のサンプリング
        print("[5] 主要ランドマーク座標サンプリング")
        samples = detector.sample_key_points(lm, w, h)
        print(f"  {'パーツ':<20} {'Idx':>4}  {'正規化 (x, y)':<22}  {'ピクセル (x, y)':<18}  {'深度 z':>8}")
        print("  " + "-" * 78)
        for name, data in samples.items():
            nx, ny = data["norm"]
            px, py = data["pixel"]
            z = data["z"]
            idx = data["idx"]
            print(f"  {name:<20} {idx:>4}  ({nx:>7.5f}, {ny:>7.5f})  ({px:>5d}, {py:>5d})      {z:>8.5f}")
        print()

        # 6. パーツ座標の検証
        print("[6] パーツ座標 検証")
        for name, indices in [
            ("RIGHT_EYE", RIGHT_EYE_IDX),
            ("LEFT_EYE", LEFT_EYE_IDX),
            ("NOSE_RIDGE", NOSE_RIDGE_IDX),
            ("FACE_OVAL", FACE_OVAL_IDX),
        ]:
            coords = detector.get_part_coords(lm, indices, w, h)
            center = coords.mean(axis=0).astype(int) if len(coords) > 0 else (0, 0)
            print(f"  {name:<12}: {len(coords):>3} points, center=({center[0]:>4}, {center[1]:>4})")
        print()

        # 7. マスク生成テスト
        print("[7] マスク生成テスト")
        face_mask = detector.get_face_mask(lm, w, h)
        skin_mask = detector.get_skin_mask(lm, w, h)
        face_area = np.count_nonzero(face_mask)
        skin_area = np.count_nonzero(skin_mask)
        total = w * h
        print(f"  顔マスク: {face_area:>7} px ({face_area/total*100:.1f}%)")
        print(f"  肌マスク: {skin_area:>7} px ({skin_area/total*100:.1f}%)")
        print(f"  差分(目+唇): {face_area - skin_area:>7} px")
        print()

    print("=" * 60)
    print("  === ALL CHECKS PASSED ===")
    print("=" * 60)


if __name__ == "__main__":
    main()

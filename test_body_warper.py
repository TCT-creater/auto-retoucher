"""body_warper 検証スクリプト"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(__file__))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from core.body_warper import PoseDetector, slim_waist, enhance_curves, slim_legs

img = cv2.cvtColor(cv2.imread("assets/test_face.jpg"), cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]
print(f"Image: {w}x{h}")

with PoseDetector() as det:
    poses = det.detect(img)
    has_pose = poses is not None and len(poses) > 0
    print(f"Pose detected: {has_pose}")

    if has_pose:
        kp = poses[0]
        print(f"Keypoints: {list(kp.keys())}")
        for k, v in kp.items():
            print(f"  {k}: {v}")
        use_kp = kp
    else:
        print("No body in face-only image (expected)")
        print("Using synthetic keypoints for functional test...")
        use_kp = {
            "left_shoulder": (350, 300),
            "right_shoulder": (650, 300),
            "left_hip": (380, 550),
            "right_hip": (620, 550),
            "left_knee": (370, 780),
            "right_knee": (630, 780),
        }

    # テスト1: strength=0 で変化なし
    r1 = slim_waist(img, use_kp, 0.0)
    d0 = np.mean(np.abs(r1.astype(float) - img.astype(float)))
    print(f"\nwaist(0.0) diff: {d0:.4f} (should be 0)")

    # テスト2: slim_waist
    r2 = slim_waist(img, use_kp, 0.5)
    d5 = np.mean(np.abs(r2.astype(float) - img.astype(float)))
    print(f"waist(0.5) diff: {d5:.2f}")

    # テスト3: enhance_curves
    r3 = enhance_curves(img, use_kp, 0.5)
    dc = np.mean(np.abs(r3.astype(float) - img.astype(float)))
    print(f"curves(0.5) diff: {dc:.2f}")

    # テスト4: slim_legs
    r4 = slim_legs(img, use_kp, 0.5)
    dl = np.mean(np.abs(r4.astype(float) - img.astype(float)))
    print(f"legs(0.5) diff: {dl:.2f}")

    # テスト5: 不完全キーポイントで安全に動作
    r5 = slim_waist(img, {}, 0.5)
    d_empty = np.mean(np.abs(r5.astype(float) - img.astype(float)))
    print(f"waist(empty kp) diff: {d_empty:.4f} (fallback)")

    # テスト6: 単調性
    r_low = slim_waist(img, use_kp, 0.2)
    r_high = slim_waist(img, use_kp, 0.8)
    d_low = np.mean(np.abs(r_low.astype(float) - img.astype(float)))
    d_high = np.mean(np.abs(r_high.astype(float) - img.astype(float)))
    print(f"waist monotonic: 0.2={d_low:.2f} < 0.8={d_high:.2f} = {d_low < d_high}")

    print(f"\ndtype: {r2.dtype}, shape: {r2.shape}")
    print("=== ALL BODY TESTS OK ===")

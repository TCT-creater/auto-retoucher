"""
ai_models/face_restorer.py
===========================
Replicate API 経由の AI 顔復元エンジン。

デプロイ先 (Railway) に GPU がないため、
AI 推論はすべて Replicate API のクラウド GPU で実行する。

モデル: tencentarc/gfpgan (GFPGAN v1.4)
コスト: ~$0.002/画像 (≈ 0.3円)
レイテンシ: ~3-5秒/画像 (コールドスタート時 ~10秒)
"""

from __future__ import annotations

import os
import io
import ssl
import urllib.request
import base64

import cv2
import numpy as np
from PIL import Image

from core.types import ImageRGB

# .env ファイルから環境変数を読み込み
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv がなくてもOS環境変数で動作可能

# ============================================================
# Replicate API 設定
# ============================================================
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")

# GFPGAN v1.4 on Replicate
_GFPGAN_MODEL = (
    "tencentarc/gfpgan:"
    "0fbacf7afc6c144e5be9767cff80f25aff23e52b0708f17e20f9879b2f21516c"
)


# ============================================================
# FaceRestorer — Replicate API クライアント
# ============================================================

class FaceRestorer:
    """Replicate API を使った AI 顔復元。

    使い方:
        restorer = FaceRestorer()
        result = restorer.restore(image, strength=0.7)

    必要な環境変数:
        REPLICATE_API_TOKEN  — Replicate のAPIトークン

    コンテキストマネージャ対応:
        with FaceRestorer() as restorer:
            result = restorer.restore(image, strength=0.7)
    """

    def __init__(self):
        if not REPLICATE_API_TOKEN:
            print("  [FaceRestorer] ⚠️ REPLICATE_API_TOKEN 未設定 — "
                  "AI復元機能は無効です")
        else:
            print("  [FaceRestorer] Replicate API (GFPGAN v1.4) 準備完了")

    @property
    def available(self) -> bool:
        """API トークンが設定されていて利用可能か。"""
        return bool(REPLICATE_API_TOKEN)

    def restore(
        self,
        image: ImageRGB,
        strength: float = 0.7,
    ) -> ImageRGB:
        """Replicate API で顔復元を実行。

        Args:
            image:    入力画像 (RGB, uint8)
            strength: 0.0(元画像) → 1.0(完全復元)

        Returns:
            復元済み画像 (RGB, uint8)
        """
        if strength <= 0.0:
            return image.copy()

        if not self.available:
            raise RuntimeError(
                "REPLICATE_API_TOKEN が設定されていません。\n"
                "Streamlit Secrets または環境変数で設定してください。"
            )

        s = min(max(strength, 0.0), 1.0)

        import replicate

        # 画像を PNG バイトに変換
        pil_img = Image.fromarray(image)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)

        # Replicate API 呼び出し
        output = replicate.run(
            _GFPGAN_MODEL,
            input={
                "img": buf,
                "version": "v1.4",
                "scale": 1,       # 拡大なし（元サイズ維持）
            },
        )

        if output is None:
            print("  [FaceRestorer] ⚠️ 復元結果なし — 元画像を返します")
            return image.copy()

        # レスポンスから画像をダウンロード
        result_np = self._download_result(str(output))

        # リサイズ（サイズが異なる場合）
        if result_np.shape[:2] != image.shape[:2]:
            result_np = cv2.resize(
                result_np,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LANCZOS4,
            )

        # strength でブレンド（0.0=元画像, 1.0=完全復元）
        blended = cv2.addWeighted(
            image, 1.0 - s,
            result_np, s,
            0,
        )
        return blended

    @staticmethod
    def _download_result(url: str) -> np.ndarray:
        """Replicate の出力 URL から画像をダウンロード。"""
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx) as resp:
            data = resp.read()

        pil = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(pil)

    def close(self):
        """クラウドモードではリソース解放不要。"""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

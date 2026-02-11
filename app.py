"""
Auto Retoucher ✨
==================
精密ポートレートレタッチアプリ — OpenCV + AI 復元エンジン
"""

import os
import sys
import time
import io

import streamlit as st
import cv2
import numpy as np
from PIL import Image

from styles import MAIN_CSS

# ============================================================
# ページ設定
# ============================================================
st.set_page_config(
    page_title="Auto Retoucher ✨",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(MAIN_CSS, unsafe_allow_html=True)


# ============================================================
# ヘッダー
# ============================================================
st.markdown("""
<div class="app-header">
    <h1>✨ おまかせレタッチ</h1>
    <p>ワンタッチで、いつもの写真をもっとキレイに 🌸</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# キャッシュ: シングルトン
# ============================================================
@st.cache_resource
def get_face_detector():
    from core.face_detector import FaceDetector
    return FaceDetector(max_faces=3, refine_landmarks=True)


@st.cache_resource
def get_pose_detector():
    from core.body_warper import PoseDetector
    return PoseDetector()


@st.cache_resource
def get_face_restorer():
    from ai_models.face_restorer import FaceRestorer
    return FaceRestorer()


# ============================================================
# サイドバー
# ============================================================

# デフォルト値
_DEFAULTS = {
    "skin_smooth": 0.4, "skin_texture": 0.2, "shine_reduce": 0.2,
    "brightness": 0.5, "contrast": 0.5, "saturation": 0.5,
    "clahe_strength": 0.3, "white_bal": 0.2, "warmth": 0.5,
    "eye_size": 0.2, "nose_slim": 0.15, "lip_plump": 0.0, "jaw_slim": 0.2,
    "bust_enhance": 0.0, "waist_slim": 0.0, "hip_curve": 0.0, "leg_slim": 0.0,
    "sharpness": 0.25,
    "ai_restore": 0.0,
    "eye_catchlight": 0.0, "eye_sclera": 0.0, "eye_iris": 0.0,
    "blemish_sensitivity": 0.0, "blemish_strength": 0.7,
}

for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


with st.sidebar:
    st.markdown("## 📷 写真を選ぶ")
    uploaded = st.file_uploader(
        "写真をドラッグ＆ドロップ",
        type=["jpg", "jpeg", "png", "webp"],
        help="JPEG / PNG / WebP に対応しています",
    )

    # ============================================================
    # ✨ おまかせレタッチボタン
    # ============================================================
    st.markdown("---")
    auto_retouch = st.button(
        "✨ おまかせレタッチ",
        type="primary",
        use_container_width=True,
        help="写真を自動で解析して、いい感じに仕上げます♪",
    )

    if auto_retouch and uploaded is not None:
        _file_bytes = uploaded.read()
        uploaded.seek(0)
        _pil = Image.open(io.BytesIO(_file_bytes)).convert("RGB")
        _img_for_analysis = np.array(_pil)
        _ah, _aw = _img_for_analysis.shape[:2]
        if max(_ah, _aw) > 800:
            _scale = 800 / max(_ah, _aw)
            _img_for_analysis = cv2.resize(
                _img_for_analysis,
                (int(_aw * _scale), int(_ah * _scale)),
                interpolation=cv2.INTER_AREA,
            )

        from core.tone_adjuster import auto_adjust_exposure
        exposure = auto_adjust_exposure(_img_for_analysis)

        # === おまかせプリセット ===
        st.session_state["skin_smooth"] = 0.40
        st.session_state["skin_texture"] = 0.20
        st.session_state["shine_reduce"] = 0.15
        st.session_state["brightness"] = exposure["brightness"]
        st.session_state["contrast"] = exposure["contrast"]
        st.session_state["saturation"] = exposure["saturation"]
        st.session_state["clahe_strength"] = exposure["clahe"]
        st.session_state["white_bal"] = exposure["white_bal"]
        st.session_state["warmth"] = exposure["warmth"]
        st.session_state["eye_size"] = 0.10
        st.session_state["nose_slim"] = 0.08
        st.session_state["lip_plump"] = 0.0
        st.session_state["jaw_slim"] = 0.10
        st.session_state["sharpness"] = 0.25
        st.session_state["ai_restore"] = 0.0      # API課金なし
        st.session_state["eye_catchlight"] = 0.15
        st.session_state["eye_sclera"] = 0.10
        st.session_state["eye_iris"] = 0.10
        st.session_state["blemish_sensitivity"] = 0.25
        st.session_state["blemish_strength"] = 0.60
        st.session_state["_auto_exposure_info"] = exposure
        st.rerun()

    elif auto_retouch and uploaded is None:
        st.warning("先に写真を選んでね 📷")

    # 自動解析結果の表示
    if "_auto_exposure_info" in st.session_state:
        exp = st.session_state["_auto_exposure_info"]
        st.markdown(f"""
        <div class="analysis-box">
            <b>🔍 自動解析の結果</b><br>
            明るさ: L̄={exp['_l_mean']:.0f} → {exp['brightness']:.2f} に調整<br>
            コントラスト: σ={exp['_l_std']:.0f} → {exp['contrast']:.2f}<br>
            色み: Δa*={exp['_a_offset']:.0f} Δb*={exp['_b_offset']:.0f} → ホワイトバランス {exp['white_bal']:.2f}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ─────────────────────────────────────────
    # 🧴 お肌の補正
    # ─────────────────────────────────────────
    st.markdown('<div class="slider-group"><h4>🧴 お肌の補正</h4></div>',
                unsafe_allow_html=True)
    skin_smooth = st.slider("なめらかさ", 0.0, 1.0, key="skin_smooth", step=0.05,
                            help="肌をふんわりなめらかに整えます")
    skin_texture = st.slider("キメの均一化", 0.0, 1.0, key="skin_texture", step=0.05,
                             help="色ムラを均一にしつつ肌のキメを維持")
    shine_reduce = st.slider("テカリ抑え", 0.0, 1.0, key="shine_reduce", step=0.05,
                             help="Tゾーンなどの光りすぎを自然に抑えます")

    # ─────────────────────────────────────────
    # 🌈 明るさ・色あい
    # ─────────────────────────────────────────
    st.markdown('<div class="slider-group"><h4>🌈 明るさ・色あい</h4></div>',
                unsafe_allow_html=True)
    brightness = st.slider("明るさ", 0.0, 1.0, key="brightness", step=0.05,
                           help="真ん中(0.5)がそのまま。右で明るく、左で暗く")
    contrast = st.slider("メリハリ", 0.0, 1.0, key="contrast", step=0.05,
                         help="写真の明暗のメリハリを調整します")
    saturation = st.slider("鮮やかさ", 0.0, 1.0, key="saturation", step=0.05,
                           help="色の鮮やかさ。真ん中がそのまま")
    clahe_strength = st.slider("立体感", 0.0, 1.0,
                               key="clahe_strength", step=0.05,
                               help="顔の立体感・奥行きを強調します")
    white_bal = st.slider("色かぶり補正", 0.0, 1.0, key="white_bal", step=0.05,
                          help="照明による色かぶりを自動で補正")
    warmth = st.slider("色温度", 0.0, 1.0, key="warmth", step=0.05,
                       help="左=クール（青み） / 右=ウォーム（暖かみ）")

    # ─────────────────────────────────────────
    # 👁️ 顔の形
    # ─────────────────────────────────────────
    st.markdown('<div class="slider-group"><h4>👁️ 顔の形</h4></div>',
                unsafe_allow_html=True)
    eye_size = st.slider("目の大きさ", 0.0, 1.0, key="eye_size", step=0.05,
                         help="目をほんの少し大きくします")
    nose_slim = st.slider("鼻すじ", 0.0, 1.0, key="nose_slim", step=0.05,
                          help="鼻のラインをすっきりさせます")
    lip_plump = st.slider("唇のふっくら感", 0.0, 1.0, key="lip_plump", step=0.05,
                          help="唇に自然なボリュームを")
    jaw_slim = st.slider("フェイスライン", 0.0, 1.0, key="jaw_slim", step=0.05,
                         help="あごのラインをシャープに（少し時間がかかります）")

    # ─────────────────────────────────────────
    # 💃 スタイル補正
    # ─────────────────────────────────────────
    st.markdown('<div class="slider-group"><h4>💃 スタイル補正</h4></div>',
                unsafe_allow_html=True)
    bust_enhance = st.slider("バストライン", 0.0, 1.0, key="bust_enhance", step=0.05,
                             help="全身が写っている場合のみ有効です")
    waist_slim = st.slider("ウエスト", 0.0, 1.0, key="waist_slim", step=0.05,
                           help="ウエストをすっきり見せます")
    hip_curve = st.slider("ヒップライン", 0.0, 1.0, key="hip_curve", step=0.05)
    leg_slim = st.slider("脚のライン", 0.0, 1.0, key="leg_slim", step=0.05)

    # ─────────────────────────────────────────
    # ✨ 仕上げ
    # ─────────────────────────────────────────
    st.markdown('<div class="slider-group"><h4>✨ 仕上げ</h4></div>',
                unsafe_allow_html=True)
    sharpness = st.slider("くっきり感", 0.0, 1.0, key="sharpness", step=0.05,
                          help="輪郭をくっきりさせて写真を鮮明にします")

    # ─────────────────────────────────────────
    # 🔬 AI 美肌復元
    # ─────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="slider-group"><h4>🔬 AI 美肌復元</h4></div>',
                unsafe_allow_html=True)

    restorer = get_face_restorer()
    if restorer.available:
        st.markdown('<span class="status-badge">🟢 AI 接続OK</span>',
                    unsafe_allow_html=True)
        ai_restore = st.slider("AI 復元の強さ", 0.0, 1.0, key="ai_restore", step=0.05,
                               help="AIが肌のキメや目の輝きを自然に復元します")
    else:
        st.markdown(
            '<span class="status-badge" style="background:rgba(239,100,100,0.12);'
            'color:#e07070;">🔴 APIキー未設定</span>',
            unsafe_allow_html=True,
        )
        st.caption("`.env` に `REPLICATE_API_TOKEN` を\n設定するとAI復元が使えます")
        ai_restore = 0.0

    # ─────────────────────────────────────────
    # 👁️ 目のキラキラ
    # ─────────────────────────────────────────
    st.markdown('<div class="slider-group"><h4>👁️ 目のキラキラ</h4></div>',
                unsafe_allow_html=True)
    eye_catchlight = st.slider("キャッチライト", 0.0, 1.0, key="eye_catchlight", step=0.05,
                                help="瞳にキラッとした輝きを入れます ✨")
    eye_sclera = st.slider("白目の透明感", 0.0, 1.0, key="eye_sclera", step=0.05,
                            help="白目をクリアにして澄んだ目に")
    eye_iris = st.slider("瞳のディテール", 0.0, 1.0, key="eye_iris", step=0.05,
                          help="虹彩のコントラストを強調します")

    # ─────────────────────────────────────────
    # 🧹 シミ・ホクロの修正
    # ─────────────────────────────────────────
    st.markdown('<div class="slider-group"><h4>🧹 シミ・ホクロの修正</h4></div>',
                unsafe_allow_html=True)
    blemish_sensitivity = st.slider("検出のつよさ", 0.0, 1.0,
                                     key="blemish_sensitivity", step=0.05,
                                     help="0=OFF / 右にするほど小さいシミも検出")
    blemish_strength = st.slider("消す強さ", 0.0, 1.0, key="blemish_strength",
                                  step=0.05,
                                  help="検出されたスポットをどの程度消すか")


# ============================================================
# メイン処理パイプライン
# ============================================================
if uploaded is not None:
    file_bytes = uploaded.read()
    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_rgb = np.array(pil_img)
    h, w = img_rgb.shape[:2]

    MAX_DIM = 1600
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w
        st.sidebar.caption(f"📐 リサイズ: {pil_img.width}×{pil_img.height} → {w}×{h}")

    progress = st.progress(0, text="🔍 お顔を探しています...")
    t_start = time.time()
    steps_log = []

    result = img_rgb.copy()

    # 1. 顔検出
    detector = get_face_detector()
    faces = detector.detect(img_rgb)
    n_faces = len(faces)
    steps_log.append(f"お顔の検出: {n_faces} 人")
    progress.progress(10, text=f"✅ {n_faces} 人のお顔を見つけました")

    # ================================================================
    # パイプライン: 形 → 肌質 → 明るさ → 仕上げ → AI復元
    # ================================================================

    # 2. 顔の形の補正
    if n_faces > 0 and (eye_size > 0 or nose_slim > 0 or lip_plump > 0 or jaw_slim > 0):
        from core.face_warper import enlarge_eyes, slim_nose, slim_jaw, plump_lips

        lm = faces[0].landmarks
        if eye_size > 0:
            result = enlarge_eyes(result, lm, w, h, eye_size)
            steps_log.append(f"目の大きさ: {eye_size:.0%}")
        if nose_slim > 0:
            result = slim_nose(result, lm, w, h, nose_slim)
            steps_log.append(f"鼻すじ: {nose_slim:.0%}")
        if lip_plump > 0:
            result = plump_lips(result, lm, w, h, lip_plump)
            steps_log.append(f"唇ふっくら: {lip_plump:.0%}")
        if jaw_slim > 0:
            progress.progress(20, text="📐 フェイスライン調整中...")
            result = slim_jaw(result, lm, w, h, jaw_slim)
            steps_log.append(f"フェイスライン: {jaw_slim:.0%}")

        progress.progress(30, text="👁️ 顔の形 完了")

    # 3. スタイル補正
    body_active = bust_enhance > 0 or waist_slim > 0 or hip_curve > 0 or leg_slim > 0
    if body_active:
        from core.body_warper import slim_waist, enhance_curves, slim_legs, enhance_bust

        pose_det = get_pose_detector()
        poses = pose_det.detect(result)
        if poses and len(poses) > 0:
            kp = poses[0]
            if bust_enhance > 0:
                result = enhance_bust(result, kp, bust_enhance)
                steps_log.append(f"バストライン: {bust_enhance:.0%}")
            if waist_slim > 0:
                result = slim_waist(result, kp, waist_slim)
            if hip_curve > 0:
                result = enhance_curves(result, kp, hip_curve)
            if leg_slim > 0:
                result = slim_legs(result, kp, leg_slim)
            steps_log.append("スタイル補正: 適用済み")
        else:
            steps_log.append("スタイル補正: 全身が写っていないためスキップ")

        progress.progress(45, text="💃 スタイル補正 完了")

    # 4. お肌の補正
    if n_faces > 0 and (skin_smooth > 0 or skin_texture > 0 or shine_reduce > 0):
        from core.skin_smoother import smooth_skin, enhance_skin_texture, reduce_shine

        for face in faces:
            lm = face.landmarks
            skin_mask = detector.get_skin_mask(lm, w, h)

            if skin_smooth > 0:
                result = smooth_skin(result, skin_mask, skin_smooth)
            if skin_texture > 0:
                result = enhance_skin_texture(result, skin_mask, skin_texture)
            if shine_reduce > 0:
                result = reduce_shine(result, skin_mask, shine_reduce)

        steps_log.append("お肌の補正: 適用済み")
        progress.progress(60, text="🧴 お肌の補正 完了")

    # 5. 明るさ・色あい
    tone_active = any([
        abs(brightness - 0.5) > 0.01,
        abs(contrast - 0.5) > 0.01,
        abs(saturation - 0.5) > 0.01,
        clahe_strength > 0,
        white_bal > 0,
        abs(warmth - 0.5) > 0.01,
    ])
    if tone_active:
        from core.tone_adjuster import (
            adjust_brightness, adjust_contrast, adjust_saturation,
            apply_clahe, adjust_white_balance, adjust_warmth,
        )
        result = apply_clahe(result, clahe_strength)
        result = adjust_brightness(result, brightness)
        result = adjust_contrast(result, contrast)
        result = adjust_saturation(result, saturation)
        result = adjust_white_balance(result, white_bal)
        result = adjust_warmth(result, warmth)
        steps_log.append("明るさ・色あい: 適用済み")
        progress.progress(75, text="🌈 明るさ・色あい 完了")

    # 6. くっきり感
    if sharpness > 0:
        from core.sharpener import sharpen
        result = sharpen(result, sharpness)
        steps_log.append(f"くっきり感: {sharpness:.0%}")
        progress.progress(78, text="✨ くっきり感 完了")

    # ================================================================
    # AI 補正
    # ================================================================

    # 7. 目のキラキラ
    eye_active = n_faces > 0 and (eye_catchlight > 0 or eye_sclera > 0 or eye_iris > 0)
    if eye_active:
        from ai_models.eye_enhancer import (
            enhance_catchlight, whiten_sclera, enhance_iris_detail
        )
        lm = faces[0].landmarks
        if eye_catchlight > 0:
            result = enhance_catchlight(result, lm, w, h, eye_catchlight)
            steps_log.append(f"キャッチライト: {eye_catchlight:.0%}")
        if eye_sclera > 0:
            result = whiten_sclera(result, lm, w, h, eye_sclera)
            steps_log.append(f"白目の透明感: {eye_sclera:.0%}")
        if eye_iris > 0:
            result = enhance_iris_detail(result, lm, w, h, eye_iris)
            steps_log.append(f"瞳のディテール: {eye_iris:.0%}")
        progress.progress(83, text="👁️ 目のキラキラ 完了")

    # 8. シミ・ホクロの修正
    if n_faces > 0 and blemish_sensitivity > 0:
        from ai_models.blemish_detector import auto_remove_blemishes
        lm = faces[0].landmarks
        skin_mask = detector.get_skin_mask(lm, w, h)
        result, blemish_mask, n_spots = auto_remove_blemishes(
            result, skin_mask, blemish_sensitivity, blemish_strength
        )
        steps_log.append(f"シミ・ホクロ修正: {n_spots} 個を検出して修正")
        progress.progress(88, text=f"🧹 シミ修正 完了（{n_spots} 個）")

    # 9. AI 美肌復元
    if ai_restore > 0 and restorer.available:
        progress.progress(90, text="🔬 AI が美肌を復元しています...")
        try:
            t_ai_start = time.time()
            result = restorer.restore(result, strength=ai_restore)
            t_ai_ms = (time.time() - t_ai_start) * 1000
            steps_log.append(f"AI 美肌復元: {ai_restore:.0%}（{t_ai_ms:.0f} ms）")
            progress.progress(97, text=f"🔬 AI 復元 完了（{t_ai_ms:.0f} ms）")
        except Exception as e:
            steps_log.append(f"⚠️ AI 復元スキップ: {type(e).__name__}")
            progress.progress(97, text="⚠️ AI スキップ — 通常仕上げで完了")

    # 処理完了
    t_elapsed = (time.time() - t_start) * 1000
    progress.progress(100, text=f"✅ 完成！（{t_elapsed:.0f} ms）")

    # ============================================================
    # 結果表示: ビフォー / アフター
    # ============================================================
    st.markdown("### 📸 ビフォー / アフター")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**もとの写真**")
        st.image(img_rgb, use_container_width=True)

    with col2:
        st.markdown("**レタッチ後 ✨**")
        st.image(result, use_container_width=True)

    # ============================================================
    # ダウンロード
    # ============================================================
    result_pil = Image.fromarray(result)
    buf = io.BytesIO()
    result_pil.save(buf, format="JPEG", quality=95)

    st.download_button(
        label="📥 レタッチ済み画像をダウンロード",
        data=buf.getvalue(),
        file_name="retouched.jpg",
        mime="image/jpeg",
    )

    # ============================================================
    # 処理統計
    # ============================================================
    st.markdown(f"""
    <div class="stats-box">
        <div class="stat-item">
            <span class="stat-label">画像サイズ</span>
            <span class="stat-value">{w} × {h} px</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">検出された顔</span>
            <span class="stat-value">{n_faces} 人</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">処理時間</span>
            <span class="stat-value">{t_elapsed:.0f} ms</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">適用した補正</span>
            <span class="stat-value">{len(steps_log)} ステップ</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📋 処理の詳細"):
        for step in steps_log:
            st.write(f"• {step}")

else:
    # 未アップロード時
    st.markdown("""
    <div class="welcome-area">
        <p class="emoji">📸</p>
        <h3>写真を選んでレタッチを始めましょう 🌸</h3>
        <p>
            左のサイドバーから写真を選んでください<br>
            「✨ おまかせレタッチ」を押すだけで<br>
            AIが自動でキレイに仕上げます♪
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🎀 できること")
    cols = st.columns(4)
    features = [
        ("🧴", "お肌の補正", "なめらか肌・テカリ抑え"),
        ("🌈", "明るさ・色あい", "自動で最適な明るさに"),
        ("👁️", "顔・目の補正", "目のキラキラ・輪郭補正"),
        ("🔬", "AI 美肌復元", "AIでキメと輝きを復元"),
    ]
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <p class="icon">{icon}</p>
                <p class="title">{title}</p>
                <p class="desc">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

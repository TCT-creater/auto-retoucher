"""
styles.py — Auto Retoucher UI テーマ
=====================================
やさしいベージュ系カラーの可愛いデザイン。
目に優しい暖色パレット。
"""

MAIN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Zen+Maru+Gothic:wght@300;400;500;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=M+PLUS+Rounded+1c:wght@300;400;500;700&display=swap');

/* === グローバル背景・フォント === */
.stApp {
    font-family: 'M PLUS Rounded 1c', 'Zen Maru Gothic', 'Hiragino Maru Gothic Pro', sans-serif;
    background: linear-gradient(165deg, #fdf6ee 0%, #f5ede3 40%, #f0e6da 70%, #ece0d4 100%) !important;
}

/* サイドバー */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #faf3ea 0%, #f3e8db 100%) !important;
    border-right: 1px solid rgba(200, 170, 140, 0.25);
}
section[data-testid="stSidebar"] * {
    color: #6b5744 !important;
}
section[data-testid="stSidebar"] .stSlider label {
    color: #8b7355 !important;
    font-weight: 500;
    font-size: 0.82rem;
}

/* スライダーのアクセントカラー */
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #d4a574 !important;
    border-color: #c99660 !important;
}
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [data-testid="stTickBar"] {
    background: rgba(200, 170, 140, 0.3) !important;
}

/* メインテキスト */
.stApp h1, .stApp h2, .stApp h3 {
    color: #5a4635 !important;
}
.stApp p, .stApp span, .stApp label, .stApp div {
    color: #6b5744;
}

/* === ヘッダー === */
.app-header {
    text-align: center;
    padding: 1.8rem 1rem 1.2rem;
    background: linear-gradient(135deg,
        rgba(212, 165, 116, 0.15),
        rgba(230, 190, 150, 0.12),
        rgba(200, 160, 120, 0.1));
    border-radius: 20px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(200, 170, 140, 0.25);
    box-shadow: 0 2px 12px rgba(180, 140, 100, 0.08);
}
.app-header h1 {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #c99660, #d4a574, #b8860b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
    letter-spacing: 0.05em;
}
.app-header p {
    color: rgba(107, 87, 68, 0.6) !important;
    font-size: 0.82rem;
    margin: 0;
    letter-spacing: 0.03em;
}

/* === サイドバー: スライダーグループ === */
.slider-group {
    background: linear-gradient(135deg,
        rgba(212, 165, 116, 0.08),
        rgba(200, 170, 140, 0.06));
    border: 1px solid rgba(200, 170, 140, 0.2);
    border-radius: 14px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.7rem;
}
.slider-group h4 {
    color: #a0805e !important;
    font-size: 0.82rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    letter-spacing: 0.04em;
}

/* === 画像コンテナ === */
.image-container {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(200, 170, 140, 0.25);
    box-shadow: 0 4px 16px rgba(160, 120, 80, 0.1);
}

/* === ステータスバッジ === */
.status-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 2px;
    background: rgba(74, 180, 120, 0.12);
    color: #4caf7a !important;
    border: 1px solid rgba(74, 180, 120, 0.25);
}

/* === 処理統計ボックス === */
.stats-box {
    background: linear-gradient(135deg,
        rgba(250, 243, 234, 0.9),
        rgba(243, 232, 219, 0.95));
    border: 1px solid rgba(200, 170, 140, 0.25);
    border-radius: 14px;
    padding: 1rem;
    margin-top: 1rem;
    box-shadow: 0 2px 8px rgba(180, 140, 100, 0.06);
}
.stats-box .stat-item {
    display: flex;
    justify-content: space-between;
    padding: 0.35rem 0;
    border-bottom: 1px solid rgba(200, 170, 140, 0.12);
    font-size: 0.82rem;
}
.stats-box .stat-label { color: rgba(107, 87, 68, 0.6) !important; }
.stats-box .stat-value { color: #a0805e !important; font-weight: 600; }

/* === ダウンロードボタン === */
div.stDownloadButton > button {
    background: linear-gradient(135deg, #d4a574, #c99660) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    box-shadow: 0 3px 12px rgba(200, 150, 96, 0.25) !important;
    font-family: 'M PLUS Rounded 1c', sans-serif !important;
}
div.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(200, 150, 96, 0.35) !important;
}

/* === 自動レタッチボタン === */
div.stButton > button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #e8b88a, #d4a574, #c99660) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.7rem 1.5rem !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em !important;
    box-shadow: 0 4px 16px rgba(200, 150, 96, 0.3) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    font-family: 'M PLUS Rounded 1c', sans-serif !important;
}
div.stButton > button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 8px 24px rgba(200, 150, 96, 0.4) !important;
}

/* === プログレスバー === */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #d4a574, #e8b88a, #c99660) !important;
    border-radius: 8px;
}

/* === expander === */
details {
    background: rgba(250, 243, 234, 0.8) !important;
    border: 1px solid rgba(200, 170, 140, 0.2) !important;
    border-radius: 12px !important;
}

/* === 自動解析ボックス === */
.analysis-box {
    background: linear-gradient(135deg,
        rgba(212, 165, 116, 0.1),
        rgba(230, 200, 170, 0.08));
    border: 1px solid rgba(200, 170, 140, 0.25);
    border-radius: 12px;
    padding: 0.6rem 0.8rem;
    font-size: 0.72rem;
    margin-bottom: 0.5rem;
    color: #8b7355 !important;
    line-height: 1.6;
}
.analysis-box b {
    color: #a0805e !important;
}

/* === フィーチャーカード === */
.feature-card {
    text-align: center;
    padding: 1.2rem 0.8rem;
    background: linear-gradient(135deg,
        rgba(250, 243, 234, 0.95),
        rgba(243, 232, 219, 0.9));
    border-radius: 16px;
    border: 1px solid rgba(200, 170, 140, 0.2);
    box-shadow: 0 2px 8px rgba(180, 140, 100, 0.06);
    transition: transform 0.2s ease;
}
.feature-card:hover {
    transform: translateY(-2px);
}
.feature-card .icon { font-size: 2rem; margin: 0; }
.feature-card .title {
    font-weight: 600;
    margin: 0.3rem 0 0.2rem;
    color: #6b5744 !important;
    font-size: 0.88rem;
}
.feature-card .desc {
    font-size: 0.72rem;
    color: rgba(107, 87, 68, 0.5) !important;
    margin: 0;
}

/* === ファイルアップローダー === */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    border: 2px dashed rgba(200, 170, 140, 0.4) !important;
    border-radius: 14px !important;
    background: #ffffff !important;
    padding: 0.5rem !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] * {
    background: transparent !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] small {
    color: #a09080 !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: rgba(212, 165, 116, 0.15) !important;
    border: 1px solid rgba(200, 170, 140, 0.3) !important;
    border-radius: 8px !important;
}

/* === 横区切り線 === */
section[data-testid="stSidebar"] hr {
    border-color: rgba(200, 170, 140, 0.2) !important;
}

/* === ウェルカムエリア === */
.welcome-area {
    text-align: center;
    padding: 3rem 1rem;
}
.welcome-area .emoji { font-size: 4rem; margin-bottom: 0.5rem; }
.welcome-area h3 {
    color: rgba(107, 87, 68, 0.7) !important;
    font-weight: 500;
}
.welcome-area p {
    color: rgba(107, 87, 68, 0.4) !important;
    font-size: 0.85rem;
    line-height: 1.8;
}
</style>
"""

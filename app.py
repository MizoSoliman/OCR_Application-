import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm
import json
import pandas as pd
import requests
import re
import os
from PIL import Image
import plotly.graph_objects as go
import gdown

# ─── Google Drive File IDs ───────────────────────────────────────
MODEL_ID   = "1sO9XEf3tKedjOZBVSlfUSnWfMwVyhWxa"
CLASSES_ID = ""   # ← حط ID بتاع class_names.json
CSV_ID     = ""   # ← حط ID بتاع egyptian_drugs_database.csv
# ────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def download_files():
    if not os.path.exists("best_model.pth"):
        with st.spinner("⬇️ جارٍ تحميل النموذج (44 MB)..."):
            gdown.download(id=MODEL_ID, output="best_model.pth", quiet=False)
    if CLASSES_ID and not os.path.exists("class_names.json"):
        gdown.download(id=CLASSES_ID, output="class_names.json", quiet=False)
    if CSV_ID and not os.path.exists("egyptian_drugs_database.csv"):
        gdown.download(id=CSV_ID, output="egyptian_drugs_database.csv", quiet=False)

download_files()

# ─── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="PharmaScan AI",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: #0a0f1e;
    font-family: 'DM Sans', sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Hero Header ── */
.hero {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b35 50%, #0a1628 100%);
    border-bottom: 1px solid rgba(56, 189, 248, 0.15);
    padding: 0.8rem 3rem 0.6rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(56,189,248,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 200px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(99,102,241,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: #f0f9ff;
    margin: 0 0 0.2rem;
    letter-spacing: -0.5px;
}
.hero-title span { color: #38bdf8; }
.hero-sub {
    color: #64748b;
    font-size: 0.88rem;
    font-weight: 300;
    letter-spacing: 0.3px;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.2);
    color: #38bdf8;
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 1.5px; text-transform: uppercase;
    padding: 3px 10px; border-radius: 20px;
    margin-bottom: 0.4rem;
}

/* ── Main Layout ── */
.main-wrap {
    padding: 2.5rem 3rem;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
    min-height: calc(100vh - 180px);
}

/* ── Column divider ── */
[data-testid="stColumns"] > div:first-child {
    border-right: 1.5px solid rgba(56,189,248,0.3) !important;
    padding-right: 2rem !important;
}
[data-testid="stColumns"] > div:last-child {
    padding-left: 2rem !important;
}

/* ── Upload Panel ── */
.panel {
    background: #0d1424;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px;
    padding: 2rem;
    position: relative;
}
.panel-label {
    font-size: 0.7rem; font-weight: 600;
    letter-spacing: 2px; text-transform: uppercase;
    color: #38bdf8; margin-bottom: 1.2rem;
    display: flex; align-items: center; gap: 8px;
}
.panel-label::after {
    content: '';
    flex: 1; height: 1px;
    background: linear-gradient(to right, rgba(56,189,248,0.3), transparent);
}

/* ── Streamlit widget overrides ── */
[data-testid="stFileUploadDropzone"] {
    background: rgba(56,189,248,0.03) !important;
    border: 2px dashed rgba(56,189,248,0.2) !important;
    border-radius: 16px !important;
    padding: 3rem 1rem !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    background: rgba(56,189,248,0.06) !important;
    border-color: rgba(56,189,248,0.5) !important;
}
[data-testid="stFileUploadDropzone"] p {
    color: #475569 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Result Card ── */
.result-name {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: #f0f9ff;
    margin: 0 0 0.4rem;
    line-height: 1.2;
}
.result-generic {
    color: #38bdf8;
    font-size: 0.85rem;
    font-weight: 400;
    margin-bottom: 1.5rem;
    opacity: 0.8;
}
.confidence-bar-wrap {
    margin-bottom: 1.5rem;
}
.conf-label {
    display: flex; justify-content: space-between;
    font-size: 0.78rem; color: #64748b;
    margin-bottom: 6px;
}
.conf-track {
    background: rgba(255,255,255,0.05);
    border-radius: 99px; height: 6px;
    overflow: hidden;
}
.conf-fill {
    height: 100%; border-radius: 99px;
    transition: width 1s ease;
}
.conf-high  { background: linear-gradient(to right, #10b981, #34d399); }
.conf-mid   { background: linear-gradient(to right, #f59e0b, #fbbf24); }
.conf-low   { background: linear-gradient(to right, #ef4444, #f87171); }

/* ── Info Cards ── */
.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1.5rem;
}
.info-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.2rem;
}
.info-card.full { grid-column: 1 / -1; }
.info-card-icon { font-size: 1.2rem; margin-bottom: 0.5rem; }
.info-card-title {
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 1.5px; text-transform: uppercase;
    color: #475569; margin-bottom: 0.6rem;
}
.info-card-body {
    font-size: 0.88rem; color: #94a3b8;
    line-height: 1.6;
}
.info-card.warn { border-color: rgba(245,158,11,0.2); background: rgba(245,158,11,0.03); }
.info-card.warn .info-card-title { color: #f59e0b; }

/* ── Top-K predictions list ── */
.topk-item {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.topk-item:last-child { border-bottom: none; }
.topk-rank {
    width: 28px; height: 28px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem; font-weight: 700;
    flex-shrink: 0;
}
.rank-1 { background: rgba(56,189,248,0.15); color: #38bdf8; }
.rank-2 { background: rgba(255,255,255,0.06); color: #64748b; }
.rank-other { background: rgba(255,255,255,0.03); color: #475569; }
.topk-name { flex: 1; font-size: 0.85rem; color: #94a3b8; }
.topk-conf { font-size: 0.8rem; font-weight: 600; color: #38bdf8; }

/* ── Source badge ── */
.source-badge {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 1px; text-transform: uppercase;
    padding: 3px 10px; border-radius: 20px;
    margin-top: 1rem;
}
.source-local { background: rgba(16,185,129,0.1); color: #10b981; border: 1px solid rgba(16,185,129,0.2); }
.source-fda   { background: rgba(56,189,248,0.1); color: #38bdf8; border: 1px solid rgba(56,189,248,0.2); }
.source-na    { background: rgba(100,116,139,0.1); color: #64748b; border: 1px solid rgba(100,116,139,0.2); }

/* ── Empty state ── */
.empty-state {
    text-align: center; padding: 4rem 2rem;
    color: #1e293b;
}
.empty-state-icon { font-size: 4rem; margin-bottom: 1rem; opacity: 0.4; }
.empty-state-text { color: #334155; font-size: 0.95rem; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #38bdf8 !important; }

/* ── Disclaimer ── */
.disclaimer {
    background: rgba(245,158,11,0.05);
    border: 1px solid rgba(245,158,11,0.15);
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    margin-top: 1rem;
    font-size: 0.78rem;
    color: #92400e;
    line-height: 1.5;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0f1e; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 99px; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ──────────────────────────────────────────────────
IMG_SIZE    = 300
MODEL_PATH  = "best_model.pth"
CLASSES_PATH = "class_names.json"
CSV_PATH    = "egyptian_drugs_database.csv"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Load Resources (cached) ────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    with open(CLASSES_PATH, encoding="utf-8") as f:
        class_names = json.load(f)
    model = timm.create_model(
        "efficientnet_b3", pretrained=False, num_classes=len(class_names)
    )
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    return model, class_names

@st.cache_data(show_spinner=False)
def load_drug_db():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df["name_lower"] = df["drug_name"].str.lower().str.strip()
        return df
    return None

# ─── Inference ──────────────────────────────────────────────────
infer_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(img: Image.Image, model, class_names, top_k=5):
    tensor = infer_tf(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)
    top_p, top_i = torch.topk(probs, min(top_k, len(class_names)))
    return [
        {"name": class_names[i], "conf": float(p)}
        for i, p in zip(top_i.squeeze(0).cpu().flatten(), top_p.squeeze(0).cpu().flatten())
    ]

# ─── Drug Info ──────────────────────────────────────────────────
def clean_name(raw: str) -> list:
    no_num = re.sub(r"\b\d+(\.\d+)?\s*(mg|ml|mcg|g|iu|%|units?)?\b", "", raw, flags=re.IGNORECASE)
    no_num = re.sub(r"[_\-]", " ", no_num).strip()
    first  = raw.split()[0]
    return list(dict.fromkeys(
        [c.strip() for c in [raw, no_num, first] if len(c.strip()) > 2]
    ))

def get_drug_info(drug_name: str, df) -> dict:
    empty = {"indications": "غير متوفر", "dosage": "غير متوفر",
             "warnings": "غير متوفر", "generic": "", "source": ""}
    # 1️⃣ Local DB
    if df is not None:
        name_l = drug_name.lower().strip()
        for name in clean_name(drug_name):
            nl = name.lower().strip()
            match = df[df["name_lower"] == nl]
            if match.empty:
                match = df[df["name_lower"].str.contains(nl.split()[0], na=False)]
            if not match.empty:
                row = match.iloc[0]
                return {
                    "indications": row.get("indications", "غير متوفر"),
                    "dosage":      row.get("dosage",      "غير متوفر"),
                    "warnings":    row.get("warnings",    "غير متوفر"),
                    "generic":     row.get("generic_name", ""),
                    "source":      "local"
                }
    # 2️⃣ OpenFDA
    url = "https://api.fda.gov/drug/label.json"
    for name in clean_name(drug_name):
        for field in ["openfda.brand_name", "openfda.generic_name"]:
            try:
                r = requests.get(url, params={"search": f'{field}:"{name}"', "limit": 1}, timeout=8)
                if r.status_code == 200:
                    data = r.json().get("results", [])
                    if data:
                        lbl = data[0]
                        return {
                            "indications": " ".join(lbl.get("indications_and_usage",     ["N/A"]))[:500],
                            "dosage":      " ".join(lbl.get("dosage_and_administration", ["N/A"]))[:500],
                            "warnings":    " ".join(lbl.get("warnings",                  ["N/A"]))[:300],
                            "generic":     ", ".join(lbl.get("openfda", {}).get("generic_name", [])),
                            "source":      "fda"
                        }
            except Exception:
                pass
    return {**empty, "source": "na"}

# ─── UI Helpers ─────────────────────────────────────────────────
def conf_class(c):
    if c >= 0.70: return "conf-high"
    if c >= 0.40: return "conf-mid"
    return "conf-low"

def source_badge(src):
    if src == "local": return '<span class="source-badge source-local">📁 Local DB</span>'
    if src == "fda":   return '<span class="source-badge source-fda">🌐 OpenFDA</span>'
    return '<span class="source-badge source-na">⚠️ Not Found</span>'

def rank_class(i):
    if i == 0: return "rank-1"
    if i == 1: return "rank-2"
    return "rank-other"

# ─── Hero ───────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div style="text-align:center;">
        <div class="hero-badge">💊 AI-Powered · EfficientNetB3 · 150 Drugs</div>
        <div class="hero-title">Pharma<span>Scan</span> AI</div>
        <div class="hero-sub">صوّر علبة الدواء — النموذج يتعرف عليه فوراً ويعرض الاستخدام والجرعة</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Load model ─────────────────────────────────────────────────
try:
    with st.spinner("جارٍ تحميل النموذج..."):
        model, CLASS_NAMES = load_model()
        df_drugs = load_drug_db()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"❌ فشل تحميل النموذج: {e}")
    st.info("تأكد من وجود `best_model.pth` و `class_names.json` في نفس مجلد التطبيق.")
    st.stop()

# ─── Main Layout ────────────────────────────────────────────────
st.markdown('<div class="main-wrap" style="padding:1rem 3rem;display:flex;gap:2rem;">', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1])

# ══════════════════════════════════════════════
# LEFT COLUMN — Upload + Image Preview
# ══════════════════════════════════════════════
with col_left:
    st.markdown('<div class="panel-label">📷 &nbsp; الصورة المدخلة</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        label="",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
        key="uploader"
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_container_width=True,
                 caption="", output_format="JPEG")

        st.markdown(f"""
        <div style="
            display:flex; gap:1rem; margin-top:1rem;
            font-size:0.78rem; color:#475569;
        ">
            <span>📐 {img.width} × {img.height}px</span>
            <span>📁 {uploaded.name}</span>
            <span>💾 {uploaded.size/1024:.1f} KB</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📸</div>
            <div class="empty-state-text">ارفع صورة علبة الدواء هنا<br>
            <span style="font-size:0.8rem;color:#1e293b;">JPG · PNG · WEBP</span></div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# RIGHT COLUMN — Results
# ══════════════════════════════════════════════
with col_right:
    st.markdown('<div class="panel-label">🔬 &nbsp; نتائج التحليل</div>', unsafe_allow_html=True)

    if not uploaded:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">💊</div>
            <div class="empty-state-text">النتائج ستظهر هنا<br>
            <span style="font-size:0.8rem;color:#1e293b;">بعد رفع صورة الدواء</span></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("جارٍ التحليل..."):
            preds = predict(img, model, CLASS_NAMES, top_k=1)
            best  = preds[0]
            info  = get_drug_info(best["name"], df_drugs)

        # ── Drug Name & Confidence ──
        cf = best["conf"]
        cc = conf_class(cf)
        generic_txt = f'<div class="result-generic">{info["generic"]}</div>' if info["generic"] else ""

        st.markdown(f"""
        <div class="result-name">{best["name"]}</div>
        {generic_txt}
        <div class="confidence-bar-wrap">
            <div class="conf-label">
                <span>Confidence</span>
                <span style="color:#f0f9ff;font-weight:600;">{cf*100:.1f}%</span>
            </div>
            <div class="conf-track">
                <div class="conf-fill {cc}" style="width:{cf*100:.1f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Drug Info Cards ──
        st.markdown(f"""
        <div class="info-grid">
            <div class="info-card full">
                <div class="info-card-icon">📋</div>
                <div class="info-card-title">الاستخدامات</div>
                <div class="info-card-body">{info['indications']}</div>
            </div>
            <div class="info-card">
                <div class="info-card-icon">💉</div>
                <div class="info-card-title">الجرعة</div>
                <div class="info-card-body">{info['dosage']}</div>
            </div>
            <div class="info-card warn">
                <div class="info-card-icon">⚠️</div>
                <div class="info-card-title">التحذيرات</div>
                <div class="info-card-body">{info['warnings']}</div>
            </div>
        </div>
        {source_badge(info['source'])}
        """, unsafe_allow_html=True)

        # ── Disclaimer ──
        st.markdown("""
        <div class="disclaimer">
            ⚕️ <strong>تنبيه طبي:</strong> هذا التطبيق للأغراض التعليمية والبحثية فقط.
            لا تعتمد عليه كبديل عن استشارة الطبيب أو الصيدلاني.
        </div>
        """, unsafe_allow_html=True)

# ─── Footer ─────────────────────────────────────────────────────
st.markdown("""
<div style="
    text-align:center; padding:2rem;
    border-top:1px solid rgba(255,255,255,0.04);
    color:#1e293b; font-size:0.75rem;
    margin-top:2rem;
">
    PharmaScan AI · EfficientNetB3 · 150 Egyptian Drug Packages
    · Built with Streamlit & PyTorch
</div>
""", unsafe_allow_html=True)

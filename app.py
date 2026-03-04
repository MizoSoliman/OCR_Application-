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
import gdown

# ─── Google Drive File IDs ───────────────────────────────────────
MODEL_ID   = "1sO9XEf3tKedjOZBVSlfUSnWfMwVyhWxa"
CLASSES_ID = ""
CSV_ID     = ""

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

# ─── CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Remove all Streamlit spacing ── */
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"]             { display: none !important; }
[data-testid="stToolbar"]            { display: none !important; }
[data-testid="stDecoration"]         { display: none !important; }
[data-testid="stMainBlockContainer"] { padding: 0 !important; }
[data-testid="stVerticalBlock"]      { gap: 0 !important; }
[data-testid="stVerticalBlock"] > *  { margin-top: 0 !important; margin-bottom: 0 !important; }
.block-container                     { padding: 0 !important; max-width: 100% !important; }
.element-container                   { margin: 0 !important; }

/* ── Base ── */
.stApp { background: #0a0f1e; font-family: 'DM Sans', sans-serif; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b35 50%, #0a1628 100%);
    border-bottom: 1px solid rgba(56,189,248,0.15);
    padding: 0.8rem 3rem 0.7rem;
    text-align: center;
}
.hero { margin-bottom: 0.6rem; }
.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.2);
    color: #38bdf8; font-size: 0.68rem; font-weight: 600;
    letter-spacing: 1.5px; text-transform: uppercase;
    padding: 3px 10px; border-radius: 20px; margin-bottom: 0.3rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem; color: #f0f9ff;
    margin: 0 0 0.2rem; letter-spacing: -0.5px;
}
.hero-title span { color: #38bdf8; }
.hero-sub { color: #64748b; font-size: 0.88rem; font-weight: 300; }

/* ── Columns ── */
[data-testid="stColumns"] { gap: 0 !important; padding: 0 !important; }
[data-testid="stColumns"] > div:first-child {
    border-right: 1.5px solid rgba(56,189,248,0.25) !important;
    padding: 1.5rem 2rem !important;
}
[data-testid="stColumns"] > div:last-child {
    padding: 1.5rem 2rem !important;
}

/* ── Panel label ── */
.panel-label {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 2px;
    text-transform: uppercase; color: #38bdf8; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 8px;
}
.panel-label::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(to right, rgba(56,189,248,0.3), transparent);
}

/* ── Upload ── */
[data-testid="stFileUploadDropzone"] {
    background: rgba(56,189,248,0.03) !important;
    border: 2px dashed rgba(56,189,248,0.2) !important;
    border-radius: 16px !important; padding: 2rem 1rem !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    background: rgba(56,189,248,0.06) !important;
    border-color: rgba(56,189,248,0.5) !important;
}
[data-testid="stFileUploadDropzone"] p { color: #475569 !important; }

/* ── Result ── */
.result-name {
    font-family: 'DM Serif Display', serif; font-size: 1.8rem;
    color: #f0f9ff; margin: 0 0 0.3rem; line-height: 1.2;
}
.result-generic { color: #38bdf8; font-size: 0.85rem; margin-bottom: 1rem; }
.conf-label { display: flex; justify-content: space-between; font-size: 0.78rem; color: #64748b; margin-bottom: 6px; }
.conf-track { background: rgba(255,255,255,0.05); border-radius: 99px; height: 6px; overflow: hidden; margin-bottom: 1.2rem; }
.conf-fill  { height: 100%; border-radius: 99px; }
.conf-high  { background: linear-gradient(to right, #10b981, #34d399); }
.conf-mid   { background: linear-gradient(to right, #f59e0b, #fbbf24); }
.conf-low   { background: linear-gradient(to right, #ef4444, #f87171); }

/* ── Info cards ── */
.info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; margin-top: 1rem; }
.info-card { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; padding: 1rem; }
.info-card.full { grid-column: 1 / -1; }
.info-card.warn { border-color: rgba(245,158,11,0.2); background: rgba(245,158,11,0.03); }
.info-card-icon { font-size: 1.1rem; margin-bottom: 0.3rem; }
.info-card-title { font-size: 0.68rem; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; color: #475569; margin-bottom: 0.4rem; }
.info-card.warn .info-card-title { color: #f59e0b; }
.info-card-body { font-size: 0.85rem; color: #94a3b8; line-height: 1.6; }

/* ── Source badge ── */
.source-badge { display: inline-flex; align-items: center; gap: 5px; font-size: 0.68rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; padding: 3px 10px; border-radius: 20px; margin-top: 0.8rem; }
.source-local { background: rgba(16,185,129,0.1); color: #10b981; border: 1px solid rgba(16,185,129,0.2); }
.source-fda   { background: rgba(56,189,248,0.1); color: #38bdf8; border: 1px solid rgba(56,189,248,0.2); }
.source-na    { background: rgba(100,116,139,0.1); color: #64748b; border: 1px solid rgba(100,116,139,0.2); }

/* ── Empty state ── */
.empty-state { text-align: center; padding: 3rem 2rem; }
.empty-state-icon { font-size: 3rem; margin-bottom: 0.8rem; opacity: 0.3; }
.empty-state-text { color: #334155; font-size: 0.9rem; }

/* ── Disclaimer ── */
.disclaimer {
    background: rgba(245,158,11,0.05); border: 1px solid rgba(245,158,11,0.15);
    border-radius: 10px; padding: 0.7rem 1rem; margin-top: 0.8rem;
    font-size: 0.78rem; color: #92400e; line-height: 1.5;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0f1e; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 99px; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ──────────────────────────────────────────────────
IMG_SIZE     = 300
MODEL_PATH   = "best_model.pth"
CLASSES_PATH = "class_names.json"
CSV_PATH     = "egyptian_drugs_database.csv"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Load Resources ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    with open(CLASSES_PATH, encoding="utf-8") as f:
        class_names = json.load(f)
    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=len(class_names))
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
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

def predict(img, model, class_names, top_k=1):
    tensor = infer_tf(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)
    top_p, top_i = torch.topk(probs, min(top_k, len(class_names)))
    return [
        {"name": class_names[i], "conf": float(p)}
        for i, p in zip(top_i.squeeze(0).cpu().flatten(), top_p.squeeze(0).cpu().flatten())
    ]

# ─── Drug Info ──────────────────────────────────────────────────
def clean_name(raw):
    no_num = re.sub(r"\b\d+(\.\d+)?\s*(mg|ml|mcg|g|iu|%|units?)?\b", "", raw, flags=re.IGNORECASE)
    no_num = re.sub(r"[_\-]", " ", no_num).strip()
    first  = raw.split()[0]
    return list(dict.fromkeys([c.strip() for c in [raw, no_num, first] if len(c.strip()) > 2]))

def get_drug_info(drug_name, df):
    empty = {"indications": "غير متوفر", "dosage": "غير متوفر", "warnings": "غير متوفر", "generic": "", "source": "na"}
    if df is not None:
        for name in clean_name(drug_name):
            nl = name.lower().strip()
            match = df[df["name_lower"] == nl]
            if match.empty:
                match = df[df["name_lower"].str.contains(nl.split()[0], na=False)]
            if not match.empty:
                row = match.iloc[0]
                return {"indications": row.get("indications","غير متوفر"), "dosage": row.get("dosage","غير متوفر"),
                        "warnings": row.get("warnings","غير متوفر"), "generic": row.get("generic_name",""), "source": "local"}
    url = "https://api.fda.gov/drug/label.json"
    for name in clean_name(drug_name):
        for field in ["openfda.brand_name", "openfda.generic_name"]:
            try:
                r = requests.get(url, params={"search": f'{field}:"{name}"', "limit": 1}, timeout=8)
                if r.status_code == 200:
                    data = r.json().get("results", [])
                    if data:
                        lbl = data[0]
                        return {"indications": " ".join(lbl.get("indications_and_usage",["N/A"]))[:500],
                                "dosage": " ".join(lbl.get("dosage_and_administration",["N/A"]))[:500],
                                "warnings": " ".join(lbl.get("warnings",["N/A"]))[:300],
                                "generic": ", ".join(lbl.get("openfda",{}).get("generic_name",[])),
                                "source": "fda"}
            except: pass
    return empty

def conf_class(c):
    return "conf-high" if c >= 0.70 else "conf-mid" if c >= 0.40 else "conf-low"

def source_badge(src):
    if src == "local": return '<span class="source-badge source-local">📁 Local DB</span>'
    if src == "fda":   return '<span class="source-badge source-fda">🌐 OpenFDA</span>'
    return '<span class="source-badge source-na">⚠️ Not Found</span>'

# ════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">💊 AI-Powered · EfficientNetB3 · 150 Drugs</div>
    <div class="hero-title">Pharma<span>Scan</span> AI</div>
    <div class="hero-sub">صوّر علبة الدواء — النموذج يتعرف عليه فوراً ويعرض الاستخدام والجرعة</div>
</div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────
try:
    model, CLASS_NAMES = load_model()
    df_drugs = load_drug_db()
except Exception as e:
    st.error(f"❌ فشل تحميل النموذج: {e}")
    st.stop()

# ── Columns ───────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="panel-label">📷 &nbsp; الصورة المدخلة</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_container_width=True, caption="", output_format="JPEG")
        st.markdown(f"""
        <div style="display:flex;gap:1rem;margin-top:0.5rem;font-size:0.75rem;color:#475569;">
            <span>📐 {img.width}×{img.height}</span>
            <span>📁 {uploaded.name}</span>
            <span>💾 {uploaded.size/1024:.1f} KB</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📸</div>
            <div class="empty-state-text">ارفع صورة علبة الدواء هنا<br>
            <span style="font-size:0.8rem;color:#1e293b;">JPG · PNG · WEBP</span></div>
        </div>""", unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="panel-label">🔬 &nbsp; نتائج التحليل</div>', unsafe_allow_html=True)

    if not uploaded:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">💊</div>
            <div class="empty-state-text">النتائج ستظهر هنا<br>
            <span style="font-size:0.8rem;color:#1e293b;">بعد رفع صورة الدواء</span></div>
        </div>""", unsafe_allow_html=True)
    else:
        preds = predict(img, model, CLASS_NAMES, top_k=1)
        best  = preds[0]
        info  = get_drug_info(best["name"], df_drugs)
        cf    = best["conf"]
        cc    = conf_class(cf)
        generic_txt = f'<div class="result-generic">{info["generic"]}</div>' if info["generic"] else ""

        st.markdown(f"""
        <div class="result-name">{best["name"]}</div>
        {generic_txt}
        <div class="conf-label">
            <span>Confidence</span>
            <span style="color:#f0f9ff;font-weight:600;">{cf*100:.1f}%</span>
        </div>
        <div class="conf-track">
            <div class="conf-fill {cc}" style="width:{cf*100:.1f}%"></div>
        </div>
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
        <div class="disclaimer">
            ⚕️ <strong>تنبيه طبي:</strong> هذا التطبيق للأغراض التعليمية فقط.
            لا تعتمد عليه كبديل عن استشارة الطبيب أو الصيدلاني.
        </div>
        """, unsafe_allow_html=True)

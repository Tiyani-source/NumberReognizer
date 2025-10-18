import os, io, math, json
import numpy as np
from typing import Dict, List, Tuple

import streamlit as st
import streamlit.components.v1 as components
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Quiet benign PyTorch class path warnings on some runtimes
import warnings
warnings.filterwarnings("ignore", message="Tried to instantiate class '__path__._path'")
warnings.filterwarnings("ignore", message="Examining the path of torch.classes")

# Optional OpenCV & SciPy (used for deskew/centering)
import cv2
try:
    from scipy.ndimage import center_of_mass, rotate
except Exception:
    from scipy.ndimage.measurements import center_of_mass  # type: ignore
    from scipy.ndimage.interpolation import rotate        # type: ignore

# ----------------------------- Page Config -----------------------------------
st.set_page_config(page_title="CalmDigit — Mood Visualizer", page_icon="static/favicon.png", layout="wide")

# ----------------------------- Styles ----------------------------------------
PALETTE = {
    "bg": "#0b1020",          # deep navy canvas
    "card": "#101735",        # card base
    "accent1": "#7c3aed",     # violet
    "accent2": "#06b6d4",     # cyan
    "accent3": "#60a5fa",     # sky
    "text": "#e6ebff",       # near-white
    "muted": "#9fb0d9"        # muted text
}

HERO_GRAD = ["#7c3aed", "#06b6d4", "#60a5fa"]

def gradient_css(colors: List[str]) -> str:
    gradient = ", ".join(colors)
    return f"""
    <style>
      :root {{
        --bg: {PALETTE['bg']};
        --card: {PALETTE['card']};
        --accent1: {PALETTE['accent1']};
        --accent2: {PALETTE['accent2']};
        --accent3: {PALETTE['accent3']};
        --text: {PALETTE['text']};
        --muted: {PALETTE['muted']};
      }}
      html, body, .stApp {{ background-color: var(--bg) !important; color: var(--text); }}
      /* Hero */
      .hero {{
        background: linear-gradient(120deg, {gradient});
        background-size: 280% 280%;
        animation: shift 18s ease-in-out infinite;
        border-radius: 22px; padding: 18px 22px 14px 22px;
        box-shadow: 0 14px 28px rgba(0,0,0,0.30);
        color: #0b1220;
      }}
      @keyframes shift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
      }}
      
      .badge {{ display:inline-block; padding: 6px 12px; border-radius: 9999px; background: rgba(255,255,255,0.25); color:#0b1220; font-weight: 800; }}
      .subtitle {{ color:#0b1220; opacity:0.9; }}

      /* Buttons — flat base, white on hover */
      div.stButton > button {{
        border-radius:14px; padding:0.7rem 1.1rem; font-weight:800;
        border:2px solid transparent; color:#0b1220;
        background: linear-gradient(135deg, var(--accent2), var(--accent3));
        transition: transform .08s ease, background .15s ease, color .15s ease, border-color .15s ease;
        box-shadow: none; /* remove glow */
      }}
      div.stButton > button:hover {{
        background:#ffffff; color:#0b1220; border-color: var(--accent2);
        transform: translateY(-1px);
      }}
      div.stButton > button:active {{ transform: translateY(0); }}

      /* Result card — remove glow, add subtle border */
      .result-card, .card {{
        border-radius: 18px; padding: 18px; color: var(--text);
        background: var(--card);
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: none; /* no outer glow */
      }}
      .result-card .title {{ font-size: 0.8rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }}
      /* Digit tile — remove glow, keep crisp */
      .big-digit {{
        display:inline-flex; align-items:center; justify-content:center;
        width: 132px; height: 132px; border-radius: 24px;
        background: linear-gradient(135deg, rgba(124,58,237,0.75), rgba(6,182,212,0.75));
        color: var(--text); font-weight:900; font-size:68px;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: none; /* remove inner glow */
        margin: 8px 0 10px 0;
      }}
      .muted {{ color: var(--muted); }}
      .hint {{ font-size: 0.98rem; color: var(--text); opacity: 0.88; }}

      /* Layout helpers */
      .container {{ width: 100%; max-width: 1600px; margin: 0 auto; padding: 0 32px; }}
      .section {{ margin: 18px auto 14px auto; }}
      .kicker {{ font-size: 0.9rem; letter-spacing: .02em; color: var(--muted); }}
      .center {{ text-align:center; }}

      /* Keypad grid — remove glow and style like buttons */
      .keypad {{ display:grid; grid-template-columns: repeat(10, minmax(40px,1fr)); gap:12px; }}

      /* Hero — brand styling */
      .logo-tile {{
        width: 96px; height: 96px; border-radius: 28px;
        display:flex; align-items:center; justify-content:center;
        background: linear-gradient(135deg, var(--accent1), var(--accent3));
        border: 1px solid rgba(255,255,255,0.10);
      }}
      .logo-tile span {{ font-size: 54px; }}
      .brand-title {{
        font-weight: 900; font-size: 2.2rem; line-height:1;
        background: linear-gradient(135deg, #ffffff, #c7d2fe);
        -webkit-background-clip: text; background-clip: text; color: transparent;
      }}
      .brand-sub {{ font-size: 1.05rem; font-weight:700; color: rgba(11,18,32,0.95); }}

      
      /* Breathing box */
      .breath-box {{ width: 96px; height: 96px; margin: 8px auto; border-radius: 14px; border: 2px solid rgba(96,165,250,0.5); animation: breathe 16s ease-in-out infinite; }}
      @keyframes breathe {{ 0% {{ transform: scale(0.9); }} 25% {{ transform: scale(1.05); }} 50% {{ transform: scale(0.9); }} 75% {{ transform: scale(1.05); }} 100% {{ transform: scale(0.9); }} }}

      /* Compact hero chips and tighter hero copy */
      .chip {{ display:inline-block; padding:6px 10px; border-radius:9999px; background: rgba(255,255,255,0.22); color:#0b1220; font-weight:700; font-size:0.85rem; margin-right:8px; }}
      .hero-title {{ font-size:1.6rem; font-weight:900; margin:6px 0 2px 0; }}
      .hero-sub {{ font-size:1rem; font-weight:600; opacity:0.9; }}

      /* Responsive */
      @media (max-width: 760px) {{
        .two-col {{ display:block !important; }}
      }}
    </style>
    """

st.markdown(gradient_css(HERO_GRAD), unsafe_allow_html=True)
st.markdown(
    """
<div class="hero-bleed">
  <div class="hero">
    <div style="display:flex; align-items:center; gap:18px;">
      <div class="logo-tile"><span>✨</span></div>
      <div>
        <div class="brand-title">CalmDigit</div>
        <div class="brand-sub">A 10-second mood check-in</div>
      </div>
    </div>
    <div style="margin-top:12px;">
      <span class="chip">Reflect</span>
      <span class="chip">Draw</span>
      <span class="chip">Breathe</span>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

# ----------------------------- Model -----------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

@st.cache_resource(show_spinner=False)
def load_model(weights_path: str = "mnist_cnn.pt"):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CNN().to(device)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights '{weights_path}' not found. Place your trained file next to app.py.")
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device

# ----------------------------- Memory (NN override) ---------------------------
FEEDBACK_PATH = "feedback_mem.npz"  # stores vecs (N x 784) + labels (N,)

def _load_memory():
    if os.path.exists(FEEDBACK_PATH):
        d = np.load(FEEDBACK_PATH)
        return d["vecs"], d["labels"]
    return np.empty((0,784), np.float32), np.empty((0,), np.int64)

def _save_memory(vecs, labels):
    np.savez(FEEDBACK_PATH, vecs=vecs.astype(np.float32), labels=labels.astype(np.int64))

def add_feedback(vec784: np.ndarray, label: int):
    V,L = _load_memory(); V = np.vstack([V, vec784.reshape(1,-1)]); L = np.concatenate([L, [label]])
    _save_memory(V,L)

def adjust_with_memory(vec784: np.ndarray, pred: int, threshold: float = 0.93):
    V,L = _load_memory()
    if V.shape[0] == 0: return pred, False
    sims = (V @ vec784) / ((np.linalg.norm(V,axis=1)+1e-8) * (np.linalg.norm(vec784)+1e-8))
    i = int(np.argmax(sims))
    if float(sims[i]) >= threshold:
        return int(L[i]), True
    return pred, False

# ----------------------------- Preprocess ------------------------------------

def _center_image(img28: np.ndarray) -> np.ndarray:
    cy, cx = center_of_mass(img28 > 0)
    if np.isnan(cx) or np.isnan(cy):
        return img28
    M = np.float32([[1,0,round(14-cx)],[0,1,round(14-cy)]])
    return cv2.warpAffine(img28, M, (28,28), flags=cv2.INTER_NEAREST, borderValue=0)

def preprocess_from_canvas(pil: Image.Image, keypad_mode: bool=True) -> torch.Tensor:
    img = pil.convert("L"); np_img = np.array(img)
    ys, xs = np.where(np_img < 220)
    if len(xs)==0: return transforms.ToTensor()(Image.new("L", (28,28), 0)).unsqueeze(0)
    x1,x2 = xs.min(), xs.max(); y1,y2 = ys.min(), ys.max()
    crop = np_img[y1:y2+1, x1:x2+1]
    side = max(crop.shape); sq = np.full((side,side), 255, np.uint8)
    h,w = crop.shape; sq[(side-h)//2:(side-h)//2+h, (side-w)//2:(side-w)//2+w] = crop
    resized = cv2.resize(sq, (28,28), interpolation=cv2.INTER_AREA)
    if resized.mean() > 127: resized = 255 - resized
    if keypad_mode:
        resized = cv2.dilate(resized, np.ones((2,2), np.uint8), 1)
        m = cv2.moments(resized)
        if abs(m["mu02"]) > 1e-3:
            skew = m["mu11"]/m["mu02"]; M = np.float32([[1, -0.3*skew, 0],[0,1,0]])
            resized = cv2.warpAffine(resized, M, (28,28), flags=cv2.INTER_LINEAR, borderValue=0)
    centered = _center_image(resized)
    to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    return to_tensor(Image.fromarray(centered)).unsqueeze(0)

def tensor_to_vec784(t: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return t.detach().cpu().numpy().reshape(-1).astype(np.float32)

# ----------------------------- Mood Map --------------------------------------
# ----------------------------- Mood Map --------------------------------------
MOOD_GRADS = {
    "low":    [PALETTE["accent3"], PALETTE["accent2"]],
    "balanced":[PALETTE["accent2"], PALETTE["accent1"]],
    "high":   [PALETTE["accent1"], PALETTE["accent3"]],
}

MOODS: Dict[str, Dict] = {
    "low": {
        "range": range(0,4),
        "label": "Low Energy",
        "palette": MOOD_GRADS["low"],
        "desc": "Feeling slowed down or heavy. Small, gentle actions help: sip water, step outside, or try a 5‑minute stretch.",
        "quotes": ["Be kind to yourself today.", "Rest is productive.", "Tiny steps are still steps."],
    },
    "balanced": {
        "range": range(4,7),
        "label": "Steady / Balanced",
        "palette": MOOD_GRADS["balanced"],
        "desc": "Grounded and steady. Try a focused task or connect with someone you care about.",
        "quotes": ["Consistency beats intensity.", "Steady is strong.", "One good choice at a time."],
    },
    "high": {
        "range": range(7,10),
        "label": "High Energy",
        "palette": MOOD_GRADS["high"],
        "desc": "There’s momentum here. Channel it into meaningful progress—batch quick wins or start a task you’ve postponed.",
        "quotes": ["Ride the wave.", "Use the spark to light the path.", "Momentum loves direction."],
    },
}

def mood_from_digit(d:int):
    for k,cfg in MOODS.items():
        if d in cfg["range"]: return k,cfg
    return "balanced", MOODS["balanced"]

# ----------------------------- Load Model ------------------------------------
try:
    model, device = load_model("mnist_cnn.pt")
except Exception as e:
    st.error(str(e))
    st.stop()

# --- Helpers to reset state and canvas ---
if "canvas_nonce" not in st.session_state:
    st.session_state["canvas_nonce"] = 0

def _reset_result_state():
    for k in [
        "has_result", "pred_digit", "pred_probs", "last_vec", "adjusted_flag"
    ]:
        st.session_state.pop(k, None)

# ----------------------------- Canvas ----------------------------------------
st.markdown('<div class="container section">', unsafe_allow_html=True)
st.subheader("1) Write below on a scale of 0 - 9 how is your energy?")

cols = st.columns([1.1, 1])
with cols[0]:
    # Big, clean canvas; use a dynamic key so we can truly clear it
    canvas_key = f"canvas_main_{st.session_state['canvas_nonce']}"
    canvas = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=22,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=380,
        width=680,
        drawing_mode="freedraw",
        display_toolbar=False,
        key=canvas_key,
    )
    # Button row centered under canvas
    bcol1, bcol2 = st.columns([1,1])
    with bcol1: clear = st.button("Clear Canvas")
    with bcol2: interpret = st.button("Interpret Digit → Mood")

with cols[1]:
    st.markdown(
        """
<div class="card">
  <h4>Reflect and write slow (time to slow down from the hustle)</h4>
  <ul>
    <li>Drawing slows the pace, anchors attention in the body, and makes a private feeling visible.</li>
  </ul>
  <h4>🧭 Guided tips</h4>
  <div class="kicker">Keep it simple and centered.</div>
  <ul>
    <li>Use one bold stroke; avoid the edges.</li>
    <li>Draw slowly. If it feels off, press <em>Clear Canvas</em> and redraw.</li>
    <li>After the result appears, tap the number you meant if we misread it.</li>
  </ul>
  <div class="kicker">Why a number? Naming energy with a digit reduces overthinking and gently anchors attention.</div>
</div>
""",
        unsafe_allow_html=True,
    )

# Explain what Clear Canvas does
st.caption("**Clear Canvas** wipes the drawing area only. It does not submit or change the mood result.")
st.markdown('</div>', unsafe_allow_html=True)

if clear:
    # Truly clear: bump the nonce so the canvas gets a fresh key
    st.session_state["canvas_nonce"] += 1
    _reset_result_state()
    st.rerun()

# ----------------------------- Predict ---------------------------------------
pred_digit, pred_probs = None, None
if interpret:
    if canvas.image_data is None:
        st.info("Please draw a digit first.")
    else:
        nd = (canvas.image_data[..., :3] * 255).astype(np.uint8)
        pil = Image.fromarray(nd)
        x = preprocess_from_canvas(pil, keypad_mode=True).to(device)
        st.session_state["last_vec"] = tensor_to_vec784(x)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).cpu().numpy().flatten()
        pred = int(np.argmax(probs))
        pred, adjusted = adjust_with_memory(st.session_state["last_vec"], pred, threshold=0.93)
        st.session_state.update({
            "has_result": True, "pred_digit": int(pred), "pred_probs": probs.tolist(), "adjusted_flag": adjusted
        })

# Recover last on rerun
if st.session_state.get("has_result") and pred_digit is None:
    pred_digit = int(st.session_state["pred_digit"]) ; pred_probs = st.session_state["pred_probs"]

# ----------------------------- Output ----------------------------------------
st.markdown('<div id="result-anchor" class="container section">', unsafe_allow_html=True)
st.subheader("2) Your mood interpretation")

if not st.session_state.get("has_result"):
    st.info("Draw a digit and click **Interpret Digit → Mood** to see your mood card.")
else:
    d = int(st.session_state["pred_digit"]) ; k,cfg = mood_from_digit(d)
    if st.session_state.get("adjusted_flag"): st.caption("Prediction adjusted using your past feedback ✧")

    if st.session_state.pop("just_corrected", False):
        st.success("Thanks for the feedback! I’ll remember similar drawings next time.")

    st.markdown(gradient_css(cfg["palette"]), unsafe_allow_html=True)
    st.markdown(f"""
<div class="result-card">
  <div class="title">Predicted digit</div>
  <div class="big-digit">{d}</div>
  <div class="title">Mood</div>
  <div style="font-size:1.25rem; font-weight:900;">{cfg['label']}</div>
  <div class="hint" style="margin-top:6px;">{cfg['desc']}</div>
  <div class="kicker" style="margin-top:6px;">“{np.random.choice(cfg['quotes'])}”</div>
</div>
""", unsafe_allow_html=True)

    # Smooth-scroll the page so the result is in view only when new result or after correction
    if st.session_state.get("scroll_to_result", True):
        components.html(
            """
            <script>
            window.addEventListener('load', function(){
              const el = window.parent.document.querySelector('#result-anchor');
              if (el && typeof el.scrollIntoView === 'function') {
                el.scrollIntoView({ behavior: 'smooth', block: 'start' });
              } else {
                window.parent.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
              }
            });
            </script>
            """,
            height=0,
        )
        st.session_state["scroll_to_result"] = False

    st.write("")
    if st.button("Draw & submit a new number"):
        st.session_state["canvas_nonce"] += 1
        _reset_result_state()
        st.rerun()

    # Feedback keypad
    st.caption("If it misread your digit, a quick correction below helps it recognize similar drawings next time.")
    st.markdown("**Was this correct?** If not, tap the digit you meant:")
    clicked = None
    kb_cols = st.columns(10)
    for i in range(10):
        with kb_cols[i]:
            st.markdown('<div class="kbtn">', unsafe_allow_html=True)
            if st.button(str(i), key=f"kb_{i}"):
                clicked = i
            st.markdown('</div>', unsafe_allow_html=True)
    if clicked is not None and st.session_state.get("last_vec") is not None:
        # Save correction to memory and replace the visible result
        add_feedback(st.session_state["last_vec"], int(clicked))
        st.session_state["pred_digit"] = int(clicked)
        st.session_state["has_result"] = True
        st.session_state["adjusted_flag"] = True  # show the adjusted note
        st.session_state["just_corrected"] = True  # one-shot toast
        st.session_state["scroll_to_result"] = True
        st.rerun()

    # Calm & Reflect
    with st.expander("Calm & reflect (1 minute)"):
        st.markdown("""
Focus on slow box-breathing: inhale ◼︎ hold ◼︎ exhale ◼︎ hold. Follow the square below.
<div class="breath-box"></div>
**Count**: 4 in → 4 hold → 4 out → 4 hold. Repeat twice.
""", unsafe_allow_html=True)

    # Why drawing? Purpose statement
    with st.container():
        st.markdown("""
<div class="glass">
  <div style="font-weight:700; margin-bottom:6px;">Why draw a number?</div>
  Drawing slows the pace, anchors attention in the body, and makes a private feeling visible.
  Choosing a single digit helps you name your energy without overthinking — a tiny act of self‑check‑in.
</div>
""", unsafe_allow_html=True)

    # Gentle nudge actions
    st.markdown("**Small next steps (pick one):**")
    steps = [
        "Sip water and stretch your shoulders",
        "Step outside for 2 minutes",
        "Write one sentence about how you feel",
        "Send a kind message to yourself or a friend",
    ]
    st.write("• " + "\n• ".join(steps))

st.markdown('</div>', unsafe_allow_html=True)

st.write("")
st.markdown("---")
st.markdown(
    "**Gentle note:** This app offers general well‑being suggestions and is not a medical tool. If you’re struggling or in crisis, please reach out to a qualified professional or local support services.")

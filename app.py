# app.py
import io
import os
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


import warnings
warnings.filterwarnings("ignore", message="Tried to instantiate class '__path__._path'")


def pick_device():
    
    if torch.backends.mps.is_available() and os.environ.get("FORCE_CPU", "0") != "1":
        return torch.device("mps")
    if torch.cuda.is_available() and os.environ.get("FORCE_CPU", "0") != "1":
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = pick_device()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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


@st.cache_resource
def load_model(weights_path: str):
    model = CNN().to(DEVICE)
    state = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


MNIST_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),   
    transforms.Resize((28, 28)),                   
    transforms.ToTensor(),                         
])

# --- Enhanced preprocessing: contrast/sharpness/binarization/optional denoise ---
def prepare_tensor_with_enhancement(
    pil_image: Image.Image,
    device,
    contrast: float = 2.0,
    sharpness: float = 1.5,
    use_binarize: bool = True,
    adaptive_ratio: float = 0.9,
    use_median: bool = False,
):
    """
    Enhance and clean handwriting before feeding to the model.
    Steps:
      - Grayscale
      - Optional median filter (denoise)
      - Contrast & sharpness boost
      - Optional adaptive binarization (pure black/white)
      - Resize + tensor like MNIST
    """
    # 1) Grayscale early
    img = pil_image.convert("L")

    # 2) Optional small denoise
    if use_median:
        img = img.filter(ImageFilter.MedianFilter(size=3))

    # 3) Contrast and sharpness
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)

    # 4) Optional adaptive binarization (convert to pure black/white)
    if use_binarize:
        img = ImageOps.autocontrast(img)
        arr = np.array(img)
        thr = float(arr.mean()) * float(adaptive_ratio)
        bw = np.where(arr > thr, 255, 0).astype(np.uint8)
        img = Image.fromarray(bw)

    # 5) Resize + ToTensor (MNIST pipeline)
    tensor = MNIST_TRANSFORM(img).unsqueeze(0)  # [1,1,28,28]
    return tensor.to(device), img

def prepare_tensor(pil_image: Image.Image) -> torch.Tensor:
    """
    Convert uploaded PIL image to a 1x1x28x28 tensor suitable for the model.
    Handles white-on-black vs black-on-white by auto-inverting if needed.
    """
    img = pil_image.convert("L")  # grayscale early
    # Heuristic: if background is light and digit is dark, keep; else invert
    if np.array(img).mean() > 127:         # background likely light
        pass
    else:
        img = ImageOps.invert(img)         # make digit light on dark (MNIST-like)

    tensor = MNIST_TRANSFORM(img).unsqueeze(0)  # shape [1,1,28,28]
    return tensor.to(DEVICE)

# ---------------------------
# Inference
# ---------------------------
def predict(model: nn.Module, tensor: torch.Tensor):
    with torch.no_grad():
        logits = model(tensor)                     # [1,10]
        probs = torch.softmax(logits, dim=1)[0]    # [10]
        pred = int(torch.argmax(probs).item())
        topk = torch.topk(probs, k=3)
        top3 = [(int(idx), float(p)) for p, idx in zip(topk.values, topk.indices)]
    return pred, probs.cpu().numpy(), top3

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="MNIST Digit Recognizer", page_icon="🔢", layout="centered")
st.title("🔢 MNIST Digit Recognizer")
st.caption(f"Device: **{DEVICE}**")

with st.sidebar:
    st.header("Model")
    weights_path = st.text_input(
        "Path to model weights (.pt)", 
        value="mnist_cnn.pt",
        help="Place your trained weights file next to app.py or provide an absolute path."
    )
    st.info("Tip: if running on CPU only, set env `FORCE_CPU=1` to bypass MPS/CUDA.")

    st.header("Preprocessing")
    pp_enabled = st.checkbox("Enable image enhancement", value=True)
    contrast = st.slider("Contrast", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    sharpness = st.slider("Sharpness", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
    use_binarize = st.checkbox("Binarize (black/white)", value=True)
    adaptive_ratio = st.slider("Adaptive threshold ratio", min_value=0.5, max_value=1.2, value=0.9, step=0.05)
    use_median = st.checkbox("Denoise (median filter)", value=False)
    show_processed = st.checkbox("Show processed 28×28", value=True)

uploaded = st.file_uploader(
    "Upload a digit image (PNG/JPG). A rough phone scribble works too.",
    type=["png", "jpg", "jpeg"]
)

if weights_path and not os.path.exists(weights_path):
    st.warning(f"Model weights not found at `{weights_path}`. "
               "Save your trained weights as mnist_cnn.pt (see instructions below).")

model = None
if weights_path and os.path.exists(weights_path):
    try:
        model = load_model(weights_path)
    except Exception as e:
        st.error(f"Failed to load model from `{weights_path}`.\n{e}")

col1, col2 = st.columns(2)

with col1:
    if uploaded is not None:
        pil = Image.open(io.BytesIO(uploaded.read()))
        # Streamlit version compatibility: older versions (<1.20) don't support use_container_width
        try:
            st.image(pil, caption="Original Upload", use_container_width=True)
        except TypeError:
            st.image(pil, caption="Original Upload")

        if model is not None:
            if pp_enabled:
                tensor, processed_img = prepare_tensor_with_enhancement(
                    pil,
                    device=DEVICE,
                    contrast=contrast,
                    sharpness=sharpness,
                    use_binarize=use_binarize,
                    adaptive_ratio=adaptive_ratio,
                    use_median=use_median,
                )
            else:
                tensor = prepare_tensor(pil)
                processed_img = None

            pred, probs, top3 = predict(model, tensor)
            st.success(f"Prediction: **{pred}**")
            st.write("Top-3:")
            for idx, p in top3:
                st.write(f"- {idx}: {p*100:.2f}%")

            # Optionally show what the model actually saw (processed 28×28)
            if show_processed:
                if processed_img is None:
                    # derive 28×28 from tensor
                    view = tensor.squeeze(0).squeeze(0).detach().cpu().numpy()
                    st.image(view, caption="Processed (28×28)", clamp=True)
                else:
                    st.image(processed_img, caption="Processed (after enhancement)")
    else:
        st.info("Upload an image to get a prediction.")

with col2:
    st.markdown("#### How to draw a clear digit")
    st.markdown(
        "- Use **white background** and **dark digit** if possible.\n"
        "- Keep the digit **centered** and relatively **thick**.\n"
        "- Avoid noisy backgrounds; crop to the digit if needed."
    )

st.markdown("---")
st.subheader("About")
st.markdown(
    "This app uses a small CNN trained on **MNIST** (28×28 grayscale) to classify digits 0–9. "
    "Uploads are resized and converted to MNIST format before prediction."
)
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from keras.models import load_model
from keras.utils import custom_object_scope
from huggingface_hub import hf_hub_download

# KONFIGURASI DASAR
st.set_page_config(
    page_title="Web Pra-Skrining Penyakit Mata",
    page_icon="👁️",
    layout="wide"
)

HF_REPO_ID = "malaaaa1/eye-model"
HF_MODEL_FILENAME = "ConvNeXtTiny_Dengan_segmentasi_ORSASM.keras"

MODEL_PATH = Path(
    hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILENAME
    )
)

CLASSES = ["Normal", "Cataract", "Pterygium", "Jaundice"]
IMG_SIZE = (224, 224)

MODEL_ACCURACY = 0.9521

# CSS TAMPILAN WEB
# ============================================================
# CSS TAMPILAN WEB
# ============================================================

st.markdown("""
<style>
    /* Hilangkan nuansa default Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: transparent;
        height: 0rem;
    }

    .stApp {
        background: linear-gradient(180deg, #edf5ff 0%, #f7fbff 100%);
    }

    .block-container {
        max-width: 1420px;
        padding-top: 3.5rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        padding-bottom: 3rem;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 34px;
    }

    .brand-icon {
        width: 28px;
        height: 28px;
        border-radius: 999px;
        background: #dbeafe;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 15px;
    }

    .brand-text {
        font-size: 18px;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.2px;
    }

    .main-title {
        font-size: 30px;
        line-height: 1.1;
        font-weight: 850;
        color: #0f172a;
        margin-bottom: 8px;
        letter-spacing: -0.7px;
    }

    .sub-title {
        color: #5b6f8c;
        font-size: 15px;
        line-height: 1.6;
        margin-bottom: 22px;
        max-width: 980px;
    }

    .section-space {
        height: 38px;
    }

    .card-title {
        font-size: 18px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 16px;
        letter-spacing: -0.2px;
    }

    .small-card-title {
        font-size: 16px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 14px;
        letter-spacing: -0.2px;
    }

    /* Card Streamlit */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid #d7e5f5 !important;
        border-radius: 20px !important;
        padding: 16px !important;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.07);
    }

    /* File uploader */
    div[data-testid="stFileUploader"] {
        background: #f7fbff;
        border: 1.5px dashed #b7d2f8;
        border-radius: 18px;
        padding: 10px;
    }

    div[data-testid="stFileUploader"] section {
        background: #eef4fb;
        border-radius: 14px;
        border: none;
    }

    /* Button */
    div[data-testid="stButton"] > button {
        width: 100%;
        border: none;
        border-radius: 999px;
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        color: white;
        font-weight: 800;
        height: 46px;
        box-shadow: 0 10px 20px rgba(37, 99, 235, 0.22);
    }

    div[data-testid="stButton"] > button:hover {
        background: linear-gradient(90deg, #1d4ed8, #1e40af);
        color: white;
        border: none;
    }

    .pred-badge {
        background: #dbeafe;
        color: #2563eb;
        border-radius: 999px;
        padding: 12px 18px;
        font-size: 20px;
        font-weight: 850;
        text-align: center;
        margin-bottom: 18px;
    }

    .metric-label {
        color: #64748b;
        font-size: 13px;
        font-weight: 750;
        margin-bottom: 4px;
    }

    .metric-value {
        color: #0f172a;
        font-size: 17px;
        font-weight: 850;
        margin-bottom: 6px;
    }

    .explain-box {
        font-size: 15px;
        line-height: 1.7;
        color: #334155;
    }

    .warning-box {
        background: #fff7ed;
        color: #9a3412;
        padding: 14px 18px;
        border-radius: 16px;
        border: 1px solid #fed7aa;
        margin-top: 14px;
        font-size: 14px;
        line-height: 1.6;
    }

    /* Responsif HP */
    @media (max-width: 900px) {
        .block-container {
            padding-top: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .main-title {
            font-size: 26px;
        }

        .brand-row {
            margin-bottom: 22px;
        }
    }
</style>
""", unsafe_allow_html=True)


# LOAD MODEL
@keras.saving.register_keras_serializable(package="Custom", name="preprocess_input")
def preprocess_input(x):
    return tf.keras.applications.convnext.preprocess_input(x)


@keras.saving.register_keras_serializable(package="Custom", name="function")
def function(x):
    return tf.keras.applications.convnext.preprocess_input(x)


@st.cache_resource
def load_best_model():
    try:
        keras.config.enable_unsafe_deserialization()
    except Exception:
        pass

    keras.saving.get_custom_objects()["preprocess_input"] = preprocess_input
    keras.saving.get_custom_objects()["function"] = function
    tf.keras.utils.get_custom_objects()["preprocess_input"] = preprocess_input
    tf.keras.utils.get_custom_objects()["function"] = function

    custom_objects = {
        "preprocess_input": preprocess_input,
        "function": function
    }

    with custom_object_scope(custom_objects):
        model = load_model(
            MODEL_PATH,
            compile=False,
            safe_mode=False,
            custom_objects=custom_objects
        )

    return model

# PREPROCESSING: ROI, CLAHE, ORSASM
def center_crop_roi(img_rgb, crop_ratio=0.82):
    h, w = img_rgb.shape[:2]

    new_h = int(h * crop_ratio)
    new_w = int(w * crop_ratio)

    y1 = max(0, (h - new_h) // 2)
    x1 = max(0, (w - new_w) // 2)

    y2 = y1 + new_h
    x2 = x1 + new_w

    roi = img_rgb[y1:y2, x1:x2]

    return roi


def apply_clahe_rgb(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))

    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    return img_clahe


def make_orsasm_mask(
    img_rgb,
    center_x=0.50,
    center_y=0.52,
    width=0.98,
    upper_height=0.50,
    lower_height=0.42,
    upper_power=0.75,
    lower_power=0.90,
    n_points=300
):
    h, w = img_rgb.shape[:2]

    cx = w * center_x
    cy = h * center_y

    half_width = (w * width) / 2.0
    top_h = h * upper_height
    bottom_h = h * lower_height

    t = np.linspace(-1, 1, n_points)
    x = cx + t * half_width

    base = 1 - np.abs(t) ** 2
    base = np.clip(base, 0, 1)

    y_top = cy - top_h * (base ** upper_power)
    y_bottom = cy + bottom_h * (base ** lower_power)

    top_points = np.stack([x, y_top], axis=1)
    bottom_points = np.stack([x[::-1], y_bottom[::-1]], axis=1)

    polygon = np.vstack([top_points, bottom_points])
    polygon = np.round(polygon).astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = (mask > 10).astype(np.uint8) * 255

    return mask


def apply_orsasm_segmentation(img_rgb):
    mask = make_orsasm_mask(img_rgb)
    segmented = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

    return segmented


def preprocessing_pipeline(img_rgb):
    """
    Pipeline inference web:
    ROI extraction → CLAHE → ORSASM → Resize.
    Normalisasi/model-specific preprocess dilakukan oleh layer model.
    """
    roi = center_crop_roi(img_rgb, crop_ratio=0.82)
    clahe = apply_clahe_rgb(roi)
    orsasm = apply_orsasm_segmentation(clahe)
    resized = cv2.resize(orsasm, IMG_SIZE, interpolation=cv2.INTER_AREA)

    return resized, orsasm

# PREDIKSI
def predict_image(model, processed_img):
    img_array = processed_img.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model(img_array, training=False).numpy()

    pred_index = int(np.argmax(preds[0]))
    pred_class = CLASSES[pred_index]
    confidence = float(np.max(preds[0]))

    return preds, pred_index, pred_class, confidence, img_array

# GRAD-CAM UNTUK MOBILE NET V3 NESTED MODEL
def find_backbone_model(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            for sub_layer in layer.layers:
                try:
                    if len(sub_layer.output.shape) == 4:
                        return layer
                except:
                    pass

    raise ValueError("Backbone model tidak ditemukan.")


def find_last_4d_layer(backbone):
    for layer in reversed(backbone.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer
        except:
            pass

    raise ValueError("Layer feature map 4D tidak ditemukan.")


def call_layer_safely(layer, x, training=False):
    try:
        return layer(x, training=training)
    except TypeError:
        return layer(x)


def make_gradcam_heatmap_nested(model, img_array, pred_index=None):
    backbone = find_backbone_model(model)
    target_layer = find_last_4d_layer(backbone)

    backbone_index = model.layers.index(backbone)

    pre_backbone_layers = model.layers[1:backbone_index]
    post_backbone_layers = model.layers[backbone_index + 1:]

    backbone_grad_model = tf.keras.models.Model(
        inputs=backbone.input,
        outputs=[target_layer.output, backbone.output]
    )

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        x = img_tensor

        for layer in pre_backbone_layers:
            x = call_layer_safely(layer, x, training=False)

        conv_outputs, x = backbone_grad_model(x, training=False)

        for layer in post_backbone_layers:
            x = call_layer_safely(layer, x, training=False)

        predictions = x

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        raise ValueError("Gradien tidak terbaca.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)

    if float(max_value) > 0:
        heatmap = heatmap / max_value

    return heatmap.numpy()


def overlay_gradcam(img_rgb, heatmap, alpha=0.45):
    heatmap = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_color, alpha, 0)

    return overlay, heatmap_color

# PENJELASAN HASIL
def get_explanation(pred_class, confidence):
    conf_percent = confidence * 100

    base = (
        f"Sistem memprediksi citra mata sebagai kelas {pred_class} "
        f"dengan confidence score sebesar {conf_percent:.2f}%. "
    )

    if pred_class == "Normal":
        detail = "Hasil ini menunjukkan bahwa citra tidak terdeteksi sebagai kelas penyakit pada model."
    elif pred_class == "Cataract":
        detail = "Hasil ini menunjukkan bahwa model mendeteksi karakteristik visual yang mengarah pada kelas Cataract."
    elif pred_class == "Pterygium":
        detail = "Hasil ini menunjukkan bahwa model mendeteksi karakteristik visual yang mengarah pada kelas Pterygium."
    elif pred_class == "Jaundice":
        detail = "Hasil ini menunjukkan bahwa model mendeteksi karakteristik visual yang mengarah pada kelas Jaundice."
    else:
        detail = ""

    closing = (
        " Visualisasi segmentasi ORSASM menampilkan area mata yang digunakan sebagai input model, "
        "sedangkan Grad-CAM menunjukkan area yang paling berpengaruh terhadap keputusan prediksi. "
        "Hasil ini digunakan sebagai pra-skrining dan bukan diagnosis medis final."
    )

    return base + detail + closing

def sync_uploaded_file_state(uploaded_file):
    if uploaded_file is None:
        st.session_state.pop("result", None)
        st.session_state.pop("uploaded_signature", None)
        return

    current_signature = (uploaded_file.name, uploaded_file.size)

    previous_signature = st.session_state.get("uploaded_signature")

    if previous_signature != current_signature:
        st.session_state["uploaded_signature"] = current_signature
        st.session_state.pop("result", None)
        
def resize_for_ui(img_rgb, max_width=420, max_height=320):
    h, w = img_rgb.shape[:2]

    scale = min(max_width / w, max_height / h, 1.0)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return resized

# LOAD MODEL UTAMA
if not MODEL_PATH.exists():
    st.error(f"Model tidak ditemukan: {MODEL_PATH}")
    st.stop()

model = load_best_model()

# HEADER
st.markdown(
    """
    <div class="brand-row">
        <div class="brand-icon">👁️</div>
        <div class="brand-text">Web Pra-Skrining Penyakit Mata</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">Unggah Citra Mata</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="sub-title">
        Pilih citra anterior segment dalam format JPG, JPEG, atau PNG.
        Sistem akan menampilkan pratinjau citra sebelum proses deteksi dijalankan.
    </div>
    """,
    unsafe_allow_html=True
)

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    with st.container(border=True):
        st.markdown('<div class="card-title">Input Citra</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Pilih File Citra",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="uploader"
        )

        sync_uploaded_file_state(uploaded_file)

        detect_button = st.button("Deteksi Sekarang", use_container_width=True)

with right_col:
    with st.container(border=True):
        st.markdown('<div class="card-title">Pratinjau Citra</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            pil_image = Image.open(uploaded_file).convert("RGB")
            preview_img = np.array(pil_image)

            preview_display = resize_for_ui(
                preview_img,
                max_width=430,
                max_height=330
            )

            st.image(preview_display)
        else:
            st.info("Pratinjau citra akan muncul setelah file diunggah.")

# PROSES DETEKSI
if detect_button:
    if uploaded_file is None:
        st.warning("Silakan unggah citra mata terlebih dahulu.")
        st.stop()

    with st.spinner("Sistem sedang memproses citra dan menjalankan prediksi..."):
        pil_image = Image.open(uploaded_file).convert("RGB")
        input_img = np.array(pil_image)

        processed_img, orsasm_img = preprocessing_pipeline(input_img)

        preds, pred_index, pred_class, confidence, img_array = predict_image(
            model,
            processed_img
        )

        heatmap = make_gradcam_heatmap_nested(
            model=model,
            img_array=img_array,
            pred_index=pred_index
        )

        gradcam_overlay, heatmap_color = overlay_gradcam(
            processed_img,
            heatmap,
            alpha=0.45
        )

        st.session_state["result"] = {
            "input_img": input_img,
            "orsasm_img": processed_img,
            "heatmap_color": heatmap_color,
            "gradcam_overlay": gradcam_overlay,
            "pred_class": pred_class,
            "confidence": confidence
        }

# ============================================================
# HALAMAN HASIL
# ============================================================

if "result" in st.session_state:
    result = st.session_state["result"]

    st.markdown('<div class="section-space"></div>', unsafe_allow_html=True)

    st.markdown('<div class="main-title">Hasil Deteksi</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sub-title">
            Halaman ini menampilkan citra input, visualisasi segmentasi ORSASM,
            visualisasi Grad-CAM, hasil prediksi, confidence score, dan akurasi model.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns([1, 1, 1, 0.95], gap="medium")

    with col1:
        with st.container(border=True):
            st.markdown('<div class="small-card-title">Citra Input</div>', unsafe_allow_html=True)

            img_show = resize_for_ui(
                result["input_img"],
                max_width=280,
                max_height=260
            )

            st.image(img_show)

    with col2:
        with st.container(border=True):
            st.markdown('<div class="small-card-title">Segmentasi ORSASM</div>', unsafe_allow_html=True)

            seg_show = resize_for_ui(
                result["orsasm_img"],
                max_width=280,
                max_height=260
            )

            st.image(seg_show)

    with col3:
        with st.container(border=True):
            st.markdown('<div class="small-card-title">Grad-CAM</div>', unsafe_allow_html=True)

            grad_show = resize_for_ui(
                result["gradcam_overlay"],
                max_width=280,
                max_height=260
            )

            st.image(grad_show)

    with col4:
        with st.container(border=True):
            st.markdown('<div class="small-card-title">Hasil Prediksi</div>', unsafe_allow_html=True)

            st.markdown(
                f'<div class="pred-badge">{result["pred_class"]}</div>',
                unsafe_allow_html=True
            )

            conf_percent = result["confidence"] * 100
            acc_percent = MODEL_ACCURACY * 100

            st.markdown('<div class="metric-label">Confidence Score</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{conf_percent:.2f}%</div>', unsafe_allow_html=True)
            st.progress(int(conf_percent))

            st.markdown('<div class="metric-label">Akurasi Model</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{acc_percent:.2f}%</div>', unsafe_allow_html=True)
            st.progress(int(acc_percent))

    st.markdown('<div style="height: 18px;"></div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="small-card-title">Penjelasan Hasil</div>', unsafe_allow_html=True)
        explanation = get_explanation(result["pred_class"], result["confidence"])
        st.markdown(f'<div class="explain-box">{explanation}</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="warning-box">
            Catatan: Sistem ini digunakan sebagai alat pra-skrining berbasis citra
            dan tidak menggantikan pemeriksaan atau diagnosis dokter.
        </div>
        """,
        unsafe_allow_html=True
    )

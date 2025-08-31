# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "garbage_classifier.h5"
CLASS_INDEX_FILE = "class_indices.txt"
IMG_SIZE = (128, 128)  # must match training

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    # class_indices.txt contains lines like: "plastic,0"
    pairs = []
    with open(CLASS_INDEX_FILE, "r") as f:
        for line in f:
            name, idx = line.strip().split(",")
            pairs.append((name, int(idx)))
    # sort by index so order is correct for softmax outputs
    pairs.sort(key=lambda x: x[1])
    return [p[0] for p in pairs]

def preprocess(image: Image.Image):
    # ensure RGB
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

st.set_page_config(page_title="Garbage Classification", page_icon="♻️")
st.title("♻️ Garbage Classification (plastic / organic / metal)")
st.caption("Upload an image; the model will classify it.")

model = load_model()
class_names = load_class_names()

uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])
if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_column_width=True)

    x = preprocess(img)
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_name = class_names[pred_idx]
    confidence = float(np.max(probs))

    st.subheader("Prediction")
    st.write(f"**Class:** {pred_name}")
    st.write(f"**Confidence:** {confidence:.2f}")

    # show per-class probabilities
    st.subheader("Per-class probabilities")
    for name, p in zip(class_names, probs):
        st.write(f"- {name}: {p:.2f}")

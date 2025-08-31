# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "garbage_classifier.h5"
CLASS_INDEX_FILE = "class_indices.txt"
IMG_SIZE = (128, 128)  # Must match your model input

# Cache model loading for performance
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# Cache class names
@st.cache_data
def load_class_names():
    pairs = []
    with open(CLASS_INDEX_FILE, "r") as f:
        for line in f:
            name, idx = line.strip().split(",")
            pairs.append((name, int(idx)))
    pairs.sort(key=lambda x: x[1])
    return [p[0] for p in pairs]

# Preprocess uploaded image
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# Streamlit app layout
st.set_page_config(page_title="Garbage Classification", page_icon="♻️")
st.title("♻️ Garbage Classification")
st.caption("Upload an image; the model will classify it as plastic, organic, or metal.")

model = load_model()
class_names = load_class_names()

uploaded_file = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    x = preprocess_image(img)
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_name = class_names[pred_idx]
    confidence = float(np.max(probs))

    st.subheader("Prediction")
    st.write(f"**Class:** {pred_name}")
    st.write(f"**Confidence:** {confidence:.2f}")

    st.subheader("Per-class probabilities")
    for name, p in zip(class_names, probs):
        st.write(f"- {name}: {p:.2f}")

        
    
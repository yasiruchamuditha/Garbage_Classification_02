def load_model(path="garbage_classifier.h5"):
    import tensorflow as tf
    return tf.keras.models.load_model(path)

def preprocess_image_pil(pil_img, model):
    import numpy as np
    # Get expected (H, W, C) from model input
    _, h, w, c = model.input_shape
    pil_img = pil_img.resize((w, h))  # resize to model input size
    arr = np.array(pil_img).astype("float32") / 255.0
    return arr.reshape((1, h, w, c))  # add batch dimension

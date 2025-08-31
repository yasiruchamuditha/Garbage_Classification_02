def load_model(path="garbage_classifier.h5"):
    import tensorflow as tf
    return tf.keras.models.load_model(path)

def preprocess_image_pil(pil_img):
    import numpy as np
    pil_img = pil_img.resize((224, 224))  # adjust to your model input size
    arr = np.array(pil_img).astype("float32") / 255.0
    return arr.reshape((1, 224, 224, 3))  # add batch dimension

def load_model(path="garbage_classifier.h5"):
    # import inside function
    import tensorflow as tf
    model = tf.keras.models.load_model(path)
    return model

def preprocess_image_pil(pil_img):
    from PIL import Image
    import numpy as np
    # do preprocessing...
    return np.array(pil_img).astype("float32") / 255.0

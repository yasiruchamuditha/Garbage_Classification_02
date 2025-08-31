import streamlit as st
from app_logic import load_model, preprocess_image_pil
from PIL import Image

def main():
    st.title("Garbage Classifier")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Load model
        model = load_model("garbage_classifier.h5")

        # Preprocess image according to model input size
        input_data = preprocess_image_pil(image, model)

        # Run prediction
        prediction = model.predict(input_data)

    # Display result in requested format
    pred_list = prediction.tolist()
    formatted = {0: {i: v for i, v in enumerate(pred_list[0])}}
    st.write("Prediction (raw):")
    st.json(formatted)

if __name__ == "__main__":
    main()

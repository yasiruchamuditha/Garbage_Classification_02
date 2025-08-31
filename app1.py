import streamlit as st
from app_logic import load_model, preprocess_image_pil
from PIL import Image

def main():
    st.title("Garbage Classifier")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        model = load_model("garbage_classifier.h5")
        input_data = preprocess_image_pil(image)

        prediction = model.predict(input_data)
        st.write("Prediction:", prediction.tolist())  # convert to list for display

if __name__ == "__main__":
    main()

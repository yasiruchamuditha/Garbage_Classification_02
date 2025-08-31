import streamlit as st
from app_logic.model import preprocess_image_pil  # safe: no TF import at top-level

def main():
    st.title("Garbage Classifier")
    # UI that calls load_model only when needed (e.g., on button click)
    if st.button("Load model"):
        model = __import__("app_logic.model").model.load_model("garbage_classifier.h5")
        st.write("Loaded!")

if __name__ == "__main__":
    main()

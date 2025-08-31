import streamlit as st


def main():
    st.title("Garbage Classifier")
    # UI that calls load_model only when needed (e.g., on button click)
    if st.button("Load model"):
        from app_logic.model import load_model
        load_model("garbage_classifier.h5")
        st.write("Loaded!")


if __name__ == "__main__":
    main()

# python -m streamlit run fruits\app.py

from __future__ import annotations


from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from fruits.config.config import Config
from fruits.inference.predictor import FruitPredictor, PredictorError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"
SUPPORTED_FILE_TYPES = ["jpg", "jpeg", "png"]


@st.cache_resource
def load_predictor(checkpoint_path: str, class_source_dir: str) -> FruitPredictor:
    """Create and cache the predictor so model loading happens once."""
    class_names = FruitPredictor.derive_class_names_from_training_dir(class_source_dir)

    return FruitPredictor(
        checkpoint_path=checkpoint_path,
        class_names=class_names,
        device=Config.DEVICE,
    )


def open_uploaded_image(uploaded_file) -> Image.Image:
    """Open uploaded image safely and validate that it is decodable."""
    try:
        image = Image.open(uploaded_file)
        image.load()
        return image
    except (UnidentifiedImageError, OSError) as exc:
        raise PredictorError(
            "Uploaded file is not a valid or decodable image."
        ) from exc


def render_prediction(prediction: dict) -> None:
    st.subheader("Prediction")
    st.write(prediction["class_name"])

    st.subheader("Confidence")
    st.write(f"{prediction['confidence'] * 100:.2f}%")

    st.subheader("Top Predictions")
    for item in prediction["top_predictions"]:
        st.write(f"{item['class_name']}: {item['confidence'] * 100:.2f}%")


def main() -> None:
    st.set_page_config(page_title="Fruit Image Classifier", layout="centered")

    st.title("Fruit Image Classifier")
    st.write("Upload a fruit image and the trained PyTorch model will classify it.")

    checkpoint_path = st.text_input(
        "Checkpoint path",
        value=str(DEFAULT_CHECKPOINT_PATH),
        help="Path to your trained model checkpoint.",
    )

    class_source_dir = st.text_input(
        "Class mapping source directory",
        value=str(Config.TRAIN_PATH),
        help=(
            "Directory containing one subfolder per class, matching the "
            "same ordering used during training."
        ),
    )

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=SUPPORTED_FILE_TYPES,
    )

    if uploaded_file is None:
        st.info("Please upload a JPG or PNG image to continue.")
        return

    extension = Path(uploaded_file.name).suffix.lower().lstrip(".")
    if extension not in SUPPORTED_FILE_TYPES:
        st.error(
            f"Unsupported file type: .{extension}. "
            "Please upload a JPG or PNG image."
        )
        return

    try:
        image = open_uploaded_image(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    except PredictorError as exc:
        st.error(str(exc))
        return

    if st.button("Predict", type="primary"):
        try:
            predictor = load_predictor(checkpoint_path, class_source_dir)
            prediction = predictor.predict(image=image, top_k=3)
            render_prediction(prediction)
        except PredictorError as exc:
            st.error(str(exc))
        except RuntimeError as exc:
            st.error(
                "Model inference failed due to an internal runtime error. "
                f"Details: {exc}"
            )
        except Exception as exc:
            st.error(f"Unexpected error during prediction: {exc}")


if __name__ == "__main__":
    main()

# Fruit Image Classifier

This project trains and runs a convolutional neural network for fruit image classification using PyTorch and Streamlit. The app accepts an uploaded fruit image and returns the predicted class along with confidence scores.

## Features

- Train a CNN model on a fruit dataset organized by class folders
- Evaluate the model during training
- Load a saved checkpoint and run inference from a web interface
- Display the top-$k$ predictions for a single uploaded image

## Project Structure

```text
fruits/
├── app.py
├── checkpoints/
│   └── best_model.pth
├── config/
│   ├── __init__.py
│   └── config.py
├── dataset/
│   ├── __init__.py
│   └── dataset.py
├── engine/
│   ├── __init__.py
│   ├── train.py
│   └── trainer.py
├── inference/
│   ├── __init__.py
│   └── predictor.py
├── load_data/
│   ├── __init__.py
│   └── load_data.py
├── models/
│   ├── __init__.py
│   └── cnn.py
├── notebook_info/
│   ├── __init__.py
│   └── nnote.py
├── README.md
└── requirements.txt
```

## Requirements

- Python 3.10+
- PyTorch
- torchvision
- Streamlit
- Pillow

Install the dependencies from the project directory:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

The training pipeline expects the dataset to be organized in the following structure:

```text
<dataset_root>/
├── Training/
│   ├── Apple/
│   ├── Banana/
│   └── Orange/
└── Test/
    ├── Apple/
    ├── Banana/
    └── Orange/
```

By default, the project points to a local dataset path defined in [config/config.py](config/config.py). If your data lives elsewhere, update the `Config.DATASET_PATH`, `Config.TRAIN_PATH`, and `Config.TEST_PATH` values accordingly.

## Training the Model

From the repository root, run:

```bash
python -m fruits.engine.train
```

This script will:

1. Load the training and validation datasets
2. Build a CNN model
3. Train it for the configured number of epochs
4. Save the best checkpoint to the `checkpoints` directory

## Running the Web App

Start the Streamlit app from the repository root:

```bash
python -m streamlit run fruits/app.py
```

Then open the local URL shown by Streamlit in your browser.

### App workflow

- Upload a JPG, JPEG, or PNG image
- Click the Predict button
- Review the predicted fruit class, confidence score, and top predictions

## Inference Notes

The app uses the checkpoint path and class mapping directory from the interface. The default checkpoint path is:

```text
checkpoints/best_model.pth
```

If you train a new model, update the checkpoint path in the app or point it to your new `.pth` file.

## Configuration

Most training settings are defined in [config/config.py](config/config.py), including:

- dataset paths
- batch size
- number of epochs
- learning rate
- device selection (CPU or GPU)

Adjust these values to fit your environment or dataset size.

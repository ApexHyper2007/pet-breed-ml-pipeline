# Oxford-IIIT Pet Breed Classification

A deep learning image classification project that identifies the breed of a cat or dog from an input image.

The project uses the Oxford-IIIT Pet dataset and a pretrained ResNet18 model with transfer learning and fine-tuning.

## Project Overview

The model classifies pet images into 37 different cat and dog breeds.

### Dataset

- Dataset: Oxford-IIIT Pet Dataset
- Number of classes: 37
- Images used: 7,349
- Image size: 224 × 224
- Training images: 5,141
- Validation images: 1,096
- Test images: 1,112

## Machine Learning Pipeline

The project follows this pipeline:

1. Dataset verification
2. Image preprocessing
3. Image resizing to 224 × 224
4. Train/validation/test splitting
5. Data augmentation
6. Image normalization
7. Transfer learning using ResNet18
8. Fine-tuning
9. Model evaluation
10. Breed prediction on new images

## Preprocessing

All images were resized to:

```text
224 × 224 pixels
```

Images are converted to tensors and normalized using ImageNet mean and standard deviation:

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

## Data Augmentation

During training, the following augmentations are applied:

- Random horizontal flipping
- Random rotation up to 15 degrees
- Random brightness adjustment
- Random contrast adjustment

Augmentation is performed during training rather than creating additional image files.

## Model

The project uses **ResNet18** with pretrained ImageNet weights.

Instead of training the entire network from scratch:

- Most pretrained layers are frozen.
- The final ResNet block (`layer4`) is fine-tuned.
- The final fully connected layer is replaced with a classifier for 37 classes.

This approach uses transfer learning to adapt a pretrained image-recognition model to the pet-breed classification task.

## Training

The model was trained using:

- Framework: PyTorch
- Optimizer: Adam
- Learning rate: 0.0001
- Batch size: 32
- Loss function: CrossEntropyLoss
- GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU

The best model checkpoint is saved as:

```text
best_pet_resnet18.pth
```

The best validation accuracy achieved during training was:

```text
88.59%
```

## Test Results

The best checkpoint was evaluated on the held-out test set.

```text
Test images:       1112
Correct:           977
Test accuracy:     87.86%
```

Overall evaluation:

| Metric | Score |
|---|---:|
| Accuracy | 0.88 |
| Macro Precision | 0.89 |
| Macro Recall | 0.88 |
| Macro F1 | 0.88 |
| Weighted Precision | 0.89 |
| Weighted Recall | 0.88 |
| Weighted F1 | 0.88 |

## Project Structure

```text
oxford-iiit-pet/
│
├── annotations/
├── images/
├── processed_images/
├── old_scripts/
│
├── dataset.py
├── model.py
├── train.py
├── test.py
├── evaluate.py
├── predict.py
│
├── train.txt
├── validation.txt
├── test.txt
│
└── best_pet_resnet18.pth
```

### Main Files

| File | Purpose |
|---|---|
| `dataset.py` | Loads images, applies augmentation and normalization |
| `model.py` | Defines the ResNet18 model |
| `train.py` | Trains and validates the model |
| `test.py` | Evaluates accuracy on the test set |
| `evaluate.py` | Generates detailed classification metrics |
| `predict.py` | Predicts the breed of a new image |

The `old_scripts/` folder contains scripts used during earlier stages of dataset checking and experimentation.

## Running the Project

The project uses a Python virtual environment.

### Train

```powershell
& ".\.venv\Scripts\python.exe" train.py
```

### Test

```powershell
& ".\.venv\Scripts\python.exe" test.py
```

### Detailed Evaluation

```powershell
& ".\.venv\Scripts\python.exe" evaluate.py
```

### Predict a New Image

```powershell
& ".\.venv\Scripts\python.exe" predict.py image.jpg
```

The prediction script returns the top 5 predicted breeds with their model confidence scores.

## Example

For an input image, the model produces results such as:

```text
Top 5 predictions:

1. Saint Bernard: 96.22%
2. American Bulldog: 0.81%
3. Shiba Inu: 0.37%
4. Japanese Chin: 0.34%
5. Beagle: 0.31%
```

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pillow
- Scikit-learn
- ResNet18
- CUDA

## Future Improvements

Possible improvements include:

- Training for more epochs
- Experimenting with different learning rates
- Fine-tuning additional ResNet layers
- Comparing ResNet18 with other architectures
- Improving classification of visually similar breeds
- Creating a simple web interface for image prediction

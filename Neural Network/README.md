
📊 Deep Learning Projects: Regression & Image Classification
This directory contains two deep learning projects:

Boston Housing Price Prediction using regularized feedforward neural networks.

Flower Classification using the Oxford Flowers 102 dataset and a pre-trained VGG16 model with transfer learning.



### 🔧 Project 1: Boston Housing Price Prediction
🏡 Overview
The goal of this project is to predict housing prices based on 13 numerical features using three different neural network models, each with distinct regularization techniques and optimizers.

🧠 Models Used
Model 1: L2 regularization with Adam optimizer

Model 2: L1 regularization with SGD optimizer

Model 3: L1 regularization with RMSprop optimizer

All models have the same architecture:

python
Copy
Edit
Sequential([
    Dense(64, activation='relu', input_shape=(13,), kernel_regularizer=...),
    Dense(32, activation='relu', kernel_regularizer=...),
    Dense(1, activation='linear')
])
⚙️ Training Details
Loss Function: Mean Squared Error

Metrics: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and MSE

Epochs: 100

Batch Size: 32

Validation Split: 20%

📈 Visualization
Model 3 was trained and visualized to compare training and validation loss across epochs:

python
Copy
Edit
plt.plot(train_loss_3, label='Training Loss')
plt.plot(valid_loss_3, label='Validation Loss')
plt.legend()
plt.show()
✅ Objective
Evaluate and compare the impact of:

Regularization types (L1 vs L2)

Optimizers (Adam vs SGD vs RMSprop)
on model performance and overfitting behavior.

### 🌸 Project 2: Flower Classification with VGG16
🖼️ Dataset
Oxford Flowers 102

102 flower categories

Varying lighting, scale, and pose

Split:

Training: 1,020 images

Validation: 1,020 images

Testing: 6,149 images

📦 Loaded using tfds.load('oxford_flowers102')

🧹 Preprocessing
python
Copy
Edit
def preprocess(img, label):
    img = tf.image.resize(img, (224, 224))
    img = tf.cast(img, tf.float32) / 255.0
    img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
    return img, label
🧠 Model Architecture
Base: Pre-trained VGG16 model (frozen base)

Top: Custom classification head

Loss: Sparse Categorical Crossentropy

Optimizer: Adam

Metrics: Accuracy

📊 Training Strategy
Epochs: 40

EarlyStopping: Patience = 3

ReduceLROnPlateau: LR halved if no val_loss improvement for 2 epochs

TensorBoard: For monitoring metrics

✅ Evaluation
plaintext
Copy
Edit
Test Loss     : 0.9004
Test Accuracy : 0.7663
Precision     : 0.7629
Recall        : 0.7923
F1 Score      : 0.7624
🔍 Observations
The model progressively improves over epochs.

Effective generalization with decent precision and recall.



### 3. CIFAR-100 Image Classification with CNNs and Optuna Hyperparameter Tuning

This project implements a Convolutional Neural Network (CNN) on the CIFAR-100 dataset using Keras and hyperparameter tuning via Optuna.

#### 🔹 Dataset

- **Name**: CIFAR-100
- **Description**: 60,000 color images (32x32), 100 classes
- **Split**: Training, Validation, Testing

#### 🔹 Features

- ✅ Data normalization and augmentation via `ImageDataGenerator`
- ✅ CNN with tunable:
  - Layers (1–3)
  - Filters (32–256)
  - Dropout (0.2–0.5)
  - L2 regularization
  - Learning rate (log-uniform)
  - Batch size (16, 32, 64)
- ✅ Optuna for hyperparameter search
- ✅ Early stopping
- ✅ Model evaluation using accuracy, precision, recall, and F1-score

#### 🔹 Tools & Libraries

- `TensorFlow`, `Keras`
- `Optuna` for tuning
- `scikit-learn` for metrics

#### 🔹 Training Highlights

- Trained with Optuna-optimized architecture
- Used 20 epochs and early stopping (patience=3)
- Final results:

```text
Training Accuracy:   96.92%
Validation Accuracy: 30.14%
Test Accuracy:       31.10%

Precision: 0.3128
Recall:    0.2982
F1 Score:  0.2965

# ACL Injury Prediction Project

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Architecture](#model-architecture)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Prediction](#prediction)
9. [User Interface](#user-interface)
10. [Results](#results)
11. [Contributing](#contributing)
12. [License](#license)

## Introduction

This project focuses on predicting the risk of ACL (Anterior Cruciate Ligament) injuries using images of individuals playing Kabaddi. Leveraging the VGG16 convolutional neural network pre-trained on ImageNet, this model identifies risky leg positions that might lead to injuries. The system provides an accessible interface for uploading images and visualizing predictions, aimed at helping coaches and medical professionals make informed decisions.

## Dataset

The dataset consists of two main categories:
- **Injury**: Images labeled as having a risk of injury.
- **No Injury**: Images labeled as having no risk of injury.

Data preprocessing includes resizing images to 224x224 pixels and normalizing pixel values. The dataset is split into training and validation sets with an 80-20 ratio.

## Installation

To run this project, you need to have Python and the following libraries installed:
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Streamlit
- Pillow
- Scikit-learn

You can install the required libraries using the following command:
```bash
pip install tensorflow keras numpy pandas matplotlib streamlit pillow scikit-learn
```

## Usage

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/acl-injury-prediction.git
    cd acl-injury-prediction
    ```

2. **Run the training script (if not using the pre-trained model):**
    ```bash
    python train_model.py
    ```

3. **Run the Streamlit application:**
    ```bash
    streamlit run main_app.py
    ```

## Model Architecture

The model is built using the VGG16 architecture pre-trained on ImageNet. Custom layers are added on top of the VGG16 base model:
- **Global Average Pooling Layer**
- **Dense Layer with 1024 units and ReLU activation**
- **Output Layer with 1 unit and Sigmoid activation**

Some of the early layers of VGG16 are frozen to retain pre-trained weights and fine-tune the rest for our specific task.

## Training

The model is trained with the following configuration:
- **Optimizer**: Adam with a learning rate of 0.0001
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Callbacks**: EarlyStopping with patience set to 3 to prevent overfitting

Training involves two phases:
1. Training with frozen VGG16 layers.
2. Fine-tuning by unfreezing some of the later layers of VGG16.

## Evaluation

The model's performance is evaluated using a confusion matrix and accuracy metrics. Visualization tools like ROC curves and loss/accuracy plots are also utilized.

## Prediction

The prediction script processes each image by resizing, normalizing, and passing it through the trained model. The output is a probability score indicating the risk of injury. Images can be uploaded through the Streamlit interface to get real-time predictions.

## User Interface

The user interface is built using Streamlit, providing an easy-to-use platform for uploading images and displaying predictions. The interface includes:
- Image upload functionality
- Display of the uploaded image
- Display of prediction results with the probability of injury risk

## Results

The model demonstrates high accuracy in distinguishing between injury and no-injury cases, with detailed performance metrics available in the evaluation section. Users can visually assess model predictions alongside actual labels for better interpretability.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests for any enhancements or bug fixes. Ensure that your contributions align with the project's goals and maintain the code quality.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

This README provides an overview and instructions to get started with the ACL Injury Prediction project. For detailed information, please refer to the source code and documentation provided within the repository.

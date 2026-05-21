# Real-Time Emotion Classification using Mini-Xception

This repository contains a PyTorch implementation of the emotion classification pipeline described in the paper "Real-time Convolutional Neural Networks for Emotion and Gender Classification" by Arriaga et al. The project focuses on deploying a highly efficient, real-time vision system capable of face detection and emotion classification on hardware-constrained setups.

## Methodology and Architecture

The core of this project is the **Mini-Xception** network. To achieve real-time performance without sacrificing accuracy, the model architecture relies on several strict design principles outlined in the paper:

* **Removal of Fully Connected Layers:** Traditional CNNs house the vast majority of their parameters in final dense layers. This architecture completely eliminates them to drastically reduce computational overhead.
* **Depth-wise Separable Convolutions:** The convolutional layers separate spatial cross-correlations from channel cross-correlations. This is achieved by applying a spatial depth-wise convolution followed by a 1x1 point-wise convolution, significantly reducing the parameter count.
* **Residual Modules:** The network utilizes residual connections (solving for H(x) = F(x) + x) to modify the desired mapping between layers, allowing for easier feature learning.
* **Global Average Pooling:** Instead of flattening feature maps into dense layers, the network applies Global Average Pooling to reduce each feature map to a single scalar value, forcing the network to extract robust global features.

Through these methods, the model achieves a footprint of roughly 60,000 parameters (an 80x reduction compared to standard CNNs for this task) while maintaining competitive accuracy.

## Project Structure

* `Xception.py`: Contains the PyTorch definitions for the `MiniXception` architecture, including the depth-wise separable convolution blocks and residual modules.
* `train.py`: The training pipeline. It utilizes the ADAM optimizer, dynamic learning rate reduction (`ReduceLROnPlateau`), and data augmentation to train the model on the FER-2013 dataset.
* `real_time.py`: The live inference script. It uses OpenCV's Haar Cascade to detect and crop human faces, which are then passed to the trained PyTorch model for real-time emotion classification.

## Requirements and Installation

Ensure you have Python 3.8+ installed. You will need the following libraries to run the training and inference scripts:

```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install opencv-python pillow numpy
```
*Note: The PyTorch installation command above assumes you are using Windows with an NVIDIA GPU (CUDA 12.1). Adjust the PyTorch installation command based on your specific hardware and OS.*

## Dataset Preparation

This project uses the **FER-2013** dataset, which consists of 48x48 pixel grayscale images of faces. The model classifies faces into 7 emotions: angry, disgust, fear, happy, neutral, sad, and surprise.

Extract the dataset into a `dataset` folder in the root directory. The directory structure must be organized for PyTorch's `ImageFolder` class:

```text
dataset/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    └── ...
```

## Training the Model

To train the model from scratch, run the training script from your terminal. The script will automatically load the dataset, apply data augmentation (flips and rotations), and save the best-performing weights to `mini_xception_best.pth`.

```bash
python train.py
```

## Real Time Inference(Webcam)

Once the model is trained and `mini_xception_best.pth` is generated, you can run the real-time inference script. This script will access your primary webcam, detect your face, draw a bounding box, and display the predicted emotion and confidence percentage on the screen.

```bash
python real_time.py
```
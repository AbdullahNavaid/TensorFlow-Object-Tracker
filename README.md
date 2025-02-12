# TensorFlow Object Tracker

This project demonstrates object tracking using TensorFlow and MobileNetV2 for detecting and tracking objects in real-time. It uses a pre-trained dataset for AI recognition to identify and track objects with high accuracy.

## Project Overview

The main objective of this project is to create an object detection and tracking system that utilizes a pre-trained TensorFlow model (`ssd_mobilenet_v2_coco_2018_03_29`) along with a pre-trained dataset to detect objects and track their movements in a video feed.

## Features

- Object detection using TensorFlow and MobileNetV2.
- Real-time object tracking with bounding boxes and object IDs.
- Utilizes a pre-trained dataset for object recognition (COCO dataset).
- Supports multiple object tracking in a single frame.

## Installation

To run the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/AbdullahNavaid/TensorFlow-Object-Tracker.git
    ```

2. Navigate to the project directory:
    ```bash
    cd TensorFlow-Object-Tracker
    ```

3. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use 'venv\Scripts\activate'
    ```

4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place your input video file in the project directory or specify the path in the script.

2. Run the object detection and tracking script:
    ```bash
    python object_tracker.py
    ```

3. The program will start processing the video and will display real-time object tracking with bounding boxes.

## Pre-Trained Dataset

This project uses a pre-trained dataset (COCO dataset) with a MobileNetV2-based SSD model. The model is trained on a large set of images and can recognize a variety of objects with high accuracy.

- **Model**: `ssd_mobilenet_v2_coco_2018_03_29`
- **Dataset**: COCO (Common Objects in Context), a large-scale object detection, segmentation, and captioning dataset.


## Acknowledgements

- [TensorFlow](https://www.tensorflow.org/) for providing pre-trained models and object detection APIs.
- [MobileNetV2](https://github.com/tensorflow/models/tree/master/research/object_detection) for the object detection model.
- [COCO Dataset](http://cocodataset.org/) for providing a large-scale dataset used for pre-training the object detection model.

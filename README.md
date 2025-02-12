# TensorFlow Object Tracker

This project demonstrates object tracking using TensorFlow and MobileNetV2 for detecting and tracking objects in real-time. It uses a pre-trained dataset for AI recognition to identify and track objects with high accuracy. Additionally, this project utilizes the **IP Webcam** app on your phone to stream video and process it for object detection.

## Project Overview

The main objective of this project is to create an object detection and tracking system that utilizes a pre-trained TensorFlow model (`ssd_mobilenet_v2_coco_2018_03_29`) along with a pre-trained dataset to detect objects and track their movements in a video feed from your phone's camera. By using the IP Webcam app, the video feed is sent over a network, enabling real-time object tracking directly from your mobile device.

## Features

- Object detection using TensorFlow and MobileNetV2.
- Real-time object tracking with bounding boxes and object IDs.
- Utilizes a pre-trained dataset for object recognition (COCO dataset).
- Supports multiple object tracking in a single frame.
- Uses **IP Webcam** app to stream video from your phone to the laptop for object detection.

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

### Streaming Video Using IP Webcam

1. Install the **IP Webcam** app on your phone from the Google Play Store.
2. Open the app and start the server. Make sure your phone and laptop are connected to the same network.
3. Once the server starts, note down the **IPv4 address** provided by the app (e.g., `http://192.168.1.x:8080`).

### Running the Object Tracking Script

1. Update the video source in the `object_tracker.py` script to use the IP Webcam stream. Replace the placeholder URL with your phone’s IPv4 address:
    ```python
    video_url = 'http://192.168.1.x:8080/video'  # Replace with your phone's IPv4 link
    ```

2. Run the object detection and tracking script:
    ```bash
    python object_tracker.py
    ```

3. The program will start processing the video from your phone's camera and display real-time object tracking with bounding boxes.

## Pre-Trained Dataset

This project uses a pre-trained dataset (COCO dataset) with a MobileNetV2-based SSD model. The model is trained on a large set of images and can recognize a variety of objects with high accuracy.

- **Model**: `ssd_mobilenet_v2_coco_2018_03_29`
- **Dataset**: COCO (Common Objects in Context), a large-scale object detection, segmentation, and captioning dataset.


## Acknowledgements

- [TensorFlow](https://www.tensorflow.org/) for providing pre-trained models and object detection APIs.
- [MobileNetV2](https://github.com/tensorflow/models/tree/master/research/object_detection) for the object detection model.
- [COCO Dataset](http://cocodataset.org/) for providing a large-scale dataset used for pre-training the object detection model.
- [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) for streaming video from the phone to the laptop.

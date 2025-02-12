# TensorFlow Object Tracker

This project demonstrates object tracking using TensorFlow and MobileNetV2 for detecting and tracking objects in real-time.

## Project Overview

The main objective of this project is to create an object detection and tracking system that utilizes a pre-trained TensorFlow model (`ssd_mobilenet_v2_coco_2018_03_29`) to detect objects and track their movements in a video feed.

## Features

- Object detection using TensorFlow and MobileNetV2.
- Real-time object tracking with bounding boxes and object IDs.
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

## Contributing

Feel free to fork the repository, submit pull requests, or open issues for improvements or bugs. Contributions are always welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [TensorFlow](https://www.tensorflow.org/) for providing pre-trained models and object detection APIs.
- [MobileNetV2](https://github.com/tensorflow/models/tree/master/research/object_detection) for the object detection model.

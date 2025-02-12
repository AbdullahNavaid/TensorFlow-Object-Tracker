import tensorflow as tf
import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
import logging


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.9
FONT_COLOR = (255, 255, 255)
FONT_THICKNESS = 2
BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2
BAR_HEIGHT = 10
BAR_THICKNESS = -1  # Filled rectangle


class BoundingBoxSmoother:
    def __init__(self, alpha: float = 0.5):
        """
        Initialize the smoother.
        Args:
            alpha (float): Smoothing factor (0 < alpha <= 1). Higher values give more weight to recent frames.
        """
        self.alpha = alpha
        self.smoothed_boxes = None

    def smooth(self, boxes: np.ndarray) -> np.ndarray:
        """
        Smooth bounding box coordinates using exponential moving average.
        Args:
            boxes (np.ndarray): Bounding box coordinates for the current frame.

        Returns:
            np.ndarray: Smoothed bounding box coordinates.
        """
        if self.smoothed_boxes is None:
            self.smoothed_boxes = boxes
        else:
            self.smoothed_boxes = self.alpha * boxes + (1 - self.alpha) * self.smoothed_boxes
        return self.smoothed_boxes


class ObjectDetector:
    def __init__(self, model_path: str, label_map_path: str, confidence_threshold: float = 0.5):
        """
        Initialize the object detector.

        Args:
            model_path (str): Path to the frozen model file.
            label_map_path (str): Path to the label map file.
            confidence_threshold (float): Confidence threshold for detections.
        """
        self.model_path = model_path
        self.label_map_path = label_map_path
        self.confidence_threshold = confidence_threshold
        self.detection_graph = self.load_frozen_model()
        self.label_map = self._load_label_map()
        self.session: Optional[tf.compat.v1.Session] = None
        self.smoother = BoundingBoxSmoother(alpha=0.5)  # Initialize smoother

    def load_frozen_model(self) -> tf.Graph:
        """Load the frozen model from disk."""
        try:
            detection_graph = tf.Graph()
            with detection_graph.as_default():
                graph_def = tf.compat.v1.GraphDef()
                with tf.io.gfile.GFile(self.model_path, 'rb') as fid:
                    serialized_graph = fid.read()
                    graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(graph_def, name='')
            logger.info("Model loaded successfully")
            return detection_graph
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _load_label_map(self) -> Dict[int, str]:
        """Load the label map from disk."""
        label_map = {}
        try:
            with open(self.label_map_path, 'r') as f:
                for line in f:
                    if 'id:' in line:
                        class_id = int(line.strip().split('id:')[-1])
                    if 'name:' in line:
                        class_name = line.strip().split('name:')[-1].strip().strip("'")
                        label_map[class_id] = class_name
            logger.info("Label map loaded successfully")
            return label_map
        except Exception as e:
            logger.error(f"Error loading label map: {str(e)}")
            raise

    def start_session(self):
        """Start TensorFlow session."""
        self.session = tf.compat.v1.Session(graph=self.detection_graph)

    def close_session(self):
        """Close TensorFlow session."""
        if self.session:
            self.session.close()

    def get_tensor_dict(self) -> Dict[str, tf.Tensor]:
        """Get dictionary of input/output tensors."""
        return {
            'image_tensor': self.detection_graph.get_tensor_by_name('image_tensor:0'),
            'detection_boxes': self.detection_graph.get_tensor_by_name('detection_boxes:0'),
            'detection_classes': self.detection_graph.get_tensor_by_name('detection_classes:0'),
            'detection_scores': self.detection_graph.get_tensor_by_name('detection_scores:0'),
            'num_detections': self.detection_graph.get_tensor_by_name('num_detections:0')
        }

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[float], List[int], List[float]]:
        """
        Process a single frame through the object detection model.
        Args:
            frame (np.ndarray): Input frame from the camera.

        Returns:
            Tuple[np.ndarray, List[float], List[int], List[float]]: Tuple containing processed boxes, classes, and scores.
        """
        tensor_dict = self.get_tensor_dict()
        image_np_expanded = np.expand_dims(frame, axis=0)

        output_dict = self.session.run(
            [tensor_dict['detection_boxes'],
             tensor_dict['detection_classes'],
             tensor_dict['detection_scores'],
             tensor_dict['num_detections']],
            feed_dict={tensor_dict['image_tensor']: image_np_expanded}
        )

        # Smooth bounding boxes
        output_dict[0] = self.smoother.smooth(output_dict[0])

        return output_dict


class CameraHandler:
    def __init__(self, camera_url: str, detector: ObjectDetector):
        """Args:
            camera_url (str): URL for the IP camera.
            detector (ObjectDetector): Instance of ObjectDetector.
        """
        self.camera_url = camera_url
        self.detector = detector
        self.cap: Optional[cv2.VideoCapture] = None

    def start_capture(self):
        """Start video capture."""
        self.cap = cv2.VideoCapture(self.camera_url)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera stream")
        logger.info("Camera stream opened successfully")

    def stop_capture(self):
        """Stop video capture."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def interpolate_color(self, score: float) -> Tuple[int, int, int]:
        """
        Interpolate color from red to green based on confidence score.
        Args:
            score (float): Confidence score (0 to 1).

        Returns:
            Tuple[int, int, int]: Tuple of (B, G, R) color values.
        """
        red = int(255 * (1 - score))
        green = int(255 * score)
        blue = 0
        return blue, green, red

    def draw_detections(self, frame: np.ndarray, boxes: np.ndarray, classes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Draw only the most confident bounding box on the frame."""
        frame_height, frame_width = frame.shape[:2]

        # Get index of the highest confidence detection above threshold
        max_score_idx = np.argmax(scores[0])

        if scores[0][max_score_idx] > self.detector.confidence_threshold:
            box = boxes[0][max_score_idx]
            ymin, xmin, ymax, xmax = box
            left = int(xmin * frame_width)
            right = int(xmax * frame_width)
            top = int(ymin * frame_height)
            bottom = int(ymax * frame_height)

            # Draw box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Get class name from label map
            class_id = int(classes[0][max_score_idx])
            class_name = self.detector.label_map.get(class_id, f"Class {class_id}")
            score = scores[0][max_score_idx]

            # Draw label with accuracy
            label = f"{class_name}: {score:.2f}"
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Draw gradient accuracy bar
            bar_length = int((right - left) * score)
            bar_color = self.interpolate_color(score)
            cv2.rectangle(frame, (left, bottom + 5), (left + bar_length, bottom + 15), bar_color, -1)

        return frame

    def run_detection_loop(self):
        """Run the main detection loop."""
        try:
            self.detector.start_session()
            self.start_capture()

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break

                # Run detection
                boxes, classes, scores, num_detections = self.detector.process_frame(frame)

                # Draw detections
                frame = self.draw_detections(frame, boxes, classes, scores)

                # Display frame
                cv2.imshow("Object Detection", frame)

                # Break loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            logger.error(f"Error in detection loop: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.stop_capture()
        self.detector.close_session()
        logger.info("Resources cleaned up")


def main():
    # Configuration
    MODEL_PATH = 'D:/Pycharm Projects/AI Item Detection/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'
    LABEL_MAP_PATH = 'D:/Pycharm Projects/AI Item Detection/ssd_mobilenet_v2_coco_2018_03_29/LabelMapPath/Label_Map_Path.txt'
    CAMERA_URL = "http://192.168.87.25:8080/video"
    CONFIDENCE_THRESHOLD = 0.6  # Increased threshold for stability

    try:
        # Initialize detector and camera handler
        detector = ObjectDetector(MODEL_PATH, LABEL_MAP_PATH, CONFIDENCE_THRESHOLD)
        camera_handler = CameraHandler(CAMERA_URL, detector)

        # Run detection loop
        camera_handler.run_detection_loop()

    except Exception as e:
        logger.error(f"Application error: {str(e)}")


if __name__ == "__main__":
    main()
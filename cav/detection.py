import cv2
import numpy as np
import tensorflow as tf

from time import time
from random import randint

from .objects import BoundingBox


class ObjectDetector:
    """
    Class for object detection using TensorFlow 1 or 2 models.
    """

    def __init__(self, path_to_graph, detection_threshold=0.5):
        self.detection_threshold = detection_threshold
        self.tf_version = int(tf.__version__.split('.')[0])

        if self.tf_version == 1:
            self.detection_graph = tf.Graph()
            with self.detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(path_to_graph, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

                self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

            self.sess = tf.Session(graph=self.detection_graph)
        else:
            self.model = tf.saved_model.load(path_to_graph)


    def _detect_tf1(self, image, timestamp=None, returnBBoxes=True, detectThreshold=None):
        if detectThreshold is None:
            detectThreshold = self.detection_threshold
        if timestamp is None:
            timestamp = time()

        with self.detection_graph.as_default():
            image_expanded = np.expand_dims(image, axis=0)

            (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_expanded})

            classes = np.squeeze(classes).astype(np.int32)
            scores = np.squeeze(scores)

            cond = scores > detectThreshold
            boxes = boxes[0][cond]
            classes = classes[cond]
            scores = scores[cond]

            if returnBBoxes:
                boxes = self.boxes2BoundingBoxes(boxes, image.shape, timestamp)

            return boxes, scores, classes


    def detect(self, image, timestamp=None, returnBBoxes=True, detectThreshold=None):
        if timestamp is None:
            timestamp = time()
        if detectThreshold is None:
            detectThreshold = self.detection_threshold

        if self.tf_version == 1:
            return self._detect_tf1(image, timestamp, returnBBoxes, detectThreshold)
        else:
            image = np.asarray(image)
            input_tensor = tf.convert_to_tensor(image)[tf.newaxis, ...]
            model_fn = self.model.signatures['serving_default']
            output_dict = model_fn(input_tensor)

            num_detections = int(output_dict.pop('num_detections'))
            output_dict = {key: value[0, :num_detections].numpy()
                           for key, value in output_dict.items()}
            output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

            scores = output_dict['detection_scores']
            classes = output_dict['detection_classes']

            # Keep only person (1) and sports ball (37)
            valid_classes = [1, 37]
            cond = (scores > detectThreshold) & np.isin(classes, valid_classes)

            boxes = output_dict['detection_boxes'][cond]
            classes = classes[cond]
            scores = scores[cond]

            if returnBBoxes:
                boxes = self.boxes2BoundingBoxes(boxes, image.shape, timestamp)

            return boxes, scores, classes


    def boxes2BoundingBoxes(self, boxes, imgshape, timestamp=None):
        """Convert normalized TF boxes into project BoundingBox objects."""
        y, x = imgshape[0], imgshape[1]
        bboxes = []
        for box in boxes:
            ymin, xmin, ymax, xmax = box.tolist()
            ymin = int(y * ymin)
            xmin = int(x * xmin)
            ymax = int(y * ymax)
            xmax = int(x * xmax)
            bboxes.append(BoundingBox(xmin, xmax, ymin, ymax, timestamp))
        return bboxes

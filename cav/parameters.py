import cv2
import json
import numpy as np

class Parameters:
    """
    Parameters used for transforming between camera view and Bird’s Eye view.
    This version is simplified for HoopStats, where only pixel-based
    transformations are required (no latitude/longitude mapping).
    """

    def __init__(self):
        # Perspective matrices
        self.unwarp_M = None       # Camera → Bird’s Eye
        self.unwarp_Minv = None    # Bird’s Eye → Camera

        # Optional parameters (for compatibility / extensions)
        self.elevation = None
        self.lanes_mask = None

    # ----------------------------------------------------------------------
    def __generate_unwarp_matrices(self, cameraPoints, birdEyePoints):
        """
        Generates homography matrices for mapping between camera and bird’s-eye view.
        Both input point sets must contain exactly four points.
        """
        if len(cameraPoints) != 4 or len(birdEyePoints) != 4:
            raise ValueError("Both 'cameraPoints' and 'birdEyePoints' must contain exactly 4 points.")

        cameraPoints = np.float32(cameraPoints)
        birdEyePoints = np.float32(birdEyePoints)

        self.unwarp_M = cv2.getPerspectiveTransform(cameraPoints, birdEyePoints)
        self.unwarp_Minv = cv2.getPerspectiveTransform(birdEyePoints, cameraPoints)
        print("✅ Perspective transform matrices generated successfully.")

    # ----------------------------------------------------------------------
    def generateParameters(self, jsonfile):
        """
        Loads camera-to-bird-eye mapping and optional parameters from a JSON file.
        Expected JSON structure:
        {
            "cameraPoints": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
            "birdEyePoints": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
            "lanes_mask": "optional_mask_path",
            "elevation": optional_value
        }
        """
        with open(jsonfile, 'r') as f:
            data = json.load(f)

        if 'cameraPoints' not in data or 'birdEyePoints' not in data:
            raise ValueError("JSON must contain 'cameraPoints' and 'birdEyePoints'.")

        self.__generate_unwarp_matrices(data['cameraPoints'], data['birdEyePoints'])

        # Optional attributes
        if 'elevation' in data:
            self.elevation = data['elevation']

        if 'lanes_mask' in data:
            mask_path = data['lanes_mask']
            try:
                self.lanes_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if self.lanes_mask is not None:
                    print(f"✅ Lanes mask loaded: {mask_path}")
                else:
                    print(f"⚠️ Unable to load lanes mask at {mask_path}")
            except Exception as e:
                print(f"⚠️ Error loading lanes mask: {e}")

        print("ℹ️ Latitude/Longitude mapping skipped (not needed for HoopStats).")

    # ----------------------------------------------------------------------
    def camera2BirdEye(self, x, y):
        """
        Transforms pixel coordinates from the camera view to the Bird’s Eye view.
        """
        if self.unwarp_M is None:
            raise ValueError("Transformation matrix not initialized.")
        src = np.array([[[x, y]]], dtype='float32')
        trans = cv2.perspectiveTransform(src, self.unwarp_M)
        return trans[0][0]

    # ----------------------------------------------------------------------
    def birdEye2Camera(self, x, y):
        """
        Transforms pixel coordinates from the Bird’s Eye view to the camera view.
        """
        if self.unwarp_Minv is None:
            raise ValueError("Inverse transformation matrix not initialized.")
        src = np.array([[[x, y]]], dtype='float32')
        trans = cv2.perspectiveTransform(src, self.unwarp_Minv)
        return trans[0][0]

    # ----------------------------------------------------------------------
    def getZoneID(self, birdEyeX, birdEyeY):
        """
        Returns zone ID based on preloaded lane mask.
        Each pixel intensity represents a unique zone.
        """
        if self.lanes_mask is None:
            return None

        x, y = int(birdEyeX), int(birdEyeY)
        if 0 <= y < self.lanes_mask.shape[0] and 0 <= x < self.lanes_mask.shape[1]:
            return int(self.lanes_mask[y, x])
        return None

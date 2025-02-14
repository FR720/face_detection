import cv2
import numpy as np

class FaceRecognizer:
    def __init__(self):
        """Initialize FaceRecognizer using haar cascade classifier"""
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def detect_faces(self, image):
        """Detect faces in image using haar cascade"""
        # Convert to grayscale for detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect faces
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convert to same format as before
        face_detections = []
        for (x, y, w, h) in faces:
            # Calculate a pseudo-confidence based on face size
            face_size = w * h
            max_size = gray.shape[0] * gray.shape[1]
            confidence = min(face_size / max_size * 2, 1.0)  # Normalize to [0,1]
            
            face_detections.append({
                'bbox': (x, y, x+w, y+h),
                'confidence': confidence
            })
        
        return face_detections
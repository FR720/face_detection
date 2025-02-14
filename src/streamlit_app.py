import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

class StreamlitApp:
    def __init__(self, face_recognizer):
        self.face_recognizer = face_recognizer
        # Initialize session state
        if 'camera_running' not in st.session_state:
            st.session_state.camera_running = False

    def run(self):
        st.title("Face Recognition System")
        
        # Sidebar settings
        st.sidebar.title("Settings")
        self.detection_threshold = st.sidebar.slider(
            "Detection Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key="detection_threshold"
        )
        
        # Display options
        st.sidebar.subheader("Display Options")
        self.show_bbox = st.sidebar.checkbox(
            "Show Bounding Box", 
            value=True,
            key="show_bbox"
        )
        self.show_confidence = st.sidebar.checkbox(
            "Show Confidence", 
            value=True,
            key="show_confidence"
        )
        
        # Input option
        option = st.radio(
            "Choose input:",
            ["Upload Image", "Use Camera"],
            key="input_option"
        )
        
        if option == "Upload Image":
            self.handle_image_upload()
        else:
            self.handle_camera()

    def handle_image_upload(self):
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png'],
            key="image_uploader"
        )
        
        if uploaded_file is not None:
            # Convert uploaded file to image
            image = Image.open(uploaded_file)
            image = np.array(image)
            
            # Process image
            self.process_image(image)

    def handle_camera(self):
        st.write("Camera Controls")
        
        # Camera control buttons
        if not st.session_state.camera_running:
            if st.button("Start Camera", key="start_camera"):
                st.session_state.camera_running = True
                st.rerun()
        else:
            if st.button("Stop Camera", key="stop_camera"):
                st.session_state.camera_running = False
                st.rerun()

        # Camera feed
        if st.session_state.camera_running:
            stframe = st.empty()
            cap = cv2.VideoCapture(0)

            while st.session_state.camera_running:
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process frame
                    processed_frame = self.process_image(frame_rgb, show_result=False)
                    
                    # Display processed frame
                    stframe.image(processed_frame)
                    
                    # Add small delay
                    time.sleep(0.1)

            cap.release()

    def process_image(self, image, show_result=True):
        # Detect faces
        faces = self.face_recognizer.detect_faces(image)
        
        # Draw results
        result_image = image.copy()
        
        for face in faces:
            if self.show_bbox:
                bbox = face['bbox']
                cv2.rectangle(
                    result_image,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (0, 255, 0),
                    2
                )
            
            if self.show_confidence:
                confidence = face['confidence']
                label = f"Confidence: {confidence:.2f}"
                cv2.putText(
                    result_image,
                    label,
                    (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    2
                )
        
        if show_result:
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Original Image")
            
            with col2:
                st.image(result_image, caption="Detected Faces")
            
            # Display detection information
            if faces:
                st.success(f"Found {len(faces)} faces in the image")
                
                # Show detailed information for each face
                for i, face in enumerate(faces, 1):
                    with st.expander(f"Face #{i} Details", key=f"face_details_{i}"):
                        st.write(f"Confidence: {face['confidence']:.2f}")
                        st.write(f"Position: {face['bbox']}")
            else:
                st.warning("No faces detected in the image")
        
        return result_image
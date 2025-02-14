import os
import sys
import streamlit as st

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.streamlit_app import StreamlitApp
from src.face_recognizer import FaceRecognizer

def main():
    try:
        # Initialize face recognizer (no need for model paths now)
        recognizer = FaceRecognizer()
        
        # Create and run app
        app = StreamlitApp(recognizer)
        app.run()
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check if all required packages are properly installed.")

if __name__ == "__main__":
    main()
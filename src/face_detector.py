import os
 

def download_face_detection_model():
    """
    Download face detection model files from Google Drive
    """
    # Create directory for models
    os.makedirs("models/face_detection", exist_ok=True)
    
    # Model files URLs (hosted on Google Drive)
    model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    config_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    
    # Local paths
    model_path = "models/face_detection/res10_300x300_ssd_iter_140000.caffemodel"
    config_path = "models/face_detection/deploy.prototxt"

    print("Downloading face detection model files...")
    
    try:
        # Download the .caffemodel file
        os.system(f'curl -o {model_path} {model_url}')
        
        # Download the .prototxt file
        os.system(f'curl -o {config_path} {config_url}')
        
        print("Download completed successfully!")
        
    except Exception as e:
        print(f"Error downloading files: {str(e)}")
        print("\nAlternative method: Please manually download the following files:")
        print("\n1. res10_300x300_ssd_iter_140000.caffemodel:")
        print("   Save to: models/face_detection/res10_300x300_ssd_iter_140000.caffemodel")
        print("\n2. deploy.prototxt:")
        print("   Save to: models/face_detection/deploy.prototxt")
        print("\nYou can find these files in the OpenCV repository or use the direct links:")
        print("- https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel")
        print("- https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt")

if __name__ == "__main__":
    download_face_detection_model()
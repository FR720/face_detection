import subprocess
import sys
import os

def setup_project():
    """Setup the face recognition project"""
    print("Setting up face recognition project...")
    
    # Create directories
    directories = [
        'src',
        'models/face_detection',
        'models/trained_models'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create requirements.txt
    requirements = """
streamlit>=1.31.0
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=10.0.0
tensorflow>=2.15.0
scikit-learn>=1.3.0
    """.strip()
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    print("Created requirements.txt")
    
    # Upgrade pip
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    print("Upgraded pip")
    
    # Install requirements
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Installed requirements")
    
    # Create __init__.py
    with open('src/__init__.py', 'w') as f:
        pass
    print("Created src/__init__.py")
    
    print("\nSetup completed successfully!")
    print("\nYou can now run the application with:")
    print("streamlit run deploy.py")

if __name__ == "__main__":
    setup_project()
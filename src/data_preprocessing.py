import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

class DataPreprocessor:
    def __init__(self, dataset_path, target_size=(299, 299), batch_size=32):
        self.dataset_path = dataset_path
        self.target_size = target_size
        self.batch_size = batch_size
        
    def get_dataset_info(self):
        """Get dataset information without loading images"""
        person_folders = []
        image_paths = []
        labels = []
        label_dict = {}
        
        print("Analyzing dataset...")
        
        for idx, person_folder in enumerate(sorted(os.listdir(self.dataset_path))):
            if person_folder.startswith('pins_'):
                person_name = person_folder[5:]
                folder_path = os.path.join(self.dataset_path, person_folder)
                
                if os.path.isdir(folder_path):
                    label_dict[idx] = person_name
                    
                    for image_file in os.listdir(folder_path):
                        image_path = os.path.join(folder_path, image_file)
                        image_paths.append(image_path)
                        labels.append(idx)
        
        print(f"Found {len(image_paths)} images in {len(label_dict)} classes")
        return image_paths, labels, label_dict

    def create_data_generators(self):
        """Create data generators for training, validation, and test sets"""
        # Get dataset information
        image_paths, labels, label_dict = self.get_dataset_info()
        
        # Split paths and labels
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            image_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels, test_size=0.2, random_state=42, 
            stratify=train_val_labels
        )
        
        # Create data generators
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Custom generators
        def generate_data(paths, labels, datagen, batch_size):
            num_samples = len(paths)
            while True:
                # Shuffle the data
                indices = np.random.permutation(num_samples)
                
                for start in range(0, num_samples, batch_size):
                    end = min(start + batch_size, num_samples)
                    batch_indices = indices[start:end]
                    
                    batch_paths = [paths[i] for i in batch_indices]
                    batch_labels = [labels[i] for i in batch_indices]
                    
                    # Load and preprocess images
                    batch_images = []
                    for path in batch_paths:
                        try:
                            img = cv2.imread(path)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, self.target_size)
                            batch_images.append(img)
                        except Exception as e:
                            print(f"Error loading {path}: {e}")
                            continue
                    
                    batch_images = np.array(batch_images)
                    batch_labels = np.array(batch_labels)
                    
                    # Apply data augmentation
                    if len(batch_images) > 0:
                        batch_images = next(datagen.flow(batch_images, batch_size=len(batch_images)))
                        yield batch_images, to_categorical(batch_labels, num_classes=len(label_dict))
        
        # Create generators
        train_generator = generate_data(train_paths, train_labels, train_datagen, self.batch_size)
        val_generator = generate_data(val_paths, val_labels, val_datagen, self.batch_size)
        test_generator = generate_data(test_paths, test_labels, test_datagen, self.batch_size)
        
        return {
            'train': train_generator,
            'val': val_generator,
            'test': test_generator,
            'train_steps': len(train_paths) // self.batch_size,
            'val_steps': len(val_paths) // self.batch_size,
            'test_steps': len(test_paths) // self.batch_size,
            'num_classes': len(label_dict),
            'label_dict': label_dict
        }
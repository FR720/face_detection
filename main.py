from src.data_preprocessing import DataPreprocessor
from src.model import FaceRecognitionModel
from src.trainer import ModelTrainer
import os

def main():
    # Create necessary directories
    os.makedirs('models/trained_models', exist_ok=True)

    # Initialize preprocessor
    dataset_path="data/105_classes_pins_dataset"
    preprocessor = DataPreprocessor(
        dataset_path=dataset_path,
        target_size=(299, 299),
        batch_size=32
    )
    
    # Create data generators
    print("Creating data generators...")
    data_generators = preprocessor.create_data_generators()
    
    # Build model
    print("Building model...")
    model = FaceRecognitionModel.build_model(num_classes=data_generators['num_classes'])
    
    # Train model
    print("Training model...")
    trainer = ModelTrainer(
        model=model,
        train_generator=data_generators['train'],
        val_generator=data_generators['val'],
        train_steps=data_generators['train_steps'],
        val_steps=data_generators['val_steps']
    )
    
    # Train the model
    history = trainer.train(epochs=1)
    
    # Save the model
    trainer.save_model()
    
    # Evaluate on test set
    print("\nEvaluating model...")
    trainer.evaluate(
        test_generator=data_generators['test'],
        test_steps=data_generators['test_steps']
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()
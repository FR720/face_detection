import tensorflow as tf
import tf2onnx
import os

class ModelTrainer:
    def __init__(self, model, train_generator, val_generator, train_steps, val_steps):
        """
        Initialize the trainer
        
        Args:
            model: Keras model
            train_generator: Training data generator
            val_generator: Validation data generator
            train_steps: Number of training steps per epoch
            val_steps: Number of validation steps per epoch
        """
        self.model = model
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.train_steps = train_steps
        self.val_steps = val_steps

    def train(self, epochs=50):
        """
        Train the model
        
        Args:
            epochs: Number of epochs to train
        
        Returns:
            Training history
        """
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=3
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='models/trained_models/best_model.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]

        # Train the model
        history = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_steps,
            validation_data=self.val_generator,
            validation_steps=self.val_steps,
            epochs=epochs,
            callbacks=callbacks
        )

        return history

    def save_model(self, save_dir='models/trained_models'):
        """
        Save the model in different formats
        """
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Save Keras model
        keras_path = os.path.join(save_dir, 'face_recognition_model.h5')
        self.model.save(keras_path)
        print(f"Keras model saved to {keras_path}")

        # Convert and save as ONNX
        try:
            spec = (tf.TensorSpec((None, 299, 299, 3), tf.float32, name="input"),)
            onnx_path = os.path.join(save_dir, 'face_recognition_model.onnx')
            
            model_proto, _ = tf2onnx.convert.from_keras(
                self.model, 
                input_signature=spec,
                output_path=onnx_path
            )
            print(f"ONNX model saved to {onnx_path}")
        except Exception as e:
            print(f"Error saving ONNX model: {e}")

        # Convert and save as TFLite
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()

            tflite_path = os.path.join(save_dir, 'face_recognition_model.tflite')
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            print(f"TFLite model saved to {tflite_path}")
        except Exception as e:
            print(f"Error saving TFLite model: {e}")

    def evaluate(self, test_generator, test_steps):
        """
        Evaluate the model on test data
        """
        results = self.model.evaluate(
            test_generator,
            steps=test_steps
        )
        
        print("\nTest Results:")
        for metric_name, value in zip(self.model.metrics_names, results):
            print(f"{metric_name}: {value:.4f}")

        return results
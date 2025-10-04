import os
import shutil
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from models.math_symbol_cnn import build_cnn_model
import kagglehub

def main():
    # ---------------- Download dataset ----------------
    print("📦 Downloading dataset from Kaggle...")
    try:
        dataset_path = kagglehub.dataset_download("sagyamthapa/handwritten-math-symbols")
        dataset_dir = os.path.join(dataset_path, "dataset")
        print(f"✅ Dataset downloaded to: {dataset_dir}")
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("Please ensure you have kagglehub configured correctly")
        return
    
    # Verify dataset exists
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return
    
    # Remove unwanted folders: x, y, z (variables) and dec (decimal)
    removed_count = 0
    folders_to_remove = ["x", "y", "z", "dec"]  # Added 'dec' to remove decimal
    
    for folder in folders_to_remove:
        path = os.path.join(dataset_dir, folder)
        if os.path.exists(path):
            shutil.rmtree(path)
            removed_count += 1
            print(f"✅ Removed folder: {folder}")
    
    if removed_count > 0:
        print(f"Removed {removed_count} unnecessary folders")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # ---------------- Hyperparameters ----------------
    IMG_SIZE = 28
    BATCH_SIZE = 32
    EPOCHS = 50  # Increased for better training
    LEARNING_RATE = 0.0003  # Lower learning rate
    
    print(f"\n🔧 Configuration:")
    print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    
    # ---------------- Enhanced Data augmentation ----------------
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,           # Reduced rotation
        width_shift_range=0.1,       # Reduced shift
        height_shift_range=0.1,
        zoom_range=0.1,              # Reduced zoom
        shear_range=0.05,            # Reduced shear
        brightness_range=[0.85, 1.15],  # Moderate brightness
        fill_mode='constant',
        cval=0,
        validation_split=0.2
    )
    
    # Validation data - only rescaling
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # ---------------- Create data generators ----------------
    print("\n📊 Loading training data...")
    train_gen = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        subset="training",
        shuffle=True,
        seed=42
    )
    
    print("📊 Loading validation data...")
    val_gen = val_datagen.flow_from_directory(
        dataset_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        subset="validation",
        shuffle=False,
        seed=42
    )
    
    NUM_CLASSES = len(train_gen.class_indices)
    print(f"\n✅ Data loaded successfully!")
    print(f"  Classes: {NUM_CLASSES}")
    print(f"  Training samples: {train_gen.samples}")
    print(f"  Validation samples: {val_gen.samples}")
    print(f"  Steps per epoch: {len(train_gen)}")
    print(f"\n📋 Class mapping:")
    for cls, idx in sorted(train_gen.class_indices.items(), key=lambda x: x[1]):
        print(f"    {idx}: {cls}")
    
    # Verify we have the right classes (should be 15 now: 0-9, add, sub, mul, div, eq)
    expected_classes = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                       'add', 'sub', 'mul', 'div', 'eq'}
    actual_classes = set(train_gen.class_indices.keys())
    
    if not expected_classes.issubset(actual_classes):
        missing = expected_classes - actual_classes
        print(f"\n⚠️ Warning: Missing classes: {missing}")
    
    unexpected = actual_classes - expected_classes
    if unexpected:
        print(f"\n⚠️ Warning: Unexpected classes found: {unexpected}")
        print("These should have been removed. Please check dataset directory.")
    
    # ---------------- Build model ----------------
    print("\n🏗️  Building enhanced model...")
    model = build_cnn_model(
        input_shape=(IMG_SIZE, IMG_SIZE, 1),
        num_classes=NUM_CLASSES
    )
    
    # Compile with Adam optimizer
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    total_params = model.count_params()
    print(f"\n📋 Model Architecture Summary:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Input Shape: (28, 28, 1)")
    print(f"  Output Classes: {NUM_CLASSES}")
    
    # ---------------- Callbacks ----------------
    checkpoint = ModelCheckpoint(
        "models/math_symbol_cnn.keras",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    earlystop = EarlyStopping(
        monitor='val_accuracy',
        patience=12,  # More patience
        restore_best_weights=True,
        verbose=1,
        mode='max',
        min_delta=0.0005
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    callbacks = [checkpoint, earlystop, reduce_lr]
    
    # ---------------- Train model ----------------
    print("\n🚀 Starting training...")
    print("=" * 70)
    print("Target: >95% validation accuracy")
    print("=" * 70 + "\n")
    
    try:
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n" + "=" * 70)
        print("✅ Training completed successfully!")
        
        # Print final metrics
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        best_val_acc = max(history.history['val_accuracy'])
        best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
        
        print(f"\n📊 Training Results:")
        print(f"  Final Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
        print(f"  Final Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
        print(f"  Final Training Loss: {final_train_loss:.4f}")
        print(f"  Final Validation Loss: {final_val_loss:.4f}")
        print(f"\n  🏆 Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%) at epoch {best_epoch}")
        
        # Check for overfitting
        if final_train_acc - final_val_acc > 0.08:
            print(f"\n⚠️  Warning: Possible overfitting detected (gap: {(final_train_acc - final_val_acc)*100:.2f}%)")
            print("    Consider adding more dropout or reducing model complexity")
        else:
            print(f"\n✅ Model generalization looks good!")
        
        # Check if accuracy is acceptable
        if best_val_acc < 0.90:
            print(f"\n⚠️  Validation accuracy below 90%. Consider:")
            print("    - Training for more epochs")
            print("    - Adjusting learning rate")
            print("    - Checking data quality")
        elif best_val_acc >= 0.95:
            print(f"\n🎉 Excellent validation accuracy achieved!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ---------------- Save class indices ----------------
    print("\n💾 Saving class indices...")
    try:
        with open("models/class_indices.json", "w") as f:
            json.dump(train_gen.class_indices, f, indent=2)
        print("✅ Class indices saved to models/class_indices.json")
    except Exception as e:
        print(f"❌ Error saving class indices: {e}")
        return
    
    # ---------------- Save training history ----------------
    print("💾 Saving training history...")
    try:
        history_dict = {
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'best_epoch': int(best_epoch),
            'best_val_accuracy': float(best_val_acc),
            'num_classes': NUM_CLASSES,
            'classes': list(train_gen.class_indices.keys())
        }
        with open("models/training_history.json", "w") as f:
            json.dump(history_dict, f, indent=2)
        print("✅ Training history saved to models/training_history.json")
    except Exception as e:
        print(f"⚠️  Could not save training history: {e}")
    
    # ---------------- Model evaluation ----------------
    print("\n📈 Evaluating model on validation set...")
    try:
        val_loss, val_accuracy = model.evaluate(val_gen, verbose=0)
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    except Exception as e:
        print(f"⚠️  Could not evaluate model: {e}")
    
    print("\n" + "=" * 70)
    print("🎉 All done! Model is ready to use.")
    print("\n📝 Next steps:")
    print("  1. Run: python test_model.py (to verify accuracy)")
    print("  2. Run: streamlit run app.py (to test the application)")
    print("  3. Write clearly with good spacing for best results")
    print("=" * 70)


if __name__ == "__main__":
    main()
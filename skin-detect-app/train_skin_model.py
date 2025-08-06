Skip to content
Navigation Menu
HitakshikaSuri10
skintight

Type / to search
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
Settings
Files
Go to file
t
skin-detect-app
app.py
ml_model.py
requirements.txt
run.py
test_app.py
static
templates
.gitignore
.python-version
README.md
app.py
ml_model.py
render.yaml
requirements.txt
run.py
test_app.py
train_skin_model.py
train_skin_model_old.py
skintight
/train_skin_model.py
Hitakshika SuriHitakshika Suri
Hitakshika Suri
and
Hitakshika Suri
Auto-download ML model from Google Drive if missing; improved trainin…
b0716d8
 · 
1 hour ago

Code

Blame
213 lines (177 loc) · 7.09 KB
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import random
from PIL import Image
import shutil

# Paths
train_dir = 'archive-2/train'
test_dir = 'archive-2/test'

# Parameters
img_height, img_width = 224, 224
batch_size = 32
epochs = 15

# Map available classes to our 6 target classes
class_mapping = {
    'Acne': ['Acne and Rosacea Photos'],
    'Eczema': ['Eczema Photos', 'Atopic Dermatitis Photos'],
    'Melanoma': ['Melanoma Skin Cancer Nevi and Moles', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions'],
    'Normal': ['Seborrheic Keratoses and other Benign Tumors'],
    'Psoriasis': ['Psoriasis pictures Lichen Planus and related diseases'],
    'Rosacea': ['Acne and Rosacea Photos']  # Will be handled specially
}

# Create a temporary directory for our 6-class dataset
temp_train_dir = 'temp_train'
temp_test_dir = 'temp_test'

if os.path.exists(temp_train_dir):
    shutil.rmtree(temp_train_dir)
if os.path.exists(temp_test_dir):
    shutil.rmtree(temp_test_dir)

os.makedirs(temp_train_dir)
os.makedirs(temp_test_dir)

# Create directories for our 6 classes
target_classes = ['Acne', 'Eczema', 'Melanoma', 'Normal', 'Psoriasis', 'Rosacea']
for class_name in target_classes:
    os.makedirs(os.path.join(temp_train_dir, class_name))
    os.makedirs(os.path.join(temp_test_dir, class_name))

print("Creating 6-class dataset...")

# Copy and organize images
for target_class, source_classes in class_mapping.items():
    for source_class in source_classes:
        # Handle train data
        source_train_path = os.path.join(train_dir, source_class)
        target_train_path = os.path.join(temp_train_dir, target_class)
        
        if os.path.exists(source_train_path):
            # Copy images, but limit to avoid class imbalance
            images = [f for f in os.listdir(source_train_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            max_images = min(500, len(images))  # Limit to 500 images per class
            selected_images = random.sample(images, max_images)
            
            for img in selected_images:
                src = os.path.join(source_train_path, img)
                dst = os.path.join(target_train_path, f"{source_class}_{img}")
                shutil.copy2(src, dst)
        
        # Handle test data
        source_test_path = os.path.join(test_dir, source_class)
        target_test_path = os.path.join(temp_test_dir, target_class)
        
        if os.path.exists(source_test_path):
            images = [f for f in os.listdir(source_test_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            max_images = min(100, len(images))  # Limit to 100 images per class for test
            selected_images = random.sample(images, max_images)
            
            for img in selected_images:
                src = os.path.join(source_test_path, img)
                dst = os.path.join(target_test_path, f"{source_class}_{img}")
                shutil.copy2(src, dst)

print("Dataset creation complete!")

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Data generators
train_generator = train_datagen.flow_from_directory(
    temp_train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    temp_test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

num_classes = len(train_generator.class_indices)
print(f"Training with {num_classes} classes: {list(train_generator.class_indices.keys())}")

# Compute class weights
class_indices = train_generator.class_indices
class_labels = list(class_indices.keys())
counts = [len(os.listdir(os.path.join(temp_train_dir, label))) for label in class_labels]
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(len(class_labels)),
    y=np.concatenate([np.full(count, i) for i, count in enumerate(counts)])
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# Build a custom CNN model
model = Sequential([
    # First Convolutional Block
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    # Second Convolutional Block
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    # Third Convolutional Block
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    # Fourth Convolutional Block
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    # Flatten and Dense Layers
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

print("Model architecture:")
model.summary()

# Callbacks
checkpoint = ModelCheckpoint('skin_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("Starting training...")
# Initial training
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stop, checkpoint],
    class_weight=class_weights
)

print("Initial training complete. Best model saved as skin_model.h5")

# Fine-tuning phase with lower learning rate
print("\n--- Starting fine-tuning phase ---")
model.compile(optimizer=Adam(learning_rate=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])

fine_tune_epochs = 10
history_finetune = model.fit(
    train_generator,
    epochs=fine_tune_epochs,
    validation_data=validation_generator,
    callbacks=[early_stop, ModelCheckpoint('skin_model_finetuned.h5', monitor='val_accuracy', save_best_only=True, mode='max')],
    class_weight=class_weights
)

print("Fine-tuning complete. Final model saved as skin_model_finetuned.h5")

# Clean up temporary directories
shutil.rmtree(temp_train_dir)
shutil.rmtree(temp_test_dir)

print("Training complete! Models saved:")
print("- skin_model.h5 (initial training)")
print("- skin_model_finetuned.h5 (fine-tuned)")
print("Your app will automatically use the fine-tuned model if available!")
skintight/train_skin_model.py at main · HitakshikaSuri10/skintight

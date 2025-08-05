import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import random
from PIL import Image

# Paths
train_dir = 'archive-2/train'
test_dir = 'archive-2/test'

# Parameters
img_height, img_width = 224, 224
batch_size = 32
epochs = 20

# Strong data augmentation for training
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

# Use only 2 distinct, important classes for a quick test
selected_classes = [
    'Acne and Rosacea Photos',
    'Warts Molluscum and other Viral Infections'
]

# Data Generators (filtered to selected classes)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    classes=selected_classes
)

# Visualize a batch of images and their labels
batch_images, batch_labels = next(train_generator)
plt.figure(figsize=(12, 8))
for i in range(min(8, batch_images.shape[0])):
    plt.subplot(2, 4, i+1)
    plt.imshow(batch_images[i])
    label_idx = batch_labels[i].argmax()
    class_name = list(train_generator.class_indices.keys())[label_idx]
    plt.title(class_name)
    plt.axis('off')
plt.tight_layout()
plt.show()

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    classes=selected_classes
)

num_classes = len(train_generator.class_indices)
print(f"Detected {num_classes} classes: {list(train_generator.class_indices.keys())}")

# --- CLASS WEIGHTS SECTION ---
# Get class indices and labels
class_indices = train_generator.class_indices
class_labels = list(class_indices.keys())

# Get the number of images per class
counts = [len(os.listdir(os.path.join(train_dir, label))) for label in class_labels]

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(len(class_labels)),
    y=np.concatenate([
        np.full(count, i) for i, count in enumerate(counts)
    ])
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# Build EfficientNetB0 model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(len(selected_classes), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('skin_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Initial training
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[early_stop, checkpoint]
)

print("Training complete. Best model saved as skin_model.h5")

# --- FINE-TUNING PHASE ---
print("\n--- Starting fine-tuning phase ---")
# Unfreeze the last 10 layers of the base model
for layer in base_model.layers[-10:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])

fine_tune_epochs = 10
history_finetune = model.fit(
    train_generator,
    epochs=fine_tune_epochs,
    validation_data=validation_generator,
    callbacks=[early_stop, ModelCheckpoint('skin_model_finetuned.h5', monitor='val_accuracy', save_best_only=True, mode='max')]
)

print("Fine-tuning complete. Final model saved as skin_model_finetuned.h5")

# --- VISUAL DATA INSPECTION ---
for class_name in selected_classes:
    class_dir = os.path.join(train_dir, class_name)
    image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    sample_files = random.sample(image_files, min(8, len(image_files)))
    plt.figure(figsize=(12, 3))
    plt.suptitle(f"Sample images from class: {class_name}")
    for i, img_file in enumerate(sample_files):
        img_path = os.path.join(class_dir, img_file)
        img = Image.open(img_path)
        plt.subplot(1, 8, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()

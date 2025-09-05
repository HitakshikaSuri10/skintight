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
train_skin_model.py
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
/train_skin_model_old.py
Hitakshika SuriHitakshika Suri
Hitakshika Suri
and
Hitakshika Suri
Clean repository with latest updates including About Me section
a31c241
 · 
15 hours ago

Code

Blame
58 lines (50 loc) · 1.37 KB
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Path to your dataset folder (train subfolder)
DATASET_DIR = 'archive-2/train'  
# Image parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data generators for training and validation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Build a simple CNN model
model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen
)

# Save the model
model.save('skin_model.h5')
print(" Model saved as skin_model.h5")
skintight/train_skin_model_old.py at main · HitakshikaSuri10/skintight 

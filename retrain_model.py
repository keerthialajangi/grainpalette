import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import os

# -------------------------------
# PARAMETERS (You can adjust)
# -------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 4       # smaller for CPU
EPOCHS = 7           # small but sufficient
NUM_CLASSES = 5      # 5 rice types
DATASET_DIR = os.path.join(os.getcwd(), "rice_data")

SAVE_PATH = os.path.join(os.getcwd(), "model", "rice_model_fixed")

# -------------------------------
# 1️⃣ Data Generator
# -------------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# -------------------------------
# 2️⃣ Build MobileNetV2 Model
# -------------------------------
base_model = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# -------------------------------
# 3️⃣ Compile Model
# -------------------------------
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# 4️⃣ Train Model
# -------------------------------
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# -------------------------------
# 5️⃣ Save Model (SavedModel format)
# -------------------------------
model.save(SAVE_PATH)
print("✅ Model retrained and saved successfully at", SAVE_PATH)

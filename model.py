import os
import csv
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten, Cropping2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# === Load driving data ===
data_path = 'data'
log_path = os.path.join(data_path, 'driving_log.csv')

lines = []
with open(log_path) as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # skip header
    for line in reader:
        if len(line) < 4:
            continue
        lines.append(line)

print(f"📊 Total CSV entries: {len(lines)}")

# ✅ Check how many valid images exist
valid_count = 0
for line in lines:
    center_path = line[0].replace('\\', '/').strip()
    filename = os.path.basename(center_path)
    full_path = os.path.join(data_path, 'IMG', filename)
    if os.path.isfile(full_path):
        valid_count += 1

print(f"🖼️ Valid image files found: {valid_count}")

# === Split data ===
train_lines, val_lines = train_test_split(lines, test_size=0.2)

# === Data generator with augmentation ===
def generator(samples, batch_size=32):
    num_samples = len(samples)
    print(f"📦 Generator initialized with {num_samples} samples")

    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch = samples[offset:offset + batch_size]
            images = []
            angles = []

            for line in batch:
                if len(line) < 4:
                    continue

                center_path = line[0].replace('\\', '/').strip()
                angle_str = line[3].strip()

                try:
                    angle = float(angle_str)
                except ValueError:
                    continue

                filename = os.path.basename(center_path)
                full_path = os.path.join(data_path, 'IMG', filename)

                if not os.path.isfile(full_path):
                    continue

                image = cv2.imread(full_path)
                if image is None:
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(angle)

                # ✅ Augment: flipped image and angle
                images.append(cv2.flip(image, 1))
                angles.append(-angle)

            if images and angles:
                yield np.array(images), np.array(angles)

# === NVIDIA model ===
def nvidia_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.3))  # Reduced to prevent underfitting
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=Adam(1e-4))
    return model

# === Prepare generators ===
train_gen = generator(train_lines, batch_size=32)
val_gen = generator(val_lines, batch_size=32)

# ✅ Check sample batch
X_sample, y_sample = next(train_gen)
print(f"✅ Sample batch: {X_sample.shape}, {y_sample.shape}")

# === Train model ===
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

model = nvidia_model()
history = model.fit(
    train_gen,
    steps_per_epoch=math.ceil(len(train_lines) / 32),
    validation_data=val_gen,
    validation_steps=math.ceil(len(val_lines) / 32),
    callbacks=[early_stop, checkpoint],
    epochs=15,
    verbose=1
)

# === Save final model ===
model.save('model.h5')
print("✅ Final model saved as model.h5")

# === Plot training history ===
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss Over Epochs")
plt.show()

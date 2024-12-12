from dotenv import load_dotenv
from second_create_target import labels
import os

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import models
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

load_dotenv()
output_path = os.getenv("output_path")
img_dir_path = output_path[:-7]

### Sort labels alphanumerical so that the image loading function can treat them correctly ###

images = os.listdir(output_path)
# Zipping images and labels together
zipped = list(zip(images, labels))
# Sorting the zipped list by the image names (alphanumeric order)
sorted_zipped = sorted(zipped, key=lambda x: x[0])
# Unzipping the sorted list back into images and labels
sorted_images, sorted_labels = zip(*sorted_zipped)
# Converting back to labels list
sorted_labels = list(sorted_labels)

### Create Train and Test Set ###
train_dataset, validation_dataset = image_dataset_from_directory(
    directory = img_dir_path,
    labels=sorted_labels,
    label_mode='int',
    class_names=None,
    color_mode='grayscale',
    batch_size=32,
    image_size=(150,150),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset = 'both'
)

### Model Architecture ###
model = Sequential()

model.add(layers.Input((150, 150, 1)))

model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation="relu"))

model.add(layers.Dropout(0.3))

model.add(layers.Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

EarlyStopper = EarlyStopping(monitor='val_loss',
                                      patience=10,
                                      verbose=0,
                                      restore_best_weights=True)

model.fit(train_dataset, validation_data=validation_dataset, epochs=20, callbacks=[EarlyStopper])


### Save Model
model.save('model_V3_windowed')

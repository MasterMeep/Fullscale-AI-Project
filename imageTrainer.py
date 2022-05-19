import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

class imageTrainer:
    def __init__(self, data_dir, validation_split = 0.2, batch_size = 32, image_sizes = 180, optimizer = 'adam', epochs = 15, export = './trainedModel'):
        self.epochs = epochs
        self.export = export

        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=validation_split,
            subset="training",
            seed=123,
            image_size=(image_sizes, image_sizes),
            batch_size=batch_size)

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=validation_split,
            subset="validation",
            seed=123,
            image_size=(image_sizes, image_sizes),
            batch_size=batch_size)


        self.classes = self.train_ds.class_names


        AUTOTUNE = tf.data.AUTOTUNE

        self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)


        num_classes = len(self.classes)

        data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                            input_shape=(image_sizes,
                                        image_sizes,
                                        3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
        )

        self.model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
        ])

        self.model.compile(optimizer=optimizer,
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    def train(self):

        history = self.model.fit(self.train_ds, validation_data=self.val_ds, epochs=self.epochs)

        #summary data
        self.acc = history.history['accuracy']
        self.val_acc = history.history['val_accuracy']

        self.loss = history.history['loss']
        self.val_loss = history.history['val_loss']

        #save model
        self.model.save(self.export)
        """self.data = {'accuracy': self.acc, 'validation accuracy': self.val_acc, 'loss': self.loss, 'validation loss': self.val_loss}
        return({'accuracy': self.acc, 'validation accuracy': self.val_acc, 'loss': self.loss, 'validation loss': self.val_loss})"""

    def classify(self, imagePath):
        img = Image.open(imagePath)
        img = img.convert('RGB')
        img.save('image.jpg', 'jpeg')

        img = tf.keras.utils.load_img(
            './image.jpg', target_size=(self.image_sizes, self.image_sizes))
        img_array  = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        return(self.classes[np.argmax(score)])

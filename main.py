import tensorflow as tf
import keras
from keras import layers, models
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

# Data Preprocessing
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Ensure image is in BGR format
    hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    if hsl[:, :, 1].dtype == np.uint8:  # Check if the lightness channel is 8-bit single-channel
        hsl[:, :, 1] = cv2.equalizeHist(hsl[:, :, 1])
    img = cv2.cvtColor(hsl, cv2.COLOR_HLS2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert back to RGB format
    return img

def preprocess_dataset(data_dir, img_size=(256, 256), batch_size=32):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        preprocessing_function=preprocess_image,
        validation_split=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=20,
        zoom_range=0.2
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator

# Define Models
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def create_vgg16_model(input_shape, num_classes):
    base_model = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the base model
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def ensemble_models(models, input_shape):
    inputs = keras.Input(shape=input_shape)
    outputs = [model(inputs) for model in models]
    y = layers.Average()(outputs)
    ensemble_model = keras.Model(inputs=inputs, outputs=y)
    return ensemble_model

# Training and Evaluation
def train_model(model, train_generator, validation_generator, epochs=2, batch_size=32, model_path='best_model.keras'):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint = keras.callbacks.ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, mode='max')
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_steps=validation_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stopping]
    )

    return history

# Main Function
if __name__ == '__main__':
    # Specify the absolute path to the dataset
    DATA_DIR = r'C:\Users\AKULA KOUSHIK\Downloads\Telegram Desktop\PlantVillage\PlantVillage'
    IMG_SIZE = (256, 256)
    BATCH_SIZE = 32
    EPOCHS = 2
    MODEL_PATH = 'best_model.keras'

    # Preprocess data
    train_generator, validation_generator = preprocess_dataset(DATA_DIR, IMG_SIZE, BATCH_SIZE)

    # Create and train individual models
    input_shape = IMG_SIZE + (3,)
    num_classes = train_generator.num_classes

    cnn_model = create_cnn_model(input_shape, num_classes)
    vgg16_model = create_vgg16_model(input_shape, num_classes)

    history_cnn = train_model(cnn_model, train_generator, validation_generator, EPOCHS, BATCH_SIZE, 'cnn_model.keras')
    history_vgg16 = train_model(vgg16_model, train_generator, validation_generator, EPOCHS, BATCH_SIZE, 'vgg16_model.keras')

    # Ensemble models
    ensemble_model = ensemble_models([cnn_model, vgg16_model], input_shape)
    ensemble_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train ensemble model
    ensemble_history = train_model(ensemble_model, train_generator, validation_generator, EPOCHS, BATCH_SIZE, MODEL_PATH)

    # Save final ensemble model
    ensemble_model.save(MODEL_PATH)

    # Plot results
    plt.plot(history_cnn.history['loss'], label='CNN train_loss')
    plt.plot(history_cnn.history['val_loss'], label='CNN val_loss')
    plt.plot(history_vgg16.history['loss'], label='VGG16 train_loss')
    plt.plot(history_vgg16.history['val_loss'], label='VGG16 val_loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(history_cnn.history['accuracy'], label='CNN train_acc')
    plt.plot(history_cnn.history['val_accuracy'], label='CNN val_acc')
    plt.plot(history_vgg16.history['accuracy'], label='VGG16 train_acc')
    plt.plot(history_vgg16.history['val_accuracy'], label='VGG16 val_acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

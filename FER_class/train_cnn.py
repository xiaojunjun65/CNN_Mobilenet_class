import tensorflow as tf
import matplotlib.pyplot as plt
from time import *

def data_load(data_dir, test_data_dir, img_height, img_width, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names

    return train_ds, val_ds, class_names


# CNN
def model_load(class_num, IMG_SHAPE=(48, 48, 3)):
    model = tf.keras.models.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=IMG_SHAPE),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    model.summary()
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# きょくせん
def show_loss_acc(history):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('FER+CNN Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('FER+CNN  Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('results/results_cnn.png', dpi=100)


def train(epochs):
    begin_time = time()
    train_ds, val_ds, class_names = data_load("/data/guojun/classinfo/cnn/data_new/train",
                                              "/data/guojun/classinfo/cnn/data_new/test", 48, 48, 256)
    print(class_names)
    model = model_load(class_num=len(class_names))
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.save("../models/cnn.h5")
    # じかん
    end_time = time()
    run_time = end_time - begin_time
    print('运行时间为：', run_time, "s")
    show_loss_acc(history)


if __name__ == '__main__':
    train(epochs=300)

from tensorflow import keras
from keras.utils import to_categorical
from keras.datasets import imdb, mnist, reuters
from keras import layers, models
import numpy as np

from functions import loses, relu_tanh, batch, neuron, lay


def first_exercise():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    network = keras.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))

    network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    network.fit(train_images, train_labels, epochs=5, batch_size=128, verbose=0)
    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print(test_acc)

    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    def vectorize_sequences(sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    print('___________________________')
    lay(1, x_train, y_train, x_test, y_test)
    lay(2, x_train, y_train, x_test, y_test)
    lay(3, x_train, y_train, x_test, y_test)
    print('___________________________')
    neuron(32, 32, x_train, y_train, x_test, y_test)
    neuron(32, 64, x_train, y_train, x_test, y_test)
    neuron(64, 64, x_train, y_train, x_test, y_test)
    neuron(64, 96, x_train, y_train, x_test, y_test)
    neuron(96, 128, x_train, y_train, x_test, y_test)
    print('___________________________')
    loses('binary_crossentropy', x_train, y_train, x_test, y_test)
    loses('mse', x_train, y_train, x_test, y_test)
    print('___________________________')
    relu_tanh('relu', x_train, y_train, x_test, y_test)
    relu_tanh('tanh', x_train, y_train, x_test, y_test)
    print('___________________________')
    batch(32, x_train, y_train, x_test, y_test)
    batch(64, x_train, y_train, x_test, y_test)
    batch(96, x_train, y_train, x_test, y_test)
    batch(128, x_train, y_train, x_test, y_test)
    batch(256, x_train, y_train, x_test, y_test)
    batch(512, x_train, y_train, x_test, y_test)
    print('////////////////////////////////////////////////////')


def second_exercise():
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

    def vectorize_sequences(sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    one_hot_train_labels = to_categorical(train_labels)
    one_hot_test_labels = to_categorical(test_labels)

    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]

    y_val = one_hot_train_labels[:1000]
    partial_y_train = one_hot_train_labels[1000:]

    model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val), verbose=0)

    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val), verbose=0)
    result = model.evaluate(x_test, one_hot_test_labels, verbose=0)
    print(result)


def third_exercise():
    pass


def main():
    #first_exercise()
    second_exercise()
    third_exercise()


if __name__ == '__main__':
    main()

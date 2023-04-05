from keras import models, layers
import matplotlib.pyplot as plt
import numpy as np


def epochs_3(num_epochs, layers_count, batch, k, train_data, num_val_samples, train_targets):
    def build_model(layers_count):
        if layers_count == 2:
            model = models.Sequential()
            model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(1))
            model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
            return model
        elif layers_count == 1:
            model = models.Sequential()
            model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
            model.add(layers.Dense(1))
            model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
            return model
        else:
            model = models.Sequential()
            model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(1))
            model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
            return model

    all_scores = []

    for i in range(k):
        print('К-блок номер #', i)
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

        partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]],
            axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]],
            axis=0)

        model = build_model(layers_count)
        model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=batch, verbose=0)
        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
        all_scores.append(val_mae)
        print('Средняя ошибка -', all_scores)
        print('Среднее значение ошибки', np.mean(all_scores))


def graph_2(history):
    loss = history.history['loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, acc, 'bo', label='Training loss')
    plt.plot(epochs, val_acc, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def graph_1(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def batch_2(batch, partial_x_train, partial_y_train, x_test, one_hot_test_labels, x_val, y_val):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    model.fit(partial_x_train, partial_y_train, epochs=4, batch_size=batch, validation_data=(x_val, y_val), verbose=0)
    results = model.evaluate(x_test, one_hot_test_labels, verbose=0)
    print(f'Выборка - {batch} ', 'Потери -', results[0], 'Точность -', results[1])


def lay_2(layers_count, partial_x_train, partial_y_train, x_test, one_hot_test_labels, x_val, y_val):
    model = models.Sequential()
    if layers_count == 1:
        model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
        model.add(layers.Dense(46, activation='softmax'))
    elif layers_count == 2:
        model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(46, activation='softmax'))
    else:
        model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(46, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    model.fit(partial_x_train, partial_y_train, epochs=4, batch_size=512, validation_data=(x_val, y_val), verbose=0)
    results = model.evaluate(x_test, one_hot_test_labels, verbose=0)
    print(f'Количество слоёв - {layers_count} ', 'Потери -', results[0], 'Точность -', results[1])


def epochs_count(epo, partial_x_train, partial_y_train, x_test, one_hot_test_labels, x_val, y_val):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    model.fit(partial_x_train, partial_y_train, epochs=epo, batch_size=512, validation_data=(x_val, y_val), verbose=0)
    results = model.evaluate(x_test, one_hot_test_labels, verbose=0)
    print(f'Количество эпох - {epo}', 'Потери -', results[0], 'Точность -', results[1])


def lay_1(mode, x_train, y_train, x_test, y_test):
    model = models.Sequential()
    if mode == 1:
        model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
        model.add(layers.Dense(1, activation='sigmoid'))
    elif mode == 2:
        model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
    elif mode == 3:
        model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=4, batch_size=512, verbose=0)
    results = model.evaluate(x_test, y_test, verbose=0)

    print(f'Количество слоёв - {mode}', 'Потери -', results[0], 'Точность -', results[1])


def loses(mode, x_train, y_train, x_test, y_test):
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss=f'{mode}', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=4, batch_size=512, verbose=0)
    results = model.evaluate(x_test, y_test, verbose=0)

    print(f'Функция потерь {mode}', 'Потери -', results[0], 'Точность -', results[1])


def relu_tanh(mode, x_train, y_train, x_test, y_test):
    model = models.Sequential()
    model.add(layers.Dense(16, activation=f'{mode}', input_shape=(10000,)))
    model.add(layers.Dense(16, activation=f'{mode}'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=4, batch_size=512, verbose=0)
    results = model.evaluate(x_test, y_test, verbose=0)

    print(f'Функция активации {mode}', 'Потери -', results[0], 'Точность -', results[1])


def batch_1(batch_size, x_train, y_train, x_test, y_test):
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=4, batch_size=256, verbose=0)
    results = model.evaluate(x_test, y_test, verbose=0)

    print(f'batch_size {batch_size}', 'Потери -', results[0], 'Точность -', results[1])


def neuron(neuron_1, neuron_2, x_train, y_train, x_test, y_test):
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=4, batch_size=512, verbose=0)
    results = model.evaluate(x_test, y_test, verbose=0)

    print(f'{neuron_1} и {neuron_2} нейрона', 'Потери -', results[0], 'Точность -', results[1])

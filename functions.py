from keras import models, layers


def lay(mode, x_train, y_train, x_test, y_test):
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


def batch(batch_size, x_train, y_train, x_test, y_test):
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

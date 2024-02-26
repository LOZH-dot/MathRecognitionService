import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical



# Загрузка набора данных CIFAR-10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Нормализация данных
X_train = X_train / 255.0
X_test = X_test / 255.0

# Преобразование меток классов в one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Создание модели нейронной сети
model = Sequential([
     Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
     MaxPooling2D((2, 2)),
     Conv2D(64, (3, 3), activation='relu'),
     MaxPooling2D((2, 2)),
     Flatten(),
     Dense(128, activation='relu'),
     Dense(10, activation='softmax')
 ])

# # Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))

# Оценка модели
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Точность модели: {accuracy}')
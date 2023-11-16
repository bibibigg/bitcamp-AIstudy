import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.datasets import fashion_mnist

# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

# 정규화
x_train, x_test = x_train/255.0, x_test/255.0

# reshape 3차원을 4차원으로 늘림
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)


#실습

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size= (4, 4),
                 activation='relu',
                   input_shape = (28, 28, 1)))
# model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (4, 4), activation = 'relu' ))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

'''
# 3. 컴파일, 훈련

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=256)

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)

# 결과
# MaxPooling2D(2,2), Dropout(0.2)
# loss :  0.27315348386764526
# acc :  0.9052000045776367
'''
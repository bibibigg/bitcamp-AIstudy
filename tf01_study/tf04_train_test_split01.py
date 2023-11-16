import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
# x= np.array(range(1, 21)) range명령어를 사용하여 1부터 20까지 작성가능
# y= np.array(range(1, 21))
# print(x.shape)
# print(y.shape)
# print(x)

x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

x_text = np.array([15, 16, 17, 18, 19, 20])
y_text = np.array([15, 16, 17, 18, 19, 20])

# 2. 모델구성

model = Sequential()
model.add = (Dense(14, input_dim=1))
model.add = (Dense(50))
model.add = (Dense(1))

# 3. 컴파일, 훈련

model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1) # 주어진 데이터로 모델 훈련

# 4. 평가, 예측

loss = model.evaluate(x_text, y_text) # 주어진 데이터를 사용하여 모델의 성능을 평가
print('loss : ', loss)

result = model.predict([21])
print('21의 예측값 : ', result)
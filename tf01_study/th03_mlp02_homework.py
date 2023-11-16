import numpy as np 
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 2, 1, 1, 2, 1.1, 1.2, 1.4, 1.5, 1.6],
              [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]) # 3개의 리스트를 인식하기 위해 3개의 리스트를 대활호로 묶음
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

# 모델구성부분부터 평가예측까지 완성하시오
# 예측값은 [[10, 1.6, 1]]

print(x.shape) #배열 x의 형태 확인
# (3, 10)
print(y.shape) #배열 y의 형태 확인
# (10,)

x = x.transpose() # 각 첫번째 리스트끼리 열로 묶음 따라서 10개의 행과 3개의 열로 나뉨
print(x.shape)
# (10, 3)

# 2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000, batch_size=32)

# 4. 평가, 예측
loss = model.evaluate(x,y)
print('loss : ', loss)

result = model.predict([[10, 1.6, 1]])
print('10과 1.6과 1의 예측값 : ', result)



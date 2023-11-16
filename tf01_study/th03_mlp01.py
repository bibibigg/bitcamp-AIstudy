import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 2, 1, 1, 2, 1.1, 1.2, 1.4, 1.5, 1.6]]) # 2개의 리스트를 인식하기 위해 2개의 리스트를 대활호로 묶음
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

print(x.shape) # (2, 10) shape명령어는 행과 열을 확인
print(y.shape) # (10,)

x = x.transpose() #y와 행을 맞추기 위해 x를 transpose로 묶음
# x = x.T 위 명령어와 같음
print(x.shape) #2개의 행과 10개의 열을 2개로 묶었기에 행은 10개 열은 2개가 됨
# (10, 2)

# 2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=2)) #2차원 배열이기에 input_dim은 2를 준다
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=500, batch_size=5)

# 4. 평가, 예측
loss = model.evaluate(x,y)
print('loss : ', loss)

result = model.predict([[10, 1.6]])
print('10과 1.6의 예측값 : ', result)

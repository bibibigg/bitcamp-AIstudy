import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x = np.array([1, 2, 3, 5, 4])
y = np.array([1, 2, 3, 4, 5])

# 2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1)) #input_dim의 의미
model.add(Dense(30))
model.add(Dense(1)) # 출력층에서 1이아닌 3을 적을 시 예측값이 3개가 나옴
# 3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam') #mae w값이 (-)일때
model.fit(x,y, epochs=500)
# 4. 예측, 평가
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([6])
print('6의 예측값 : ', result)
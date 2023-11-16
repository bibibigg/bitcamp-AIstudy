# 1. 데이터
import numpy as np 
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

# 2. 모델구성
from keras.models import Sequential 
from keras.layers import Dense      

model = Sequential()
model.add(Dense(10, input_dim=1))   # 입력층 뉴런 10개
model.add(Dense(20))                # 히든 레이어 뉴런 50개
model.add(Dense(3))                # 히든 레이어가 증가할수록 표현력이 증가하나 
                                    # 모델의 복잡도가 높아지고 과적합의 문제가 발생할 수 있으므로 
                                    # 적절한 히든 레이어의 수를 선택하기 위해 실험과 검증을 통해 최적의 구조를 찾아야함
model.add(Dense(1))                 #출력층 뉴런 1개

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') # mse는 손실함수로 평균 제곱오차 사용 옵티마이저로 Adam 사용
model.fit(x, y, epochs=500) # x,y 데이터를 사용하여 훈련 반복횟수는 100

# 4. 예측, 평가
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([4])
print('4의 예측값 : ', result)
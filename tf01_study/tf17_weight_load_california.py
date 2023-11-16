import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# from sklearn.datasets import load_boston # 윤리적문제로 제공하지않음
from sklearn.datasets import fetch_california_housing # california_housing 의 데이터 로드
import time
#1. 데이터
# datasets = load_boston()

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(datasets.feature_names) # 데이터셋의 특성 정보 출력
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(datasets.DESCR)
# :속성 정보:
# - 그룹의 중위수 소득
# - 그룹의 주택 연령 중위수
# - 가구당 평균 객실 수
# - 평균 가구당 침실 수
# - 모집단 블럭 그룹 모집단
# - 평균 가구원수
# - Latitude 블록 그룹 위도
# - 경도 블록 그룹 경도

print(x.shape)  # (20640, 8)
print(y.shape)  # (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size= 0.2,
    random_state=100,
    shuffle=True
)

print(x_train.shape) # (14447, 8)
print(y_train.shape) # (14447,)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=8))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

model.load_weights('./_save/tf17_weight_california.h5')
# weight로드시 모델과 컴파일은 있어야 함


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
'''
## earlyStopping
from keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min',
                              verbose=1, restore_best_weights=True) #restore_best_weights는 기본값은 False이므로 True로 반드시 변경
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=5000, batch_size=200,
          validation_split=0.2,
          callbacks=[earlyStopping], verbose=1) # validation data =>(train 0.6, test 0.2)
end_time = time.time() - start_time

model= load_model('./_save/tf16_california.h5') # 모델에 관련된 것만 로드, time은 로드되지 않음 
                                                # earlystopping은 model.fit에 설정되어있기에 model에 정보가 들어있어서 실행가능
'''
# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict) # y_test의 실제값과 y_predict의 예측값 사이의 스코어를 계산하여 출력
print('r2 스코어 : ', r2)
# print('끝난 시간 : ', end_time) #타임을 주석처리 하는 이유는 로드한 것은 모델에 관한 것만 로드하였기에
                                #time설정한것은 로드가 되지 않아서 오류가 나오기에 주석처리

#===============================================================#
# patience=100
# Epoch 460: early stopping
# loss :  0.6002389788627625
# r2 스코어 :  0.5545195934748055
# 끝난 시간 :  42.316638469696045


#patience=50
# Epoch 136: early stopping
# loss :  0.6408074498176575
# r2 스코어 :  0.5244108604210477
# 끝난 시간 :  13.825413227081299


# Epoch 214: early stopping
# r2 스코어 :  0.5338618154923176
# 끝난 시간 :  26.296260595321655

#load_model 사용
# Epoch 259: early stopping
# r2 스코어 :  0.5462146575286781
# 끝난 시간 :  30.51783537864685

# load_model2
# loss :  0.6381087899208069
# r2 스코어 :  0.5264138103484519
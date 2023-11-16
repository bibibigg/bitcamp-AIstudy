import numpy as np
from keras.models import Sequential
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
    test_size= 0.2, #model.fit에 valiation_split을 0.2로 설정하였기에 자동으로 train_size는 나머지 0.6으로 설정
    random_state=100,
    shuffle=True
)

print(x_train.shape) # (14447, 8)
print(y_train.shape) # (14447,)

# 2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=8))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

## earlyStopping
from keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min',
                              verbose=1, restore_best_weights=True)
    #model_checkpoint
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                          save_best_only=True,
                           filepath='./_mcp/tf18_california.hdf5' ) #모델 체크포인트 저장 파일은 h5가 아닌 hdf5
# 모델체크포인트를 사용하여 훈련중간 모델의 가중치를 저장 가능
# 위 명령어를 사용하여 훈련과정 중 최상의 성능을 보이는 모델 가중치를 저장할 수 있음

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=5000, batch_size=200,
          validation_split=0.2,
          callbacks=[earlyStopping, mcp], verbose=1) # validation data =>(train 0.6, test 0.2)
end_time = time.time() - start_time



# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict) # y_test의 실제값과 y_predict의 예측값 사이의 스코어를 계산하여 출력
print('r2 스코어 : ', r2)
print('끝난 시간 : ', end_time)

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

# Epoch 139: early stopping
# loss :  0.6381087899208069
# r2 스코어 :  0.5264138103484519

# r2 스코어 :  0.5283928555385713
# 끝난 시간 :  17.35898494720459

# Epoch 187: early stopping
# loss :  0.6310914754867554
# r2 스코어 :  0.5316218378789006
# 끝난 시간 :  24.612561464309692
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score # 회귀분석이 아닌 분류분석을 사용하기에 accruracy_score사용
from sklearn.datasets import load_breast_cancer
import time

# 1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR) 
print(datasets.feature_names) 
# 'mean radius' 'mean texture' 'mean perimeter' 'mean area'
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'       
#  'radius error' 'texture error' 'perimeter error' 'area error'        
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'    
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'        
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension'
x = datasets.data
y = datasets.target
print(x.shape) # (569, 30)
print(y.shape) # (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.6, test_size=0.2, random_state=100, shuffle=True
)


# 2. 모델 구성
model = Sequential()
model.add(Dense(100,activation='linear', input_dim=30))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1, activation='sigmoid')) # 이진분류는 무조건 아웃풋 레이어의 활성화 함수를
                                            # 'sigmoid'로 해줘야 한다.

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy', 'mse'])
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=200,
          validation_split=0.2, verbose=2) #verbose=0을 하면 훈련과정을 안볼 수 있음
end_time = time.time() - start_time


# 4. 평가, 예측
loss, acc, mse = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print(y_predict)
print('loss : ', loss)
print('acc : ', acc)
print('mse : ', mse)
# 실습 accuracy_score 출력
# y_predict 반올림하기

# y_predict = np.where(y_predict > 0.5, 1, 0) # np.where을 사용하여 반올림 0.5보다 크면 1 아니면 0
y_predict = np.round(y_predict)
acc_test = accuracy_score(y_test, y_predict)
print('loss : ', loss) # loss :  [0.31341370940208435, 0.9415204524993896, 0.057326801121234894] loss가 세가지가 나오는 이유는 컴파일에서 metrics = ['accuracy', mse']를 주었기 때문
print('acc : ', acc) # 0.847953216374269(where 사용) acc :  0.9239766081871345(round 사용)
print('걸린 시간 : ', end_time) # 걸린 시간 :  1.0465044975280762 time명령어를 사용하여 걸린 시간이 나옴 time명령어는 fit 앞뒤에 설정


import matplotlib.pyplot as plt
plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], marker = '.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker ='.', c='blue', label='val_loss')
plt.title('Loss & Val_loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()
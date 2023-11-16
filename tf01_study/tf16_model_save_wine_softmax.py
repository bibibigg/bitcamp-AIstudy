# 실습 loss = 'sparse_categorical_crossentropy'를 사용하여 분석
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
import time

# 1. 데이터
datasets = load_wine()
print(datasets.DESCR)
print(datasets.feature_names)
# ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'] 

x = datasets['data']
y = datasets.target

print(x.shape) # (178, 13)
print(y.shape) # (178,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.6, test_size=0.2, random_state=100, shuffle=True
)

print(x_train.shape)
print(y_train.shape)
print(y_test.shape)

# 2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(3, activation='softmax'))


# 3. 컴파일, 훈련

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min',
                              restore_best_weights=True, verbose=1)
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=5000, batch_size=10,
                 validation_split=0.2, callbacks=[earlyStopping])
end_time = time.time() - start_time

model.save('./_save/tf16_wine.h5')
# 4. 평가, 예측 
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)
print('걸린 시간', end_time)


#=======================================================#
# patience=100
# loss :  0.11825362592935562
# acc :  0.9166666865348816
# 걸린 시간 17.69651699066162
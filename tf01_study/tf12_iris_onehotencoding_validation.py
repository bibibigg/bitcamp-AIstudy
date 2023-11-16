import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import time

# 1.데이터
datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# x = datasets.data
x = datasets['data']
y = datasets.target
print(x.shape)  # (150, 4)
print(y.shape) # (150,)

###one hot encoding
from keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)  # (150, 3)

x_train, x_test, y_train, y_test  = train_test_split(
    x, y, train_size=0.6, test_size=0.2, random_state=100, shuffle=True
)
print(x_train.shape, y_train.shape) # (105, 4) (105,)
print(x_test.shape, y_test.shape)   # (45, 4) (45,)
print(y_test)



# 2. 모델 구성
model = Sequential()
model.add(Dense(20, input_dim=4))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=50, batch_size=100,
           validation_split=0.2, verbose=1)
end_time = time.time() - start_time
print('걸린 시간 : ', end_time)

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test) # model.evaluate()에 loss와 acc를 반환 acc는 정확도를 의미
print('loss : ', loss)
print('acc : ', acc)
import matplotlib.pyplot as plt
plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.title('Loss & Val_loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Flatten, Dropout
from keras.datasets import imdb
from keras_preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping ,ModelCheckpoint

# 1. 데이터
(x_train, y_train), (x_test, y_test) = imdb.load_data(
num_words = 10000 ) #임베딩 레이어의 input_dim

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print(np.unique(y_train, return_counts=True))
print(len(np.unique(y_train))) # 2

# 최대길이와 평균길이
print('리뷰의 최대 길이 : ', max(len(i) for i in x_train))
print('리뷰의 평균 길이 : ', sum(map(len, x_train)) / len(x_train))


print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# pad_sequences
x_train = pad_sequences(x_train, padding='pre',
                       maxlen=100)
x_test = pad_sequences(x_test, padding='pre',
                      maxlen=100)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 2. 모델 구성
model = Sequential()
model.add(Embedding(input_dim = 10000, output_dim=100))
model.add(LSTM(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

model.summary()

#[실습] 코드 완성하기

# 3. 컴파일, 훈련

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

earlyStopping = EarlyStopping(monitor='var_loss', patience=10, mode='min',
                              restore_best_weights=True, verbose=1)

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                          save_best_only=True,
                           filepath='./_mcp/tf21_imdb.hdf5' )

model.fit(x_train, y_train, epochs=100, batch_size=256, validation_split=0.2,
          callbacks=[earlyStopping,mcp])

# 4.평가 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)

# y_predict = model.predict(y_test, )
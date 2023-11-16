import numpy as np
import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.callbacks import EarlyStopping,ModelCheckpoint



# 1. 데이터

path = 'C:\\Users\\bitcamp\\Desktop\\credit_card_prediction\\'
datasets = pd.read_csv(path + 'train.csv')


x = datasets[['ID', 'Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code',
       'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active',
       'Is_Lead']]
y = datasets[['Is_Lead']]

print(x.info())

print(x.describe())

x['NAN'] = np.nan
print(x)


print(x.shape) # (245725, 11)
print(y.shape) # (245725, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.6, test_size=0.2,
      random_state=100, shuffle=True
)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


#3. 모델 구성
model = Sequential()
model.add(Dense(256,activation='linear', input_dim=11))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
earlyStopping = EarlyStopping(monitor='val_loss', patience=100,
                              mode='min', verbose=1, restore_best_weights=True)
#model_checkpoint
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                          save_best_only=True,
                           filepath='./mini_project/mini11.hdf5' )
model.fit(x_train, y_train, epochs=100, batch_size=200,
          verbose=1,  validation_split=0.2, callbacks=[earlyStopping, mcp]) 
model.save('./mini_project/mini11.h5')

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)
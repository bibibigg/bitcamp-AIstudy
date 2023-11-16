import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder

# 1. 데이터 # 불러올 파일이 단일
path = 'C:\\Users\\bitcamp\\Desktop\\credit_card_prediction\\'
datasets = pd.read_csv(path + 'train.csv')

print(datasets.info())
print(datasets.describe())


# 결측값
# mean_value = datasets['Credit_Product'].mean()
# datasets['Credit_Product'].fillna(mean_value, inplace=True)

print(datasets.columns)
    # ID  Gender  Age Region_Code     Occupation Channel_Code  Vintage Credit_Product  Avg_Account_Balance Is_Active  Is_Lead
# print(datasets.head(7))

x = datasets[['Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code', 'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active']]
y = datasets[['Is_Lead']]

# x['NAN'] = np.nan
x = x.fillna(method='ffill') 

print(x.info())



cat_cols = list(x.dtypes[x.dtypes=="object"].index)
for col in cat_cols: 
    if x[col].dtypes == 'O': # dtype이 'O'는 문자열을 의미합니다.
        le = LabelEncoder()
        x[col] = le.fit_transform(x[col].values)



print(x, y)
print(x.shape, y.shape)     # (245725, 9) (245725, 1)

print(x.head(10))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size =0.2, shuffle=True, random_state=77
)

print(x_train.shape, y_train.shape)     # (196580, 9) (196580, 1)
print(x_test.shape, y_test.shape)       # (49145, 9) (49145, 1)


# 2. 모델구성
model = Sequential()
model.add(Dense(128, input_dim=9, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일
model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy', 'mse'])


earlyStopping = EarlyStopping(monitor='val_loss', patience=100,
                              mode='min', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                          save_best_only=True,
                           filepath='./_mcp/mini01.hdf5' )

hist = model.fit(x_train, y_train, epochs=100,
          batch_size=64,
          validation_split=0.2,
          callbacks=[earlyStopping,mcp])


# 평가 
loss, acc, mse = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print(y_predict)
r2 = r2_score(y_test, y_predict)
print('loss : ', loss)
print('acc : ', acc)
print('mse : ', mse)
print('r2 스코어 : ', r2)

#시각화
import matplotlib.pyplot as plt
plt.figure(figsize=[10, 6])
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.title('loss & val_loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()
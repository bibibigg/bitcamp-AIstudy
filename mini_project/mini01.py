import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터

path = './_data/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')


x_train = train[['ID', 'Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code',
       'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active',
       'Is_Lead']]
y_train = train[['Is_Lead']]

x_test = test[['ID', 'Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code',
       'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active']]

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)

print(x_train.info()) #info() 컬럼명, null값, 데이터타입 확인

x_train['NAN'] = np.nan
x_test['NAN'] = np.nan
print(x_train)
print(x_test)

print(x_train.describe())
print(x_test.describe())

# NaN 처리하기
# x = x_train.fillna(0)
# y = x_test.fillna(0)# 0으로 채우기
# x = x_train.fillna(x.mean()[0]) #해당 컬럼의 mean값, mode, min, max값 채우기
# y = x_test.fillna(x.mean()[0])
# x = x_train.fillna(method='ffill') # 해당 컬럼의 바로 앞 데이터의 값으로 채우기
# y = x_test.fillna(method='ffill')
x_train = x_train.fillna(method='bfill') # 해당 컬럼의 바로 뒤 데이터의 값으로 채우기
x_test = x_test.fillna(method='bfill')
print(x_train, x_test)

print(x_train.describe())

ob_col_train = list(x_train.dtypes[x_train.dtypes=="object"].index)
ob_col_test = list(x_test.dtypes[x_test.dtypes=="object"].index)

for col in ob_col_train:
    x_train[col] = LabelEncoder().fit_transform(x_train[col].values)
    
for col in ob_col_test:
    x_test[col] = LabelEncoder().fit_transform(x_test[col].values)

print(x_train.describe())

sns.set(font_scale = 1.2)
sns.set(rc = {'figure.figsize':(9, 6)})
sns.heatmap(data = train.corr()
    square = True,
    annot = True,
    cbar = True
)
plt.show()
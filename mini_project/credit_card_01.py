import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder

# 1. 데이터 # 불러올 파일이 단일
datasets = pd.read_csv('C:\\ai_study\\_data\\train.csv')
test_datasets = pd.read_csv('C:\\ai_study\\_data\\test.csv')

datasets = datasets.fillna(method="ffill")  # NAN값을 바로앞 데이터의 값으로채우기
test_datasets = datasets.fillna(method="ffill")


ob_col_train = list(datasets.dtypes[datasets.dtypes=="object"].index)
for col in ob_col_train:
    datasets[col] = LabelEncoder().fit_transform(datasets[col].values)

ob_col_train = list(test_datasets.dtypes[test_datasets.dtypes=="object"].index)
for col in ob_col_train:
    test_datasets[col] = LabelEncoder().fit_transform(test_datasets[col].values)
'''
# 상관계수 히트 맵 
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale = 1.2)
sns.set(rc = {'figure.figsize':(10, 9)})
sns.heatmap(data = test_datasets.corr(), 
            square = True,                  # 정사각형으로 View
            annot = True,                   # 각 cell의 값 표기 유무
            cbar = True)                    #  colorbar의 유무
plt.show()
'''

# ID Gender Age Region_Code Occupation Channel_Code Vintage Credit_Product Avg_Account_Balance Is_Active Is_Lead
##오류나옴 
x = datasets.data     # 독립변수 (원인)
y = test_datasets.target                                                           # 종속변수 (결과) target

print(x.shape, y.shape)                     # (245725, 9) (245725, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size =0.2, shuffle=True, random_state=77
)

print(x_train.shape, y_train.shape)         # (196580, 9) (196580, 1)
print(x_test.shape, y_test.shape)           # (49145, 9) (49145, 1)


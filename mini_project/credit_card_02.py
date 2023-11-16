import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import cross_val_predict, cross_val_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1. 데이터 # 불러올 파일이 단일
datasets = pd.read_csv('C:\\ai_study\\_data\\train.csv')


# datasets = datasets.fillna(method="ffill")  # NAN값을 바로앞 데이터의 값으로채우기

print(datasets.head(10))                    # nan값 확인 ID : ETQCZFEJ 의 Credit_Product 값이 nan -> No 변경

ob_col_train = list(datasets.dtypes[datasets.dtypes=="object"].index)
for col in ob_col_train:
    datasets[col] = LabelEncoder().fit_transform(datasets[col].values)




# print(datasets.columns)                     # 컬럼 확인
# ID Gender Age Region_Code Occupation Channel_Code Vintage Credit_Product Avg_Account_Balance Is_Active Is_Lead

x = datasets[[ 'Age', 'Region_Code', 'Occupation', 'Channel_Code',
              'Vintage', 'Avg_Account_Balance', ]]     # 독립변수 (원인)
y = datasets[['Is_Lead']]                                                           # 종속변수 (결과) target

# print(x.shape, y.shape)                     # (245725, 9) (245725, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42
)

#stratifiedkfold 분류모델에서는 일반적으로 stratifiedKFold사용
n_splits = 5
random_state = 42
kfold = StratifiedKFold(n_splits = n_splits, random_state = random_state, shuffle=True)

#Scaler 적용
# scaler = StandardScaler()
# scaler  = MinMaxScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train.shape, y_train.shape)         # (196580, 9) (196580, 1)
# print(x_test.shape, y_test.shape)           # (49145, 9) (49145, 1)

# 2 모델
# model = XGBClassifier()
model = LGBMClassifier()
# model = CatBoostClassifier()

# 3 훈련
model.fit(x_train, y_train)

# 4 평가, 예측
result = model.score(x_test, y_test)
print('결과 acc : ', result)



score = cross_val_score(model, x_train,
                        y_train, cv=kfold) #cv = cross_validation

y_predict = cross_val_predict(model,
                              x_test, y_test,
                              cv=kfold)
acc = accuracy_score(y_test, y_predict)
print('cv pred acc : ', acc)


############ feature importance ################
print(model, " : ", model.feature_importances_)

#시각화
import matplotlib.pyplot as plt
n_features = x.shape[1] #'x'를 사용하여 변수 'x'의 shape에서 feature의 개수를 가져와야 합니다.
                            #따라서 n_features = x.shape[1]
plt.barh(range(n_features), model.feature_importances_, align='center')
plt.yticks(np.arange(n_features), x.columns) #x'를 사용하여 변수 'x'의 열 이름을 가져와야 합니다.
                                                # 따라서 plt.yticks(np.arange(n_features), x.columns)
plt.title('credit Feature Importances')
plt.ylabel('Feature')
plt.xlabel('Importances')
plt.ylim(-1, n_features)

plt.show()



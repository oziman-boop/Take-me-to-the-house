import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from datetime import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

import random
# from branca.element import Template, MacroElement
#from geopy.distance import geodesic
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
#from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error, make_scorer

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 290)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_csv('hemnet_lastt.csv')
df = df_.copy()

df.head()

translation_dict={
'Äganderätt':'Freehold',
'Bostadsrätt':'Cooperative apartment',
'Annat':'Other',
'Andelsboende':'Cooperative housing',
'Tomträtt':'Landlease',
'Andelibostadsförening':'Share in housing association',
'Informationsaknas':'Information missing',
'Hyresrätt':'Rental'
}
df['ownership_type_']=df['ownership_type'].map(translation_dict)

df.drop(columns = ['property_type'], inplace=True)
df.drop(columns = 'ownership_type_', inplace=True)

df2 = df[df['balcony'].notna()]
df2['balcony'] = df2['balcony'].map({'Ja' : 1, 'Nej' : 0})

df3 = df[df['balcony'].isna()]
df3 = df3.sample(72000, random_state=42)

dflast = pd.concat([df2, df3], axis=0)

one_hot_columns = ['housing_form', 'ownership_type', 'county', 'sold_year', 'sold_month']
drop_columns = ['street', 'build_year', 'land_area', 'area', 'latitude', 'longitude', 'operating_cost',
                'association', 'broker', 'story', 'fee', 'url', 'build_year', 'sold_date', 'floor', 'construction_date']

dfdlast = pd.get_dummies(dflast, columns=one_hot_columns, drop_first=True, dtype=int)
dfdlast.drop(columns = drop_columns, inplace = True)

dfdlast['price'] = dfdlast['price'] / 100000
dfdlast['wanted_price'] = dfdlast['wanted_price'] / 100000

train_b = dfdlast[dfdlast['balcony'].notna()]
test_b = dfdlast[dfdlast['balcony'].isna()]

X_train_b = train_b.drop(['balcony'], axis=1)
X_test_b = test_b.drop(['balcony'], axis=1)
y_train_b = train_b['balcony']
y_test_b = test_b['balcony']

Mm = MinMaxScaler()
X_train_b = Mm.fit_transform(X_train_b)
X_test_b = Mm.transform(X_test_b)

from xgboost import XGBClassifier
xgbc = XGBClassifier()
xgbc.fit(X_train_b, y_train_b)

xgb_pred = xgbc.predict(X_test_b)

df3['balcony'] = xgb_pred


df4 = df3.copy()
df4['price'] = np.nan

dflast2 = pd.concat([df2, df4], axis=0)

dflast2.shape
dflast2.columns

one_hot_columns = ['housing_form', 'ownership_type', 'county', 'sold_year', 'sold_month']
drop_columns = ['street', 'build_year', 'land_area', 'area', 'latitude', 'longitude', 'operating_cost',
                'association', 'broker', 'story', 'fee', 'url', 'build_year', 'sold_date', 'floor', 'construction_date']


dflast2 = pd.get_dummies(dflast2, columns=one_hot_columns, drop_first=True, dtype=int)
dflast2.drop(columns = drop_columns, inplace = True)

train = dflast2[dflast2['price'].notna()]
test = dflast2[dflast2['price'].isna()]
X_train = train.drop(['price'], axis=1)
X_test = test.drop(['price'], axis=1)
y_train = train['price']
y_test = test['price']

robust = RobustScaler()

X_train = robust.fit_transform(X_train)
X_test = robust.transform(X_test)

xgb = XGBRegressor()
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

df3['predicted_price'] = xgb_pred

r2 = round(r2_score(df3['price'], df3['predicted_price']))

sns.regplot(df3, x = 'price', y = 'predicted_price')
plt.savefig('xgboost.png')
plt.show()


# R2, MAE, MSE ve RMSE hesaplamaları
r2_xgb = r2_score(df3['price'], xgb_pred)
mae_xgb = mean_absolute_error(df3['price'], xgb_pred)
mse_xgb = mean_squared_error(df3['price'], xgb_pred)
rmse_xgb = mean_squared_error(df3['price'], xgb_pred, squared=False)

# Sonuçların yazdırılması
print(f'R2: {r2_xgb:.3f}')
print(f'MAE: {mae_xgb:.3f}')
print(f'MSE: {mse_xgb:.3f}')
print(f'RMSE: {rmse_xgb:.3f}')
#
# R2: 0.614
# MAE: 578352.015
# MSE: 1686484454492.209
# RMSE: 1298647.163

from lightgbm import LGBMRegressor
lgb = LGBMRegressor(verbose = -1)
lgb.fit(X_train, y_train)
lgb_pred = lgb.predict(X_test)

r2_lgb = r2_score(df3['price'], lgb_pred)
mae_lgb = mean_absolute_error(df3['price'], lgb_pred)
mse_lgb = mean_squared_error(df3['price'], lgb_pred)
rmse_lgb = mean_squared_error(df3['price'], lgb_pred, squared = False)
print(f'R2: {r2_lgb:.3f}')
print(f'MAE: {mae_lgb:.3f}')
print(f'MSE: {mse_lgb:.3f}')
print(f'RMSE: {rmse_lgb:.3f}')

df3['predicted_price'] = lgb_pred
sns.regplot(df3, x = 'price', y = 'predicted_price')
plt.savefig('lgbm.png')
plt.show()

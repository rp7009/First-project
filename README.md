# First-project
Kaggle-Titanic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data=pd.read_csv('D:/Rp/Personal/kaggle/train.csv')
data_train_new=data[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 
       'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces',  
       'GarageYrBlt', 'GarageCars', 'GarageArea', 
       'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 
       'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SalePrice']]
elec=data_train_new['Electrical'].mode()
elec[0]
data_train_new['Electrical'].fillna(elec[0], inplace=True)
mvty=data_train_new['MasVnrType'].mode()
mvty[0]
data_train_new['MasVnrType'].fillna(mvty[0], inplace=True)
data_train_X=data_train_new[['MSSubClass', 'LotFrontage', 'LotArea', 'Street',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 
       'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces',  
       'GarageYrBlt', 'GarageCars', 'GarageArea', 
       'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 
       'MiscVal', 'MoSold', 'YrSold', 'SaleType']]
   data_train_y=data_train_new['SalePrice']
   data_test=pd.read_csv('D:/Rp/Personal/kaggle/house-price/test.csv')
   data_test_new=data_test[['MSSubClass', 'LotFrontage', 'LotArea', 'Street',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 
       'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces',  
       'GarageYrBlt', 'GarageCars', 'GarageArea', 
       'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 
       'MiscVal', 'MoSold', 'YrSold', 'SaleType']]
 nl=['KitchenQual','Exterior2nd','Utilities','BsmtHalfBath','Functional','MasVnrType','GarageYrBlt']
for i in nl:
    exmo=data_train_new[i].mode()
    data_test_new[i].fillna(exmo[0],inplace=True)
label_encoder_i = preprocessing.LabelEncoder()
data_train_X['Street']=label_encoder_i.fit_transform(data_train_X['Street'])
data_test_new['Street']=label_encoder_i.transform(data_test_new['Street'])
from sklearn.ensemble import GradientBoostingRegressor
grid_gra={'n_estimators':[1500,1000,700],'learning_rate':[0.095,0.085,0.0875],'alpha':[1,0.999,0.997]}
from sklearn.model_selection import GridSearchCV 
model_gb=GradientBoostingRegressor()
gridcv_gra=GridSearchCV(estimator=model_gb,param_grid=grid_gra)
gridcv_gra.fit(data_train_X,data_train_y)
y_prd=gridcv_gra.predict(data_test_new)

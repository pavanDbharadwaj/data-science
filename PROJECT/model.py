import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import RobustScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pickle as pkl
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error,root_mean_squared_error



df=pd.read_csv(r'C:\Users\pavan\Desktop\data science\PROJECT\car_sales_data.csv')


cat_col=df.select_dtypes(exclude='number').columns
df[cat_col]=df[cat_col].astype('category')


x=df.drop(columns='Price')
y=df['Price']


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=42)


num=x_train.select_dtypes(include='number').columns
cat=x_train.select_dtypes(exclude='number').columns


num_pipleline=Pipeline(
    steps=[
        ('num_scaling',RobustScaler())
    ]
)


cat_pipleline=Pipeline(
    steps=[
        ('car_encoding',OneHotEncoder(handle_unknown='ignore'))
    ]
)



preprocessing=ColumnTransformer(
    transformers=[
        ('num_pipleline',num_pipleline,num),
        ('cat_pipleline',cat_pipleline,cat)
    ]
)



main_pipeline=Pipeline(
    steps=[
        ("preprocessing",preprocessing),
        ('model',XGBRegressor())
    ]
)



main_pipeline.fit(x_train,y_train)



main_pipeline.score(x_train,y_train)

main_pipeline.score(x_test,y_test)


y_pred = main_pipeline.predict(x_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", root_mean_squared_error(y_test, y_pred))


with open('model.pkl', 'wb') as file:
    pkl.dump(main_pipeline, file)
    print('successfull')
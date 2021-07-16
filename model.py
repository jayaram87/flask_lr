import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
import pickle

boston_data = load_boston()

df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
df['PRICE'] = boston_data.target

X_train, X_test, y_train, y_test = train_test_split(df[[i for i in df.columns if i != 'PRICE']].values, df['PRICE'].values, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(mean_squared_error(y_pred, y_test))

pickle.dump({'model': model}, open('model'+'.pkl', 'wb'))

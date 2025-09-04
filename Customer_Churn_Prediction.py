import kagglehub
import matplotlib as mat
import pandas as pd
import numpy as np
# Download latest version
#path = kagglehub.dataset_download("rjmanoj/credit-card-customer-churn-prediction")

#print("Path to dataset files:", path)

import os

# print("Files in dataset folder:", os.listdir(path))
df=pd.read_csv("D:\Federico II\Data Mining and Machine Learning\Module B\Campus_X\Churn_Modelling.csv")
print(df)

df.head()
print(df.shape)
df.info()

print("Duplicated values are :",df.duplicated().sum())

print("Customers who left the bank :",df['Exited'].value_counts())

print("cuntories in the dataset",df['Geography'].value_counts())

#removing the starting three columns

df.drop(columns=['RowNumber','CustomerId','Surname'],inplace=True)#inplaces means permanant changes
print(df.head())
# now we have to convert the categorical variable into numerical

#using one hot encoding

df=pd.get_dummies(df,columns=['Geography','Gender'],drop_first=True).astype(int)

#now all data is in numerical form

#now we want to scale the data  

X=df.drop(columns=['Exited']) #temporarily drop the last column and stores remaining into X
y=df['Exited'] 
from sklearn.model_selection import train_test_split
X_train ,X_test, y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=1)


print(X)
print(y)

print(X_train.shape)


from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

print("Scaled training shape:", X_train_scaled.shape)
print("Scaled test shape:", X_test_scaled.shape)

import tensorflow
from tensorflow import keras 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense 


model = Sequential()

model.add(Dense(3,activation='sigmoid',input_dim=11))
model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history =model.fit(X_train_scaled,y_train,epochs=2,validation_split=0.5)


#how to check the weights of the layes

print (model.layers[0].get_weights())

y_log=model.predict(X_test_scaled)

y_pred=np.where(y_log>0.5,1,0)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

import matplotlib.pyplot as plt
print(history.history)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

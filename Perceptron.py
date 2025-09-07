import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


print("Perceptron")

df=pd.read_csv("student_dataset.csv")

print(df.shape)
print(df.head())

sns.scatterplot(x=df['CGPA'],y=df['Resume'],hue=df['Placed'])
#plt.show()

x=df.iloc[:,0:2]
y=df.iloc[:,-1]
print(x,y)

from sklearn.linear_model import Perceptron
p=Perceptron()
p.fit(x,y)

p.coef_ #weights
p.intercept_  #bais

print(p.coef_,p.intercept_)

from mlxtend.plotting import plot_decision_regions
plot_decision_regions(x.values,y.values,clf=p,legend=2)
plt.show()
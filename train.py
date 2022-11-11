import pandas as pd
df = pd.read_csv('cardio_train.csv', sep = ";")

# print("null data:")
# print(df.isnull().sum())

# dp = df.duplicated()
# print("duplicate data: ",dp.sum())



import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# print(df.corr())
sns.heatmap(df.corr(),annot=True)

X = df[['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'active']]
y = df.iloc[: ,-1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)



clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_predict = clf.predict(X_test).ravel()
result = pd.DataFrame({'Actual':y_test["cardio"], 'Predict': y_predict})
print(result)


acc_score = accuracy_score(y_test, y_predict)
print(acc_score)
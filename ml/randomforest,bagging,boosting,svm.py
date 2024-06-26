# -*- coding: utf-8 -*-
"""RandomForest,Bagging,Boosting,SVM.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ztJ2yOkJlksVv3dUca6ex9-S9ESlxbyB
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('/content/customer.csv')
df.head()

df.isna().sum()

df.dropna(inplace=True)
print(df)

print(df.nunique())

for col in df.columns:
    print(f"Value counts in column {col}: {df[col].value_counts()}")

df['Education'] = df['Education'].map({'Hot':0,'Mild':1,'Cool':2})

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Education'] = le.fit_transform(df['Education'])
df['Education'].unique()

df['Property_Area'] = le.fit_transform(df['Property_Area'])
df['Self_Employed'] = le.fit_transform(df['Self_Employed'])

df['Gender'] = le.fit_transform(df['Gender'])
df['Married'] = le.fit_transform(df['Married'])

df['Loan_Status'] = le.fit_transform(df['Loan_Status'])

df['Dependents'] = le.fit_transform(df['Dependents'])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
cols=['ApplicantIncome','CoapplicantIncome','LoanAmount']
for col in cols:
    df[col]=sc.fit_transform(df[[col]])

df.head()

X = df.drop(['Loan_Status','Loan_ID'],axis=1)
y = df['Loan_Status']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
rfmodel = RandomForestClassifier(n_estimators=50)
dec_model = DecisionTreeClassifier()

"""()"""

dec_model.fit(X_train,y_train)

rfmodel.fit(X_train,y_train)

rfmodel.score(X_test,y_test)

dec_model.score(X_test,y_test)

y_pred = rfmodel.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test,y_pred))

y_pred = dec_model.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test, y_pred)
print(cm)
import seaborn as sns
# f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(cm, annot=True, fmt="d")
plt.title('Confusion matrix of the classifier')
plt.xlabel('Predicted')
plt.ylabel('True')

cm = confusion_matrix(y_test,y_pred)
import seaborn as sns
sns.heatmap(cm,annot=True,fmt='d')
plt.title("Confusion matrix")
plt.xlabel('Predicted')
plt.ylabel('Actual')

!pip install graphviz pydotplus

import matplotlib.pyplot as plt
from sklearn import tree
fig = plt.figure(figsize=(15,10))
_ = tree.plot_tree(dec_model,
                   feature_names=X_train.columns,
                   class_names=['Class 0', 'Class 1'],
                   filled=True)
plt.show()

import matplotlib.pyplot as plt
from sklearn import tree
fig = plt.figure(figsize=(15,10))
_ = tree.plot_tree(dec_model,feature_names = X_train.columns,class_names=['Class 0','Class 1'],filled=True)
plt.show()

fig.savefig('decision_tree.png')

from sklearn.svm import SVC
sv_linear = SVC(kernel='linear')
sv_rbf = SVC(kernel='rbf')
sv_gaussian = SVC(kernel='rbf',gamma='scale')

sv_linear.fit(X_train,y_train)

sv_rbf.fit(X_train,y_train)

sv_gaussian.fit(X_train,y_train)

sv_linear.score(X_test,y_test)

sv_rbf.score(X_test,y_test)

sv_gaussian.score(X_test,y_test)

y_pred = sv_linear.predict(X_test)
cmsv = confusion_matrix(y_test,y_pred)
sns.heatmap(cmsv,annot=True,fmt="d")
plt.title('Confusion matrix of the classifier')
plt.xlabel('Predicted')
plt.ylabel('True')

def bagging():

  indexes = np.random.randint(0, len(X_train), len(X_train))
  bag = X_train[indexes]
  target = y_train[indexes]
  return (bag,target)

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
tree = DecisionTreeClassifier()
bagging_clf = BaggingClassifier(base_estimator=tree, n_estimators=1500, random_state=42)
bagging_clf.fit(X_train, y_train)
def evaluate(model, X_train, X_test, y_train, y_test):
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    print("TRAINIG RESULTS: \n===============================")
    clf_report = pd.DataFrame(classification_report(y_train, y_train_pred, output_dict=True))
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_train, y_train_pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(y_train, y_train_pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")

    print("TESTING RESULTS: \n===============================")
    clf_report = pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True))
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, y_test_pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(y_test, y_test_pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")

evaluate(bagging_clf, X_train, X_test, y_train, y_test)

from sklearn.ensemble import AdaBoostClassifier

ada_boost_clf = AdaBoostClassifier(n_estimators=30)
ada_boost_clf.fit(X_train, y_train)
evaluate(ada_boost_clf, X_train, X_test, y_train, y_test)


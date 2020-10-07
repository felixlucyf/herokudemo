import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
import pickle

col_names = ['age', 'parents', 'bmi', 'ailment', 'financial', 'selfeval', 'pscore', 'gender', 'activities', 'family','label']
pdf = pd.read_csv(r"C:\mydata.csv", header = None, names = col_names)
pdf.head()
feature_cols = ['parents', 'family','ailment','financial','activities', 'selfeval', 'gender','pscore']
X = pdf[feature_cols] # Features
y = pdf.label # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.21, random_state = 109)
clf = svm.SVC(kernel='linear')
clf = clf.fit(X_train,y_train)

pickle.dump(clf,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
tmp=[1,5,1,1,1,1,1,5]
print(model.predict([tmp]))



y_pred = clf.predict(X_test)
from sklearn import metrics
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("Precision: ",metrics.precision_score(y_test, y_pred))
print("Recall: ",metrics.recall_score(y_test, y_pred))
print("Log-Loss: ", metrics.log_loss(y_test, y_pred))
print("Confusion Matrix: ", metrics.confusion_matrix(y_test,y_pred))
print("Area Under ROC Curve: ", metrics.roc_auc_score(y_test,y_pred))



import time
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#%%
iris = datasets.load_iris()

X = iris.data[:, :] 
y = iris.target
#%%


X_train, X_test, y_train, y_test = train_test_split(
    X, y,test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#%%
t0 = time.time()

lrc = LogisticRegression(n_jobs=-1,solver='lbfgs', 
    multi_class='multinomial')

lrc.fit(X_train, y_train)
y_pred = lrc.predict(X_test)
model_acc = lrc.score(X_test, y_test)
test_acc = accuracy_score(y_test, y_pred)

print('\nLRC Trained Classifier Accuracy: ', model_acc)
print('\nPredicted Values: ',y_pred)
print('\nAccuracy of Classifier on Validation Images: ',test_acc)
run_time = time.time() - t0
print("\nTook ", run_time, "Seconds to complete")

#%%
t0 = time.time()

rfc = RandomForestClassifier(n_jobs=-1, n_estimators=100)

rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
model_acc = rfc.score(X_test, y_test)
test_acc = accuracy_score(y_test, y_pred)

print('\nRFC Trained Classifier Accuracy: ', model_acc)
print('\nPredicted Values: ',y_pred)
print('\nAccuracy of Classifier on Validation Images: ',test_acc)
run_time = time.time() - t0
print("\nTook ", run_time, "Seconds to complete")

#%%
t0 = time.time()

svc = LinearSVC()

svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
model_acc = svc.score(X_test, y_test)
test_acc = accuracy_score(y_test, y_pred)

print('\nSVM Trained Classifier Accuracy: ', model_acc)
print('\nPredicted Values: ',y_pred)
print('\nAccuracy of Classifier on Validation Images: ',test_acc)
run_time = time.time() - t0
print("\nTook ", run_time, "Seconds to complete")

#%%



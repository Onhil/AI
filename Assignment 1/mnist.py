#%%
import time
from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#%%
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

#%%
train_samples = 10000


X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_samples, test_size=10000)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#%%
t0 = time.time()

lrc = LogisticRegression(multi_class='multinomial',penalty='l2', 
            solver='saga', tol=0.1, max_iter=1000, n_jobs=-1)

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

rfc = RandomForestClassifier(n_jobs=-1, n_estimators=500,)

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

svc = SVC(gamma=0.001, kernel="poly", C=100)

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

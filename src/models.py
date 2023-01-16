from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def multinomialnb(X_train, y_train, X_test):
    global y_predicted1
    clf = Pipeline([  ('clf', MultinomialNB())])
    clf.fit(X_train, y_train)
    y_predicted1 = clf.predict(X_test)
    return y_predicted1

def svc(X_train, y_train, X_test):
    global y_predicted2
    clf = Pipeline([  ('clf', SVC())])
    clf.fit(X_train, y_train)
    y_predicted2 = clf.predict(X_test)
    return y_predicted2

def logReg(X_train, y_train, X_test):
    global y_predicted3
    clf = Pipeline([  ('clf', LogisticRegression())])
    clf.fit(X_train, y_train)
    y_predicted3 = clf.predict(X_test)
    return y_predicted3

def evaluate(y_predicted, y_test):
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_predicted))


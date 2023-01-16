from data_extract import *
from data_processing import *
from models import *
from traintest_split import *

df=extract_data("../data/failure.db")
df=remove_duplicates(df)
df=dropCarID(df)
df=dropnull(df)
df=convert_model(df)
df=convert_factory(df)
df=textToNumbers(df)
df=dropUnrelatedColumns(df)
X_train, X_test, y_train, y_test=traintest_split(df)
X_train, y_train=balance_data(X_train, y_train)
y_predicted1=multinomialnb(X_train, y_train, X_test)
y_predicted2=svc(X_train, y_train, X_test)
y_predicted3=logReg(X_train, y_train, X_test)
evaluate(y_predicted1, y_test)
evaluate(y_predicted2, y_test)
evaluate(y_predicted3, y_test)
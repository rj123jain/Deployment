import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
# Every data you work uoy have path in local right now that pacticula file have some server
url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv" 
print('imported everything')
# Creating all the column names
names=['preg','plas','pres','skin','test','mass','pedi','age','class']
df=pd.read_csv(url,names=names)
print(df)
array=df.values
X=array[:,0:8]
y=array[:,8]
X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2,random_state=101)
model=LogisticRegression()
model.fit(X_train,y_train)

#accuracy
result=model.score(X_test,y_test)
print(f'the accuracy of the resule is {result}')

# At the end we save the model
pickle.dump(model,open('dibatic_80.pkl','wb'))# the is call model saving

# differce  in pickle and joblib is 
# syntax rest all are same

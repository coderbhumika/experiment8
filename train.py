import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df=pd.read_csv("Iris.csv")

df=df.drop("Id",axis=1)

x=df.drop("Species",axis=1)
y=df['Species']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)

IrisPredModel=LogisticRegression()

IrisPredModel.fit(x_train,y_train)

pred=IrisPredModel.predict(x_test)
print("Prediction of First Sample:",pred[0])

accuracy=accuracy_score(pred,y_test)
print("Accuracy is: ", accuracy)

joblib.dump(IrisPredModel, "model.pkl")  #model saving using joblib
joblib.load("model.pkl") #model loading using joblib

print("Model saved and loaded successfully")
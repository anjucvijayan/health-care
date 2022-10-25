from tkinter import Y
import pandas as pd
import numpy as np
import pickle



#reading dataset
data = pd.read_csv("healthcare-dataset-stroke-data.csv")
data.drop(['id'],axis=1,inplace=True)

#handling missing values
data['bmi']=data['bmi'].fillna(data['bmi'].median())
#handling outliers
for col in ['avg_glucose_level','bmi']:
    data[col]=np.log(data[col])

#encoding data
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
data['smoking_status'] =label_encoder.fit_transform(data['smoking_status'])
data['work_type'] = label_encoder.fit_transform(data['work_type'])
data['gender'] =label_encoder.fit_transform(data['gender'])
data = pd.get_dummies(data,columns=['ever_married','Residence_type'],drop_first=True)


#splitting data
x=data.drop('stroke',axis=1)
y=data['stroke']

#scaling unsing minmax scalar
#from sklearn import preprocessing
#min_max = preprocessing.MinMaxScaler(feature_range=(0,1))
#x=min_max.fit_transform(x)
#x=pd.DataFrame(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state = 42)
#print(x_test)

#print(x)
#SVM Model
from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state = 0)
model =classifier.fit(x_train,y_train)

pickle.dump(classifier,open('model.pkl','wb'))


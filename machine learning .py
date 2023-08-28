#this python program is for  getting the data set using  the pandas module and the  segeration it into testing and training data
import pandas as pd


#this is used for the reqading the csv fil from the git hub using the link
df= pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')
print(df)
# printing the url data  in the python




#segration the data from the csv for the convenince


x=df.drop('logS',axis=1)#this Drop function helps us to drop the particular column from the data set and shows the remaining  data
print(x)
y=df['logS']
print(y)#printing logS alone



#importing thr sklearn for the segrating the training and data set  for the model

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 200)
print("training set data")
print(x_train)




#MODEL BUILDING  (MAIN)

#linear regression

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train,y_train)


#applying the model to make the predection

y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)



#evaluating the model performance

from sklearn.metrics import mean_squared_error,r2_score
lr_train_mse = mean_squared_error(y_train,y_lr_train_pred)
lr_train_r2 = r2_score(y_train,y_lr_train_pred)


lr_test_mse = mean_squared_error(y_test,y_lr_test_pred)
lr_test_r2 = r2_score(y_test ,y_lr_test_pred)

lr_results = pd.DataFrame(['Linear regression',lr_train_mse,lr_train_r2,lr_test_mse,lr_test_r2]).transpose()
lr_results.colums = ["methods","training mse","traning r2","test mse","test r2"]

print(lr_results)
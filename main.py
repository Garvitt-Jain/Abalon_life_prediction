# -*- coding: utf-8 -*-
"""Ds3-As5

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ICJynJaXT-magZRb2MDGHt1o9h2cscZz

Data Science Assignment - 5
"""

# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Question 1
# Reading the csv file
#reading given csv life   
steelplatefaults=pd.read_csv('SteelPlateFaults-2class.csv')
#dividing the given csv file on the basic of classes i.e 0 and 1 
steelplatefaults_label_0=steelplatefaults[steelplatefaults['Class']==0]
steelplatefaults_label_1=steelplatefaults[steelplatefaults['Class']==1]
#splitting each class data into train and test data into the ration 70:30
[x_train_0,x_test_0,x_label_train_0,x_label_test_0]=train_test_split(steelplatefaults_label_0,steelplatefaults_label_0['Class'],test_size=0.3,random_state=42,shuffle=True)
[x_train_1,x_test_1,x_label_train_1,x_label_test_1]=train_test_split(steelplatefaults_label_1,steelplatefaults_label_1['Class'],test_size=0.3,random_state=42,shuffle=True)
#concatinating each class data into a single train_data
train_data=pd.concat([x_train_0,x_train_1],axis=0)

train_data_label=pd.concat([x_label_train_0,x_label_train_1],axis=0)
#concatinating each#concatinating each class train data label into a single train_data class data into a single train_data
test_data=pd.concat([x_test_0,x_test_1],axis=0)
#concatinating each class test data label into a single test_data
test_data_label=pd.concat([x_label_test_0,x_label_test_1],axis=0)
Ftest_df =  test_data
Ftrain_df = train_data
y_test = Ftest_df["Class"]
F_train0_df = Ftrain_df[Ftrain_df["Class"]==0]
F_train1_df = Ftrain_df[Ftrain_df["Class"]==1]
F_train0_df = F_train0_df.drop(['X_Minimum', 'Y_Minimum', 'TypeOfSteel_A300', 'TypeOfSteel_A400',"Class"],axis = 1)
F_train1_df = F_train1_df.drop(['X_Minimum', 'Y_Minimum', 'TypeOfSteel_A300', 'TypeOfSteel_A400',"Class"],axis = 1)
x_test = Ftest_df.drop(['X_Minimum', 'Y_Minimum', 'TypeOfSteel_A300', 'TypeOfSteel_A400',"Class"],axis = 1)
Prior_prob0 = len(F_train0_df.index)/len(Ftrain_df.index)
Prior_prob1 = len(F_train1_df.index)/len(Ftrain_df.index)
warnings.filterwarnings("ignore")

# No of gaussian components
Q_list = [2,4,8,16]
for q in Q_list:
  # Making gaussian mixture model for class 0 and class 1
  GMM0 = GaussianMixture(n_components=q, covariance_type='full',reg_covar=0.00001)
  GMM1 = GaussianMixture(n_components=q, covariance_type='full',reg_covar=0.00001)
  # Fitting the gaussian mixture model for each class on train data.
  GMM0.fit(F_train0_df)
  GMM1.fit(F_train1_df)
  # Calculating logarithmic probabilities  for test data
  lp_0 = GMM0.score_samples(x_test)
  lp_1 = GMM1.score_samples(x_test)
  # Making emplty list to append predicted class labels
  y_pred = []
  for x0,x1 in zip(lp_0,lp_1):
  # Test vectors for class 0 and 1
    x0 = np.exp(x0)
    x1 = np.exp(x1)
    postprob0 = (x0*Prior_prob0)/(x1*Prior_prob1 + x0*Prior_prob0)
    postprob1 = (x1*Prior_prob1)/(x1*Prior_prob1 + x0*Prior_prob0)
    
    # Classifying according to posterior probability.
    if(postprob0>postprob1):
      y_pred.append(0)    
    else:
      y_pred.append(1)
  # Converting list into numpy array
  y_pred = np.array(y_pred)
  print("----------------------- --------------------------------=-------------------------------------------")
  # Printing confusion matrix of prediction
  print("The confusion matrix for the above bayes classification is : ")
  # Printing accuracy score of prediction
  print((confusion_matrix(y_test, y_pred)))
  print("The  classification accuracy percentage for this bayes classification  is "+str(accuracy_score(y_test, y_pred)*100))
  print("----------------------- --------------------------------=-------------------------------------------")

# Building a data frame for comparing result of knn , knn with normalised data, bayes classifier.

d =[89.58 ,97.023,93.819,95.23]
index=['knn','knn with normalized data','bayes classifier','bayes classifier using GMM']
df=pd.DataFrame(d,index=index)
df.columns=['Accuracy in percentage']
fig = plt.figure(figsize = (8, 2))
ax = fig.add_subplot(111)

ax.table(cellText = df.values,
          rowLabels = df.index,
          colLabels = df.columns,
          loc = "center"
         )
ax.set_title("Comparison of classification accuracy")

ax.axis("off");

# Defining function to calculate root mean square error between predicted and observed data values.
def calrmse(pred , observed):
  error = [(p - o) for p, o in zip(pred, observed)]
  square_error = [e**2 for e in error]
  mean_square_error = sum(square_error)/len(square_error)
  root_mean_square_error = mean_square_error**0.5
  return root_mean_square_error

# Question 1
# Reading the csv file  
df_abalone =pd.read_csv('abalone.csv')
#splitting each class data into train and test data into the ratio 70:30
df_abalone_train,df_abalone_test =train_test_split(df_abalone,test_size=0.3,random_state=42,shuffle=True)
# Saving the test and train data as  csv file.
df_abalone_train.to_csv('abalone-train.csv',index=False)
df_abalone_test.to_csv('abalone-test.csv',index=False)

# Q1 Simple linear regression
# Calculating correlation coefficient of all attributes with no of Rings.
corr = dict(df_abalone_train[df_abalone_train.columns[:]].corr()['Rings'][:])
# Removing The key rings.
corr.pop("Rings")
# Finding the attribute which is highest correlated to rings.
print("The highest ccorrelation coefficient  is  "+str(max(corr.values()))+" corresponding to attribute - "+str(max(corr,key = corr.get)))
#Q1) A -part Plotting the best fit line on training data.
plt.figure(figsize=(5,5))
plt.scatter(df_abalone_train["Shell weight"],df_abalone_train["Rings"])
m = (np.cov(df_abalone_train["Shell weight"],df_abalone_train["Rings"])[0][1]/np.var(df_abalone_train["Shell weight"]))
i = np.mean(df_abalone_train["Rings"]) - m*np.mean(df_abalone_train["Shell weight"])
y = m*df_abalone_train["Shell weight"] + i
plt.plot(df_abalone_train["Shell weight"],y)
plt.legend(["Best fit Line","Scateer plot"])
plt.xlabel('Shell weight ')
plt.ylabel('No of Rings on the Shell')
plt.title("Scatter plot between Shell weight and no of rings on Shell")
plt.grid(True)
print("----------------------------------------------------------------------------------------------")
# Building simple regression model using Shell weight
reg = LinearRegression().fit(np.array(df_abalone_train["Shell weight"]).reshape(-1,1),df_abalone_train["Rings"] )
#Q1 Part2 Calculating rmse for train data
y_pred_train = reg.predict(np.array(df_abalone_train["Shell weight"]).reshape(-1,1)) 
print("The root mean square eroor for predicting no of rings in train data using simple regression model is "+str(calrmse(list(y_pred_train),list(df_abalone_train["Rings"]))))
print("----------------------------------------------------------------------------------------------")
#Q1 Part3 Calculating rmse for test data
y_pred_test = reg.predict(np.array(df_abalone_test["Shell weight"]).reshape(-1,1)) 
print("The root mean square eroor for predicting no of rings in test data using simple regression model is "+str(calrmse(list(y_pred_test),list(df_abalone_test["Rings"]))))
print("----------------------------------------------------------------------------------------------")
# Q1 Part4 Scatter plot between predicted rings and and actual rings in test data
plt.figure(figsize=(5, 5))
plt.scatter(df_abalone_test["Rings"],y_pred_test)
plt.xlabel('Actual no of rings on abalone ')
plt.ylabel('Predicted No of Rings on the Shell')
plt.title("Scatter plot between Actual no of rings and predicted no of rings using Simple linear regression model")
plt.grid(True)

# Q2 Multivariate linear regression model using all the attributes except ring
reg = LinearRegression().fit(np.array(df_abalone_train.loc[:,:"Shell weight"]),df_abalone_train["Rings"] )
#Q2 Part1 Calculating rmse for train data in Multivariate linear regression model
y_pred_train = reg.predict(np.array(df_abalone_train.loc[:,:"Shell weight"]))
print("The root mean square eroor for predicting no of rings in train data using Multiple linear regression model is "+str(calrmse(list(y_pred_train),list(df_abalone_train["Rings"]))))
print("----------------------------------------------------------------------------------------------")
#Q2 Part2 Calculating rmse for test data using  Multivariate linear regression model
y_pred_test = reg.predict(np.array(df_abalone_test.loc[:,:"Shell weight"]))
print("The root mean square eroor for predicting no of rings in test data using Multiple linear regression model is "+str(calrmse(list(y_pred_test),list(df_abalone_test["Rings"]))))
print("----------------------------------------------------------------------------------------------")
# Q2 Part3 Scatter plot between predicted rings and and actual rings in test data using Multivariate linear regression model
plt.figure(figsize=(5, 5))
plt.scatter(df_abalone_test["Rings"],y_pred_test)
plt.xlabel('Actual no of rings on abalone ')
plt.ylabel('Predicted No of Rings on the Shell')
plt.title("Scatter plot between Actual no of rings and predicted no of rings using Multiple linear regression model ")
plt.grid(True)

# Q3) building  a simple nonlinear regression model using polynomial curve  fitting to predict Rings.
p_value = [2,3,4,5]
# Dictionaries to store corresponding rmse values for degree
rmse_train ={}
rmse_test = {}
#List to store the most correct predicted values.
cpredtrain = []
cpredtest = []
# Sample data for finding curve.
myline = np.linspace(0,1,1000)
# Building regression model for each value of p
for d in p_value:
  poly_features = PolynomialFeatures(d)
  # Fitting the data acccording to p
  x_poly_train = poly_features.fit_transform(np.array(df_abalone_train["Shell weight"]).reshape(-1,1))
  x_poly_test = poly_features.fit_transform(np.array(df_abalone_test["Shell weight"]).reshape(-1,1))
  regressor = LinearRegression()
  regressor.fit(x_poly_train,df_abalone_train["Rings"])
  # Predecting the no of rigs in train and test data
  y_pred_train = regressor.predict(x_poly_train)
  y_pred_test = regressor.predict(x_poly_test)
  acc_train = calrmse(list(y_pred_train),list(df_abalone_train["Rings"]))
  acc_test = calrmse(list(y_pred_test),list(df_abalone_test["Rings"]))
  #Q3 Part 1 Finding rmse of train data acccording to p
  print("The root mean square eroor for predicting no of rings in train data using simple polynomial regression model of degree "+str(d)+" is "+str(acc_train))
  print("----------------------------------------------------------------------------------------------")
  #Q3 Part 2 Finding rmse of test data acccording to p
  print("The root mean square eroor for predicting no of rings in test data using simple regression model of degree "+str(d)+" is "+str(acc_test))
  print("----------------------------------------------------------------------------------------------")
  rmse_train[d] = (calrmse(y_pred_train,df_abalone_train["Rings"]))
  rmse_test[d] = (calrmse(y_pred_test,df_abalone_test["Rings"])) 
  # Checkig if the degree is corresponding to minimum RMSE.
  if acc_train == min(rmse_train.values()):
    cpredtrain = y_pred_train
    # Predecting the y values to plot best fit curve
    y = regressor.predict(poly_features.fit_transform(myline.reshape(-1,1)))
  if acc_test == min(rmse_test.values()):
    cpredtest = y_pred_test
# Finding the best value of p to be chosen for test and train data.
print("----------------------------------------------------------------------------------------------")
print("The minimum root mean square eroor for predicting no of rings of train data in Simple NON Linear regression model  is  "+str(min(rmse_train.values()))+" corresponding to degree d equal to "+str(min(rmse_train,key = rmse_train.get)))
print("----------------------------------------------------------------------------------------------")
print("The minimum root mean square eroor for predicting no of rings  of test data Simple NON Linear regression model is  "+str(min(rmse_test.values()))+" corresponding to degree d equal to "+str(min(rmse_test,key = rmse_test.get)))
# Creating the bar plot of RMSE  for train data predicted using Simple NON Linear regression model
plt.figure(figsize=(5,5))
plt.bar(list(rmse_train.keys()), rmse_train.values(), color ='maroon',width = 0.4)
plt.xlabel("Degree of polynomial used for Polynomial regression for train data")
plt.ylabel("Root mean square error in predicting the rings for train data")
plt.title("Bar plot of Root mean square error for polynomial degree used for regression for train data")
plt.show()
# Creating the bar plot of RMSE  for test data predicted using Simple NON Linear regression model
plt.figure(figsize=(5,5))
plt.bar(rmse_test.keys(), rmse_test.values(), color ='maroon',width = 0.4)     
plt.xlabel("Degree of polynomial used for Polynomial regression for test data")
plt.ylabel("Root mean square error in predicting the rings for test data")
plt.title("Bar plot of Root mean square error for polynomial degree used for regression for test data")
plt.show()
# Q3 Part3 Plotting The best fit curve on training data predicted using Simple NON Linear regression model
plt.figure(figsize=(5,5))
plt.scatter(df_abalone_train["Shell weight"],df_abalone_train["Rings"])
plt.plot(myline,y,color='red')
plt.legend(["Best fit Curve","Scateer plot"])
plt.xlabel('Shell weight ')
plt.ylabel('No of Rings on the Shell')
plt.title("Scatter plot between Shell weight and no of rings on Shell")
plt.grid(True)
# Q3 Part4 Scatter plot betwenn predicted no of rings and actual no of rings in test data predicted using Simple NON Linear regression model
plt.figure(figsize=(5, 5))
plt.scatter(df_abalone_test["Rings"],cpredtest)
plt.xlabel('Actual no of rings on abalone ')
plt.ylabel('Predicted No of Rings on the Shell')
plt.title("Scatter plot between Actual no of rings and predicted no of rings using Simple Polynomial regression model")
plt.grid(True)

# Q4 building a multivariate non - linear regression model using polynomial regression
# Part1 
# Q4) building  a Multiple nonlinear regression model using polynomial curve  fitting to predict Rings.
p_value = [2,3,4,5]
# Dictionaries to store corresponding rmse values for degree
rmse_train ={}
rmse_test = {}
#List to store the most correct predicted values.
cpredtrain = []
cpredtest = []


for d in p_value:
  poly_features = PolynomialFeatures(d)
  x_poly_train = poly_features.fit_transform(np.array(df_abalone_train.loc[:,:"Shell weight"]))
  x_poly_test = poly_features.fit_transform(np.array(df_abalone_test.loc[:,:"Shell weight"]))
  regressor = LinearRegression()
  regressor.fit(x_poly_train,df_abalone_train["Rings"])
  y_pred_train = regressor.predict(x_poly_train)
  y_pred_test = regressor.predict(x_poly_test)
  # Finding accuracy as RMSE
  acc_train = calrmse(list(y_pred_train),list(df_abalone_train["Rings"]))
  acc_test = calrmse(list(y_pred_test),list(df_abalone_test["Rings"]))
  #Q4 Part 1 Finding rmse of train data acccording to p using Multiple NON linear regression model
  print("The root mean square eroor for predicting no of rings in train data using Multiple polynomial regression model of degree "+str(d)+" is "+str(acc_train))
  print("----------------------------------------------------------------------------------------------")
  #Q4 Part 2 Finding rmse of test data acccording to p using Multiple NON linear regression model
  print("The root mean square eroor for predicting no of rings in test data using Multiple regression model of degree "+str(d)+" is "+str(acc_test))
  print("----------------------------------------------------------------------------------------------")
  rmse_train[d] = (calrmse(y_pred_train,df_abalone_train["Rings"]))
  rmse_test[d] = (calrmse(y_pred_test,df_abalone_test["Rings"]))
  # Checkig if the degree is corresponding to minimum RMSE.
  if acc_train == min(rmse_train.values()):
    cpredtrain = y_pred_train
    
  if acc_test == min(rmse_test.values()):
    cpredtest = y_pred_test
# Finding the best value of p to be chosen for test and train data prediced using Multiple polynomial regression model
print("----------------------------------------------------------------------------------------------")
print("The minimum root mean square eroor for predicting no of rings of train data is  "+str(min(rmse_train.values()))+" corresponding to degree d equal to "+str(min(rmse_train,key = rmse_train.get)))
print("----------------------------------------------------------------------------------------------")
print("The minimum root mean square eroor for predicting no of rings  of test data is  "+str(min(rmse_test.values()))+" corresponding to degree d equal to "+str(min(rmse_test,key = rmse_test.get)))
# creating the bar plot of rmse for train data  prediced using Multiple polynomial regression model
plt.figure(figsize=(5,5))
plt.bar(list(rmse_train.keys()), rmse_train.values(), color ='maroon',width = 0.4)
plt.xlabel("Degree of polynomial used for Polynomial regression for train data")
plt.ylabel("Root mean square error in predicting the rings for train data")
plt.title("Bar plot of Root mean square error for polynomial degree used for regression for train data")
plt.show()
# creating the bar plot of rmse  for test data  prediced using Multiple polynomial regression model
plt.figure(figsize=(5,5))
plt.bar(rmse_test.keys(), rmse_test.values(), color ='maroon',width = 0.4)     
plt.xlabel("Degree of polynomial used for Polynomial regression for test data")
plt.ylabel("Root mean square error in predicting the rings for test data")
plt.title("Bar plot of Root mean square error for polynomial degree used for regression for test data")
plt.show()
# Q4 PArt 3 Scatter plot betwenn predicted no of rings and actual no of rings in test data
plt.figure(figsize=(5, 5))
plt.scatter(df_abalone_test["Rings"],cpredtest)
plt.xlabel('Actual no of rings on abalone ')
plt.ylabel('Predicted No of Rings on the Shell')
plt.title("Scatter plot between Actual no of rings and predicted no of rings using Multiple Polynomial regression model")
plt.grid(True)
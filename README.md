
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df=pd.read_csv("/content/Churn_Modelling.csv", index_col="RowNumber")
df

df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df

df.isnull().sum()

df.duplicated()

df.describe()

scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1

x=df1.iloc[:,:-1].values
x

y=df1.iloc[:,-1].values
y

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))

## OUTPUT:
DATASET:
![420863162-a35e4788-9992-4790-b149-8ff8eba05b67](https://github.com/user-attachments/assets/5abab1a6-db8f-4ea3-bd85-4e2ec465607b)

DROPPING THE UNWANTED DATASET:
![420863273-7529e14c-021c-44e0-b686-e2db15804ead](https://github.com/user-attachments/assets/82d4dcf4-8143-40d2-8b27-1a3c21bf79e6)

CHECKING NULL VALUES:
![420863308-51983d5f-602f-491d-a82c-793a68658267](https://github.com/user-attachments/assets/1a089849-038d-47af-9e3f-dfe6adf6f639)

CHECKING FOR DUPLICATION:
![420863342-888daf01-95f4-4c83-ad80-7465cc4cfe7b](https://github.com/user-attachments/assets/40c31e65-8153-41af-aea2-e92c95866b9e)

DESCRIBING THE DATASET:
![420863380-1a633ece-a020-487c-9187-f1ed5607ddba](https://github.com/user-attachments/assets/e8450524-dd61-40eb-8929-97a88545b19d)

SCALING THE DATASET:
![420863400-e6025ef5-f15f-468c-b124-535417f6ca23](https://github.com/user-attachments/assets/68adf7a4-b7f3-4998-8022-267fbe5a3169)

X FEATURES:
![420863422-6a279396-d46c-4299-9dcd-02c2a1d638d1](https://github.com/user-attachments/assets/482f4215-4a5a-4d54-a705-261431fd5a1b)

Y FEATURES:
![420863446-49cf7fe8-f7e8-4253-829b-23ca374dd9a1](https://github.com/user-attachments/assets/0ea3848b-5823-4f2f-a78a-04d6e2a7694a)

SPLITTING THE TRAINING AND TESTING DATASET:
![420863473-e04487b2-f0d5-4ed8-9cf5-13f08887687d](https://github.com/user-attachments/assets/80d986fa-2ce5-4bdc-bba9-cfa2e92d44a1)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.




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
![1 1](https://github.com/user-attachments/assets/16d8cdb9-0227-497e-b618-6e19064385d0)

DROPPING THE UNWANTED DATASET:
![Screenshot (727)](https://github.com/user-attachments/assets/672ed404-7028-4298-911e-bdcbad5b277d)

CHECKING NULL VALUES:
![Screenshot (728)](https://github.com/user-attachments/assets/77c629b8-7af3-4314-acb7-ae2b5d44b3df)


CHECKING FOR DUPLICATION:
![Screenshot (729)](https://github.com/user-attachments/assets/c48c3b8f-1d42-4fe4-847d-219808855e4a)

DESCRIBING THE DATASET:
![Screenshot (730)](https://github.com/user-attachments/assets/16c8ee20-1e5b-49df-a69b-ddfcdb2cbe28)

SCALING THE DATASET:
![Screenshot (731)](https://github.com/user-attachments/assets/e7dc6217-709c-4462-b050-b1a9db601777)

X FEATURES:
![Screenshot (732)](https://github.com/user-attachments/assets/24bc5efb-5186-444b-b518-9c9dd57c0cb7)

Y FEATURES:
![Screenshot (733)](https://github.com/user-attachments/assets/48b50547-e2f4-4084-87bd-4c5a04b79582)

SPLITTING THE TRAINING AND TESTING DATASET:
![Screenshot (734)](https://github.com/user-attachments/assets/7bf68830-c321-4a55-bee5-1a2f7a443d30)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.



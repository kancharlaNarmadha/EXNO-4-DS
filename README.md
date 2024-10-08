# EXNO:4-DS

# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv")
data
```
![image](https://github.com/user-attachments/assets/b278f507-4d60-480b-974d-ba110582958b)

```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/8a194726-3e20-4a4e-b810-f7de97556862)

```
missing=data[data.isnull().any(axis=1)]
missing
```

![image](https://github.com/user-attachments/assets/c67d4618-c9ba-4533-8f1e-c2db0813e5f7)

```
data2=data.dropna(axis=0)
data2
```

![image](https://github.com/user-attachments/assets/30f76e0c-4468-4e6b-b24a-9eb8ad9f3f59)

```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```

![image](https://github.com/user-attachments/assets/fc456f8f-0822-4a36-b628-6138b4aff0fc)

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/85a30de9-5196-4f62-b96f-0efaf45d20cc)

```
data2
```

![image](https://github.com/user-attachments/assets/7c880916-10c1-4368-968e-fddbff141faf)

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```

![image](https://github.com/user-attachments/assets/59d7ebe8-8a35-4dcb-a408-014fc1839567)

```
columns_list=list(new_data.columns)
print(columns_list)
```

![image](https://github.com/user-attachments/assets/fa100185-2f95-4f33-92e4-7b425678ec78)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```

![image](https://github.com/user-attachments/assets/377c8b48-c6fb-48cc-8f71-aeea894be090)
```
y=new_data['SalStat'].values
print(y)
```

![image](https://github.com/user-attachments/assets/2571c16a-8536-43c5-88b1-44ce24f10d35)

```
x=new_data[features].values
print(x)
```
![image](https://github.com/user-attachments/assets/ee7e65b5-c64e-432c-86ea-fb610a97b78d)

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```

![image](https://github.com/user-attachments/assets/02e8056f-29f3-43bd-9097-f560fc401463)

```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```

![image](https://github.com/user-attachments/assets/5f9bfcd0-23ce-443e-bfb2-b41364b698dc)

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```

![image](https://github.com/user-attachments/assets/f6047509-3b9a-4277-94d6-e51f7ae08358)

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```

![image](https://github.com/user-attachments/assets/5fb890ff-46b4-4511-a1af-b423eb363928)

```
data.shape
```

![image](https://github.com/user-attachments/assets/113986c0-6ec4-4fef-9bc6-b5ff0194981f)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```

![image](https://github.com/user-attachments/assets/5382bbed-79ae-4832-a252-9b904a3f3e2c)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

![image](https://github.com/user-attachments/assets/b3ac63e5-74cc-4a33-8114-00a7f756d7da)

```
tips.time.unique()
```

![image](https://github.com/user-attachments/assets/9937f476-be0b-4cbc-8ddb-6a20d7ac0868)

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

![image](https://github.com/user-attachments/assets/815c9dc7-3418-4ab7-be8e-61718cc6edbb)

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```

![image](https://github.com/user-attachments/assets/f86093e2-9a12-4907-bffa-206761482db7)


# RESULT:
       Thus, Feature selection and Feature scaling has been used on thegiven dataset.

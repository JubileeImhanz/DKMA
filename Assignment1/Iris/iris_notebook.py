#!/usr/bin/env python
# coding: utf-8

# # ECE 657A ASSIGNMENT 1
# ## Iris Dataset
# 
# ###### Jubilee Imhanzenobe and Olohireme Ajayi

# In[525]:


# importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


# In[526]:


# defining function for plotting correlation heatmap
def plot_heatmap(correlation):
    plt.figure(figsize=(15, 8))
    ax = sns.heatmap(correlation,annot=True,fmt='.3f',linewidths=0.3,annot_kws={"size": 18})
    plt.xticks(fontsize=12) 
    plt.yticks(fontsize=12) 
    plt.title('Correlation between features', fontsize=20)
    ax.figure.axes[-1].tick_params(labelsize=18) # To increase fontsize of colorbar ticks
    lim = len(correlation.columns)
    ax.set_ylim([0,lim]) # to make the map display correctly without trimming the edges
    plt.show()
    return


# In[527]:


# importing the dataset
data = pd.read_csv("iris_dataset_missing.csv")


# ## Question 1

# ### CM1

# ##### Plotting the pairs plot

# In[528]:

plt.figure(figsize=(15, 8))
ax = sns.pairplot(data, hue="species")
plt.show()


# ### CM2

# ##### Reporting the correlation coefficient for selected features

# In[529]:


# changing species from object to int
data.species.replace({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}, inplace = True)


# In[530]:


plot_heatmap(data.corr())


# - petal_width has the highest correlation with the target (0.955)
# - sepal_width has a negative correlation with the target (-0.324)
# - sepal_width has the lowest correlation with the target (-0.324)
# - petal_width, petal_length and sepal_length have positive correlation with the target
# - the highest interfeature correlation is 0.958 and it exists between petal_width and petal_length.
# - the lowest interfeature correlation is -0.032 and it exists between sepal_width and sepal_length.

# ##### Calculating the mean, variance, skew, kurtosis for the datasets

# In[531]:


# comparing for the entire dataset
for column in data.iloc[:,:-1].columns:
    mean = round(data[column].mean(), 2)
    variance = round(data[column].var(), 2)
    skew = round(data[column].skew(), 2)
    kurtosis = round(data[column].kurt(), 2)
    print("***", column.title(), "***")
    print("Mean -", mean)
    print("Variance -", variance)
    print("Skew -", skew)
    print("Kurtosis -", kurtosis)
    print()


# In[532]:


# comparing for the different target
for specie in data["species"].unique():
    print("*** Specie = ", ['Iris-setosa','Iris-versicolor','Iris-virginica'][specie], "***")
    for column in data.iloc[:,:-1].columns:
        mean = round(data[data['species'] == specie][column].mean(), 2)
        variance = round(data[data['species'] == specie][column].var(), 2)
        skew = round(data[data['species'] == specie][column].skew(), 2)
        kurtosis = round(data[data['species'] == specie][column].kurt(), 2)
        print(column, "Mean -", mean)
        print(column, "Variance -", variance)
        print(column, "Skew -", skew)
        print(column, "Kurtosis -", kurtosis)
        print()


# ##### Nature of the data and observations

# In[ ]:





# ### CM3

# ##### Checking for outliers

# In[533]:


# Using histogram
for column in data.iloc[:,:-1].columns:
    plt.figure()
    sns.histplot(data = data, x = column, bins = 20)
    plt.show()


# from the histogram plots,
# - There are few outliers in petal width with values less than 0
# - There are also a few outliers in petal_length

# In[534]:


# finer detection using box plot
for column in data.columns[:-1]:
    plt.figure()
    ax = sns.boxplot(x="species", y=column, data=data)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['Iris-setosa','Iris-versicolor','Iris-virginica'])
    plt.show()


# From the boxplot
# - There are outliers in sepal_width of the Iris-virginica specie
# - There are outliers in petal_length and petal_width of the Iris-setosa specie

# In[535]:


# Handling negative values in petal_width by replacing with nan
for index in data[data["petal_width"] < 0].index:
    data.loc[index, "petal_width"] = np.nan 


# In[536]:


# Handling Outliers by replacing them with nan
outliers_dict = {"sepal_width":2, "petal_length":0, "petal_width":0}
for column, specie in outliers_dict.items():
    Q1 = data[column][data["species"] == specie].quantile(0.25)
    Q3 = data[column][data["species"] == specie].quantile(0.75)
    IQR = Q3 - Q1 #Interquartile range
    fence_low = Q1 - (1.5 * IQR)
    fence_high = Q3 + (1.5 * IQR)
    
    df2 = pd.DataFrame(data[data['species'] == specie][column])
    
    for index in df2[df2[column] < fence_low].index:
        data.loc[index, column] = np.nan
    for index in df2[df2[column] > fence_high].index:
        data.loc[index, column] = np.nan


# The outliers in the data were values lower than their group lower fence or values higher than the higher fence and these have been replaced with nan values 

# In[537]:


# Replotting box plot to confirm outlier removal
for column in data.columns[:-1]:
    plt.figure()
    ax = sns.boxplot(x="species", y=column, data=data)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['Iris-setosa','Iris-versicolor','Iris-virginica'])
    plt.show()


# The new boxplot show that the outliers in the data have been removed

# ### CM5

# ##### Data Cleaning

# In[538]:


# checking for missing values in columns
for column in data.columns:
    print(column.upper(), "-", data[column].isna().sum())


# In[539]:


# checking for missing values in rows
for i in range(len(data)):
    row = data.iloc[i, :]
    if row.isna().sum() > 0:
        print(i , ": ", row.isna().sum())


# - for the different features, petal_lenght has the highest number of missing values
# - no row has more than 2 missing value so we can use estimation method for the missing values. If any row had more than 2 missing values, the row would have been dropped

# In[540]:


# Handling missing values for numeric features
group_mean = data.groupby("species").mean()
for column in data.columns[:-1]:
    for index in data[data[column].isna()].index:
        specie = data.iloc[index,-1]
        data.loc[index, column] = round(group_mean[column][specie], 2)


# The approximation method used in replacing missing values was using the mean of the feature grouped by the specie since all the features are numeric.

# ##### Plotting the correlation plot to see the effect of data cleaning on the dataset

# In[541]:


plot_heatmap(data.corr())


# After Data cleaning
# - The correlation between petal_width and species increased from 0.955 to 0.957
# - The correlation between petal_length and species increased from 0.949 to 0.953
# - The correlation between sepal_width and species increased from -0.324 to -0.365
# - The correlation between sepal_length and species remained unchanged

# ## Question 2 

# ### Building the KNN Model

# In[542]:


data.head()


# In[543]:


# separating ths dataset into matrix of features and target
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

# Splitting the data into train, test, and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=275)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=275)


# In[544]:


# Developing the Classification Model
classifier = KNeighborsClassifier()
classifier.fit(X_train,y_train)

# Predicting the test set result
y_pred = classifier.predict(X_test)

# Evaluating the Model
cm = confusion_matrix(y_test,y_pred)

accuracy_1 = round(100 * accuracy_score(y_test,y_pred), 2)
f1_score_1 = round(f1_score(y_test, y_pred, average = "weighted"), 2)
#auc_1 = roc_auc_score(y_test, y_pred, average = "macro", multi_class = "ovo")

y_pred_train  = classifier.predict(X_train)

print("Training Set Evaluation")
print("Accuracy: ", round(100 * accuracy_score(y_train, y_pred_train), 2))
print("F1_score: ", round(f1_score(y_train, y_pred_train, average = 'weighted'), 2))
print()
print("Test Set Evaluation")
print("Accuracy: ", accuracy_1)
print("F1_score: ", f1_score_1)
#print("AUC: ", auc_1)


# ##### Finding best parameter by tuning k

# In[545]:


k_list = [1,5,10,15,20,25,30,35]

accuracy = {}
for k in k_list:
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train,y_train)
    
    # Predicting the test set result
    y_pred = classifier.predict(X_val)
    
    model_accuracy = accuracy_score(y_val, y_pred)
    
    accuracy[k] = round(model_accuracy * 100, 2)


# ### CM6

# In[546]:


# plotting the parameter vs accuracy graph
sns.lineplot(x = k_list, y = accuracy.values())


# The highest accuracy accurs at k = {1, 5, 10} but we will select k = 5 as the optimal k because in KNN, a small value of k will make our model highly susceptible to noise which will cause high variations in the performance of the model on different sets of unobserved data and a high value of k wil lead to higher computational cost.

# ##### Building the model with optimal k

# In[547]:


classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train,y_train)

# Predicting the test set result
y_pred = classifier.predict(X_test)

# Evaluating the model
accuracy_2 = round(100 * accuracy_score(y_test, y_pred), 2)
f1_score_2 = round(f1_score(y_test, y_pred, average = "weighted"), 2)
#auc_2 = roc_auc_score(y_test, y_pred, average = "macro", multi_class = "ovo")

print("Test Set Evaluation")
print("Accuracy: ", accuracy_2)
print("F1_score: ", f1_score_2)
#print("AUC: ", auc_2)


# ##### Improving the model

# In[548]:


# Normalizing numerical features
X_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
X_val = X_scaler.transform(X_val)
X_test = X_scaler.transform(X_test)


# The normalization teschnique used is z-score normalization which modifies the data to have a mean of 0 and a standard deviation of 1 thus making the aggregate data have the properties of a normal distribution.

# ##### parameter tuning of weight and distance metrics

# In[549]:


# finding optimal distance metric
weights = ["uniform", "distance"]
distance_metrics = [1, 2]
evaluation = []

for weight in weights:
    for p in distance_metrics:     
        classifier = KNeighborsClassifier(n_neighbors = 5, weights = weight, p = p)
        classifier.fit(X_train,y_train)

        # Testing on the validation set
        y_pred = classifier.predict(X_val)

        # Evaluating the model
        accuracy_3 = round(100 * accuracy_score(y_val, y_pred), 2)
        f1_score_3 = round(f1_score(y_val, y_pred, average = "weighted"), 2)
        #auc_3 = roc_auc_score(y_val, y_pred, average = "macro", multi_class = "ovo")
        
        evaluation.append({'weight':weight, 'p':p, 'accuracy':accuracy_3, 'F1':f1_score_3})


# In[550]:


for eval in evaluation:
    print(eval)


# ##### Building the optimized model  (uniform weight and manhattan distance)

# In[551]:


classifier = KNeighborsClassifier(n_neighbors = 5, weights = "distance", p = 2)
classifier.fit(X_train, y_train)


# ### CM7

# In[552]:


# Predicting the test set result with optimized model
y_pred = classifier.predict(X_test)

# Evaluating the model
accuracy_4 = round(100 * accuracy_score(y_test, y_pred), 2)
f1_score_4 = round(f1_score(y_test, y_pred, average = "weighted"), 2)
#auc_4 = roc_auc_score(y_test, y_pred, average = "macro", multi_class = "ovo")
print("Test Set Evaluation")
print("Accuracy: ", accuracy_4)
print("F1_score: ", f1_score_4)
#print("AUC: ", auc_4)


# In[ ]:





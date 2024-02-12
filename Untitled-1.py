# %%
#Making imports

import numpy as np
import pandas as pd

# %%
#Cloning the dataset repo

# %% [markdown]
# The following function takes the .txt or .fasta file as input and breaks the individual sequences into a numpy array with its respective Accession ID.

# %%
"""
Function that parses data from a text file and returns a np array
the np array has the following format
[name, fasta_sequence]
"""

def readData(name_dataset, path):
  data = []
  laststring = ""
  lastcount = -1
  with open(path, "r") as f:
    count = -1
    s = ""
    for line in f:
      if (line[0] == '>'):
        count += 1
        temp = [f"{name_dataset}{count}", s]
        data.append(temp)
        s = ""
      else:
        s += line.strip()
    lastcount = count + 1
    laststring = s

  data.append([f"{name_dataset}{lastcount}", laststring])
  data.pop(0)
  data = np.array(data)

  return data

# %% [markdown]
# Downloading the data and converting it to a usable form i.e. a numpy array. ALgpred data has duplicates which will be dealt with later. Allertop and Algpred data has non allergens which ideally should not be in the positive dataset (needs to be confirmed). Taking into account the updated databases, the unpreprocessed data has an overflow of 1033 samples as compared to unpreprocessed Algpred 2.0 server.

# %%
"""
processing the algpred dataset into a numpy array

encoding the same data into a new column
allergen -> 1
non-allergen -> 0
algpred -> done

compare_fasta(all allergens) -> done
allertop_allergen(all allergens) -> done
allertop_nonallergen(all non-allergens) -> done
uniprot_allergen(all allergens) -> done
allergen_online(all allergens) -> done
"""

#ALGPRED DATA
df_algpred = pd.read_csv('wk2_ip_datasets/processed.csv')
algpred_data = df_algpred.values

#cleaning the algpred data to get rid of the \n values from the end
algpred_data = np.vectorize(lambda x: x.strip('\n'))(algpred_data)
condition = np.char.startswith(algpred_data[:,0].astype(str), 'A')
new_column = np.where(condition, 1, 0)
algpred_data = np.column_stack((algpred_data, new_column))
print(algpred_data.shape)
print(algpred_data)
print("\n\n\n")

#COMPARE_ALLERGEN DATA
compare_allergen_data = readData("compare_allergen", "wk2_ip_datasets/COMPARE-2023-FastA-Seq.txt")
rows_compareallergen = compare_allergen_data.shape[0]
cols_compareallergen = compare_allergen_data.shape[1]
ones_column = np.full((rows_compareallergen, 1), "1")
compare_allergen_data = np.hstack((compare_allergen_data, ones_column))
print(compare_allergen_data.shape)
print(compare_allergen_data)
print("\n\n\n")

#ALLERTOP_ALLERGEN DATA
allertop_allergen_data = readData("allertop_allergen", "wk2_ip_datasets/allergens_allertop.txt")
rows_allertopallergen = allertop_allergen_data.shape[0]
cols_allertopallergen = allertop_allergen_data.shape[1]
ones_column = np.full((rows_allertopallergen, 1), "1")
allertop_allergen_data = np.hstack((allertop_allergen_data, ones_column))
print(allertop_allergen_data.shape)
print(allertop_allergen_data)
print("\n\n\n")

#ALLERTOP_NONALLERGEN DATA
allertop_nonallergen_data = readData("allertop_nonallergen", "wk2_ip_datasets/nonallergens_allertop.txt")
rows_allertopnonallergen = allertop_nonallergen_data.shape[0]
cols_allertopnonallergen = allertop_nonallergen_data.shape[1]
zeros_column = np.full((rows_allertopnonallergen, 1), "0")
allertop_nonallergen_data = np.hstack((allertop_nonallergen_data, zeros_column))
print(allertop_nonallergen_data.shape)
print(allertop_nonallergen_data)
print("\n\n\n")

#UNIPROT_KB ALLERGEN DATA
uniprot_df = pd.read_excel(r"wk2_ip_datasets/uniprotkb_allergen_AND_reviewed_true_2023_08_30.xlsx")
print(uniprot_df)
uniprot_df = uniprot_df.values
uniprot_data = []
rows_uniprot = uniprot_df.shape[0]
cols_uniprot = uniprot_df.shape[1]
for i in range(rows_uniprot):
  uniprot_data.append([f"uniprot_allergen{i+1}", uniprot_df[i][cols_uniprot-1]])
uniprot_data = np.array(uniprot_data)
ones_column = np.full((rows_uniprot, 1), "1")
uniprot_data = np.hstack((uniprot_data, ones_column))
print(uniprot_data.shape)
print(uniprot_data)
print("\n\n\n")

#ALLERGEN_ONLINE DATA
df_allergenonline = pd.read_csv("wk2_ip_datasets/allergen_online.csv")
allergenonline_data = df_allergenonline.values

rows_allergenonline = allergenonline_data.shape[0]
cols_allergenonline = allergenonline_data.shape[1]

for i in range(rows_allergenonline):
  allergenonline_data[i][0] = f"allergen_online{i+1}"

ones_column = np.full((rows_allergenonline, 1), "1")
allergenonline_data = np.hstack((allergenonline_data, ones_column))
print(allergenonline_data.shape)
print(allergenonline_data)

# %%
"""
STORING ALL THE DATA IN A SINGLE NUMPY ARRAY
The format for the same is as follows:
[name ,  fasta sequence  ,  encoded value]
encoded value is "0" if non allergen and "1" if allergen
"""

all_data = np.vstack((algpred_data, compare_allergen_data, allertop_allergen_data, allertop_nonallergen_data, uniprot_data, allergenonline_data))

print(all_data.shape)
print(all_data)
df=pd.DataFrame(all_data)
df.to_csv('all_data.csv')

# %%
# !--NotebookApp.iopub_data_rate_limit

# %%
"""
Importing all_data with aac features added
The aac features were extracted using the pfeature library locally
"""

df = pd.read_csv("wk2_ip_datasets/all_data_with_aac.csv")
all_data = df.to_numpy()
new_data=[i[1:len(i)] for i in all_data]
print(all_data.shape)
print(new_data)
print(len(new_data))

# %%
#Getting rid of duplicates

print(len(new_data))
new_data=np.array(new_data)
tuple_of_tuples = tuple(map(tuple, new_data))
s=set(tuple_of_tuples)
print(s)
org_data=[]
for i in s:
  org_data.append(list(i))
for i in org_data:
  print(i)
org_data=np.array(org_data)
df=pd.DataFrame(org_data)
df.to_csv('real_aac.csv',index=False,header=False)



# %% [markdown]
# This final dataframe has all the samples, not preprocessed.

# %%
df.shape

# %% [markdown]
# The dataframe was preprocessed manually in excel by removing all the sequences which have 'BJOUXZ' amino acids in the sequences. All the samples with a sequence of length of less than 50 were also removed and this should make the total number of samples be 11280. However, we have not considered non allergens for positive dataset, so our size of the dataset was 8335 for positive after the preprocessing.

# %% [markdown]
# We have generated two cluster files using CD-HIT, the server used in the paper. There was a sequence similarity of 40% as the cutoff. We have generated the cluster files and have extracted the clusters and split the dataset in 80:20 for training and testing (the split was approximate and not exact as dividing it in clusters which have the required sum was not possible).

# %%
cluster_dict = {}

with open("wk2_ip_datasets/1694200452.fas.1.clstr.sorted", 'r') as file:
    cluster_data = file.read().split(">Cluster")

    for cluster in cluster_data:
        lines = cluster.strip().split('\n')
        if not lines:
            continue
        cluster_name = lines[0]  # The first line contains the cluster name
        sequence_count = len(lines) - 1  # Subtract 1 for the cluster name line

        cluster_dict[cluster_name] = sequence_count

# Now, cluster_dict contains cluster names as keys and the number of sequences in each cluster as values
# print(cluster_dict)
sum = 0
del cluster_dict['']
import random

target_sum = 6668
selected_clusters = []

# Create a list of shuffled cluster names (keys)
shuffled_clusters = list(cluster_dict.keys())
random.shuffle(shuffled_clusters)

current_sum = 0

for cluster_name in shuffled_clusters:
    if current_sum + cluster_dict[cluster_name] <= target_sum:
        selected_clusters.append(cluster_name)
        current_sum += cluster_dict[cluster_name]
    if current_sum == target_sum:
        break

# selected_clusters now contains the randomly selected clusters that add up to 6668
print(len(selected_clusters))
#cluster_data has all the sequences in that specific cluster

sequence_ids =[]
for i in selected_clusters:
    lines = (cluster_data[int(i)].split('>'))
    for line in lines[1:]:
        id_part = line.split('...')[0].strip()  # Split by '...' and take the first part
        if id_part:  # Check if the ID part is not empty
            sequence_ids.append(id_part)

# %%
validation = []
for i in cluster_dict.keys():
  if i not in selected_clusters:
    validation.append(i)

sequence_ids_2 =[]
for i in validation:
    lines = (cluster_data[int(i)].split('>'))
    for line in lines[1:]:
        id_part = line.split('...')[0].strip()  # Split by '...' and take the first part
        if id_part:  # Check if the ID part is not empty
            sequence_ids_2.append(id_part)

# %% [markdown]
# The preprocessed positive dataset(8335) was uploaded to the git repo. The length distribution of the positive dataset was followed to extract the samples randomly for the negative dataset.

# %%
df_positive = pd.read_csv("wk2_ip_datasets/positive.csv")

# %%
df_negative = pd.read_csv("wk2_ip_datasets/new_negative.csv")

# %%
df_positive.columns

# %%
plus_set=df_positive.to_numpy()
minus_set=df_negative.to_numpy()
temp_set=np.concatenate((plus_set,minus_set),axis=0)
# print(temp_set[0:1])
DATA_set=np.delete(temp_set,[0,1],axis=1)
# print(DATA_set)

# %%
new_column_names = {'name': 'IDs', 'sequence': 'Sequence'}
df_negative=df_negative.rename(columns=new_column_names)
Data_set_df=pd.concat([df_positive,df_negative])
Data_set_df=Data_set_df.drop(columns=['IDs','Sequence'])
Data_set_df=Data_set_df.rename(columns={'Unnamed: 22':'Predicted'})

# %%
training_positive_df = df_positive[df_positive['IDs'].isin(sequence_ids)]
validation_positive_df = df_positive[df_positive['IDs'].isin(sequence_ids_2)]

# %%
print(training_positive_df.shape)
print(validation_positive_df.shape)
# print(validation_positive_df)

# %%
cluster_dict = {}

with open("wk2_ip_datasets/1696966524.fas.1.clstr.sorted", 'r') as file:
    cluster_data = file.read().split(">Cluster")

    for cluster in cluster_data:
        lines = cluster.strip().split('\n')
        if not lines:
            continue
        cluster_name = lines[0]  # The first line contains the cluster name
        sequence_count = len(lines) - 1  # Subtract 1 for the cluster name line

        cluster_dict[cluster_name] = sequence_count

# Now, cluster_dict contains cluster names as keys and the number of sequences in each cluster as values
# print(cluster_dict)
sum = 0
del cluster_dict['']
import random

target_sum = 6668
selected_clusters = []

# Create a list of shuffled cluster names (keys)
shuffled_clusters = list(cluster_dict.keys())
random.shuffle(shuffled_clusters)

current_sum = 0

for cluster_name in shuffled_clusters:
    if current_sum + cluster_dict[cluster_name] <= target_sum:
        selected_clusters.append(cluster_name)
        current_sum += cluster_dict[cluster_name]
    if current_sum == target_sum:
        break

# selected_clusters now contains the randomly selected clusters that add up to 6668
print(len(selected_clusters))
#cluster_data has all the sequences in that specific cluster

sequence_ids =[]
for i in selected_clusters:
    lines = (cluster_data[int(i)].split('>'))
    for line in lines[1:]:
        id_part = line.split('...')[0].strip()  # Split by '...' and take the first part
        if id_part:  # Check if the ID part is not empty
            sequence_ids.append(id_part)

validation = []
for i in cluster_dict.keys():
  if i not in selected_clusters:
    validation.append(i)

sequence_ids_2 =[]
for i in validation:
    lines = (cluster_data[int(i)].split('>'))
    for line in lines[1:]:
        id_part = line.split('...')[0].strip()  # Split by '...' and take the first part
        if id_part:  # Check if the ID part is not empty
            sequence_ids_2.append(id_part)

# %%
training_negative_df = df_negative[df_negative['IDs'].isin(sequence_ids)]
validation_negative_df = df_negative[df_negative['IDs'].isin(sequence_ids_2)]

print(training_negative_df.shape)
print(validation_negative_df.shape)

# %%
#Converting the splits into csv files to feed into the pfeature library and extract their AAC features

training_negative_df.to_csv('training_negative_dataset.csv')
validation_negative_df.to_csv('validation_negative_datatset.csv')

# %%
#Importing the csv files into a np array with the extracted AAC features

training_negative_df = pd.read_csv("wk2_ip_datasets/training_neg.csv")
validation_negative_df = pd.read_csv("wk2_ip_datasets/valid_neg.csv")

print(training_negative_df.shape)
print(validation_negative_df.shape)

#Converting all the above data to numpy arrays
train_pos = training_positive_df.to_numpy()
val_pos = validation_positive_df.to_numpy()
train_neg = training_negative_df.to_numpy()
val_neg = validation_negative_df.to_numpy()

# %%
#Printing their shapes
print("The shape of the postive training dataset is: ", end=" ")
print(train_pos.shape)

print("The shape of the positive validation dataset is: ", end=" ")
print(val_pos.shape)

print("The shape of the negative training dataset is: ", end=" ")
print(train_neg.shape)

print("The shape of the negative validation dataste is: ", end=" ")
print(val_neg.shape)

# %%
#Dropping the FASTA sequence column from all the datasets

#Getting rid of serial number column from train_neg and val_neg
train_neg = np.delete(train_neg, 0, 1)
val_neg = np.delete(val_neg, 0, 1)

train_neg = np.delete(train_neg, 1, 1)
val_neg = np.delete(val_neg, 1, 1)

train_pos = np.delete(train_pos, 1, 1)
val_pos = np.delete(val_pos, 1, 1)


print(train_pos.shape)
print(train_neg.shape)

# %%
#Making one training dataset

#Getting rid of the protein_name and storing it elsewhere
name_pos = train_pos[:,0]
name_neg = train_neg[:,0]

#Getting rid of the protein_name column from both the train datasets
train_pos = np.delete(train_pos, 0, 1)
train_neg = np.delete(train_neg, 0, 1)

#Converting nan values of the train_neg and val_neg into 0
train_neg[:,-1] = 0
val_neg[:,-1] = 0

#Converting all the data in string format inside the array into float
train_pos = train_pos.astype(float)

final_train = np.vstack((train_pos, train_neg))
print(final_train.shape)

# %%
"""
So far, we have

1) Imported all our data
2) we've gotten rid of duplicates
3) we've gotten rid of redundant data (clustered and compared using fasta sequence similarity)
4) we've extracted features from our data
5) we've labeled our data
6) we've made the train-test split

NOW, we shall fit our data into our ML model
"""

# %%
#Fitting the model with Decision Trees

y_dt = final_train[:,-1]
x_dt = final_train[:,:20]

y_dt = y_dt.astype(int)

from sklearn.tree import DecisionTreeClassifier
dt_clf =DecisionTreeClassifier(max_features="log2", max_depth=10, criterion="entropy")

dt_clf.fit(x_dt, y_dt)

# %%
import sklearn as sk

#Testing with positive validation set
dt_val_pos_name = val_pos[:,0]
dt_val_pos_target = val_pos[:,-1]

dt_val_pos = val_pos[:, 1:val_pos.shape[1]-1]

dt_val_pos = dt_val_pos.astype(float)
dt_val_pos_target = dt_val_pos_target.astype(float)

dt_pos_pred = dt_clf.predict(dt_val_pos)

#Calculating the accuracy for the positive val set
print("The accuracy for the positive dataset is: ", end=" ")
sdt1 = (sk.metrics.accuracy_score(dt_pos_pred, dt_val_pos_target))
print(sdt1*100)

#Testing with negative validation set
dt_val_neg_name = val_neg[:,0]
dt_val_neg_target = val_neg[:,-1]

dt_val_neg = val_neg[:, 1:val_neg.shape[1]-1]

dt_val_neg = dt_val_neg.astype(float)
dt_val_neg_target = dt_val_neg_target.astype(float)

dt_neg_pred = dt_clf.predict(dt_val_neg)

#Calculating the accuracy for the negative val set
print("The accuracy for the negative dataset is: ", end=" ")
sdt2 = (sk.metrics.accuracy_score(dt_neg_pred, dt_val_neg_target))
print(sdt2*100)

#Printing the total accuracy
print("The total accuracy is: ", end=" ")
print((sdt1+sdt2)*50)

# %%
#KNN
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
error_rate = []
knn_val_pos_name = val_pos[:,0]
knn_val_pos_target = val_pos[:,-1]
knn_val_pos = val_pos[:, 1:val_pos.shape[1]-1]
knn_val_pos_target = knn_val_pos_target.astype(float)
knn_val_pos = knn_val_pos.astype(float)

knn_val_neg_name = val_neg[:,0]
knn_val_neg_target = val_neg[:,-1]

knn_val_neg = val_neg[:, 1:val_neg.shape[1]-1]

knn_val_neg = knn_val_neg.astype(float)
knn_val_neg_target = knn_val_neg_target.astype(float)

for i in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_dt,y_dt)
    pos_pred_i = knn.predict(knn_val_pos)
    sdt1 = (sk.metrics.accuracy_score(pos_pred_i, knn_val_pos_target))
    neg_pred_i=knn.predict(knn_val_neg)
    sdt2=(sk.metrics.accuracy_score(neg_pred_i, knn_val_neg_target))
    error_rate.append(((sdt1+sdt2)/2)*100)

plt.figure(figsize=(10,6))
plt.figure(figsize=(10,6))
plt.plot(range(1,11),error_rate)
plt.show()

# %%
#KNN FOR neigbhours=9
knn = KNeighborsClassifier(n_neighbors=84, p=2, weights='distance')
knn.fit(x_dt,y_dt)
pos_pred_i = knn.predict(knn_val_pos)
sdt1 = (sk.metrics.accuracy_score(pos_pred_i, knn_val_pos_target))
neg_pred_i=knn.predict(knn_val_neg)
sdt2=(sk.metrics.accuracy_score(neg_pred_i, knn_val_neg_target))
print("Accuracy is ",(sdt1+sdt2)*50)

# %%
#Fitting the model with random forest

y_rf = final_train[:,-1]
x_rf = final_train[:,:20]

y_rf = y_rf.astype(int)

from sklearn.ensemble import RandomForestClassifier
rf_clf =RandomForestClassifier(n_estimators=1000, max_features="sqrt", max_depth=100, criterion="gini", random_state=42)
rf_clf.fit(x_rf, y_rf)

# %%
#Testing with positive validation set
rf_val_pos_name = val_pos[:,0]
rf_val_pos_target = val_pos[:,-1]

rf_val_pos = val_pos[:, 1:val_pos.shape[1]-1]

rf_val_pos = rf_val_pos.astype(float)
rf_val_pos_target = rf_val_pos_target.astype(float)

rf_pos_pred = rf_clf.predict(rf_val_pos)

#Calculating the accuracy for the positive val set
print("The accuracy for the positive dataset is: ", end=" ")
srf1 = (sk.metrics.accuracy_score(rf_pos_pred, rf_val_pos_target))
print(srf1*100)

#Testing with negative validation set
rf_val_neg_name = val_neg[:,0]
rf_val_neg_target = val_neg[:,-1]

rf_val_neg = val_neg[:, 1:val_neg.shape[1]-1]

rf_val_neg = rf_val_neg.astype(float)
rf_val_neg_target = rf_val_neg_target.astype(float)

rf_neg_pred = rf_clf.predict(rf_val_neg)

#Calculating the accuracy for the negative val set
print("The accuracy for the negative dataset is: ", end=" ")
srf2 = (sk.metrics.accuracy_score(rf_neg_pred, rf_val_neg_target))
print(srf2*100)

#Printing the total accuracy
print("The total accuracy is: ", end=" ")
print((srf1+srf2)*50)

# %%
from sklearn import svm
y_svm=final_train[:,-1]
x_svm=final_train[:,:20]
x_svm=x_svm.astype(float)
y_svm=y_svm.astype(int)
svm_val_pos_name = val_pos[:,0]
svm_val_pos_target = val_pos[:,-1]
svm_val_pos = val_pos[:, 1:val_pos.shape[1]-1]
svm_val_pos_target = svm_val_pos_target.astype(float)
svm_val_pos = svm_val_pos.astype(float)

svm_val_neg_name = val_neg[:,0]
svm_val_neg_target = val_neg[:,-1]

svm_val_neg = val_neg[:, 1:val_neg.shape[1]-1]

svm_val_neg = svm_val_neg.astype(float)
svm_val_neg_target = svm_val_neg_target.astype(float)
clf = svm.SVC(kernel="rbf", gamma=0.05, C=0.1,probability=True)
clf.fit(x_svm, y_svm)
pred_pos = clf.predict(svm_val_pos)
pred_neg=clf.predict(svm_val_neg)
acc_pos_svm=sk.metrics.accuracy_score(svm_val_pos_target,pred_pos)
acc_neg_svm=sk.metrics.accuracy_score(svm_val_neg_target,pred_neg)
print("Accuracy",(acc_pos_svm+acc_neg_svm)*50)

# %%
from sklearn.metrics import precision_score, recall_score, f1_score

# %%
def svm_model(X_train,Y_train,X_test,Y_test):
  predictor=svm.SVC(kernel='linear')
  predictor.fit(X_train,Y_train)
  predicted_val=predictor.predict(X_test)
  conf_mat=confusion_matrix(predicted_val, Y_test)
  precision = precision_score(predicted_val,Y_test)
  sensitivity = recall_score(predicted_val,Y_test)
  specificity=conf_mat[0,0]/(conf_mat[0,1]+conf_mat[0,0])
  f1 = f1_score(predicted_val,Y_test)
  accuracy=accuracy_score(predicted_val,Y_test)
  error=0
  for i in range(len(Y_train)):
    if Y_train[i]!=predicted_val[i]:
      error+=1
  return accuracy,error,precision,sensitivity,specificity,f1

# %%
def rf_model(X_train,Y_train,X_test,Y_test):
  predictor=RandomForestClassifier(n_estimators = 1000, max_depth = 200, random_state = 42)
  predictor.fit(X_train,Y_train)
  predicted_val=predictor.predict(X_test)
  conf_mat=confusion_matrix(predicted_val, Y_test)
  precision = precision_score(predicted_val,Y_test)
  sensitivity = recall_score(predicted_val,Y_test)
  specificity=conf_mat[0,0]/(conf_mat[0,1]+conf_mat[0,0])
  f1 = f1_score(predicted_val,Y_test)
  accuracy=accuracy_score(predicted_val,Y_test)
  error=0
  for i in range(len(Y_train)):
    if Y_train[i]!=predicted_val[i]:
      error+=1
  return accuracy,error,precision,sensitivity,specificity,f1

# %%
def knn_model(X_train,Y_train,X_test,Y_test):
  predictor=KNeighborsClassifier(n_neighbors=9)
  predictor.fit(X_train,Y_train)
  predicted_val=predictor.predict(X_test)
  conf_mat=confusion_matrix(predicted_val, Y_test)
  precision = precision_score(predicted_val,Y_test)
  sensitivity = recall_score(predicted_val,Y_test)
  specificity=conf_mat[0,0]/(conf_mat[0,1]+conf_mat[0,0])
  f1 = f1_score(predicted_val,Y_test)
  accuracy=accuracy_score(predicted_val,Y_test)
  error=0
  for i in range(len(Y_train)):
    if Y_train[i]!=predicted_val[i]:
      error+=1
  return accuracy,error,precision,sensitivity,specificity,f1


# %%
def DT_model(X_train,Y_train,X_test,Y_test):
  predictor=DecisionTreeClassifier(random_state = 42)
  predictor.fit(X_train,Y_train)
  predicted_val=predictor.predict(X_test)
  conf_mat=confusion_matrix(predicted_val, Y_test)
  precision = precision_score(predicted_val,Y_test)
  sensitivity = recall_score(predicted_val,Y_test)
  specificity=conf_mat[0,0]/(conf_mat[0,1]+conf_mat[0,0])
  f1 = f1_score(predicted_val,Y_test)
  accuracy=accuracy_score(predicted_val,Y_test)
  error=0
  for i in range(len(Y_train)):
    if Y_train[i]!=predicted_val[i]:
      error+=1
  return accuracy,error,precision,sensitivity,specificity,f1


# %%
import seaborn as sns
import matplotlib.pyplot as plt


# %%
correlation_matrix = Data_set_df.corr()
# print(correlation_matrix)
plt.figure(figsize=(25, 25))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap for Binary Classification")
plt.savefig("heatmap.png")
plt.show()

# %%
from sklearn.feature_selection import SelectKBest

# %%
test = SelectKBest(k=20)
Y = Y.values.flatten()
selected = test.fit(X,Y)

# %%
import matplotlib.pyplot as plt
indices = np.argsort(selected.scores_)[::-1]
features = []
for i in range(20):
    features.append(X.columns[indices[i]])

# Now plot
plt.figure()
plt.bar(features, selected.scores_[indices[range(20)]])
plt.xticks(rotation=45, ha="right")  # Rotate labels for better readability

# Set plot title and labels
plt.title("Feature Importance Scores")
plt.xlabel("Features")
plt.ylabel("SelectKBest Scores")
plt.savefig("kbest.png")
plt.show()

# %%
X=Data_set_df.drop(columns=['Predicted'])
columns_df=Data_set_df.columns
Y=Data_set_df.drop(columns=[i for i in columns_df if i!='Predicted'])
X=X.astype('float')
Y=Y.astype('int')

# %%
Y.shape

# %%
from sklearn.manifold import TSNE as tsne
import pandas as pn
ts = tsne(n_components=2).fit_transform(X)
reduced_df = pn.DataFrame(ts,columns=['xcol','ycol'])
reduced_df['Predicted']= Y.values.flatten()


# %%
import seaborn as sb
sb.scatterplot(data=reduced_df,x="xcol",y="ycol",hue="Predicted")
plt.savefig("TSNE.png")

# %%
X_dataset=X.to_numpy()
Y_dataset=Y.to_numpy()
print(X_dataset.dtype)
print(Y_dataset.dtype)

# %% [markdown]
# Frequency Distribution of AAC
# 

# %%
features=X.columns
for i in features:
  plt.figure(figsize=(100, 10))
  frequency_distribution = X[i].value_counts()
  sns.barplot(x=frequency_distribution.index, y=frequency_distribution.values)
  plt.xlabel(i)
  plt.ylabel("Frequency")
  plt.title("Frequency Distribution")
  plt.show()

# %%
#Making one final train set and using k-fold cross validation on it
kf_set = np.vstack((train_pos , val_pos[:, 1:] , train_neg , val_neg[:, 1:]))
kfX= kf_set[:,:-1]
kfY = kf_set[:,-1]
kfX=kfX.astype(float)
kfY = kfY.astype(int)

from sklearn.model_selection import train_test_split
kfX_train, kfX_test, kfY_train, kfY_test = train_test_split(kfX, kfY, test_size = 0.2, random_state=42)

# %%
kfX_train


# %%
kfY_train
print(np.sum(kfY_train==1))

# %%
print(kfX_train.shape, kfY_train.shape, kfX_test.shape, kfY_test.shape)

# %%
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import roc_auc_score as auc

# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# %%
X_train = scaler.fit_transform(kfX_train)
X_test = scaler.transform(kfX_test)

# %%
from sklearn.model_selection import cross_validate
from sklearn import metrics


#precision_macro -> precision
#recall_macro -> senstivity
#f1_macro -> f1 score

scoring = ['precision_macro', 'recall_macro', 'f1_macro','accuracy']
scores = cross_validate(clf, kfX_train, 
                        kfY_train, cv=5, scoring=scoring)
score_names = sorted(scores.keys())

for i in range(2,len(score_names)):
  curr_name = score_names[i]
  print(scores[curr_name])
  print(f"With svm we have %f {curr_name} " % (scores[curr_name].mean()))

# %%
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix

def calc_specificity(y_true,y_pred):
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  specificity = tn / (tn + fp)
  return specificity

# %%
clf.fit(kfX_train,kfY_train)
y_pred=clf.predict(kfX_test)
precision = precision_score(y_pred, kfY_test, average='macro')
recall = recall_score(y_pred, kfY_test, average='macro')
f1 = f1_score(y_pred, kfY_test, average='macro')
accuracy = accuracy_score(kfY_test, y_pred)
print(f1,precision,recall,f1,accuracy)
print(mcc(y_pred,kfY_test))
print(auc(y_pred,kfY_test))
print(calc_specificity(y_pred,kfY_test))

# %%
from sklearn.model_selection import cross_validate
from sklearn import metrics

# %%
#precision_macro -> precision
#recall_macro -> senstivity
#f1_macro -> f1 score

scoring = ['precision_macro', 'recall_macro', 'f1_macro','accuracy']
scores = cross_validate(dt_clf, kfX_train, kfY_train, cv=5, scoring=scoring)
score_names = sorted(scores.keys())

for i in range(2,len(score_names)):
  curr_name = score_names[i]
  print(scores[curr_name])
  print(f"With decision trees we have %f {curr_name} " % (scores[curr_name].mean()))

# %%
dt_clf.fit(kfX_train,kfY_train)

# %%
y_pred=dt_clf.predict(kfX_test)
precision = precision_score(y_pred, kfY_test, average='macro')
recall = recall_score(y_pred, kfY_test, average='macro')
f1 = f1_score(y_pred, kfY_test, average='macro')
accuracy = accuracy_score(kfY_test, y_pred)
print(f1,precision,recall,accuracy)
print(mcc(y_pred,kfY_test))
print(auc(y_pred,kfY_test))
print(calc_specificity(y_pred,kfY_test))

# %%
#precision_macro -> precision
#recall_macro -> senstivity
#f1_macro -> f1 score

scoring = ['precision_macro', 'recall_macro', 'f1_macro','accuracy']
scores = cross_validate(knn, kfX_train, kfY_train, cv=5, scoring=scoring)
score_names = sorted(scores.keys())

for i in range(2,len(score_names)):
    curr_name = score_names[i]
    print(scores[curr_name])
    print(f"With knn we have %f {curr_name} " % (scores[curr_name].mean()))

# %%
knn.fit(kfX_train,kfY_train)

# %%
y_pred=knn.predict(kfX_test)
precision = precision_score(y_pred, kfY_test, average='macro')
recall = recall_score(y_pred, kfY_test, average='macro')
f1 = f1_score(y_pred, kfY_test, average='macro')
accuracy = accuracy_score(kfY_test, y_pred)
print(f1,precision,recall,f1,accuracy)
print(mcc(y_pred,kfY_test))
print(auc(y_pred,kfY_test))
print(calc_specificity(y_pred,kfY_test))

# %%
#precision_macro -> precision
#recall_macro -> senstivity
#f1_macro -> f1 score

rf_clf =RandomForestClassifier(n_estimators=1000, max_features="sqrt", max_depth=100, criterion="gini", random_state=42)
scoring = ['precision_macro', 'recall_macro', 'f1_macro','accuracy']
scores = cross_validate(rf_clf, kfX_train, kfY_train, cv=5, scoring=scoring)
score_names = sorted(scores.keys())

for i in range(2,len(score_names)):
  curr_name = score_names[i]
  print(scores[curr_name])
  print(f"With random forests we have %f {curr_name} " % (scores[curr_name].mean()))

# %%
rf_clf.fit(kfX_train,kfY_train)
y_pred=rf_clf.predict(kfX_test)
precision = precision_score(y_pred, kfY_test, average='macro')
recall = recall_score(y_pred, kfY_test, average='macro')
f1 = f1_score(y_pred, kfY_test, average='macro')
accuracy = accuracy_score(kfY_test, y_pred)
print(precision,recall,f1,accuracy)
print(mcc(y_pred,kfY_test))
print(auc(y_pred,kfY_test))
print(calc_specificity(y_pred,kfY_test))

# %%
#using naive bayes
from sklearn.naive_bayes import GaussianNB
nb_clf=GaussianNB()
scoring = ['precision_macro', 'recall_macro', 'f1_macro','accuracy']
scores = cross_validate(nb_clf, kfX, kfY, cv=5, scoring=scoring)
score_names = sorted(scores.keys())

for i in range(2,len(score_names)):
  curr_name = score_names[i]
  print(scores[curr_name])
  print(f"With random forests we have %f {curr_name} " % (scores[curr_name].mean()))

# %%
#using enssemble learning
from sklearn.ensemble import VotingClassifier
ensemble1 = VotingClassifier(
    estimators=[
        ('svm', clf),
        ('dt', dt_clf)
    ],
    voting='hard'
)
ensemble2 = VotingClassifier(
    estimators=[
        ('knn', knn),
        ('rf', rf_clf)
    ],
    voting='hard'
)


# %%
scoring = ['precision_macro', 'recall_macro', 'f1_macro','accuracy']
scores = cross_validate(ensemble1, kfX_train, kfY_train, cv=5, scoring=scoring)
score_names = sorted(scores.keys())

for i in range(2,len(score_names)):
  curr_name = score_names[i]
  print(scores[curr_name])
  print(f"Ensemble {curr_name} " % (scores[curr_name].mean()))

# %%
ensemble1.fit(kfX_train,kfY_train)

# %%
ensemble2.fit(kfX_train,kfY_train)

# %%
y_pred=ensemble1.predict(kfX_test)
precision = precision_score(y_pred, kfY_test, average='macro')
recall = recall_score(y_pred, kfY_test, average='macro')
f1 = f1_score(y_pred, kfY_test, average='macro')
accuracy = accuracy_score(kfY_test, y_pred)
print(f1,precision,recall,f1,accuracy)
print(mcc(y_pred,kfY_test))
print(auc(y_pred,kfY_test))

# %%
y_pred=ensemble2.predict(kfX_test)
precision = precision_score(y_pred, kfY_test, average='macro')
recall = recall_score(y_pred, kfY_test, average='macro')
f1 = f1_score(y_pred, kfY_test, average='macro')
accuracy = accuracy_score(kfY_test, y_pred)
print(f1,precision,recall,f1,accuracy)
print(mcc(y_pred,kfY_test))
print(auc(y_pred,kfY_test))
print(calc_specificity(y_pred,kfY_test))

# %%
ensemble = VotingClassifier(
    estimators=[
        ('ensemble1',ensemble1),
        ('ensemble2',ensemble2)
    ],
    voting='hard'
)

# %%
ensemble.fit(kfX_train,kfY_train)


# %%
y_pred=ensemble.predict(kfX_test)
precision = precision_score(y_pred, kfY_test, average='macro')
recall = recall_score(y_pred, kfY_test, average='macro')
f1 = f1_score(y_pred, kfY_test, average='macro')
accuracy = accuracy_score(kfY_test, y_pred)
print(f1,precision,recall,f1,accuracy)
print(mcc(y_pred,kfY_test))
print(auc(y_pred,kfY_test))

# %%
print(calc_specificity(y_pred,kfY_test))

# %%
from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef(y_pred,kfY_test)
print(mcc)

# %%
from sklearn.metrics import roc_auc_score

# %%
auc_score=roc_auc_score(y_pred,kfY_test)
print(auc_score)

# %%
 #K Best feature to get the ranking of features
from sklearn.feature_selection import RFECV
scoring = ['precision_macro', 'recall_macro', 'f1_macro','accuracy']
scores = cross_validate(rf_clf, kfX, kfY, cv=5, scoring=scoring,return_estimator=True)
score_names = sorted(scores.keys())
for i in range(2,len(score_names)):
  curr_name = score_names[i]
  print(scores[curr_name])
  print(f"With random forests we have %f {curr_name} " % (scores[curr_name].mean()))

feature_importance_scores = []
for fold_estimator in scores['estimator']:
    importance_scores = fold_estimator.feature_importances_
    feature_importance_scores.append(importance_scores)

# Calculate average feature importance scores over all folds
average_importance_scores = sum(feature_importance_scores) / cv

# Print the feature ranking
for rank, importance_score in enumerate(average_importance_scores, start=1):
    print(f"Rank {rank}: Importance Score: {importance_score:.4f}")

# %% [markdown]
# Parameter Tuning
# 

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV as RCV

# %%
from sklearn import svm

# %%
#FOR SVM
param_grid = {
    'C': np.logspace(-3,3,10),
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': np.logspace(-3,3,10),
    'degree': [2, 3, 4],
    'class_weight': [None, 'balanced']
}
grid_search =RCV(estimator=svm.SVC(),param_distributions=param_grid,n_iter=10,cv=5,n_jobs=-1,verbose=3)
grid_search.fit(kfX_train,kfY_train)

# %%
print(grid_search.best_params_)

# %%
indices=np.random.choice(kfX_train.shape[0], 100, replace=False)
KfX_background = kfX_train[indices]
explainer = shap.KernelExplainer(clf.predict,kfX_train)
shap_values = explainer.shap_values(kfX_train)
feature_importance = np.abs(shap_values).mean(axis=0)
feature_ranking = np.argsort(-feature_importance)
print("Ranked Features:", feature_ranking)

# %%
from yellowbrick.model_selection import learning_curve

# %%
rf_clf =RandomForestClassifier(n_estimators=1000, max_features="sqrt", max_depth=100, criterion="gini", random_state=42)

# %%
print(learning_curve(rf_clf, kfX_train, kfY_train, cv=5, scoring='accuracy'))

# %%
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin

# %%
class TFClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs=100, batch_size=32):
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X, y):
        input_dim = X.shape[1]
        X = X.astype(np.float32)
        y = y.astype(np.int32)
        self.model.add(Dense(units=1, activation='sigmoid', kernel_initializer=glorot_normal(), bias_initializer='zeros'))
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
        return self

    def predict(self, X):
        X = X.astype(np.float32)
        return (self.model.predict(X) > 0.5).astype(int)

tf_classifier = TFClassifier(epochs=1000, batch_size=1)
scoring = ['precision_macro', 'recall_macro', 'f1_macro','accuracy']
from sklearn.preprocessing import StandardScaler

# Create a StandardScaler instance
scores = cross_validate(tf_classifier, kfX_train, kfY_train, cv=5, scoring=scoring,n_jobs=-1)
score_names = sorted(scores.keys())
for i in range(2,len(score_names)):
  curr_name = score_names[i]
  print(scores[curr_name])
  print(f"With logistic Regression we have %f {curr_name} " % (scores[curr_name].mean()))


# %%
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# %%
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif, k=20)
selector.fit(kfX, kfY)
feature_scores = selector.scores_
# Get the feature ranking by sorting the indices of features based on their scores
feature_ranking = sorted(range(len(feature_scores)), key=lambda i: feature_scores[i], reverse=True)
feature_score=[i for i in sorted(feature_scores,reverse=True)]
print(feature_score)
print("Feature Ranking:", feature_ranking)

# %%
#param tuning for random forest
from sklearn.ensemble import RandomForestClassifier
params = {'n_estimators': [i for i in range(80,101)], 'min_samples_split': [i for i in range(2,10)], 'max_depth': [None,10,25,30,40,55]}
rf = RandomForestClassifier(n_jobs = -1)
gs_rf = GridSearchCV(rf,params)
gs_rf.fit(kfX_train,kfY_train)

# %%
gs_rf.best_params_

# %%
y_svm=gr

# %%
y_rf =gs_rf.predict(kfX_test)

# %%
from sklearn.metrics import accuracy_score
precision = precision_score(y_rf, kfY_test, average='macro')
recall = recall_score(y_rf, kfY_test, average='macro')
f1 = f1_score(y_rf, kfY_test, average='macro')
accuracy = accuracy_score(kfY_test, y_rf)
auc_score=roc_auc_score(y_rf,kfY_test)
mcc = mcc(y_rf,kfY_test)
print("Precision: ",precision)
print("Recall: ",recall)
print("f1:",f1)
print("accuracy:",accuracy)
print("specifity:",calc_specificity(kfY_test, y_rf))
print("AUC: ",auc_score)
print("Mcc: ",mcc)

# %%
#param tuning for xtra
from sklearn.ensemble import ExtraTreesClassifier
params = {'n_estimators': [i for i in range(80,500,20)], 'min_samples_split': [i for i in range(4,20)], 'max_depth': [None,10,25,30,40]}
et = ExtraTreesClassifier(n_jobs = -1)
gs_et = RCV(estimator=et,param_distributions=params,n_iter=20,cv=5,n_jobs=-1,verbose=3)
gs_et.fit(kfX_train,kfY_train)

# %%
gs_et.best_params_

# %%
y_et = gs_et.predict(kfX_test)
from sklearn.metrics import accuracy_score
precision = precision_score(y_et, kfY_test, average='macro')
recall = recall_score(y_et, kfY_test, average='macro')
f1 = f1_score(y_et, kfY_test, average='macro')
accuracy = accuracy_score(kfY_test, y_et)
auc_score=auc(y_et,kfY_test)
y_et = y_et.astype(float)
mcc_score = mcc(y_et,kfY_test)
print("Precision: ",precision)
print("Recall: ",recall)
print("f1:",f1)
print("accuracy:",accuracy)
print("specifity:",calc_specificity(kfY_test, y_et))
print("AUC: ",auc_score)
print("MCC: ",mcc_score)

# %%
#END C=46.41588833612773, class_weight=balanced, degree=2, gamma=0.1, kernel=rbf;, score=0.949 total time= 1.7min

# %%
svm_opt=svm.SVC(C=46.41588833612773, class_weight='balanced', degree=2, gamma=0.1, kernel='rbf')
svm_opt.fit(kfX_train,kfY_train)

# %%
Y_pred=svm_opt.predict(kfX_test)

# %%
precision = precision_score(Y_pred, kfY_test, average='macro')
recall = recall_score(Y_pred, kfY_test, average='macro')
f1 = f1_score(Y_pred, kfY_test, average='macro')
accuracy = accuracy_score(kfY_test, Y_pred)
print(f1,precision,recall,f1,accuracy)
print(mcc(Y_pred,kfY_test))
print(auc(Y_pred,kfY_test))

# %%
#xgb
from xgboost import XGBClassifier

# %%
xgb = XGBClassifier(n_jobs = -1)
params = {'n_estimators': [i for i in range(80,500,20)],'min_child_weight':[i for i in range(1,10)],'max_depth': [None,10,25,30,40]}
#gs_xgb = GridSearchCV(xgb,params)
gs_xgb = RCV(estimator = xgb, param_distributions = params, n_iter = 20, cv = 5, n_jobs = -1, verbose = 3)
gs_xgb.fit(kfX_train,kfY_train)

# %%
gs_xgb.best_params_

# %%
Y_pred=gs_xgb.predict(kfX_test)
precision = precision_score(Y_pred, kfY_test, average='macro')
recall = recall_score(Y_pred, kfY_test, average='macro')
f1 = f1_score(Y_pred, kfY_test, average='macro')
accuracy = accuracy_score(kfY_test, Y_pred)
print(f1,precision,recall,f1,accuracy)
print(mcc(Y_pred,kfY_test))
print(auc(Y_pred,kfY_test))

# %%
#label propagation
from sklearn.semi_supervised import LabelPropagation

# %%
lp = LabelPropagation(n_jobs = -1)
params = {'kernel': [kernel for kernel in ['knn', 'rbf']], 'gamma': [i for i in range(0,50)],'n_neighbors':[i for i in range(3,20)]}
gs_lp = RCV(estimator = lp, param_distributions = params, n_iter = 20, cv = 5, n_jobs = -1, verbose = 3)
gs_lp.fit(kfX_train,kfY_train)

# %%
gs_lp.best_params_

# %%
Y_pred=gs_lp.predict(kfX_test)
precision = precision_score(Y_pred, kfY_test, average='macro')
recall = recall_score(Y_pred, kfY_test, average='macro')
f1 = f1_score(Y_pred, kfY_test, average='macro')
accuracy = accuracy_score(kfY_test, Y_pred)
print(f1,precision,recall,f1,accuracy)
print(mcc(Y_pred,kfY_test))
print(auc(Y_pred,kfY_test))

# %%
from sklearn.neighbors import KNeighborsClassifier

# %%
knn_opt = KNeighborsClassifier(n_jobs = -1)
params = {'n_neighbors': [i for i in range(3,20)], 'weights': [weight for weight in ['uniform', 'distance']],'algorithm':[algorithm for algorithm in ['ball_tree', 'kd_tree', 'brute']], 'leaf_size': [i for i in range(3,30)], 'p': [power for power in [1,2]]}
gs_knn = RCV(estimator = knn_opt, param_distributions = params, n_iter = 20, cv = 5, n_jobs = -1, verbose = 3)
gs_knn.fit(kfX_train,kfY_train)

# %%
gs_knn.best_params_

# %%
Y_pred=gs_knn.predict(kfX_test)
precision = precision_score(Y_pred, kfY_test, average='macro')
recall = recall_score(Y_pred, kfY_test, average='macro')
f1 = f1_score(Y_pred, kfY_test, average='macro')
accuracy = accuracy_score(kfY_test, Y_pred)
print(f1,precision,recall,accuracy)
print(mcc(Y_pred,kfY_test))
print(auc(Y_pred,kfY_test))

# %%
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelPropagation
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

# %%
SVM_TUNED=SVC(C=46.5,class_weight='balanced',degree=2,gamma=0.01,kernel='rbf',probability=True)
XBG_TUNED=XGBClassifier(n_estimators=360,min_child_weight=1,max_depth=25,n_jobs = -1)
KNN_TUNED=KNeighborsClassifier(weights='distance',p=2,n_neighbors=4,leaf_size=14,algorithm='kd_tree',n_jobs=-1)
LabelProp_TUNED=LabelPropagation(kernel='rbf',gamma=19,n_neighbors=7,n_jobs=-1)
ET_TUNED=ExtraTreesClassifier(n_estimators=460,min_samples_split=5,max_depth=40,n_jobs=-1)
RF_TUNED=RandomForestClassifier(n_estimators=92,max_depth=25,min_samples_split=3,n_jobs=-1)

# %%
from sklearn.ensemble import VotingClassifier

# %%
Ensemble_TUNED=VotingClassifier(estimators=[('SVM',SVM_TUNED),('XBG',XBG_TUNED),('KNN',KNN_TUNED),('LabelProp',LabelProp_TUNED),('XtraTree',ET_TUNED),('RF',RF_TUNED)],voting='hard',n_jobs=-1)

# %%
Ensemble_TUNED.fit(kfX_train,kfY_train)

# %%
Y_pred=Ensemble_TUNED.predict(kfX_test)

# %%
precision = precision_score(Y_pred, kfY_test, average='macro')
recall = recall_score(Y_pred, kfY_test, average='macro')
f1 = f1_score(Y_pred, kfY_test, average='macro')
accuracy = accuracy_score(kfY_test, Y_pred)
specificity=calc_specificity(kfY_test, Y_pred)
mcc_s=mcc(Y_pred,kfY_test)
auc_s=auc(Y_pred,kfY_test)
print('F1: ',f1,' Precision:',precision,' Recall',recall,' Specificity:',specificity,' Accuracy:',accuracy,' MCC:',mcc_s,' AUC:',auc_s) 

# %%
Ensemble_TUNED1=VotingClassifier(estimators=[('XBG',XBG_TUNED),('LabelProp',LabelProp_TUNED),('XtraTree',ET_TUNED),('RF',RF_TUNED)],voting='hard',n_jobs=-1)

# %%
Ensemble_TUNED1.fit(kfX_train,kfY_train)

# %%
Y_pred=Ensemble_TUNED1.predict(kfX_test)

# %%
precision = precision_score(Y_pred, kfY_test, average='macro')
recall = recall_score(Y_pred, kfY_test, average='macro')
f1 = f1_score(Y_pred, kfY_test, average='macro')
accuracy = accuracy_score(kfY_test, Y_pred)

print(f1,precision,recall,f1,accuracy)
print(mcc(Y_pred,kfY_test))
print(auc(Y_pred,kfY_test))


# %%
Ensemble_TUNED3=VotingClassifier(estimators=[('SVM',SVM_TUNED),('XBG',XBG_TUNED),('KNN',KNN_TUNED),('LabelProp',LabelProp_TUNED),('XtraTree',ET_TUNED),('RF',RF_TUNED)],voting='hard',weights=[1,2,1,2,4,2],n_jobs=-1)

# %%
Ensemble_TUNED3.fit(kfX_train,kfY_train)

# %%
Y_pred=Ensemble_TUNED3.predict(kfX_test)

# %%
precision = precision_score(Y_pred, kfY_test, average='macro')
recall = recall_score(Y_pred, kfY_test, average='macro')
f1 = f1_score(Y_pred, kfY_test, average='macro')
accuracy = accuracy_score(kfY_test, Y_pred)
print(f1,precision,recall,f1,accuracy)
print(mcc(Y_pred,kfY_test))
print(auc(Y_pred,kfY_test))

# %%
ET_TUNED.fit(kfX_train,kfY_train)

# %%
Y_pred=ET_TUNED.predict(kfX_test)

# %%
precision = precision_score(Y_pred, kfY_test, average='macro')
recall = recall_score(Y_pred, kfY_test, average='macro')
f1 = f1_score(Y_pred, kfY_test, average='macro')
accuracy = accuracy_score(kfY_test, Y_pred)
print(f1,precision,recall,f1,accuracy)
print(mcc(Y_pred,kfY_test))
print(auc(Y_pred,kfY_test))

# %%
Y_pred=Ensemble_TUNED.predict(kfX_train)

# %%
precision = precision_score(Y_pred, kfY_train, average='macro')
recall = recall_score(Y_pred, kfY_train, average='macro')
f1 = f1_score(Y_pred, kfY_train, average='macro')
accuracy = accuracy_score(kfY_train, Y_pred)
print(f1,precision,recall,f1,accuracy)
print(calc_specificity(kfY_train, Y_pred))
print(mcc(Y_pred,kfY_train))
print(auc(Y_pred,kfY_train))

# %%
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.metrics import make_scorer

# %%
scoring = {
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall',
    'accuracy': 'accuracy',
    'specificity': make_scorer(calc_specificity),
    'roc_auc': make_scorer(auc),
    'mcc': make_scorer(mcc)
}

# %%
scores = cross_validate(SVM_TUNED, kfX_train, kfY_train, cv=5, scoring=scoring,n_jobs=-1)
score_names = sorted(scores.keys())

for i in range(2,len(score_names)):
  curr_name = score_names[i]
  print(scores[curr_name])
  print(f"Ensemble {curr_name} " % (scores[curr_name].mean()))

# %%
scores = cross_validate(KNN_TUNED, kfX_train, kfY_train, cv=5, scoring=scoring,n_jobs=-1)
score_names = sorted(scores.keys())

for i in range(2,len(score_names)):
  curr_name = score_names[i]
  print(scores[curr_name])
  print(f"Ensemble {curr_name} " % (scores[curr_name].mean()))

# %%
scores = cross_validate(ET_TUNED, kfX_train, kfY_train, cv=5, scoring=scoring,n_jobs=-1)
score_names = sorted(scores.keys())

for i in range(2,len(score_names)):
  curr_name = score_names[i]
  print(scores[curr_name])
  print(f"Ensemble {curr_name} " % (scores[curr_name].mean()))

# %%
scores = cross_validate(RF_TUNED, kfX_train, kfY_train, cv=5, scoring=scoring,n_jobs=-1)
score_names = sorted(scores.keys())

for i in range(2,len(score_names)):
  curr_name = score_names[i]
  print(scores[curr_name])
  print(f"Ensemble {curr_name} " % (scores[curr_name].mean()))

# %%
scores = cross_validate(XBG_TUNED, kfX_train, kfY_train, cv=5, scoring=scoring,n_jobs=-1)
score_names = sorted(scores.keys())

for i in range(2,len(score_names)):
  curr_name = score_names[i]
  print(scores[curr_name])
  print(f"Ensemble {curr_name} " % (scores[curr_name].mean()))

# %%
scores = cross_validate(LabelProp_TUNED, kfX_train, kfY_train, cv=5, scoring=scoring,n_jobs=-1)
score_names = sorted(scores.keys())

for i in range(2,len(score_names)):
  curr_name = score_names[i]
  print(scores[curr_name])
  print(f"Ensemble {curr_name} " % (scores[curr_name].mean()))

# %%
scores = cross_validate(Ensemble_TUNED, kfX_train, kfY_train, cv=5, scoring=scoring,n_jobs=-1)
score_names = sorted(scores.keys())

for i in range(2,len(score_names)):
  curr_name = score_names[i]
  print(scores[curr_name])
  print(f"Ensemble {curr_name} " % (scores[curr_name].mean()))

# %%
#importing raghava's dataset

import pandas as pd

rag_Xpos = pd.read_csv("./wk2_ip_datasets/training_positive.csv")
rag_Xneg = pd.read_csv("./wk2_ip_datasets/training_negative.csv")
rag_valpos = pd.read_csv("./wk2_ip_datasets/positive_validation.csv")
rag_valneg = pd.read_csv("./wk2_ip_datasets/validation_negative.csv")

print ("the shape of positive training dataset is: ", rag_Xpos.shape)
print ("the shape of negative training dataset is: ", rag_Xneg.shape)
print ("the shape of positive validation dataset is: ", rag_valpos.shape)
print ("the shape of negative validation dataset is: ", rag_valneg.shape)

# %%
rag_Xpos = rag_Xpos.to_numpy()
rag_Xneg = rag_Xneg.to_numpy()
rag_valpos = rag_valpos.to_numpy()
rag_valneg = rag_valneg.to_numpy()

#Converting the NaN values column to labels

rag_Xpos[:,-1] = 1
rag_Xneg[:,-1] = 0
rag_valpos[:,-1] = 1
rag_valneg[:,-1] = 0

print ("the shape of positive training dataset is: ", rag_Xpos.shape)
print ("the shape of negative training dataset is: ", rag_Xneg.shape)
print ("the shape of positive validation dataset is: ", rag_valpos.shape)
print ("the shape of negative validation dataset is: ", rag_valneg.shape)

# %%
#vertically stacking the train sets and val sets

rag_train = np.vstack((rag_Xpos, rag_Xneg))
rag_val = np.vstack((rag_valpos, rag_valneg))

print(rag_train.shape)
print(rag_val.shape)

# %%
SVM_TUNED=SVC(C=46.5,class_weight='balanced',degree=2,gamma=0.01,kernel='rbf',probability=True)
XBG_TUNED=XGBClassifier(n_estimators=360,min_child_weight=1,max_depth=25,n_jobs = -1)
KNN_TUNED=KNeighborsClassifier(weights='distance',p=2,n_neighbors=4,leaf_size=14,algorithm='kd_tree',n_jobs=-1)
LabelProp_TUNED=LabelPropagation(kernel='rbf',gamma=19,n_neighbors=7,n_jobs=-1)
ET_TUNED=ExtraTreesClassifier(n_estimators=460,min_samples_split=5,max_depth=40,n_jobs=-1)
RF_TUNED=RandomForestClassifier(n_estimators=92,max_depth=25,min_samples_split=3,n_jobs=-1)

# %%
SVM_TUNED.fit(kfX_train,kfY_train)

# %%
Y_pred=SVM_TUNED.predict(kfX_test)

# %%
precision = precision_score(Y_pred, kfY_test, average='macro')
recall = recall_score(Y_pred, kfY_test, average='macro')
f1 = f1_score(Y_pred, kfY_test, average='macro')
accuracy = accuracy_score(kfY_test, Y_pred)
specificity=calc_specificity(kfY_test, Y_pred)
mcc_s=mcc(Y_pred,kfY_test)
auc_s=auc(Y_pred,kfY_test)
print('F1: ',f1,' Precision:',precision,' Recall',recall,' Specificity:',specificity,' Accuracy:',accuracy,' MCC:',mcc_s,' AUC:',auc_s) 

# %%
XBG_TUNED.fit(kfX_train,kfY_train)

# %%
Y_pred=XBG_TUNED.predict(kfX_test)

# %%
precision = precision_score(Y_pred, kfY_test, average='macro')
recall = recall_score(Y_pred, kfY_test, average='macro')
f1 = f1_score(Y_pred, kfY_test, average='macro')
accuracy = accuracy_score(kfY_test, Y_pred)
specificity=calc_specificity(kfY_test, Y_pred)
mcc_s=mcc(Y_pred,kfY_test)
auc_s=auc(Y_pred,kfY_test)
print('F1: ',f1,' Precision:',precision,' Recall',recall,' Specificity:',specificity,' Accuracy:',accuracy,' MCC:',mcc_s,' AUC:',auc_s) 

# %%
KNN_TUNED.fit(kfX_train,kfY_train)

# %%
Y_pred=KNN_TUNED.predict(kfX_test)

# %%
precision = precision_score(Y_pred, kfY_test, average='macro')
recall = recall_score(Y_pred, kfY_test, average='macro')
f1 = f1_score(Y_pred, kfY_test, average='macro')
accuracy = accuracy_score(kfY_test, Y_pred)
specificity=calc_specificity(kfY_test, Y_pred)
mcc_s=mcc(Y_pred,kfY_test)
auc_s=auc(Y_pred,kfY_test)
print('F1: ',f1,' Precision:',precision,' Recall',recall,' Specificity:',specificity,' Accuracy:',accuracy,' MCC:',mcc_s,' AUC:',auc_s) 

# %%
LabelProp_TUNED.fit(kfX_train,kfY_train)

# %%
Y_pred=LabelProp_TUNED.predict(kfX_test)

# %%
precision = precision_score(Y_pred, kfY_test, average='macro')
recall = recall_score(Y_pred, kfY_test, average='macro')
f1 = f1_score(Y_pred, kfY_test, average='macro')
accuracy = accuracy_score(kfY_test, Y_pred)
specificity=calc_specificity(kfY_test, Y_pred)
mcc_s=mcc(Y_pred,kfY_test)
auc_s=auc(Y_pred,kfY_test)
print('F1: ',f1,' Precision:',precision,' Recall',recall,' Specificity:',specificity,' Accuracy:',accuracy,' MCC:',mcc_s,' AUC:',auc_s) 

# %%
ET_TUNED.fit(kfX_train,kfY_train)

# %%
Y_pred=ET_TUNED.predict(kfX_test)

# %%
precision = precision_score(Y_pred, kfY_test, average='macro')
recall = recall_score(Y_pred, kfY_test, average='macro')
f1 = f1_score(Y_pred, kfY_test, average='macro')
accuracy = accuracy_score(kfY_test, Y_pred)
specificity=calc_specificity(kfY_test, Y_pred)
mcc_s=mcc(Y_pred,kfY_test)
auc_s=auc(Y_pred,kfY_test)
print('F1: ',f1,' Precision:',precision,' Recall',recall,' Specificity:',specificity,' Accuracy:',accuracy,' MCC:',mcc_s,' AUC:',auc_s) 

# %%
RF_TUNED.fit(kfX_train,kfY_train)

# %%
Y_pred=RF_TUNED.predict(kfX_test)

# %%
precision = precision_score(Y_pred, kfY_test, average='macro')
recall = recall_score(Y_pred, kfY_test, average='macro')
f1 = f1_score(Y_pred, kfY_test, average='macro')
accuracy = accuracy_score(kfY_test, Y_pred)
specificity=calc_specificity(kfY_test, Y_pred)
mcc_s=mcc(Y_pred,kfY_test)
auc_s=auc(Y_pred,kfY_test)
print('F1: ',f1,' Precision:',precision,' Recall',recall,' Specificity:',specificity,' Accuracy:',accuracy,' MCC:',mcc_s,' AUC:',auc_s) 

# %%
Ensemble_TUNED_4=VotingClassifier(estimators=[('XBG',XBG_TUNED),('LabelProp',LabelProp_TUNED),('XtraTree',ET_TUNED),('RF',RF_TUNED)],voting='hard',n_jobs=-1)

# %%
Ensemble_TUNED_4.fit(kfX_train,kfY_train)

# %%
Y_pred=Ensemble_TUNED_4.predict(kfX_test)

# %%
precision = precision_score(Y_pred, kfY_test, average='macro')
recall = recall_score(Y_pred, kfY_test, average='macro')
f1 = f1_score(Y_pred, kfY_test, average='macro')
accuracy = accuracy_score(kfY_test, Y_pred)
specificity=calc_specificity(kfY_test, Y_pred)
mcc_s=mcc(Y_pred,kfY_test)
auc_s=auc(Y_pred,kfY_test)
print('F1: ',f1,' Precision:',precision,' Recall',recall,' Specificity:',specificity,' Accuracy:',accuracy,' MCC:',mcc_s,' AUC:',auc_s) 

# %%
#SHAP Analysis
import shap
import matplotlib.pyplot as plt

# %%
def shap_plot(model,model_name):
    explainer = shap.Explainer(model.predict,kfX_test)
    shap_values = explainer(kfX_test)
    shap.plots.beeswarm(shap_values)
    summary_plot_filename = f'{model_name}_summary_plot.png'
    plt.savefig(summary_plot_filename)


# %%
shap_plot(SVM_TUNED,"SVM")

# %%
shap_plot(XBG_TUNED,"XBG")

# %%
shap_plot(KNN_TUNED,"KNN")

# %%
shap_plot(LabelProp_TUNED, "Label_Propagation")

# %%
shap_plot(ET_TUNED,"XtraTrees")

# %%
shap_plot(RF_TUNED,"Random Forest")

# %%
shap_plot(Ensemble_TUNED_4,"Ensemble_Model")

# %%
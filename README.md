# bank-marketing
https://github.com/MachineLearning2022/bank-marketing

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
train = pd.read_csv('data/bank.csv')
train.head()

train['job'].replace(['unknown'],train['job'].mode(),inplace=True)
train['marital'].replace(['unknown'],train['marital'].mode(),inplace=True)
train['education'].replace(['unknown'],train['education'].mode(),inplace=True)
train['default'].replace(['unknown'],train['default'].mode(),inplace=True)
train['housing'].replace(['unknown'],train['housing'].mode(),inplace=True)

train['y'].replace(to_replace = 'no', value = 0, inplace = True)
train['y'].replace(to_replace = 'yes', value = 1, inplace = True)
train['y'].value_counts()

def missing_data(data):
    total = data.isin(['unknown']).sum()
    #count计算总行数，计算0和1个数；sum计算所以0和1的加和
    percent = (data.isin(['unknown']).sum()/data.isin(['unknown']).count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['types'] = types
    return(np.transpose(tt))
missing_data(train)

train['job'].replace(['unknown'],train['job'].mode(),inplace=True)
train['marital'].replace(['unknown'],train['marital'].mode(),inplace=True)
train['education'].replace(['unknown'],train['education'].mode(),inplace=True)
train['default'].replace(['unknown'],train['default'].mode(),inplace=True)
train['housing'].replace(['unknown'],train['housing'].mode(),inplace=True)
train['loan'].replace(['unknown'],train['loan'].mode(),inplace=True)

train.drop('default',inplace=True,axis=1)

import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
labels ="no", "yes"
count = train["y"].value_counts()
textprops = {"fontsize":15}

plt.pie(count,  autopct='%1.2f%%', labels=labels,  startangle=25, textprops =textprops)

plt.title("Percentage of Saving", fontsize=15)
plt.show()

import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
labels ="no", "yes"
count = train["loan"].value_counts()
textprops = {"fontsize":15}

plt.pie(count,  autopct='%1.2f%%', labels=labels,  startangle=25, textprops =textprops)

plt.title("Percentage of loan", fontsize=15)
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
fg = sns.FacetGrid(data=train,height=6,hue='y')
fg = (fg.map(sns.distplot,'age',bins=[20,25,30,35,40,45,50,55,60,65,70,75,80,85,90]).add_legend())

fg=sns.catplot(data=train,x='job',kind='bar',y='y',ci=None)
fg=(fg.set_xticklabels(rotation=90)
    .set_axis_labels('','purchase rate')
    .despine(left=True))
    
fg=sns.catplot(data=train,x='marital',kind='bar',y='y',ci=None,aspect=.8)
fg=(fg.set_axis_labels('marital','purchase rate')
    .despine(left=True))
   
fg=sns.catplot(data=train,x='contact',kind='bar',y='y',ci=None,aspect=.8)
fg=(fg.set_axis_labels('marital','purchase rate')
    .despine(left=True))
fg=sns.catplot(data=train,x='education',kind='bar',y='y',ci=None)
fg=(fg.set_xticklabels(rotation=90)
    .set_axis_labels('','purchase rate')
    .despine(left=True))
fg=sns.catplot(data=train,x='housing',kind='bar',y='y',ci=None,order=['yes','no'])
fg.set_axis_labels("housing",'purchase rate')

fg=sns.catplot(data=train,x='campaign',kind='bar',y='y',ci=None,aspect=2)
fg.set_axis_labels("campaign",'purchase rate')
fg=sns.catplot(data=train,x='day_of_week',kind='bar',y='y',ci=None,aspect=2)
fg.set_axis_labels("campaign",'purchase rate')
fg=sns.catplot(data=train,x='poutcome',kind='bar',y='y',ci=None,aspect=2)
fg.set_axis_labels("poutcome",'purchase rate')
fg = sns.FacetGrid(data=train,height=6,hue='y')
fg = (fg.map(sns.distplot,'duration',bins=40) .add_legend())

import seaborn as sns
sns.set(rc={'figure.figsize':(12,8)})
sns.set_style('ticks') 

sns.boxplot(x = "y", y = "duration", data = train) 

train.drop('day_of_week',inplace=True,axis=1)
train.drop('housing',inplace=True,axis=1)
train.drop('emp.var.rate',inplace=True,axis=1)
train.drop('nr.employed',inplace=True,axis=1)
corr = train.corr()
plt.figure(figsize = (50, 50))
ax = sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns, linewidth = 1.2, cmap = 'YlGnBu', annot = True, annot_kws={"fontsize":17})
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 20)
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 20)
plt.show()

train=pd.get_dummies(train,drop_first = True)
train.head()
feature_cols = [column for column in train if column != 'y']
feature_cols
from sklearn import preprocessing
train['y'] = preprocessing.LabelEncoder().fit_transform(train['y'])
plt.figure(figsize=(15,10))
corr = train.corr()
train.corr()['y'].sort_values(ascending = False).plot(kind='barh')
train_data = train.copy()
train_data.drop('y',inplace=True,axis=1)

# 随机生成训练集和测试集
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(train_data,train['y'],test_size = 0.25,random_state = 1, stratify = train['y'])

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# 正则化
LR=LogisticRegression(penalty = 'l2',solver='saga')
LR.fit(X_train,y_train)
LR.score(X_train,y_train)

x_pre_test=LR.predict(X_test)
x_pre_test
y_pred = LR.predict(X_test)
y_pred_prob = LR.predict_proba(X_test) 
y_pred_prob

log_odds = LR.coef_[0]
a = pd.DataFrame.from_dict(dict(zip(train_data.columns,log_odds)),orient='index')
a
test_accuracy = LR.score(X_test, y_test)
print("Accuracy", test_accuracy)

from sklearn import metrics
test_precision = metrics.precision_score(y_test,y_pred,average = 'macro')
print("Precision", test_precision)
test_recall = metrics.recall_score(y_test, y_pred,average = 'macro')
print("Recall:", test_recall)
test_f1 = metrics.f1_score(y_test, y_pred,average = 'macro')
print("test_f1:", test_f1)

test_auc_roc = metrics.roc_auc_score(y_test, y_pred_prob[::,1])
print('Testing AUC:',test_auc_roc)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

plt.figure(figsize=(9,9))
sns.heatmap(cnf_matrix, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(test_accuracy)
plt.title(all_sample_title, size = 15);

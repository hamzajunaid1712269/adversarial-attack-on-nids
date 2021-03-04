import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno

from keras.models import Sequential
from keras.layers import Dense , Dropout
from keras.optimizers import RMSprop , Adam
from cleverhans.utils_keras import KerasModelWrapper
from keras.callbacks import EarlyStopping
from cleverhans . utils_tf import model_train , model_eval , batch_eval
from tensorflow import keras
from cleverhans.attacks import jsma



from cleverhans.attacks_tf import jacobian_graph

import tensorflow.compat.v1 as tf

from tensorflow . python . platform import flags

from sklearn.preprocessing import MinMaxScaler

from scipy import stats
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import  SGDClassifier
from sklearn.svm import LinearSVC  # SVM
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer  # Scoring functions
from sklearn.metrics import auc, f1_score, roc_curve, roc_auc_score  # Scoring fns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV  # Cross validation



df1 = pd.read_csv("UNSW_NB15_training-set.csv")
df2 = pd.read_csv("UNSW_NB15_testing-set.csv")
df= pd.concat([df1, df2]).drop(['id'],axis=1)
normal = df[df['label']==0]
anomaly = df[df['label']==1]
print(df.isnull().sum())
missingno.matrix(df)
plt.show()



method = "pearson"

corr_mat = df.corr(method=method)

plt.figure(figsize=(12,12)) 
sns.heatmap(corr_mat, square=True)
plt.show()


limit = 0.95

columns = corr_mat.columns
for i in range(corr_mat.shape[0]):
    for j in range(i+1, corr_mat.shape[0]):
        if corr_mat.iloc[i, j] >= 0.95:
            print(f"{columns[i]:20s} {columns[j]:20s} {corr_mat.iloc[i, j]}")
            
            
def reduce_column(s, to_keep):
   
    s = s.lower().strip()
    if s not in to_keep:
        return "others"
    else:
        return s  
        
            
def col_countplot(col,data=df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.set_style('whitegrid')
  
    ax = sns.countplot(x=col, hue='label', data=data)
    ax.legend(loc="upper right", labels=('normal', 'attack'))
    ax.set_title("Data")
    plt.xticks(rotation=45)
    plt.show()


def numeric_plot(col, data1=normal, data2=anomaly, label1='normal', label2='anomaly',):
     sns.set_style('whitegrid')
     sns.distplot(data1[col], label=label1, hist=False, rug=True)
     sns.distplot(data2[col], label=label2, hist=False, rug=True)
     plt.legend()
 

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
   
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    
    
    return errors

print(normal['proto'].nunique(), anomaly['proto'].nunique())
print(df['proto'].value_counts().head(10)*100/df.shape[0])


to_keep = ['tcp', 'udp', 'unas', 'arp', 'ospf']
df['proto_reduced'] = df['proto'].apply(reduce_column, args=(to_keep,))
col_countplot('proto_reduced')


col_countplot('service')
df['service'].unique() 
df['service']= np.where(df['service'] == '-', 'None', df['service'])
print(df['service'].unique())
col_countplot('service')

print(df['state'].nunique())
print(df['state'].value_counts())
to_keep = ['int', 'fin', 'con', 'req']
df['state_reduced'] = df['state'].apply(reduce_column, args=(to_keep,))
col_countplot('state_reduced')

col_countplot('ct_flw_http_mthd')


df.drop(columns=['proto_reduced', 'state_reduced'], inplace=True)
print(df.shape)

plt.figure(figsize=(20,4))
plt.subplot(121)
numeric_plot('sbytes')
plt.show()

plt.figure(figsize=(20,4))
plt.subplot(121)
numeric_plot('ct_srv_src')
plt.show()






le = preprocessing.LabelEncoder()

print(df.select_dtypes(exclude=np.number).columns)
df['proto'] = le.fit_transform(df['proto'])
df['service'] = le.fit_transform(df['service'])
df['state'] = le.fit_transform(df['state'])
df.drop(columns='attack_cat', inplace=True)




corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

print("columns to drop")
print(to_drop)

df.drop(columns=to_drop, inplace=True)
print(df.describe())

cat_col = ['proto', 'service', 'state']
num_col = list(set(df.columns) - set(cat_col))
"""
scaler = StandardScaler()

scaler = scaler.fit(df[num_col])
df[num_col] = scaler.transform(df[num_col])
"""


data_x = df.drop(['label'], axis=1) 

data_y = df['label'].values
data_y=data_y.astype('int')


scaler = MinMaxScaler().fit ( data_x )
data_x_scaled = np . array ( scaler . transform ( data_x))



X_train, X_test, y_train, y_test = train_test_split(data_x_scaled , data_y, test_size=.20,random_state = 42, stratify=data_y)




def mlp_model():
     model = Sequential()
     model.add(Dense(round(X_train.shape[1]/2), kernel_initializer='glorot_uniform',  activation = 'relu', input_dim = X_train.shape[1]))
    
    # Adding the second hidden layer
     model.add(Dense(round(X_train.shape[1]/2),  kernel_initializer='glorot_uniform',activation = 'relu'))


   
    
   
     # Binary classification task
        # Adding the output layer
     model.add(Dense( 1, kernel_initializer='glorot_uniform',activation = 'sigmoid'))
        # Compiling the ANN
     model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
     print(model.summary())
    
     return model



tf.compat.v1.disable_eager_execution()
X_placeholder = tf.placeholder(tf.float32 , shape=(None , X_train.shape[1]))
Y_placeholder = tf.placeholder(tf.float32 , shape=(None))

tf.set_random_seed(42)
model1 = mlp_model()
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

predictions = model1(X_placeholder)


def mlp_model_train(X, Y, val_split, batch_size, epochs_count):
    # Callback to stop if validation loss does not decrease
    callbacks = [EarlyStopping(monitor='val_loss', patience=2)]

    # Fitting the ANN to the Training set
    history = model1.fit(X, Y,
                   callbacks=callbacks,
                   validation_split=val_split,
                   batch_size = batch_size,
                   epochs = epochs_count,
                   shuffle=True)

    print(history.history)
    print(model1.summary())
    return history

def mlp_model_eval(X, Y, history, flag):
    # Predicting the results given instances X
    Y_pred = model1.predict(X)
    Y_pred = (Y_pred > 0.5)


    # Making the cufusion Matrix
    cm = confusion_matrix(Y, Y_pred)
    print("Confusion Matrix:\n", cm)
    print("Accuracy: ", accuracy_score(Y, Y_pred))

    if(len(np.unique(Y))) == 2:
        print("F1: ", f1_score(Y, Y_pred, average='binary'))
      

history = mlp_model_train(X_train, y_train,
                0.1, # Validation Split
                64, # Batch Size
                50 # Epoch Count
  
                )

source_samples = X_test.shape[0]
print(source_samples)
results = np.zeros((1, source_samples), dtype=float)
perturbations = np.zeros((1, source_samples), dtype=float)
grads = jacobian_graph(predictions , X_placeholder, 1)


jsma_params = {'theta': 1., 'gamma': 0.1,
                 'clip_min': 0., 'clip_max': 1.,
                 'y_target': None}

X_adv = np.zeros((source_samples, X_test.shape[1]))

for sample_ind in range(0, source_samples):
   
    current_class = int(np.argmax(y_test[sample_ind]))
    
    # Target the benign class
    for target in [0]:
        
        if (current_class == 0):
           break
        
        # This call runs the Jacobian-based saliency map approac
        adv_x = jsma.generate_np(sample_ind, **jsma_params)
        
        X_adv[sample_ind] = adv_x
        
"""

RFC = RandomForestClassifier()
#param = {'n_estimators':[100, 200, 300, 400]}
#grid = GridSearchCV(estimator=RFC,param_grid=param,cv = 3 ,n_jobs=-1)
#grid_result = grid.fit(X_train, y_train)
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
"""

RFC1 = RandomForestClassifier(criterion='gini', max_depth=22, min_samples_split=6, n_estimators=300, n_jobs=-1)
model=RFC1.fit(X_train, y_train)
y_test_pred=model.predict(X_test)
pred = model.score(X_test, y_test)
print("Acc:",(pred))

y_test_pred8=model.predict(X_adv)
pred12 = model.score(X_adv, y_test)
print("Acc:",(pred12))


"""

dt_clf = DecisionTreeClassifier()

param1 = {'max_depth':[8, 10, 12, 14],
         'min_samples_split':[2, 4, 6],'min_samples_leaf':[8,9,10]}


grid1 = GridSearchCV(estimator=dt_clf, param_grid=param1, cv = 3, n_jobs=-1)


grid_result1 = grid1.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result1.best_score_, grid_result1.best_params_))
best_grid1 = grid_result1.best_estimator_
grid_accuracy = evaluate(best_grid1, X_test, y_test)

"""

dt_param = {'max_depth': 14, 'min_samples_split':10, 'min_samples_leaf':6}
dt_best_clf = DecisionTreeClassifier(**dt_param)


model1=dt_best_clf.fit(X_train, y_train)

y_test_pred1=dt_best_clf.predict(X_test)
pred1 = model1.score(X_test, y_test)
print("Acc:",(pred1))

y_advtest_pred1=dt_best_clf.predict(X_adv)
print (" Accuracy score adversarial :", accuracy_score ( y_test , y_advtest_pred1 ))

clf = SGDClassifier()
"""
param2 = {'alpha':[10**x for x in range(-5,3)],  # Values for alpha
         'penalty':['l1', 'l2']}

grid2 = GridSearchCV(estimator=clf, param_grid=param2, cv = 3, n_jobs=-1)
grid_result2 = grid2.fit(X_train, y_train.values.ravel())
print("Best: %f using %s" % (grid_result2.best_score_, grid_result2.best_params_))
best_grid2 = grid_result2.best_estimator_
grid_accuracy1 = evaluate(best_grid2, X_test, y_test)
"""
param_best = {'alpha':0.0001,  
         'penalty':'l2'} 
clf1 = SGDClassifier(loss='hinge',**param_best)
model2=clf1.fit(X_train, y_train)
y_test_pred2=clf1.predict(X_test)
pred2=model2.score(X_test,y_test)
print("Acc:",(pred2))

y_advtest_pred2=clf1.predict(X_adv)
pred2_adv=model2.score(X_adv,y_test)
print("Acc:",(pred2_adv))

mlp_model_eval(X_test, y_test, history, 1)
mlp_model_eval(X_adv, y_test, history, 2)




cmap=sns.light_palette("blue")
labels= ['non-attack', 'attack']
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.heatmap(confusion_matrix(y_test,y_test_pred),annot=True, cmap=cmap, fmt='d',xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random forest on normal data")
plt.show()


cmap=sns.light_palette("blue")
labels= ['non-attack', 'attack']
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.heatmap(confusion_matrix(y_test,y_test_pred8),annot=True, cmap=cmap, fmt='d',xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Test Confusion Matrix")
plt.show()


cmap=sns.light_palette("blue")
labels= ['non-attack', 'attack']
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.heatmap(confusion_matrix(y_test,y_test_pred1),annot=True, cmap=cmap, fmt='d',xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("adv Test Confusion Matrix")
plt.show()

cmap=sns.light_palette("blue")
labels= ['non-attack', 'attack']
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.heatmap(confusion_matrix(y_test,y_advtest_pred1),annot=True, cmap=cmap, fmt='d',xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("adv Test Confusion Matrix")
plt.show()


cmap=sns.light_palette("blue")
labels= ['non-attack', 'attack']
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.heatmap(confusion_matrix(y_test,y_test_pred2),annot=True, cmap=cmap, fmt='d',xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Test Confusion Matrix")
plt.show()


cmap=sns.light_palette("blue")
labels= ['non-attack', 'attack']
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.heatmap(confusion_matrix(y_test,y_advtest_pred2),annot=True, cmap=cmap, fmt='d',xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Test_adv Confusion Matrix")
plt.show()



feats = dict ()
total = 0
orig_attack = X_test - X_adv
for i in range (0 , orig_attack . shape [0]) :

    ind = np . where ( orig_attack [i , :] != 0) [0]
    total += len ( ind )
    for j in ind :
      if j in feats :
        feats [j] += 1
    else :
       feats [j] = 1

  # The number of features that where changed for the adversarial samples
print (" Number of unique features changed :", len ( feats . keys () ))
print (" Number of average features changed per datapoint ", total / len ( orig_attack ))

top_10 = sorted ( feats , key = feats . get , reverse = True ) [:10]
top_20 = sorted ( feats , key = feats . get , reverse = True ) [:20]
print (" Top ten features :", df. columns [ top_10 ])

top_10_val = [100* feats [k ] / y_test . shape [0] for k in top_10 ]
plt . figure ( figsize =(16 , 12) )
plt . bar ( np . arange (10) , top_10_val , align ='center')
plt . xticks ( np . arange (10) , df . columns [ top_10 ], rotation ='vertical')
plt . title ('Feature participation in adversarial examples' )
plt . ylabel ('Percentage (%)')
plt . xlabel ('Features' )

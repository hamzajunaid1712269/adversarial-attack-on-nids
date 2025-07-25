import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import missingno

import cv2 as cv
from PIL import Image,ImageTk

from tkinter import Tk,Label,Button,Canvas,PhotoImage,font,Label, Frame, Entry,StringVar,Toplevel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import ImageTk, Image

from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import  SGDClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 



df1 = pd.read_csv("UNSW_NB15_training-set.csv")
df2 = pd.read_csv("UNSW_NB15_testing-set.csv")
df= pd.concat([df1, df2]).drop(['id'],axis=1)

normal = df[df['label']==0]
anomaly = df[df['label']==1]
print(df.isnull().sum())
missingno.matrix(df)
plt.savefig("missingno.png",bbox_inches='tight')
method = "pearson"

corr_mat = df.corr(method=method)

plt.figure(figsize=(12,12)) 
sns.heatmap(corr_mat, square=True)
plt.savefig("Heatmap.png",bbox_inches='tight')


limit = 0.95

columns = corr_mat.columns
for i in range(corr_mat.shape[0]):
    for j in range(i+1, corr_mat.shape[0]):
        if corr_mat.iloc[i, j] >= 0.95:
            print(f"{columns[i]:20s} {columns[j]:20s} {corr_mat.iloc[i, j]}")
     











def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
   
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    
    
    return errors
def reduce_column(s, to_keep):
   
    s = s.lower().strip()
    if s not in to_keep:
        return "others"
    else:
        return s  
        

def col_countplot(col,data=df):
    fig, ax = plt.subplots(figsize=(12,8))
    sns.set_style('whitegrid')
  
    ax = sns.countplot(x=col, hue='label', data=data)
    ax.legend(loc="upper right", labels=('normal', 'attack'))
    ax.set_title("Data")
    plt.xticks(rotation=45)
    plt.savefig("count.png",bbox_inches='tight')
    plt.show()
    


def numeric_plot(col,data1=normal, data2=anomaly, label1='normal', label2='anomaly',):
     sns.set_style('whitegrid')
     
     sns.distplot(data1[col].apply(np.log1p), label=label1, hist=False, rug=True)
     sns.distplot(data2[col].apply(np.log1p), label=label2, hist=False, rug=True)
     plt.savefig("count2.png",bbox_inches='tight')
     
def pair(x,y):
    sns.set_style('whitegrid')
    sns.pairplot(df[[x,y]])
    plt.savefig("count1.png",bbox_inches='tight')
    
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


def plot_heatmap(feature):
    print() 
    attack_categories_sorted = np.sort(df.attack_cat.unique())
    attack_category_heatmap = []
    
    feature_unique_values_df = df[feature].value_counts().sort_values(ascending=False)
    
    if feature_unique_values_df.shape[0] > 10:
        print('Showing only top 10 unique values out of ' + str(feature_unique_values_df.shape[0]) + ' values\n')
        feature_unique_values = feature_unique_values_df.index[:10]
    else:
        feature_unique_values = feature_unique_values_df.index
    
    for attack_category in attack_categories_sorted:

        
        attack_cat_dataset = df[df.attack_cat==attack_category]
        
        attack_cat_field_counts = []
        for index in feature_unique_values:
            attack_cat_field_counts.append(attack_cat_dataset[attack_cat_dataset[feature] == index].shape[0])

        attack_category_heatmap.append(attack_cat_field_counts)

    
    attack_category_heatmap = np.array(attack_category_heatmap).transpose().tolist()
    plt.figure(figsize = (15, 10))
    sns.heatmap(attack_category_heatmap, 
                xticklabels = attack_categories_sorted, 
                yticklabels = feature_unique_values, 
                cmap='RdYlBu_r',
                linewidth = 1,
                annot=True, fmt='g')
    plt.savefig(feature+"heat.png",bbox_inches='tight')


plot_heatmap('proto')
plot_heatmap('service')
plot_heatmap('state')

plt.show()

corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

print("columns to drop")
print(to_drop)


df.drop(columns=to_drop, inplace=True)
print(df.describe())

train_class_distribution = df['attack_cat'].value_counts().tolist()






percentage=[]
for i in train_class_distribution :
    print('Number of data points in class',  ':', '(', np.round((i/len(df)*100), 3), '%)')
    percentage.append( np.round((i/len(df)*100), 3))

for i in percentage:
    if i < 6:
        percentage.remove(i)

percentage.remove(1.039)
percentage.remove(0.586)


other=sum(percentage)
other=100-other
percentage.append(other)
print(percentage)

attack=['normal','generic','exploit','Fuzzers','DOS','Others']




plt.pie(percentage, labels = attack,autopct='%1.1f%%')
plt.axis('equal')
plt.savefig("pie.png",bbox_inches='tight')

 



col_countplot('attack_cat')


df.drop('attack_cat', axis=1, inplace=True)

cat_col = ['proto', 'service', 'state']
num_col = list(set(df.columns) - set(cat_col))
le = preprocessing.LabelEncoder()


print(df.select_dtypes(exclude=np.number).columns)
df['proto'] = le.fit_transform(df['proto'])
df['service'] = le.fit_transform(df['service'])
df['state'] = le.fit_transform(df['state'])

data_x = df.drop(['label'], axis=1) 

data_y = df['label'].values
data_y=data_y.astype('int')



scaler = MinMaxScaler().fit ( data_x )
data_x_scaled = np . array ( scaler . transform ( data_x))


"""

X_train, X_test, y_train, y_test = train_test_split(data_x_scaled , data_y, test_size=.20,random_state = 42, stratify=data_y)






dt_clf = DecisionTreeClassifier()

param1 = {'max_depth':[8, 10, 12, 14],
         'min_samples_split':[2, 4, 6],'min_samples_leaf':[8,9,10]}


grid1 = GridSearchCV(estimator=dt_clf, param_grid=param1, cv = 3, n_jobs=-1)


grid_result1 = grid1.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result1.best_score_, grid_result1.best_params_))
best_grid1 = grid_result1.best_estimator_
grid_accuracy = evaluate(best_grid1, X_test, y_test)



dt_param = {'max_depth': 14, 'min_samples_split':10, 'min_samples_leaf':6}
dt_best_clf = DecisionTreeClassifier(**dt_param)


model1=dt_best_clf.fit(X_train, y_train)

y_test_pred1=dt_best_clf.predict(X_test)
pred1 = model1.score(X_test, y_test)
print("Acc:",(pred1))


features = df.columns
importances = model1.feature_importances_
indices = np.argsort(importances)

# customized number 
num_features = 10 

plt.figure(figsize=(12,8))


# only plot the customized number of features
plt.barh(range(num_features), importances[indices[-num_features:]], color='b', align='center')
plt.yticks(range(num_features), [features[i] for i in indices[-num_features:]])
plt.xlabel('Relative Importance')
plt.savefig("fi-D.png",bbox_inches='tight')

clf = SGDClassifier()

param2 = {'alpha':[10**x for x in range(-5,3)],  # Values for alpha
         'penalty':['l1', 'l2']}

grid2 = GridSearchCV(estimator=clf, param_grid=param2, cv = 3, n_jobs=-1)
grid_result2 = grid2.fit(X_train, y_train)
print("Best: %f using %s" % (grid_result2.best_score_, grid_result2.best_params_))
best_grid2 = grid_result2.best_estimator_
grid_accuracy1 = evaluate(best_grid2, X_test, y_test)

param_best = {'alpha':0.0001,  
         'penalty':'l2'} 
clf1 = SGDClassifier(loss='hinge',**param_best)
model2=clf1.fit(X_train, y_train)
y_test_pred2=clf1.predict(X_test)
pred2=model2.score(X_test,y_test)
print("Acc:",(pred2))




features = df.columns
importances  = model2.coef_[0]
indices = np.argsort(importances)

# customized number 
num_features = 10 

plt.figure(figsize=(12,8))
plt.title('Feature Importances')

# only plot the customized number of features
plt.barh(range(num_features), importances[indices[-num_features:]], color='b', align='center')
plt.yticks(range(num_features), [features[i] for i in indices[-num_features:]])
plt.xlabel('Relative Importance')
plt.savefig("fi-sgd.png",bbox_inches='tight')



cmap=sns.light_palette("blue")
labels= ['non-attack', 'attack']
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.heatmap(confusion_matrix(y_test,y_test_pred1),annot=True, cmap=cmap, fmt='d',xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Decision tree Confusion Matrix")
plt.savefig("c1.png",bbox_inches='tight')


cmap=sns.light_palette("blue")
labels= ['non-attack', 'attack']
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.heatmap(confusion_matrix(y_test,y_test_pred2),annot=True, cmap=cmap, fmt='d',xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Test Confusion Matrix")

plt.savefig("c2.png",bbox_inches='tight')
"""

class MyFirstGUI:
    def __init__(self, master):
        self.master = master
        
    
        
        

        
        
       

        
        self.Attack_button1 = Button(master, text="Decision Tree",fg='white',padx = 10, pady = 2,width=11,height=2,bg="#000000", command=self.dt,borderwidth=6)
        
        
         
        self.Attack_button1.pack()
        
        self.Attack_button2 = Button(master, text="SVM",fg='white',padx = 10, pady = 2,width=11,height=2,bg="#000000", command=self.svm,borderwidth=6)
        self.Attack_button2.pack()
        
         
        self.Attack_button3 = Button(master, text="Dataset",fg='white',padx = 10, pady = 2,width=11,height=2,bg="#000000", command=self.details,borderwidth=6)
        self.Attack_button3.pack()
        
        self.Attack_button4 = Button(master, text="Attacks",fg='white',padx = 10, pady = 2,width=11,height=2,bg="#000000", command=self.Attacks,borderwidth=6)
        self.Attack_button4.pack()
        
          
        self.Attack_button5 = Button(master, text="Heatmaps",fg='white',padx = 10, pady = 2,width=11,height=2,bg="#000000", command=self.Heatmap,borderwidth=6)
        self.Attack_button5.pack()
        
        self.Attack_button6 = Button(master, text="Catagorical",fg='white',padx = 10, pady = 2,width=11,height=2,bg="#000000", command=self.openNewWindow,borderwidth=6)
        self.Attack_button6.pack()
        
         
        self.Attack_button7 = Button(master, text="Pairplot",fg='white',padx = 10, pady = 2,width=11,height=2,bg="#000000", command=self.openNewWindow1,borderwidth=6)
        self.Attack_button7.pack()
        
             
        self.Attack_button8 = Button(master, text="Numeric",fg='white',padx = 10, pady = 2,width=11,height=2,bg="#000000", command=self.openNewWindow2,borderwidth=6)
        self.Attack_button8.pack()
        
        
        
      

        
        self.close_button = Button(master, text="Close",width=11,height=2,bg="#808080", command=master.destroy,borderwidth=6)
        self.close_button.pack()
        
        

        

    def dt(self):
         print("Tain_Confusion Matrix")
         
         img1 = cv.imread("c1.png")
         img4 = cv.imread("fi-D.png")
         cv.namedWindow("Matrix")
         cv.namedWindow("Feature")
         
         cv.imshow('Matrix',img1)
         cv.imshow('feature',img4)
         
         
    def svm(self):
         print("Attacked_Confusion Matrix")
         
         
         img1 = cv.imread("c2.png")
         img4 = cv.imread("fi-sgd.png")
         cv.namedWindow("Matrix")
         cv.namedWindow("Feature")
         
         cv.imshow('Matrix',img1)
         cv.imshow('feature',img4)
        
    def  details(self):
         
         print("Details")
         
         img1 = cv.imread("Heatmap.png")
         img4 = cv.imread("missingno.png")
         cv.namedWindow("Heatmap")
         cv.namedWindow("Size")
         
         cv.imshow('Heatmap',img1)
         cv.imshow('Size',img4)
        
         cv.waitKey(0)
    def  Attacks(self):
         print("Details")
         
         img1 = cv.imread("pie.png")
         img4 = cv.imread("attack_cat.png")
         cv.namedWindow("piechart")
         cv.namedWindow("distribution")
         
         cv.imshow('Piechart',img1)
         cv.imshow('distribution',img4)
         
    def  Heatmap(self):
         print("Heat")
         
         img1 = cv.imread("protoheat.png")
         img4 = cv.imread("serviceheat.png")
         img7 = cv.imread("stateheat.png")
         cv.namedWindow("Proto")
         cv.namedWindow("Service")
         cv.namedWindow("State")
         
         cv.imshow('Proto',img1)
         cv.imshow('Service',img4)
         cv.imshow('State',img7)
         col_countplot('proto')
        
    def openNewWindow(self):
        
      
        root = Tk()
       
        newWindow = Toplevel(root)
  
    
        newWindow.title("New Window")
  
    
        Label(newWindow, 
        text ="This is a new window").pack()
        newWindow.geometry("200x200")
    
    
        state2= Label(newWindow, text="Feature")
    

        state2.pack()
   




        statevalue = StringVar()
   


        e1 = Entry(newWindow, textvariable=statevalue)
        


        e1.pack()
        
        def  Count():
             x=(e1.get())
             col_countplot(x)
             if x == 'proto':
                 img1 = cv.imread("proto_reduced.png")
         
                 cv.namedWindow("count")
         
         
                 cv.imshow('count',img1)
            
             elif x =='state':
                    img1 = cv.imread("state_reduced.png")
         
                    cv.namedWindow("count")
         
         
                    cv.imshow('count',img1)
                    
             else:
                 
                
                 img1 = cv.imread("count.png")
         
                 cv.namedWindow("count")
         
         
                 cv.imshow('count',img1)
            
             
             
             
           
         
      
        Button(newWindow, text="Submit ", command=Count).pack()
        root.mainloop()
        
    def openNewWindow1(self):
        
      
        root = Tk()
       
        newWindow = Toplevel(root)
  
    
        newWindow.title("New Window")
  
    
        Label(newWindow, 
        text ="This is a new window").pack()
        newWindow.geometry("200x200")
    
    
        state2= Label(newWindow, text="Feature")
        state= Label(newWindow, text="Feature")
    

        state2.pack()
        state.pack()
   




        statevalue = StringVar()
        statevalue1 = StringVar()
   


        e1 = Entry(newWindow, textvariable=statevalue)
        e2 = Entry(newWindow, textvariable=statevalue1)
        


        e1.pack()
        e2.pack()
        
        def  Count():
             x=(e1.get())
             y=(e2.get())
             pair(x,y)
             
             img1 = cv.imread("count1.png")
         
             cv.namedWindow("count")
         
         
             cv.imshow('count',img1)
            
             
             
             
           
         
      
        Button(newWindow, text="Submit ", command=Count).pack()
        root.mainloop()
        
        
    
    def openNewWindow2(self):
        
      
        root = Tk()
       
        newWindow = Toplevel(root)
  
    
        newWindow.title("New Window")
  
    
        Label(newWindow, 
        text ="This is a new window").pack()
        newWindow.geometry("200x200")
    
    
        state2= Label(newWindow, text="Feature")
        
    

        state2.pack()
        
   




        statevalue = StringVar()
        
   


        e1 = Entry(newWindow, textvariable=statevalue)
  
        


        e1.pack()
        
        
        def  Count():
             x=(e1.get())
             
             numeric_plot(x)
             
             img1 = cv.imread("count2.png")
         
             cv.namedWindow("count")
         
         
             cv.imshow('count',img1)
            
             
             
             
           
         
      
        Button(newWindow, text="Submit ", command=Count).pack()
        root.mainloop()
        
        
        
        
            
          
        
def main():
    root = Tk()
    root.geometry('100x440')
    """
    load=Image.open(r'Images\\w22.jpg')
    render=ImageTk.PhotoImage(load)
    img=Label(root,image=render)
    img.place(x=0,y=0)
    """
    root.configure(bg="gray")
    MyFirstGUI(root)
    root.mainloop()
    
    
    
 
    
    

   
    
if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 15:22:31 2022

@author: Danno
"""
import time
import numpy                         as     np
import matplotlib.pyplot             as     plt
from   sklearn                       import metrics
from   sklearn.model_selection       import train_test_split
from   sklearn                       import tree
from   sklearn.metrics               import confusion_matrix
from   sklearn.tree                  import DecisionTreeClassifier
import seaborn                       as     sns


def csv_import(f_path: str):
    x = np.loadtxt(f_path,dtype = np.float16, delimiter=',',skiprows=1,usecols=[1,2])
    y = np.loadtxt(f_path,dtype = np.str_, delimiter=',',skiprows=1,usecols=[3])
    return x,y


def visualize_data(tree_clf,cm_train,cm_test,accuracy_train,accuracy_test,Xn,Yn):
    #---plot the tree classifer---
   
    #plt.figure(figsize=(30,30))
    fig,axes = plt.subplots(1,4,gridspec_kw={'width_ratios': [1, 1, 1,1]},figsize=(15,5))

    # current_ax = plt.axes()
    tree.plot_tree(tree_clf,filled=(True),ax = axes[0])
    
    #display heatmap of confusion matrix: y_train == y_train_pred
    axes[1].set_title(f'Train Tree Classifier Accuracy = {accuracy_train:0.4f}')
    #plot confusion matrix results and accuracy 
    heat_map_axis = sns.heatmap(cm_train,annot=True, linewidth=10,cmap='icefire',fmt='d',cbar = False,ax = axes[1])
    heat_map_axis.set(xlabel = '$\hat{y} \quad estimated$',ylabel = 'y real')    
    
    #display heatmap of confusion matrix: y_test == y_test_pred
    axes[2].set_title(f'Test Tree Classifier Accuracy = {accuracy_test:0.4f}')
    #plot confusion matrix results and accuracy 
    heat_map_axis = sns.heatmap(cm_train,annot=True, linewidth=10,cmap='icefire',fmt='d',cbar = False,ax = axes[2])
    heat_map_axis.set(xlabel = '$\hat{y} \quad estimated$',ylabel = 'y real')     
   
    
    #----gimel 4.3----
    #fig =  plt.figure(figsize=(5,5))
    axes[3].set_xlabel('X1')
    axes[3].set_ylabel('X2')
    #axes[3].set_xlim(-1,8)
    #axes[3].set_ylim(-1,8)
    axes[3].scatter(Xn[Yn=='x',0],Xn[Yn=='x',1],label='x')
    axes[3].scatter(Xn[Yn=='o',0],Xn[Yn=='o',1],label='o')   
    axes[3].legend()
    fig.tight_layout()   
   
    return

def decision_tree_classifier(Xn,Yn):
    #---split data into 0.7 train and 0.3 test---
    x_train,x_test,y_train,y_test = train_test_split(Xn,Yn,test_size=0.4,random_state=0)
    
    #---classifier trainning---
    tree_clf = DecisionTreeClassifier(criterion='entropy',min_samples_leaf=1)
    #create tree classifier = tree_clf
    tree_clf.fit(x_train,y_train)    #train classifer on training data
    
    #---EVALUATE TREE CLASSIFIER---
    #TRAIN data - evaluate predication
    y_train_pred = tree_clf.predict(x_train)
    cm_train = metrics.confusion_matrix(y_train, y_train_pred)
    acc_train_pred = cm_train.diagonal().sum() / cm_train.sum()
    
    #TEST data - evaluate predication
    y_test_pred = tree_clf.predict(x_test)
    cm_test = confusion_matrix(y_test,y_test_pred)
    acc_test_pred = cm_test.diagonal().sum() / cm_test.sum()
    
    visualize_data(tree_clf,cm_train,cm_test,acc_train_pred,acc_test_pred,Xn,Yn)
    return












def binary_classifier():
    # load csv files
    Xn, Yn = csv_import(f_path=r"C:\Users\fello\Desktop\ML\hw4\data.csv")

    decision_tree_classifier(Xn,Yn)







if __name__ == "__main__":
    st = time.time()
    binary_classifier()
    print(f'Elapsed Time:{(time.time() - st)*1000:.03f} [ms]')
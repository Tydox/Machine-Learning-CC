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
from   sklearn.tree                  import DecisionTreeClassifier
from   sklearn                       import tree
from   sklearn.metrics               import confusion_matrix
import seaborn                       as     sns

from sklearn.ensemble                import RandomForestClassifier



def csv_import(f_path: str):
    x = np.loadtxt(f_path,dtype = np.float16, delimiter=',',skiprows=1,usecols=[2,3])
    y = np.loadtxt(f_path,dtype = np.str_, delimiter=',',skiprows=1,usecols=[4])

    return x,y


def visualize_data(tree_clf,cm_train,cm_test,accuracy_train,accuracy_test):
    #---plot the tree classifer---
   
    #plt.figure(figsize=(30,30))
    fig,axes = plt.subplots(1,3,gridspec_kw={'width_ratios': [1, 1, 1]},figsize=(15,5))
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
    return

def decision_tree_classifier():
    # load csv files
    Xn, Yn = csv_import(f_path=r"C:\Users\fello\Desktop\ML\Trees\iris.csv")
    
    #---split data into 0.7 train and 0.3 test---
    x_train,x_test,y_train,y_test = train_test_split(Xn,Yn,test_size=0.3,random_state=0)
    
    #---classifier trainning---
    tree_clf = DecisionTreeClassifier(criterion='entropy',max_depth=(2))    #create tree classifier = tree_clf
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
    
    visualize_data(tree_clf,cm_train,cm_test,acc_train_pred,acc_test_pred)
    return


def dtree_cval():
    # load csv files
    Xn, Yn = csv_import(f_path=r"C:\Users\fello\Desktop\ML\Trees\iris.csv")
    
    max_depth = [num for num in range(1,16)]    
    acc_train_mat = []
    acc_test_mat = []

    for k in range(1,4): #K=3 folds
        acc_train_list = []
        acc_test_list = []
        #---split data into 0.7 train and 0.3 test---
        x_train,x_test,y_train,y_test = train_test_split(Xn,Yn,test_size=0.3,random_state=k)
    
        for max_d in max_depth:                
            #---classifier trainning---
            tree_clf = DecisionTreeClassifier(criterion='entropy',max_depth=(max_d))    #create tree classifier = tree_clf
            tree_clf.fit(x_train,y_train)    #train classifer on training data
            
            #---EVALUATE TREE CLASSIFIER---
            #TRAIN data - evaluate predication
            y_train_pred = tree_clf.predict(x_train)
            cm_train = metrics.confusion_matrix(y_train, y_train_pred)
            acc_train_list.append(cm_train.diagonal().sum() / cm_train.sum()) 
            
            #TEST data - evaluate predication
            y_test_pred = tree_clf.predict(x_test)
            cm_test = confusion_matrix(y_test,y_test_pred)
            acc_test_list.append(cm_test.diagonal().sum() / cm_test.sum())
            
        
        acc_train_mat.append(acc_train_list)
        acc_test_mat.append(acc_test_list)

    acc_train = np.array(acc_train_mat).mean(axis=0)
    acc_test = np.array(acc_test_mat).mean(axis=0)

    plt.figure(figsize=(15,5))
    plt.plot(max_depth,acc_train,label='Train')
    plt.plot(max_depth,acc_test,label='Test')
    plt.xlabel("Tree's Maximal Depth")
    plt.ylabel('Accuracy')
    plt.title(label = f"Accuracy VS Tree's Maximal Depth")
    plt.legend(loc = "center right")
    plt.tight_layout()   
    plt.savefig('max_depth.jpg', dpi=300)

        
    min_leafs = [num for num in range(1,16)]    
    acc_train_mat = []
    acc_test_mat = []
    for k in range(1,4): #K=3 folds
        acc_train_list = []
        acc_test_list = []
        #---split data into 0.7 train and 0.3 test---
        x_train,x_test,y_train,y_test = train_test_split(Xn,Yn,test_size=0.3,random_state=k)
    
        for min_leaf in min_leafs:                
            #---classifier trainning---
            tree_clf = DecisionTreeClassifier(criterion='entropy',min_samples_leaf=(min_leaf))    #create tree classifier = tree_clf
            tree_clf.fit(x_train,y_train)    #train classifer on training data
            
            #---EVALUATE TREE CLASSIFIER---
            #TRAIN data - evaluate predication
            y_train_pred = tree_clf.predict(x_train)
            cm_train = metrics.confusion_matrix(y_train, y_train_pred)
            acc_train_list.append(cm_train.diagonal().sum() / cm_train.sum()) 
            
            #TEST data - evaluate predication
            y_test_pred = tree_clf.predict(x_test)
            cm_test = confusion_matrix(y_test,y_test_pred)
            acc_test_list.append(cm_test.diagonal().sum() / cm_test.sum())
        
        
        acc_train_mat.append(acc_train_list)
        acc_test_mat.append(acc_test_list)

    acc_train = np.array(acc_train_mat).mean(axis=0)
    acc_test = np.array(acc_test_mat).mean(axis=0)

    plt.figure(figsize=(15,5))
    plt.plot(min_leafs,acc_train,label='Train')
    plt.plot(min_leafs,acc_test,label='Test')
    plt.xlabel("Minimal Numbers of Samples in a leaf")
    plt.ylabel('Accuracy')
    plt.title(label = f'Accuracy VS Minimal Numbers of Samples in a leaf')
    plt.legend(loc = "center right")
    plt.tight_layout()   
    plt.savefig('min_leafs.jpg', dpi=300)


    max_leafs = [num for num in range(2,16)]   
    acc_train_mat = []
    acc_test_mat = []
    for k in range(1,4): #K=3 folds
        acc_train_list = []
        acc_test_list = []
        #---split data into 0.7 train and 0.3 test---
        x_train,x_test,y_train,y_test = train_test_split(Xn,Yn,test_size=0.3,random_state=k)
    
        for max_leaf in max_leafs:                
            #---classifier trainning---
            tree_clf = DecisionTreeClassifier(criterion='entropy',max_leaf_nodes=(max_leaf))    #create tree classifier = tree_clf
            tree_clf.fit(x_train,y_train)    #train classifer on training data
            
            #---EVALUATE TREE CLASSIFIER---
            #TRAIN data - evaluate predication
            y_train_pred = tree_clf.predict(x_train)
            cm_train = metrics.confusion_matrix(y_train, y_train_pred)
            acc_train_list.append(cm_train.diagonal().sum() / cm_train.sum()) 
            
            #TEST data - evaluate predication
            y_test_pred = tree_clf.predict(x_test)
            cm_test = confusion_matrix(y_test,y_test_pred)
            acc_test_list.append(cm_test.diagonal().sum() / cm_test.sum())
            

        
        acc_train_mat.append(acc_train_list)
        acc_test_mat.append(acc_test_list)

    acc_train = np.array(acc_train_mat).mean(axis=0)
    acc_test = np.array(acc_test_mat).mean(axis=0)

    plt.figure(figsize=(15,5))
    plt.plot(max_leafs,acc_train,label='Train')
    plt.plot(max_leafs,acc_test,label='Test')
    plt.xlabel("Maximal Numbers of leafs")
    plt.ylabel('Accuracy')
    plt.title(label = f'Accuracy VS Maximal Numbers of leafs')
    plt.legend(loc = "center right")
    plt.tight_layout()   
    plt.savefig('max_leafs.jpg', dpi=300)

    return




if __name__ == "__main__":
    st = time.time()
    dtree_cval()
    print(f'Elapsed Time:{(time.time() - st)*1000:.03f} [ms]')

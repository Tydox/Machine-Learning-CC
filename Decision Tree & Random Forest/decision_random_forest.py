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
from sklearn.ensemble                import RandomForestClassifier
from sklearn.metrics                 import ConfusionMatrixDisplay

# class Forest:
#     forest_clf = None
#     cm_train = None
#     acc_train_pred = None
#     y_test_pred= None
#     cm_test = None
#     acc_test_pred = None



def csv_import_forest(f_path: str):
    x = np.loadtxt(f_path,dtype = np.float16, delimiter=',',skiprows=1,usecols=[0,1,2,3,4,5,6,7])
    y = np.loadtxt(f_path,dtype = np.str_, delimiter=',',skiprows=1,usecols=[8])

    return x,y


# def visualize_data(tree_clf,cm_train,cm_test,accuracy_train,accuracy_test):
#     #---plot the tree classifer---
#     #plt.figure(figsize=(30,30))
#     fig,axes = plt.subplots(1,3,gridspec_kw={'width_ratios': [1, 1, 1]},figsize=(15,5))
#    # current_ax = plt.axes()
#     tree.plot_tree(tree_clf,filled=(True),ax = axes[0])
#     #display heatmap of confusion matrix: y_train == y_train_pred
#     axes[1].set_title(f'Train Tree Classifier Accuracy = {accuracy_train:0.4f}')
#     #plot confusion matrix results and accuracy 
#     heat_map_axis = sns.heatmap(cm_train,annot=True, linewidth=10,cmap='icefire',fmt='d',cbar = False,ax = axes[1])
#     heat_map_axis.set(xlabel = '$\hat{y} \quad estimated$',ylabel = 'y real')    
#     #display heatmap of confusion matrix: y_test == y_test_pred
#     axes[2].set_title(f'Test Tree Classifier Accuracy = {accuracy_test:0.4f}')
#     #plot confusion matrix results and accuracy 
#     heat_map_axis = sns.heatmap(cm_train,annot=True, linewidth=10,cmap='icefire',fmt='d',cbar = False,ax = axes[2])
#     heat_map_axis.set(xlabel = '$\hat{y} \quad estimated$',ylabel = 'y real')     
#     return

def visualize_forest():
    
    
    return

def decision_forest_classifier():
    
    # load csv files
    Xn, Yn = csv_import_forest(f_path=r"C:\Users\Danno\Desktop\hw3\pima.csv")
    
    #---split data into 0.7 train and 0.3 test---
    x_train,x_test,y_train,y_test = train_test_split(Xn,Yn,test_size=0.2,random_state=0)
    
    #---classifier trainning---
    forest_size = 10
    forest_clf = RandomForestClassifier(criterion='entropy',n_estimators=forest_size)    #create tree classifier = forest_clf
    forest_clf.fit(x_train,y_train)    #train classifer on training data
    
    #---EVALUATE TREE CLASSIFIER---
    #TRAIN data - evaluate predication
    y_train_pred = forest_clf.predict(x_train)
    cm_train = metrics.confusion_matrix(y_train, y_train_pred)
    acc_train_pred = cm_train.diagonal().sum() / cm_train.sum()
    
    
    #TEST data - evaluate predication
    y_test_pred = forest_clf.predict(x_test)
    cm_test = confusion_matrix(y_test,y_test_pred)
    acc_test_pred = cm_test.diagonal().sum() / cm_test.sum()
    
    delta = abs(acc_train_pred-acc_test_pred)
    print(f'Accuracy Train: {acc_train_pred:0.4f}\nAccuracy Test: {acc_test_pred:0.4f}\nDelta: {delta:0.4f}')
    # fig,axes = plt.subplots(1,3,gridspec_kw={'width_ratios': [1, 1, 3]},figsize=(30,10))
    # cm_disp = ConfusionMatrixDisplay.from_estimator(forest_clf,x_train,y_train,ax = axes[0],cmap='Blues')
    # cm_disp.im_.colorbar.remove()
    # cm_disp = ConfusionMatrixDisplay.from_estimator(forest_clf,x_test,y_test, ax = axes[1],cmap='Blues')
    # cm_disp.im_.colorbar.remove()
    #features_names = ['preg','plas','pres','skin','test','mass','pedi','age']
    #classes_names = ['0','1']
    #tree.plot_tree(forest_clf.estimators_[1],feature_names=features_names,class_names=classes_names,filled=(True),ax = axes[2])
        
    
    fig,axes = plt.subplots(nrows = 1,ncols = 3,gridspec_kw={'width_ratios': [1, 1,2]},figsize=(21,5))
    ConfusionMatrixDisplay.from_estimator(forest_clf,x_train,y_train,ax = axes[0],cmap='Blues',colorbar=(False))
    ConfusionMatrixDisplay.from_estimator(forest_clf,x_test,y_test, ax = axes[1],cmap='Blues',colorbar=(False))
    axes[0].set_title(f'Train Classifier Accuracy = {acc_train_pred:0.4f}')
    axes[1].set_title(f'Test Classifier Accuracy = {acc_test_pred:0.4f}')
    
    feature_importances = np.array([TREE.feature_importances_ for TREE in forest_clf.estimators_])
    features_names = ['preg','plas','pres','skin','test','mass','pedi','age']
    
    features_mean = feature_importances.mean(axis=0)
    features_variance = feature_importances.std(axis=0)
    axes[2].yaxis.grid(True,zorder=1)
    p1 = axes[2].bar(features_names,features_mean,yerr = features_variance,capsize=10,ecolor='black',zorder=2, align='center')
    axes[2].set_title(f'Feature Importances, Assesed over {forest_size} Trees')
    axes[2].bar_label(p1,label_type='edge')
    axes[2].set_xlabel('Feature Importance')
    axes[2].set_ylabel('Feature Name')
   
    fig.tight_layout()
    #fig.savefig('Basic.png', dpi=300)
    
    depth_list = [1,2,3,4,5,10,20]
    forest_size_list = [1,2,3,4,5,10,20,30,50,100]
    
    acc_train = [] #empty list
    acc_test = [] #empty list
    start_time = time.time()
    
    
    # doesnt fully work
    for max_depth in depth_list:
        acc_train0 = [] #empty list
        acc_test0 = [] #empty list
        for forest_size in forest_size_list:
            for _ in range(1):
                #---split data into 0.7 train and 0.3 test---
                x_train,x_test,y_train,y_test = train_test_split(Xn,Yn,test_size=0.2,random_state=0)
                    
                #---classifier trainning---
                forest_clf = RandomForestClassifier(criterion='entropy',
                                                    n_estimators=forest_size,
                                                    max_depth=max_depth)
                #train classifer on training data
                forest_clf.fit(x_train,y_train)   
                
                #---EVALUATE TREE CLASSIFIER---
                #TRAIN data - evaluate predication
                y_train_pred = forest_clf.predict(x_train)
                cm_train = metrics.confusion_matrix(y_train, y_train_pred)
                acc_train_pred = cm_train.diagonal().sum() / cm_train.sum()
                
                
                #TEST data - evaluate predication
                y_test_pred = forest_clf.predict(x_test)
                cm_test = confusion_matrix(y_test,y_test_pred)
                acc_test_pred = cm_test.diagonal().sum() / cm_test.sum()                    
                
                acc_train0.append(acc_train_pred)
                acc_test0.append(acc_test_pred)
        acc_train.append(acc_train0)
        acc_test.append(acc_test0)
    
    print(f'Loop Elapsed Time:{(time.time() - start_time):.03f} [sec]\n')          
    
    avg_acc_train = np.mean(np.array(acc_train),axis=0)
    avg_acc_test = np.mean(np.array(acc_test),axis=0)
    
    print(f'Acc Train: {acc_train}\nAcc Test: {acc_test}\n\n')
    print(f'Acc Train: {avg_acc_train}\nAcc Test: {avg_acc_test}')
   
    dlist = [str(x) for x in depth_list]
    flist = [str(x) for x in forest_size_list]
    d =["depth=" + strg for strg in dlist]
    f =["Forest Size=" + strg for strg in flist]
    g = d+f
    
    
    t = acc_train + acc_test

    
    fig, ax = plt.subplots()
    ax.boxplot(t)
    
    


    return




if __name__ == "__main__":
    st = time.time()
    decision_forest_classifier()
    print(f'Elapsed Time:{(time.time() - st)*1000:.03f} [ms]')
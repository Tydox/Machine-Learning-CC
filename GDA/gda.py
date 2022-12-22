# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 09:36:25 2022

@author: Tydox
"""
import time
import seaborn                       as     sns
import numpy                         as     np
import matplotlib.pyplot             as     plt
from   sklearn                       import metrics
from   sklearn.naive_bayes           import GaussianNB
from   sklearn.model_selection       import train_test_split
from   sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from   sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def plot_data(_data):
    #plot first 50 data (1x64 data Xn per label Yn)
    plt.figure(figsize = (12,6))
    plt.imshow(_data[0:50],cmap='gray')
    plt.xlabel('pixel index',fontsize=15)
    plt.ylabel('image index (first 50 images)',fontsize=15)
    
    return


def LoadIrisData(_features_file_name,_labels_file_name):
#np..loadtxt(fname,delimiter=',',usecols=nparrange(65))

    # read the features data from the csv file
    _x = np.loadtxt(_features_file_name,dtype=int, delimiter=',')
    # read the labels data from the csv file
    _y = np.loadtxt(_labels_file_name,dtype=int, delimiter=',')

    return _x,_y


def linear_gda(x_train,x_test,y_train,y_test):
    print('---Linear GDA---')

    #create a Linear Discriminant Analysis classifier
    linear_classifier = LinearDiscriminantAnalysis()
    
    #train linear classifier using Xtrain real data
    linear_classifier.fit(x_train,y_train)
    
    #based on the train linear classifier, test it using Xtrain data
    y_train_predicted = linear_classifier.predict(x_train)
    #create a matrix to compare the results of the linear classifier to the real classifications\labels
    confusion_matrix_train = metrics.confusion_matrix(y_train, y_train_predicted)
    #print(f'Confusion matrix - Train\n {confusion_matrix_train}')
    plt.figure(figsize = (15,10))
    #create a heatmap of the confustion matrix to visualize how well the linear classifier did
    heat_map_axis = sns.heatmap(confusion_matrix_train,annot=True, linewidth=0.5,cmap='hot',fmt='d')
    heat_map_axis.set(xlabel = '$\hat{y} \quad estimated$',ylabel = 'y real')
    plt.title('Linear GDA - Train')
    train_accuracy = confusion_matrix_train.diagonal().sum() / confusion_matrix_train.sum()
    print(f'Accuracy on Train:\n {train_accuracy:.03f}')
    
    
    #based on the train linear classifier, test it using Xtrain data
    y_test_predicted = linear_classifier.predict(x_test)
    #create a matrix to compare the results of the linear classifier to the real classifications\labels
    confusion_matrix_test = metrics.confusion_matrix(y_test, y_test_predicted)
    #print(f'Confusion matrix - Train\n {confusion_matrix_test}')
    plt.figure(figsize = (15,10))
    #create a heatmap of the confustion matrix to visualize how well the linear classifier did
    heat_map_axis = sns.heatmap(confusion_matrix_test,annot=True, linewidth=0.5,cmap='hot',fmt='d')
    heat_map_axis.set(xlabel = '$\hat{y} \quad estimated$',ylabel = 'y real')
    plt.title('Linear GDA - Test')

    test_accuracy = confusion_matrix_test.diagonal().sum() / confusion_matrix_test.sum()
    print(f'Accuracy on Test:\n {test_accuracy:.03f}')


def quadratic_gda(x_train,x_test,y_train,y_test):
    print('---Quadratic GDA---')

    #create a Quadratic Discriminant Analysis classifier
    quadratic_classifier = QuadraticDiscriminantAnalysis()
    
    #train quadratic classifier using Xtrain real data
    quadratic_classifier.fit(x_train,y_train)
    
    #based on the train quadratic classifier, test it using Xtrain data
    y_train_predicted = quadratic_classifier.predict(x_train)
    #create a matrix to compare the results of the quadratic classifier to the real classifications\labels
    confusion_matrix_train = metrics.confusion_matrix(y_train, y_train_predicted)
    #print(f'Confusion matrix - Train\n {confusion_matrix_train}')
    plt.figure(figsize = (15,10))
    #create a heatmap of the confustion matrix to visualize how well the quadratic classifier did
    heat_map_axis = sns.heatmap(confusion_matrix_train,annot=True, linewidth=0.5,cmap='hot',fmt='d')
    heat_map_axis.set(xlabel = '$\hat{y} \quad estimated$',ylabel = 'y real')
    plt.title('Quadratic GDA - Train')

    train_accuracy = confusion_matrix_train.diagonal().sum() / confusion_matrix_train .sum()
    print(f'Accuracy on Train:\n {train_accuracy:.03f}')
    
    
    #based on the train quadratic classifier, test it using Xtrain data
    y_test_predicted = quadratic_classifier.predict(x_test)
    #create a matrix to compare the results of the quadratic classifier to the real classifications\labels
    confusion_matrix_test = metrics.confusion_matrix(y_test, y_test_predicted)
    #print(f'Confusion matrix - Train\n {confusion_matrix_test}')
    plt.figure(figsize = (15,10))
    #create a heatmap of the confustion matrix to visualize how well the quadratic classifier did
    heat_map_axis = sns.heatmap(confusion_matrix_test,annot=True, linewidth=0.5,cmap='hot',fmt='d')
    heat_map_axis.set(xlabel = '$\hat{y} \quad estimated$',ylabel = 'y real')
    plt.title('Quadratic GDA - Test')

    test_accuracy = confusion_matrix_test.diagonal().sum() / confusion_matrix_test.sum()
    print(f'Accuracy on Test:\n {test_accuracy:.03f}')
    
def guassian_naive_bayes(x_train,x_test,y_train,y_test):
    print('---Guassian Naive Bayes GDA---')
    
    #create a guassian_naive_bayes Discriminant Analysis classifier
    guassian_naive_bayes_classifier = GaussianNB()
    
    #train guassian_naive_bayes classifier using Xtrain real data
    guassian_naive_bayes_classifier.fit(x_train,y_train)
    
    #based on the train guassian_naive_bayes classifier, test it using Xtrain data
    y_train_predicted = guassian_naive_bayes_classifier.predict(x_train)
    #create a matrix to compare the results of the guassian_naive_bayes classifier to the real classifications\labels
    confusion_matrix_train = metrics.confusion_matrix(y_train, y_train_predicted)
    #print(f'Confusion matrix - Train\n {confusion_matrix_train}')
    plt.figure(figsize = (15,10))
    #create a heatmap of the confustion matrix to visualize how well the guassian_naive_bayes classifier did
    heat_map_axis = sns.heatmap(confusion_matrix_train,annot=True, linewidth=0.5,cmap='hot',fmt='d')
    heat_map_axis.set(xlabel = '$\hat{y} \quad estimated$',ylabel = 'y real')
    plt.title('Guassian Naive Bayes GDA - Train')

    train_accuracy = confusion_matrix_train.diagonal().sum() / confusion_matrix_train.sum()
    print(f'Accuracy on Train:\n {train_accuracy:.03f}')
    
    
    #based on the train guassian_naive_bayes classifier, test it using Xtrain data
    y_test_predicted = guassian_naive_bayes_classifier.predict(x_test)
    #create a matrix to compare the results of the guassian_naive_bayes classifier to the real classifications\labels
    confusion_matrix_test = metrics.confusion_matrix(y_test, y_test_predicted)
    #print(f'Confusion matrix - Train\n {confusion_matrix_test}')
    plt.figure(figsize = (15,10))
    #create a heatmap of the confustion matrix to visualize how well the guassian_naive_bayes classifier did
    heat_map_axis = sns.heatmap(confusion_matrix_test,annot=True, linewidth=0.5,cmap='hot',fmt='d')
    heat_map_axis.set(xlabel = '$\hat{y} \quad estimated$',ylabel = 'y real')
    plt.title('Guassian Naive Bayes GDA - Test')

    test_accuracy = confusion_matrix_test.diagonal().sum() / confusion_matrix_test.sum()
    print(f'Accuracy on Test:\n {test_accuracy:.03f}')


if __name__ == "__main__":
    # load csv files
    features_file_name = r"C:\Users\Danno\Desktop\hw2\features.csv"
    labels_file_name = r"C:\Users\Danno\Desktop\hw2\labels.csv"
    Xn, Yn = LoadIrisData(features_file_name,labels_file_name)
    
    #split data into train and test/
    x_train,x_test,y_train,y_test = train_test_split(Xn,Yn,test_size=0.2,random_state=0)
    
    #plot first 50 data (1x64 data Xn per label Yn)
    plot_data(Xn)
    
    st = time.time()
    linear_gda(x_train,x_test,y_train,y_test)
    print(f'Elapsed Time:{(time.time() - st)*1000:.03f} [ms]')
    
   
    st = time.time()
    quadratic_gda(x_train,x_test,y_train,y_test)
    print(f'Elapsed Time:{(time.time() - st)*1000:.03f} [ms]')


    st = time.time()
    guassian_naive_bayes(x_train,x_test,y_train,y_test)
    print(f'Elapsed Time:{(time.time() - st)*1000:.03f} [ms]')








    #for col in range(0,10):
     #   for row in range(0,10):
      #      ax.text(row+0.5, col+0.5, confusion_matrix_train[col,row],ha="center", va="center", color="b")
    #plt.show()
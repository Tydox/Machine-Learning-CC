# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 10:01:57 2022

@author: Tydox
"""

import time
import numpy                         as     np
import matplotlib.pyplot             as     plt
import pandas as pd
from numpy.linalg import norm


def importdata():
    #import data
    df = pd.read_csv(r'C:\Users\fello\Desktop\ML\poly regresssion\PolynomialRegressionExercice_TEST_DATA.txt')
    data = df.values
    Xn_test = data[:,0]
    Yn_test = data[:,1]
   
    
    #import data
    df = pd.read_csv(r'C:\Users\fello\Desktop\ML\poly regresssion\PolynomialRegressionExercice_TRAIN_DATA.txt')
    data = df.values
    Xn_train = data[:,0]
    Yn_train = data[:,1]

    return Xn_test,Yn_test,Xn_train,Yn_train

def plotdata(Xn_test,Yn_test,Xn_train,Yn_train):
    plt.figure()
    plt.scatter(Xn_test, Yn_test,color = 'r',label = "Test Data")    
    plt.scatter(Xn_train, Yn_train,color = 'b',label = "Train Data")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.legend()
    plt.title("Data Scatter")
    plt.tight_layout()   
    return

def plotRegressor(X,W):
    Y = X @ W
    plt.plot(X,Y,label = "regressor")
    plt.legend()
    return



def calcX(_x,_power):
    _X = np.array([_x**d for d in range(_power+1)]).T
    return _X

def calcY(_y):
    return _y[:,np.newaxis]


def calcW(_X,_Y):
    return np.linalg.inv((_X.T @ _X)) @ _X.T @ _Y


def calcLoss(_x,_y,_w):
    X = _x
    Y = _y
    W = _w
    N = len(_y)
    _Loss = ((Y - X@W).T @ (Y - X@W))/N
    return _Loss

def calcLossPoly(X,Y,W):
    loss = (((Y - X@W).T @ (Y-X@W)) / len(Y))[0]
    #print(loss)
    return loss

def maina():
    Xn_test,Yn_test,Xn_train,Yn_train = importdata()
    plotdata(Xn_test,Yn_test,Xn_train,Yn_train)
    
    #manual check
    # xtr = calcX(Xn_train, 2)
    # ytr = calcY(Yn_train)
    # wtr = calcW(xtr,ytr)
    # loss_tr = calcLoss(xtr,ytr,wtr)
    # print(loss_tr)

    # xte = calcX(Xn_test,2)
    # yte = calcY(Yn_test)
    # loss_te = calcLoss(xte,yte,wtr)
    # print(loss_te)
    
    #calc w* & loss
    D = [d for d in range(12)]
    
    loss_tr = []
    loss_te = []
    
    for d in D:
        xtr = calcX(Xn_train, d)
        ytr = calcY(Yn_train)
        wtr = calcW(xtr,ytr)
        loss_tr.append((calcLoss(xtr,ytr,wtr))[0][0])
        
        xte = calcX(Xn_test, d)
        yte = calcY(Yn_test)
        loss_te.append(calcLoss(xte,yte,wtr)[0][0])
        
        #plot regressor on data
        plotdata(Xn_test,Yn_test,Xn_train,Yn_train)
        plt.plot(Xn_train,xtr @ wtr,label = f'regressor D={d}')
        plt.legend()
        
        
    #print(loss_tr)
    #print(loss_te)
    
    #graph 7
    plt.figure()
    plt.plot(D,loss_tr,label = 'Train Error',marker = 'o')    
    plt.plot(D,loss_te,label = 'Test Error',marker = 'o')
    plt.legend()    
    plt.yscale('log')
    plt.tight_layout()
    plt.xlabel("D")
    plt.ylabel("Loss - MSE")
    plt.title("Polynomial Regression: MSE vs Polynom Degree")
    return




def main8():
    Xn_test,Yn_test,Xn_train,Yn_train = importdata()
    
    loss_tr = []
    loss_te = []
    lamds = np.logspace(-4,8,20)
    idMat = np.identity(12)

    xtr = calcX(Xn_train, 11) #vec Nx(D+1)
    ytr = calcY(Yn_train) #vec Nx1
    xte = calcX(Xn_test,11) #vec Nx(D+1)
    yte = calcY(Yn_test) #vec Nx1
    
    for lam in lamds:
        wrig = (np.linalg.inv( (xtr.T @ xtr) + lam*idMat ) @ (xtr.T @ ytr))
        loss_tr.append( calcLossPoly(xtr, ytr, wrig))
        loss_te.append( calcLossPoly(xte, yte, wrig))
        

        
    plt.figure(figsize=(15,6))
    # plt.plot(lamds,loss_tr,label = 'Train Error',marker = 'o')    
    # plt.plot(lamds,loss_te,label = 'Test Error',marker = 'o')
    # #plt.yscale('log')
    # plt.xscale('log')

    plt.semilogx(lamds,np.log(loss_tr),label = 'Train Error',marker = 'o')
    plt.semilogx(lamds,np.log(loss_te),label = 'Test Error',marker = 'o')
    plt.tight_layout()
    plt.xlabel("$\lambda$")
    plt.ylabel("Loss - MSE")
    plt.title("Polynomial Regression: MSE vs $\lambda$ ")
    plt.legend()    
    
    return


def main9():
    Xn_test,Yn_test,Xn_train,Yn_train = importdata()
    
    loss_tr = []
    loss_te = []
    lamds = np.logspace(-4,8,20)
    idMat = np.identity(12)

    xtr = calcX(Xn_train, 11)
    ytr = calcY(Yn_train)
    wvec = []
    plt.figure(figsize=(15,6))

    for lam in lamds:
        wrig = (np.linalg.inv(((xtr.T @ xtr) + lam*idMat )) @ xtr.T @ ytr).flatten().tolist()
        wvec.append(wrig)
   
    wmat = []
    for col in range(12):
        wlist = []

        for row in range(20):
            wlist.append(wvec[row][col])    
       
        wmat.append(wlist)
        plt.plot(lamds,wlist,label = f'W{col}',marker = 'o')    

        
        
        
    
    plt.legend()    
    #plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()
    plt.xlabel("$\lambda$")
    plt.ylabel("W*")
    plt.title("Ridge Weights Paths")
    
    
    return



if __name__ == "__main__":
    maina()
    main8()
    main9()
    # #print(Yn.shape)#array not vector!
    # #print(Yn[:,np.newaxis].shape) #check that it is a vector col or row
    



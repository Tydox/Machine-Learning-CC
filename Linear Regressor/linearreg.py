# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 10:01:57 2022

@author: Tydox
"""

import time
import numpy                         as     np
import matplotlib.pyplot             as     plt
import pandas as pd



def loadData():
    #import data
    df = pd.read_csv(r'C:\Users\fello\Desktop\ML\linear regression\HousePriceVsSizeLinearRegression.csv')
    data = df.values
    Xn = data[:,0][:,np.newaxis]
    Yn = data[:,1][:,np.newaxis]
    return Xn,Yn

def plotData(Xn,Yn):    
    plt.figure(figsize = (16,5))
    plt.scatter(Xn, Yn,color = 'r',label = "Observed Data")
    plt.xlabel("Size [${m}^2$]")
    plt.ylabel("Price [${M}\$$]")
    plt.title("House Prices vs Square Size")
    plt.legend()
    plt.tight_layout()

    return

def initX(Xn):
    return np.array(Xn).mean()

def initY(Yn):
    return np.array(Yn).mean()


def calcW1(Xn,X,Yn,Y):
    xvec = Xn - X
    yvec = Yn - Y
    adi = xvec * yvec
    numi = np.array(adi).sum()
    denum = (xvec**2).sum()
    return numi / denum

def calcW0(Y,X,W1):
    return Y-W1*X

def plotRegressor(Xn,W0,W1):
    Y = W0 + W1*Xn
    plt.plot(Xn,Y,label = "regressor")
    plt.legend()
    return

def calcLoss(Yn,W0,W1,Xn):
    xvec = W0 + W1*Xn
    sqErr = (Yn - xvec)**2
    return sqErr.mean()
    

def plotTrainTest(Xtrain,Ytrain,Xtest,Ytest,size):
    plt.figure(figsize = (16,5))
    plt.scatter(Xtrain,Ytrain,label = "Train",c = 'blue')
    plt.scatter(Xtest,Ytest,label = "Test",c = 'red')
    plt.legend()
    plt.title(label = f"House Prices vs Square Size - Split data train={size}")
    plt.xlabel("Size [${m}^2$]")
    plt.ylabel("Price [${M}\$$]")
    plt.tight_layout()

    return

def main():
    #2
    Xn,Yn = loadData()
    #3
    plotData(Xn,Yn)
    #4
    Y = initY(Yn)
    X = initX(Xn)
    W1 = calcW1(Xn,X,Yn,Y)
    W0 = calcW0(Y,X,W1)
    print(f"W0 = {W0}\nW1 = {W1}\n---\n")
    #5
    plotRegressor(Xn,W0,W1)
    loss_data = calcLoss(Yn,W0,W1,Xn)
    print(f"\nRegressor All Data Loss is: {loss_data}\n")
    
    #6
    Xtrain = Xn[:35]
    Ytrain = Yn[:35]
    
    Xtest = Xn[35:]
    Ytest = Yn[35:]
    plotTrainTest(Xtrain,Ytrain,Xtest,Ytest,size=35)
    
    #iii
    Y = initY(Ytrain)
    X = initX(Xtrain)
    W1 = calcW1(Xtrain,X,Ytrain,Y)
    W0 = calcW0(Y,X,W1)
    #5
    plotRegressor(Xn,W0,W1)
    lossTrain = calcLoss(Ytrain,W0,W1,Xtrain)
    print(f"\nRegressor Trained Loss is: {lossTrain}\n")
    lossTest = calcLoss(Ytest,W0,W1,Xtest)
    print(f"\nRegressor Tested Loss is: {lossTest}\n")
        
    return
    
def main7i():
    #2
    Xn,Yn = loadData()
    #3
    plotData(Xn,Yn)
    #4
    Y = initY(Yn)
    X = initX(Xn)
    W1 = calcW1(Xn,X,Yn,Y)
    W0 = calcW0(Y,X,W1)
    #5
    plotRegressor(Xn,W0,W1)
    loss_data = calcLoss(Yn,W0,W1,Xn)
    print(f"\nRegressor All Data Loss is: {loss_data}\n")
    
    #6
    Xtrain = Xn[:10]
    Ytrain = Yn[:10]
    
    Xtest = Xn[10:]
    Ytest = Yn[10:]
    plotTrainTest(Xtrain,Ytrain,Xtest,Ytest,size=10)
    
    #iii
    Y = initY(Ytrain)
    X = initX(Xtrain)
    W1 = calcW1(Xtrain,X,Ytrain,Y)
    W0 = calcW0(Y,X,W1)
    #5
    plotRegressor(Xn,W0,W1)
    lossTrain = calcLoss(Ytrain,W0,W1,Xtrain)
    print(f"\nRegressor Trained Loss is: {lossTrain}\n")
    lossTest = calcLoss(Ytest,W0,W1,Xtest)
    print(f"\nRegressor Tested Loss is: {lossTest}\n")
    
    return
    

def main7ii():
    #2
    Xn,Yn = loadData()
    #3
    plotData(Xn,Yn)
    #4
    Y = initY(Yn)
    X = initX(Xn)
    W1 = calcW1(Xn,X,Yn,Y)
    W0 = calcW0(Y,X,W1)
    #5
    plotRegressor(Xn,W0,W1)
    loss_data = calcLoss(Yn,W0,W1,Xn)
    print(f"\nRegressor All Data Loss is: {loss_data}\n")
    
    #6
    Xtrain = Xn[:3]
    Ytrain = Yn[:3]
    
    Xtest = Xn[3:]
    Ytest = Yn[3:]
    plotTrainTest(Xtrain,Ytrain,Xtest,Ytest,size=3)
    
    #iii
    Y = initY(Ytrain)
    X = initX(Xtrain)
    W1 = calcW1(Xtrain,X,Ytrain,Y)
    W0 = calcW0(Y,X,W1)
    #5
    plotRegressor(Xn,W0,W1)
    lossTrain = calcLoss(Ytrain,W0,W1,Xtrain)
    print(f"\nRegressor Trained Loss is: {lossTrain}\n")
    lossTest = calcLoss(Ytest,W0,W1,Xtest)
    print(f"\nRegressor Tested Loss is: {lossTest}\n")
    
    #extra
    plt.figure(figsize = (5,5))
    plt.bar(['All Data - Loss','Train- Loss','Test - Loss'],[loss_data,lossTrain,lossTest])
    plt.tight_layout()
    plt.title(label = "Regressor Loss Table")
    plt.ylabel("Loss (MSE)")
    yy = [loss_data,lossTrain,lossTest]
    xlocs, xlabs = plt.xticks()
    for i, v in enumerate(yy):
        txt = f"{v:.03f}"
        plt.text(xlocs[i] - 0.25, v + 0.03,txt)

    
    return


def main89():
    
    #Calc By MATRIX FORMULA
    Xn,Yn = loadData()
    #3
    plotData(Xn,Yn)
    #4
    N = len(Xn)
    ones = np.ones(shape=(N,))
    arr = []
    for i in range(len(Xn)):
        arr.append([ones[i],Xn[i][0]])
        
    X = np.array(arr)
    Ws = np.linalg.inv(X.T @ X) @ X.T @ Yn
    print(f"\nMatrix:\nW0 = {float(Ws[0])}\nW1 = {float(Ws[1])}\n")
    
    
    
    
 
    #Calc using Non Matrix Formulas
    Ynm = initY(Yn)
    Xnm = initX(Xn)
    Wnm1 = calcW1(Xn,Xnm,Yn,Ynm)
    Wnm0 = calcW0(Ynm,Xnm,Wnm1)
    W0 = round(Wnm0,8)
    W1 = round(Wnm1,8)
    print(f"Non Matrix:\nW0 = {W0}\nW1 = {W1}\n")

    #8
    #Calc Delta Between Matix Non Matrix formulas
    deltaW0 = float(abs(W0-Ws[0]))
    deltaW1 = float(abs(W1-Ws[1]))
    print(f"Delta W0 = {deltaW0:.08f}\nDelta W1 = {deltaW1:.08f}")
    
    
    LossNM = calcLoss(Yn,W0,W1,Xn)

    #Loss Matrix
    N = len(Xn)
    LossMatrix = float(((Yn - X @ Ws).T @ (Yn- X @ Ws))/N)
    print("\n-------------------------------\n")
    print(f"Loss Non Matrix = {LossNM}\nLoss Matrix calc = {LossMatrix}\nDelta Loss = {abs(LossNM-LossMatrix):.010f}")

    return

if __name__ == "__main__":
    main()
    main7i()
    main7ii() 
    main89()
    
    pass
    


def playtest():
    # x_v = np.array(Xn).mean()
    # N = len(Xn)
    # #x_v = sum(Xn)/N
    
    # y_n = np.array(Yn).mean()
    # x_vec = np.array([np.ones(shape=(N,)),Xn]).T
    # #print(x_vec.shape)
    # #print(x_vec)
    # W = np.linalg.inv((x_vec.T @ x_vec))@x_vec.T@(Yn[:,np.newaxis])
    # #print(W[0])
    # Y_reg = (x_vec@W)
    # plt.plot(Xn,Y_reg)
    
    # Yn = Yn[:,np.newaxis]
    # Loss = np.array(((Yn - Y_reg)))
    
    #print(f"Loss = {Loss}")
    
    #print(Yn.shape)#array not vector!
    #print(Yn[:,np.newaxis].shape) #check that it is a vector col or row
    return
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 09:56:33 2022

@author: Tydox
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1.42,5.01,2.45,1.92,1.41,4.83,1.81])
mue_real = 2
sig_real =  1.5



def calcMuSigML(data):
    #uml = data.sum() / len(x)
    mue = data.mean()
    sig = np.sqrt(((data - mue)**2).mean())
    return mue,sig

def probCalc(_data,_mue,_sig):
    return np.exp((-(_data - _mue)**2) / (2*_sig**2) )  / np.sqrt(2*np.pi*((_sig)**2))
    


#print(calcMuSigML(x))

def PlotML(data,mue_ml,sig_ml):

    
    x_real = np.linspace(mue_real - 3*sig_real, mue_real + 3*sig_real, 500)
    prob_real =  probCalc(x_real, mue_real, sig_real)
    
    
    x_est = np.linspace(mue_ml - 3*sig_ml, mue_ml + 3*sig_ml, 500)
    prob_est = probCalc(x_est, mue_ml, sig_ml)
    
    plt.figure(figsize=(14,5))
    plt.plot(x_est,prob_est,label = 'Y estimated: $\hat{\sigma}$ =' + str(float(f'{sig_ml:.2f}')) + '  $\hat {\mu}$ =' + str(float(f'{mue_ml:.2f}')))
    plt.plot(x_real,prob_real,label = 'Y real: $\sigma$ =' + str(sig_real) + '  $\mu$ =' + str(mue_real))
    plt.xlabel('X[n]')
    plt.ylabel('P(X[n])')
    plt.title('Probability')
    plt.legend()
    #plt.xlim(-10,10)
    #plt.ylim(0,0.3)
    return







def mainq4_3():
    mue_ml,sig_ml = calcMuSigML(x)
    x_real = np.linspace(mue_real - 3*sig_real, mue_real + 3*sig_real, 500)
    prob_real =  probCalc(x_real, mue_real, sig_real)
    
    x_est = np.linspace(mue_ml - 3*sig_ml, mue_ml + 3*sig_ml, 500)
    prob_est = probCalc(x_est, mue_ml, sig_ml)
    
    plt.figure(figsize=(14,5))
    plt.plot(x_est,prob_est,label = 'Y estimated: $\hat{\sigma}$ =' + str(float(f'{sig_ml:.2f}')) + '  $\hat {\mu}$ =' + str(float(f'{mue_ml:.2f}')))
    plt.plot(x_real,prob_real,label = 'Y real: $\sigma$ =' + str(sig_real) + '  $\mu$ =' + str(mue_real))
    plt.xlabel('X[n]')
    plt.ylabel('P(X[n])')
    plt.title('Probability')
    plt.legend()
    
def mainq4_4():
    xn = np.random.randint(1,7,30)
    mue_ml,sig_ml = calcMuSigML(xn)
    x_real = np.linspace(mue_real - 3*sig_real, mue_real + 3*sig_real, 500)
    prob_real =  probCalc(x_real, mue_real, sig_real)
    
    
    x_est = np.linspace(mue_ml - 3*sig_ml, mue_ml + 3*sig_ml, 500)
    prob_est = probCalc(x_est, mue_ml, sig_ml)
    
    plt.figure(figsize=(14,5))
    plt.plot(x_est,prob_est,label = 'Y estimated: $\hat{\sigma}$ =' + str(float(f'{sig_ml:.2f}')) + '  $\hat {\mu}$ =' + str(float(f'{mue_ml:.2f}')))
    plt.plot(x_real,prob_real,label = 'Y real: $\sigma$ =' + str(sig_real) + '  $\mu$ =' + str(mue_real))
    plt.xlabel('X[n]')
    plt.ylabel('P(X[n])')
    plt.title('Probability')
    plt.legend()    
    
    

def mainq4_5():
    plt.figure(figsize=(20,15))

    x_real = np.linspace(mue_real - 3*sig_real, mue_real + 3*sig_real, 500)
    prob_real =  probCalc(x_real, mue_real, sig_real)
    plt.plot(x_real,prob_real,'red',label = 'Y real: $\sigma$ =' + str(sig_real) + '  $\mu$ =' + str(mue_real))


    for est_sample in range(0,10):
        xn = np.random.uniform(0,5,30)
        mue_ml,sig_ml = calcMuSigML(xn)
        x_est = np.linspace(mue_ml - 3*sig_ml, mue_ml + 3*sig_ml, 500)
        prob_est = probCalc(x_est, mue_ml, sig_ml)        
        plt.plot(x_est,prob_est,'green',label = 'Y estimated: $\hat{\sigma}$ =' + str(float(f'{sig_ml:.2f}')) + '  $\hat {\mu}$ =' + str(float(f'{mue_ml:.2f}')))

    plt.xlabel('X[n]')
    plt.ylabel('P(X[n])')
    plt.title('Probability')
    plt.legend()
    plt.ylim(0,0.5)



def mainq4_6():
    plt.figure(figsize=(20,15))

    for est_sample in range(0,10):
        xn = np.random.uniform(0,5,3000) #start,end,number of random x'im
        mue_ml,sig_ml = calcMuSigML(xn)
        x_est = np.linspace(mue_ml - 3*sig_ml, mue_ml + 3*sig_ml, 500)
        prob_est = probCalc(x_est, mue_ml, sig_ml)
        plt.plot(x_est,prob_est,'g')
        #plt.plot(x_est,prob_est,label = 'Y estimated: $\hat{\sigma}$ =' + str(float(f'{sig_ml:.2f}')) + '  $\hat {\mu}$ =' + str(float(f'{mue_ml:.2f}')))
    
    x_real = np.linspace(mue_real - 3*sig_real, mue_real + 3*sig_real, 500)
    prob_real =  probCalc(x_real, mue_real, sig_real)
    plt.plot(x_real,prob_real,'r',label = 'Y real: $\sigma$ =' + str(sig_real) + '  $\mu$ =' + str(mue_real))

    plt.xlabel('X[n]')
    plt.ylabel('P(X[n])')
    plt.title('Probability')
    plt.legend(loc='upper right')
    plt.ylim(0,0.5)


if __name__ == "__main__":
    mainq4_3()
    mainq4_4()
    mainq4_5()
    mainq4_6()
    
    





"""
sum = 0

for a in x:
    sum += ((a-2.62)**2)

print(sum/7)    
num= np.sqrt(sum / 7)
print(num)




def UML(data):
    tmp = 0
    for num in data:
        tmp += num
    return (tmp / len(data))

def SML(data):
    tmp = 0
    u = UML(data)
    for num in data:
        tmp += ((num - u)**2)
    return (np.sqrt(tmp / len(data)))




print(UML(x))
print(SML(x))

"""




























        
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 12:57:17 2022

@author: Tydox
"""
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
import time


def plotbinom(k,n,p):
    pmf = binom.pmf(k, n, p) 
    plt.plot(p,pmf, label = f'K = {k}, N = {n}')
    plt.xlabel('p - Head')
    plt.ylabel('pmf of Head')
    plt.title('Binominal Distribution of Coin Flip')
    plt.legend()
    plt.xlim(0,1)
    plt.ylim(0,1)
    #print(type(pmf))


def plotpmax(k,n,p):
    pmf = binom.pmf(k, n, p) 
    plt.plot(p,pmf,'ro')
    plt.xlabel('p - Head')
    plt.ylabel('pmf of Head')
    plt.title('Binominal Distribution of Coin Flip')
    plt.legend()
    plt.xlim(0,1)
    plt.ylim(0,1)
    #print(type(pmf))



def TsetupGame():

        list = ['door 1','door 2','door 3']
       # print(f'List:{list})')
        prize = {'Car':'','Goat1':'','Goat2':''}
       # print(f'Dic Before:\t {prize}')
        for pz in prize:
            doorNum = list.pop()
         #   print(f'popped value:\t {doorNum}')
            prize.update({pz: doorNum})    
        #print(f'Dic after:\t{prize}')
        
        return prize


def setupGame():

        list = ['door 1','door 2','door 3']
        np.random.shuffle(list)
       # print(f'List:{list})')
        prize = {'Car':'','Goat1':'','Goat2':''}
       # print(f'Dic Before:\t {prize}')
        for pz in prize:
            doorNum = list.pop()
         #   print(f'popped value:\t {doorNum}')
            prize.update({pz: doorNum})    
        #print(f'Dic after:\t{prize}')
        
        return prize
 
def winDoor(gameSetup):
       return gameSetup.get('Car')


def playerDoor(gameSetup,hsDoor = None, playdoor = None,pl2 = None):
    
    lst = ['Car','Goat1','Goat2']
    if hsDoor != None:    
        lst.remove(hsDoor)
        print(f'New Doors: {lst}')
    
    np.random.shuffle(lst)
    pld = gameSetup.get(lst[0])
   # print('\nold' + pld)
    
    
    if pld == playdoor and pl2 != None:
        pld = gameSetup.get(lst[1])
    #    print('swap' + pld +'\n')

    return pld

    
def hostDoor(plrdoor,gameSetup):
    
    list = ['Goat1','Goat2']
    np.random.shuffle(list)
    hostc = gameSetup.get(list[0])
    
    if hostc == plrdoor:
        hostc = gameSetup.get(list[1])    
    
    return hostc

def isWinner(gameSetup,pldr):
    playerItem = gameSetup.get(pldr)
    if (playerItem == 'Car'):
        return True
    
    return False

def swapKeyVal(gameSetup):
    return {value:key for key, value in gameSetup.items()}



def main1():
    st = time.time()     # get the start time
    p = np.linspace(0, 1,1000)  #probability H - success
    n = 10  #total num of flips
    k = 3   #how many successful heads
    #plotbinom(k = how many successful heads, n = total num of flips, p)
    plt.figure()

    karr = np.array([3,15,90])
    narr = np.array([10,50,300])
    for i in range(0,3):
        plotbinom(karr[i], narr[i], p)
        plotpmax(karr[i], narr[i], karr[i] / narr[i])

        
    #plotbinom(karr[0], narr[0], p)
    #plotbinom(karr[1], narr[1], p)
    #plotbinom(karr[2], narr[2], p)
    elapsed_time = time.time() - st     # get the execution time
    print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    final_res = elapsed_time * 1000
    print('Execution time:', final_res, 'milliseconds')

    
    
def main2a():    
    st = time.time()     # get the start time
    
    game = setupGame()
    
    #find winning door
    carDoor = winDoor(game)
    print(f'Winning Door: {carDoor}')
    p1 = playerDoor(game)
    print(f'Player Door: {p1}')
    hstDoor = hostDoor(p1,game)
    print(f'Host Door: {hstDoor}\n------------------------\n')
    
    game = swapKeyVal(game)
  #  print(f'GameSwapped: {game}')
    p1win = isWinner(game,p1)
    print(f"Player 1:{p1}: {game.get(p1)}, Winner? {p1win}")    
    
    
    game = swapKeyVal(game)
    p2 = playerDoor(game,game.get(hstDoor),p1)
    game = swapKeyVal(game)
   # print(f'GameSwapped: {game}')
    p2win = isWinner(game,p2)
    print(f"Player 2:{p2}: {game.get(p2)}, Winner? {p2win}")    
    
    
    game = swapKeyVal(game)
    p3 = playerDoor(game,game.get(hstDoor),p1)
    game = swapKeyVal(game)
    #print(f'GameSwapped: {game}')
    p3win = isWinner(game,p3)
    print(f"Player 3:{p3}: {game.get(p3)}, Winner? {p3win}")   
    
    
    
    
    #player swapped
    




    elapsed_time = time.time() - st     # get the execution time
    print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    final_res = elapsed_time * 10000
    print('Execution time:', final_res, 'milliseconds')
    return






def main2b():
    st = time.time()     # get the start time
    choiceUnchanged=[]
    
    for gameitr in range(0,1000):
        game = setupGame()

        carDoor = winDoor(game)        #find winning door
        #print(f'Winning Door: {carDoor}')
        p1 = playerDoor(game)
        #print(f'Player Door: {p1}')
        hstDoor = hostDoor(p1,game)
        #print(f'Host Door: {hstDoor}\n------------------------\n')
        
        game = swapKeyVal(game)
      #  print(f'GameSwapped: {game}')
        choiceUnchanged.append(isWinner(game,p1))
     #   print(choiceUnchanged)
    #print(f"Player 1:{p1}:Winner? {choiceUnchanged}")    
    avg = []

    for k in range(1,1001):
        avg.append(np.array(choiceUnchanged[:k]).mean())
    print(type(avg))
    plt.plot(range(0,1000) ,avg)
    
    ag = plt.gca()
    ag.set_ylim(0,1)
    ag.set_xlim(0,1000)
    ag.set_xticks(([1] + list(np.arange(100,1001,100))))
    ag.set_xlabel(f'Number of Rounds')
    ag.set_ylabel(f'Winning Average')
    ag.set_title(f'Player A - Winning Mean')
    
    

    elapsed_time = time.time() - st
    print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    final_res = elapsed_time * 1000
    print('Execution time:', final_res, 'milliseconds')
    return



def main2c():
    st = time.time()     # get the start time
    ch1=[]
    ch2=[]
    ch3=[]
    
    avg1 = []
    avg2 = []
    avg3 = []
    
    kmax = 1000

    for gameitr in range(0,kmax):
        game = setupGame()
        carDoor = winDoor(game)
        #print(f'---------\nRound {gameitr+1}\nWinning Door: {carDoor}')
        p1 = playerDoor(game)
        hstDoor = hostDoor(p1,game)
        #print(f'Host Door: {hstDoor}')

        game = swapKeyVal(game)
        ch1.append(isWinner(game,p1))
        avg1.append(np.array(ch1).mean())

        #print(f'Player1 Door: {p1}\tAVG1:\n{avg1}\n')
        
        tmp = game.get(hstDoor)
        #print(tmp)
        game = swapKeyVal(game)
        p2 = playerDoor(game,tmp,p1,1)
        game = swapKeyVal(game)
        ch2.append(isWinner(game,p2))
        avg2.append(np.array(ch2).mean())

        #print(f'Player2 Door: {p2}\tAVG2:\n{avg2}')

        tmp = game.get(hstDoor)
        game = swapKeyVal(game)
        p3 = playerDoor(game,tmp,p1)
        game = swapKeyVal(game)
        ch3.append(isWinner(game,p3))
        avg3.append(np.array(ch3).mean())

        #print(f'Player3 Door: {p3}\tAVG3:\n{avg3}\n')
        
        
        
    
    
    
  #  for k in range(1,kmax + 1):
   #     avg1.append(np.array(ch1[:k]).mean())
    #    avg2.append(np.array(ch2[:k]).mean())
     #   avg3.append(np.array(ch3[:k]).mean())
        
        
    x = range(0,kmax) 
    plt.figure()
    plt.plot(x,avg1, label = 'Player A - Never Change Door')
    plt.plot(x,avg2, label = 'Player B - Always Change Door')
    plt.plot(x,avg3, label = 'Player C - 50/50 Door Change')
    
    plt.legend()
    ag = plt.gca()
    ag.set_ylim(0,1)
    ag.set_xlim(0,kmax)
    #ag.set_xticks(([1] + list(np.arange(100,1001,100))))
    ag.set_xlabel(f'Number of Rounds')
    ag.set_ylabel(f'Winning Average')
    ag.set_title(f'Player A - Winning Mean')
    
    

    elapsed_time = time.time() - st
    print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    final_res = elapsed_time * 1000
    print('Execution time:', final_res, 'milliseconds')
    return





if __name__ == "__main__":
    main2c()
    
    
    
    
    
    

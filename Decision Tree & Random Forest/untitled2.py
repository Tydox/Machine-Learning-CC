# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 20:24:33 2022

@author: Danno
"""

depth_list = [1,2,3,4,5,10,20]
forest_size_list = [1,2,3,4,5,10,20,30,50,100]

dlist = [str(x) for x in depth_list]
flist = [str(x) for x in forest_size_list]
d =["depth=" + strg for strg in dlist]
f =["Forest Size=" + strg for strg in flist]
g = d+f
print(g)


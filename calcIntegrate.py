#!/usr/bin/env python
# coding=utf-8

from sympy import *
import numpy as np
import pandas as pd

x = symbols('x')
vl, vh = 100.0,200.0
lambs = [1.0,2.0,3.0,4.0]
T = 7.0
f = 1.0/(vh-vl)

Rs = []
for lamb in lambs:
	R = []
	for r in xrange(int(vl),int(vh),2):
		#integral = integrate((x-vl)/(vh-vl)+x/(vh-vl)*exp(-lamb*T*(1.0-(r-vl)/(vh-vl))*(1.0-(x-vl)/(vh-vl))),(x,r,vh))
		integral = integrate(((x-vl)/(vh-vl)+x/(vh-vl))*exp(-lamb*T*(1-(r-vl)/(vh-vl))*(1-(x-vl)/(vh-vl))),(x,r,vh))
                R.append(lamb*T*integral)
	Rs.append(R)
Rs = np.array(Rs)
Rs = np.transpose(Rs)

df = pd.DataFrame({
	'1':Rs[:,0],
	'2':Rs[:,1],
	'3':Rs[:,2],
	'4':Rs[:,3]
	})

print df.head()

df.to_csv('T=7vl=100vh=200.csv',index=False )

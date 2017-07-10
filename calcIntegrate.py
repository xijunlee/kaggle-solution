#!/usr/bin/env python
# coding=utf-8

from sympy import *

x, r = symbols('x r')
expr1 = 1-1/sqrt(2*pi)*integrate(exp(-x**2/2),(x,-oo,r))
expr2 = 1/sqrt(2*pi)*exp(-r**2/2)
result = solve(r-expr1/expr2,r)
print result

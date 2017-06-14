#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

auction = pd.read_csv("./auction.csv")

#print auction[0:3]
#print"price item sold for",auction.groupby('item')['price'].mean()


#auction.dropna(how='all',inplace=True)

#sns.boxplot(x=auction['item'],y=auction['price'])

#print auction.shape # the number of rows and of columns
#print auction['auction_type'].value_counts()
#print auction.groupby(['auction_type']).count()
#(auction['auction_type'].value_counts()*100/auction.shape[0]).plot(kind='bar',title='Perc of auctions by auction type')

#plt.show()
#print pd.crosstab(auction['item'],auction['auction_type'])
#print pd.crosstab(auction['item'],auction['auction_type']).iloc[0,0]

#pd.crosstab(auction['item'],auction['auction_type']).apply(lambda x:100*x/sum(x),axis=1).plot(kind='bar').set_title("Number of items sold by auction")
#plt.show()

group = auction.groupby(['auctionid']).count().reset_index()
ids = list(group[group['bid']>40]['auctionid'])

k = 4
f, axes = plt.subplots(k, k, figsize=(10, 10), sharex=False)
print ids
for i in xrange(16):
    id40 = auction[auction['auctionid']==ids[i]]['bidtime'].reset_index()
    sns.distplot(id40['bidtime'],kde=False,ax=axes[i/k,i%k],axlabel=False)
plt.show()


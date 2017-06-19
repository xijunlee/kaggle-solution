#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

players = pd.read_csv('./Players.csv')
seasons = pd.read_csv('./Seasons_Stats.csv')
# preview the data 
#print seasons.head()
#print players_seasons.head(2)

#players_seasons = pd.merge(players ,seasons ,on="Player").groupby('Player')

# data cleaning
players.drop('Unnamed: 0' ,axis=1,inplace=True)

players.dropna(how='all',inplace=True) #dropping the player whose value is null

players.set_index('Player',inplace=True) #setting the player name as the dataframe index

#print players.head()


'''
print('The Tallest Player in NBA History is:',players['height'].idxmax(),' with height=',players['height'].max(),' cm')
print('The Heaviest Player in NBA History is:',players['weight'].idxmax(),' with weight=',players['weight'].max(),' kg')

print('The Shortest Player in NBA History is:',players['height'].idxmin(),' with height=',players['height'].min(),' cm')
print('The Lightest Player in NBA History is:',players['weight'].idxmin(),' with weight=',players['weight'].min(),' kg')


print('The average height of NBA Players is ',players['height'].mean())
print('The average weight of NBA Players is ',players['weight'].mean())
'''

'''
bins=range(150,250,10)
plt.hist(players["height"],bins,histtype="bar",rwidth=1.2,color='#0ff0ff')
plt.xlabel('Height in Cm')
plt.ylabel('Count')
plt.axvline(players["height"].mean(), color='b', linestyle='dashed', linewidth=2)
plt.show()
'''
'''
bins=range(60,180,10)
plt.hist(players["weight"],bins,histtype="bar",rwidth=1.2,color='#4400ff')
plt.xlabel('Weight in Kg')
plt.ylabel('Count')
plt.axvline(players["weight"].mean(), color='black', linestyle='dashed', linewidth=2)
plt.plot()
'''
'''
college=players.groupby(['collage'])['height'].count().reset_index().sort_values(by='height',ascending=False)[:10]
college.set_index('collage',inplace=True)
college.columns=['Count']
ax=college.plot.bar(width=0.8)
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+0.35))
plt.show()
'''
'''
# written by myself
#college = players.groupby(['collage']).count().reset_index().sort_values(by='height',ascending=False)
#college=players.groupby(['collage'])['height'].count().reset_index().sort_values(by='height',ascending=False)
#college.set_index('collage',inplace=True)
#college.columns=['Count']
#print college.head(10)
college = players.groupby(['collage'])['height'].count().sort_values(ascending=False)[:10]
ax=college.plot.bar(width=0.8)
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.5))
plt.show()
'''

city=players.groupby(['birth_state'])['height'].count().reset_index().sort_values(by='height',ascending=False)[:10]
city.set_index('birth_state',inplace=True)
city.columns=['Count']
ax=city.plot.bar(width=0.8,color='#ab1abf')
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.show()

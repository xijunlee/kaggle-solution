#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import pdb

#读取文件
df = pd.read_csv('./TrainingData.csv')

#丢弃不需要的列
df = df.drop(['Line','Rel Time (Sec)','Status','Er','Tx','Description','Network','Node','Trgt','Src','B1','B2','B3','B4','B5','B6','B7','B8','Value','Trigger'],axis=1)

#按照FE,711,FC生成三张子表
df_FE = df[df['PT'].isin(['FE'])]
df_711 = df[df['PT'].isin(['711'])]
df_FC = df[df['PT'].isin(['FC'])]

#结果数组，找到相应的结果将保存到这些数组中
ABSTime = []
Long_Accel = []
Lat_Accel = []
Target_Speed = []
EngRPM = []
RRWheelSpeed = []
RLWheelSpeed = []

#按照ABSTime去找相应的条目
for i1 in xrange(df_FE.shape[0]):
    
    i2, i3 =  -1, -1
     
    time = df_FE.iat[i1,0]
    for j in xrange(df_FC.shape[0]):
        if df_FC.iat[j,0]>=time:
            #如果在FC表中找到了时间第一个大于time的条目，将其序号i2记录下来
            i2 = j
            break
    if i2 >= 0:
        for j in xrange(df_711.shape[0]):
            if df_711.iat[j,0]>=time:
            #如果在711表中找到了时间第一个大于time的条目，将其序号i2记录下来
                i3 = j
                break
    #如果在三个表中都找到对应的条目，那么将所需结果分别保存在各自数组中
    if i2 >= 0 and i3 >= 0:
        Target_Speed.append(df_711.iat[i3,3])
        EngRPM.append(df_711.iat[i3,5])
        RRWheelSpeed.append(df_FC.iat[i2,7])
        RLWheelSpeed.append(df_FC.iat[i2,9])
        ABSTime.append(time)
        Long_Accel.append(df_FE.iat[i1,3])
        Lat_Accel.append(df_FE.iat[i1,5])


#每50个生成结果小文件

for i in xrange(len(ABSTime),50):
    if i+50 < len(ABSTime):
        result = pd.DataFrame({'ABSTime':ABSTime[i:i+50],
                               'Long_Accel':Long_Accel[i:i+50],
                               'Lat_Accel':Lat_Accel[i:i+50],
                               'Target_Speed':Target_Speed[i:i+50],
                               'EngRPM':EngRPM[i:i+50],
                               'RRWheelSpeed':RRWheelSpeed[i:i+50],
                               'RLWheelSpeed':RLWheelSpeed[i:i+50]
             })

    #保存成.csv文件
    result.to_csv('result'+str(i)+'.csv',index=False)



import pandas as pd
import matplotlib.pyplot as plt
import re

#df = pd.read_csv('/Electric_Production.csv', sep=',')
df = pd.read_csv('/daily-minimum-temperatures-in-me.csv', sep=',')

values = [tuple(x) for x in df.values]
dates, dataIn = zip(*values) 
data = [float(x) for x in dataIn if re.match(r'^-?\d+(?:\.\d+)$', x) is not None]
faulty = [x for x in dataIn if re.match(r'^-?\d+(?:\.\d+)$', x) is None]
print(faulty)
print(len(faulty))
#Change m to change window size (bigger = shoother and shorter)
m=50
start=m+1
alfa=float(1/(2*m+1))
Sum=0
lenght=len(data)

for i in range(start,lenght-start):
  for j in range(-m,m):
    Sum+=data[i+j]*alfa
  if(i==start):
    resultSum = [Sum]*(m)
  resultSum.append(Sum);
  Sum=0
#print(resultSum)
plt.plot(range(lenght),data)
plt.plot(range(len(resultSum)),resultSum)

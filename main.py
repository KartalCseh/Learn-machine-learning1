# Firstly I import the necessary packages
import pandas as pd
import numpy as np
import sklearn as skl
import seaborn as sns
import matplotlib.pyplot as plt

teams = pd.read_csv('teams.csv')
# print(teams)

print(teams[['athletes', 'medals']].corr())

sns.lmplot(x='athletes', y='medals', data=teams, fit_reg=True, ci=None)     # Plotting the linear regression between nummbers of athletes and medals
plt.show()  # this line is needed using pycharm, but not in Jupiter notebook
# As we can see yes, there is a correlation, so we can use the athletes to predict the number of medals

'''sns.lmplot(x='athletes', y='medals', data=teams, fit_reg=True, ci=None)     # Plotting the linear regression between nummbers of athletes and medals
plt.show()  # later I want to use this'''

#teams.plot.hist(y='medals')
#plt.show()

#Remove those data, where are missing values

#teams = [teams.isnull().any(axis=1)]
#print(teams)

teams = teams.dropna()
# 2.Building model
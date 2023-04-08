# Firstly I import the necessary packages
import pandas as pd
import numpy as np
import sklearn as skl
import seaborn as sns
import matplotlib.pyplot as plt

teams = pd.read_csv('teams.csv')
# print(teams)

print(teams[['athletes', 'medals']].corr())

#sns.lmplot(x='athletes', y='medals', data=teams, fit_reg=True, ci=None)     # Plotting the linear regression between nummbers of athletes and medals
# plt.show()  # this line is needed using pycharm, but not in Jupiter notebook
# As we can see yes, there is a correlation, so we can use the athletes to predict the number of medals

'''sns.lmplot(x='athletes', y='medals', data=teams, fit_reg=True, ci=None)     # Plotting the linear regression between nummbers of athletes and medals
plt.show()  # later I want to use this'''

#teams.plot.hist(y='medals')
#plt.show()

#Remove those data, where are missing values

#teams = [teams.isnull().any(axis=1)]
#print(teams)

teams = teams.dropna()
# 2.Building model,firstly we need to split our data to train and test (usually 80:20)
trainset = teams[teams['year'] < 2012].copy()
testset = teams[teams['year'] >= 2012].copy()

from sklearn.linear_model import LinearRegression
reg = LinearRegression()

predictors = ['athletes', 'prev_medals']
target = 'medals'

# Fit the model on the training data
reg.fit(trainset[predictors], trainset[target])

# Generate predictions on the test data
predictions = reg.predict(testset[predictors])

#print(predictions)
'''This is in the video, but it gave error: 
reg.fit(trainset[predictors], trainset['medals'])
LinearRegression()
predictons = reg.predict(test, predictors)
print(predictons)'''
# make prediction round (beaause it has mean, there is no 1.4 medal
testset['predictions'] = predictions
testset.loc[testset['predictions'] < 0, 'predictions'] = 0
testset['predictions'] = testset['predictions'].round()

from sklearn.metrics import  mean_absolute_error
error = mean_absolute_error(testset['medals'], testset['predictions'])
print('Error:', error)
describe_medals = teams.describe()['medals']
print(describe_medals)

# This was the measurimnr errorsmodel

USA = testset[testset['team'] == 'USA']
print('USA medals: \n', USA['medals'],'\nUSA predictions:\n', USA['predictions'])

errors = (testset['medals'] - testset['predictions']).abs()
print(errors)
error_by_team = errors.groupby(testset['team']).mean()
print("Error by team", error_by_team)

medals_by_team = testset['medals'].groupby(testset['team']).mean()
error_ratio = error_by_team / medals_by_team
error_ratio = error_ratio[~pd.isnull(error_ratio)]
error_ratio = error_ratio[np.isfinite(error_ratio)]
print('Error ratio:\n', error_ratio)

error_ratio.plot.hist()
plt.show()
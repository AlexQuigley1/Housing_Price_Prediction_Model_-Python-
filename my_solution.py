import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt



'''
1. show summary statistics                                                      Done
2. summary of possible missing values                                           Done

3. format date column and numeric columns                                       Done

4. visualise data   salePrice against others, neighbourhood or time             Done

5. treat missing values and possible outliers                                   Done
6. use scatter plots to determine wether that column affects house price        Done

7. check min and max of numeric data, if too big a range                        Done             
8. carry out data normalization or scaling                                      Done

9. highlight features of relevance to sale price

'''
#load in dataset
data = pd.read_csv("Manhattan12.csv", header=4)
columns = ['BOROUGH', 'NEIGHBORHOOD', 'BUILDING CLASS CATEGORY', 'TAX CLASS AT PRESENT', 'BLOCK', 'EASE-MENT', 'LOT', 'BUILDING CLASS AT PRESENT', 'ADDRESS', 'APART\nMENT\nNUMBER', 'ZIP CODE', 'RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'TOTAL UNITS', 'GROSS SQUARE FEET', 'YEAR BUILT', 'TAX CLASS AT TIME OF SALE', 'BUILDING CLASS AT TIME OF SALE', 'SALE\nPRICE', 'SALE DATE']
nonNumerical = ['BOROUGH', 'NEIGHBORHOOD', 'BUILDING CLASS CATEGORY', 'BLOCK', 'LOT', 'BUILDING CLASS AT PRESENT', 'ADDRESS', 'APART\nMENT\nNUMBER', 'ZIP CODE', 'BUILDING CLASS AT TIME OF SALE', 'EASE-MENT']
numerical = ["SALE\nPRICE", "LAND SQUARE FEET", "GROSS SQUARE FEET", "TOTAL UNITS", "RESIDENTIAL UNITS", "YEAR BUILT", "TAX CLASS AT TIME OF SALE", 'TAX CLASS AT PRESENT']


print(data.dtypes)

#remove spaces and make empty spaces np.NaN
for col in columns:
    try:
        data[col] = data[col].str.strip()
    except:
        print("not a string")

data = data.replace("", np.NaN)


#format numerical columns
commasInData = ["SALE\nPRICE", "LAND SQUARE FEET", "GROSS SQUARE FEET", "TOTAL UNITS", "RESIDENTIAL UNITS"]
for com in commasInData:
    data[com] = data[com].str.replace(',', '')

data['SALE\nPRICE'] = data['SALE\nPRICE'].str.replace('$', '')

convertToNumeric = ["SALE\nPRICE", "LAND SQUARE FEET", "GROSS SQUARE FEET", "TOTAL UNITS", "RESIDENTIAL UNITS"]
for obj in convertToNumeric:
    data[obj] = pd.to_numeric(data[obj])

#format date column
data['SALE DATE'] = pd.to_datetime(data['SALE DATE'])


data['TAX CLASS AT PRESENT'] = data['TAX CLASS AT PRESENT'].replace(r'[A-Z]$', np.NaN, regex = True)

data['TAX CLASS AT PRESENT'] = pd.to_numeric(data['TAX CLASS AT PRESENT'])
print(data.describe())

'''
VISUALISE
THE
DATA
'''
'''
def scatterMatrix(plot_cols, df):
    from pandas.plotting import scatter_matrix
    fig = plt.figure(1, figsize=(10, 10))
    fig.clf()
    ax = fig.gca()
    scatter_matrix(df[plot_cols], alpha=0.3, diagonal='hist', ax = ax)
    plt.show()
    return('Done')
scatterMatrix(numerical, data)
'''

#scatter graph
'''
data.plot.scatter(x='GROSS SQUARE FEET',y="SALE\nPRICE", title="Original data")
plt.xticks(rotation=90)
#plt.xticks(np.arange(data['TAX CLASS AT TIME OF SALE'].min(), data['TAX CLASS AT TIME OF SALE'].max()+1, 1))
plt.show()
'''



data.drop(nonNumerical, axis=1, inplace=True)


data.drop(['TAX CLASS AT PRESENT'], axis=1, inplace=True)

print("removed non numerical shape: ")
print(data.shape)

data["SALE\nPRICE"] = data["SALE\nPRICE"].replace(0, np.NaN)
data.dropna(subset=['SALE\nPRICE'], axis=0, inplace = True )
print("removed 0s in sale price shape: ")
print(data.shape)

zeroList = ["LAND SQUARE FEET", "GROSS SQUARE FEET"]

data[zeroList] = data[zeroList].fillna(method="ffill")
'''
for zero in zeroList:
    data[zero] = data[zero].replace(0, np.NaN)
'''
print("forward filling the empty cell data points")
data["LAND SQUARE FEET"] = data["LAND SQUARE FEET"].replace(0, np.NaN)
data.dropna(subset=["LAND SQUARE FEET"], axis=0, inplace = True)
print("removed 0s in Land Square Feet :")
print(data.shape)

data["GROSS SQUARE FEET"] = data["GROSS SQUARE FEET"].replace(0, np.NaN)
data.dropna(subset=["GROSS SQUARE FEET"], axis=0, inplace = True)
print("removed 0s in Gross Square Feet :")
print(data.shape)

data.dropna(subset=["YEAR BUILT"], axis=0, inplace = True)
print("removed 0s in Year Built:")
print(data.shape)

zeroList = ["SALE\nPRICE", "LAND SQUARE FEET", "GROSS SQUARE FEET", "YEAR BUILT"]



#data.dropna(subset=zeroList, axis=0, inplace = True)


p75, p25 = np.quantile(data["SALE\nPRICE"], [0.75, 0.25])
iqr = p75-p25
print("quartile 3:")
print(p75)
print("quartile 1:")
print(p25)
print("interquartile range:")
print(iqr)

std = np.std(data['SALE\nPRICE'])

mean = data['SALE\nPRICE'].median()

upperQuartile = p75+1.5*iqr
lowerQuartile = p25-1.5*iqr
print("upper limit for outliers")
print(upperQuartile)
print("lower limit for outliers")
print(lowerQuartile)

data = data.loc[(data["SALE\nPRICE"] > 50000) & (data["SALE\nPRICE"] < upperQuartile)]


#Removing outliers
def id_outlier(df, value):
    ## Create a vector of 0 of length equal to the number of rows
    temp = [0] * df.shape[0]
    ## test each outlier condition and mark with a 1 as required
    p75, p25 = np.quantile(df[value], [0.75, 0.25])
    iqr = p75-p25
    upperQuartile = p75+1.5*iqr
    lowerQuartile = p25-1.5*iqr

    #df = df.loc[(df[value] > lowerQuartile) & (df[value] < upperQuartile)]
    
    
    for i, x in enumerate(df[value]):
        if (x > upperQuartile) or (x < lowerQuartile): temp[i] = 1

    df['outlier'] = temp # append a column to the data frame
    df = df[df.outlier == 0] # filter for outliers
    df.drop('outlier', axis = 1, inplace = True)
    
    
    return df


for val in zeroList:
    data = id_outlier(data, val)  # mark outliers       


'''
#Removing outliers
def id_outlier2(df):
    ## Create a vector of 0 of length equal to the number of rows
    temp = [0] * df.shape[0]
    ## test each outlier condition and mark with a 1 as required
    for i, x in enumerate(df["SALE\nPRICE"]):
        if (x < 100000): temp[i] = 1

    df['outlier'] = temp # append a column to the data frame
    df = df[df.outlier == 0] # filter for outliers
    df.drop('outlier', axis = 1, inplace = True)
    return df


data = id_outlier2(data)

'''
print("data after removing outliers")
print(data.shape)

def normalize(df):
    #select numerical columns
    num_cols = df.select_dtypes(include=[np.number]).copy()
    df_norm = ((num_cols-num_cols.min())/(num_cols.max()-num_cols.min()))
    return df_norm

data['logPrice'] = np.log(data['SALE\nPRICE'])




data_norm = normalize(data)

print("after normalising data")
print(data.shape)

#scatter plot
'''
#data = data.groupby('TAX CLASS AT TIME OF SALE')
data_norm.plot.scatter(x='SALE DATE',y="SALE\nPRICE", title="Removed 0s data")
plt.xticks(rotation=90)
#plt.xticks(np.arange(data['TAX CLASS AT TIME OF SALE'].min(), data['TAX CLASS AT TIME OF SALE'].max()+1, 1))
plt.show()
'''

#scatter graph matrix


def scatterMatrix(plot_cols, df):
    from pandas.plotting import scatter_matrix
    fig = plt.figure(1, figsize=(10, 10))
    fig.clf()
    ax = fig.gca()
    scatter_matrix(df[plot_cols], alpha=0.3, diagonal='hist', ax = ax)
    plt.show()
    return('Done')
scatterMatrix(convertToNumeric, data)



print(data.shape)

from sklearn import svm, feature_selection, linear_model
from sklearn.model_selection import train_test_split
df = data_norm.select_dtypes(include=[np.number]).copy()
feature_cols = df.columns.values.tolist()
feature_cols.remove('SALE\nPRICE')
feature_cols.remove('logPrice')
XO = df[feature_cols]
YO = df['logPrice']
estimator = svm.SVR(kernel="linear")
selector = feature_selection.RFE(estimator, 5, step=1)
selector = selector.fit(XO, YO)
# From the ranking you can select your predictors with rank 1
# Model 1; let us select the folowing features as predictors:
select_features = np.array(feature_cols)[selector.ranking_ == 1].tolist()
print(select_features)
X = df[select_features]
Y = df['logPrice']
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2)
lm = linear_model.LinearRegression()
lm.fit(trainX, trainY)
# Inspect the calculated model equations
print("Y-axis intercept {}".format(lm.intercept_))
print("Weight coefficients:")
for feat, coef in zip(select_features, lm.coef_):
    print(" {:>20}: {}".format(feat, coef))
# The value of R^2
print("R squared for the training data is {}".format(lm.score(trainX, trainY)))
print("Score against test data: {}".format(lm.score(testX, testY)))

data_norm.rename(columns={'LAND SQUARE FEET' : 'landSquareFeet', 'RESIDENTIAL UNITS' : 'residentialUnits', 'COMMERCIAL UNITS' : 'commercialUnits', 'TOTAL UNITS' : 'totalUnits', 'GROSS SQUARE FEET' : 'grossSquareFeet', 'YEAR BUILT' : 'yearBuilt', 'TAX CLASS AT TIME OF SALE' : 'TaxClassAtTimeOfSale', 'SALE\nPRICE' : 'salePrice'}, inplace=True)

import statsmodels.formula.api as smf
model = smf.ols(formula='logPrice ~ landSquareFeet + residentialUnits + commercialUnits + grossSquareFeet + totalUnits', data=data_norm).fit()
print(model.params)
print('R squared is: ', model.rsquared)
model.summary()




import matplotlib.pylab as plb
pred_trainY = lm.predict(trainX)
plt.figure(figsize = (10, 5))
plt.plot(trainY, pred_trainY, 'o')
plb.xlabel('Actual Price')
plt.ylabel('Predicted Prices')
plt.title = "Predicted vs Actual"
plt.show()

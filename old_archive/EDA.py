import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from load import train

#All numerical columns to a seperate data frame
train_numeric = train.select_dtypes(include=[np.number])

#All non-numeric columns to a seperate data frame
train_categorical = train.select_dtypes(exclude=[np.number])

# Correlation of all numerical columns
corr = train_numeric.corr()

def eda_uni(df, col, bins=20, kde=False, target='SalePrice'):
    '''
    Univariate analysis

    Arguments:

    df:   Data frame
    col:  Column of the data frame
    categorized: True if column is a categorized column. Default=False
    bins: Number of bins for the histogram. Default = 20
    kde:  Boolean whether to plot the exact count or the density
    target: Target column of the data frame (Y)
    '''
    
    print('\n', '*' * 100, '\n')
    print('Describe:\n')
    print(df[col].describe())
    print('\n', '*' * 100, '\n')

    nulls = df[col].isnull().sum()
    print('Null: {}'.format(nulls))
    print('\n', '*' * 100, '\n')
    
    if(col in train_numeric):
        
        #Correlation
        corre = round(corr[target][col], 3)
        print('Correlation with target: {}'.format(corre))
        #print(round(corr[target][col], 3))
        print('\n', '*' * 100, '\n')

        #Skewness
        skew = round(np.abs(df[col].skew()), 3)
        print('Skewness: {}\n'.format(skew))

        if( skew > 1 ):
            print('Data array is HIGHLY skewed')
        elif( skew > 0.5 ):
            print('Data array is MODERATELY skewed')
        else:
            print('Data array is NOT skewed')
        print('\n', '*' * 100, '\n')

        #Histogram
        print('Histogram\n')
        sns.distplot(df[col], bins=bins, kde=kde)
        plt.show()
        print('\n', '*' * 100, '\n')

        #Regression plot
        print('Regression Plot\n')
        sns.regplot(x=col, y=target, data=df)
        plt.show()
        print('\n', '*' * 100, '\n')


    if(col in train_categorical):
        
        #Value counts
        print('Number of values for each class:\n')
        print(df[col].value_counts(sort=False))
        print('\n', '*' * 100, '\n')

    #Boxplot
    print('Boxplot\n')
    sns.catplot(x=col, y=target, kind='box', data=df)
    plt.show()
    print('\n', '*' * 100, '\n')



def zscore(df, col, threshold=3):
    '''
    Univariate analysis

    Returns a data frame with columns as index, value and Z-score of the input data frame column

    Arguments:

    df:   Data frame
    col:  Column of the data frame
    threshold: Threshold value for the Z-score. Higher than threshold indicates outlier.
    '''

    try:
        df[col]
    except:
        raise Exception('Column not found in data frame')
    try:
        mask = df[col].notnull() #Eliminate nulls. Otherwise zscore func returns null
        mycol = df[mask][col].copy()
        z = pd.Series(np.abs(stats.zscore(mycol)))
    except:
        raise ValueError('Column must be numeric')

    ret = pd.concat([mycol,z],axis=1)
    ret = ret.reset_index()
    ret.columns = [['index','value','zscore']]

    # Return index and value of data frame rows which z-scrore gt 3
    mask2 = np.array(ret['zscore'] > threshold)
    return ret[mask2]


    # for i, v in enumerate(z):
    #     if v > threshold:
    #         ret = ret.append(pd.Series([i, df[col][i], v], index=ret.columns), ignore_index=True)
    
    # return ret


def dummify(df, col):

    '''
    Dummify a column of a data frame and return data frame with dummified column 

    Arguments:

    df: Data frame name
    col: Column name
    '''

    dum = pd.get_dummies(df[col], prefix=col)
    dum = dum.drop(col+'_1.0', axis=1)
    df = pd.concat([df.drop(col, axis=1), dum], axis=1)
    return df
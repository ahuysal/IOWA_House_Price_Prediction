import os
import pandas as pd

project_path = 'C:\\Users\\hilmiuysal\\Desktop\\NYC Data Science Academy\\Projects\\Kaggle_HousePrices'
os.chdir(project_path)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print('Training and test data is loaded')
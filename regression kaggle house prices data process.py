import numpy as np
import pandas as pd

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

path = 'C:/Users/Val/Desktop/Kaggle/test_smaller.csv'
raw_dataset = pd.read_csv(path)
dataset = raw_dataset.copy()

paveddrive = dataset.pop('PavedDrive')
dataset['Paved'] = (paveddrive == 'Y')*1.0
dataset['Partial'] = (paveddrive == 'P')*1.0
dataset['Dirt/Gravel'] = (paveddrive == 'N')*1.0

dataset['PoolQC'] = (dataset['PoolQC'] == 'Fa')*1.0 + \
                    (dataset['PoolQC'] == 'TA')*2.0 + (dataset['PoolQC'] == 'Gd')*3.0 + (dataset['PoolQC'] == 'Ex')*4.0

dataset['Fence'] = (dataset['Fence'] == 'MnWw')*1.0 + \
                   (dataset['Fence'] == 'GdWo')*2.0 + (dataset['Fence'] == 'MnPrv')*3.0 + \
                   (dataset['Fence'] == 'GoodPrv')*4.0

dataset['MasVnrArea'] = dataset['MasVnrArea'].fillna(0)
dataset['GarageYrBlt'] = dataset['GarageYrBlt'].fillna(0)
dataset.to_csv('C:/Users/Val/Desktop/test_smaller_processed.csv')

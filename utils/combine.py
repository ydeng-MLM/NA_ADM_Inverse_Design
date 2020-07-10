import os
import glob
import numpy as np
import pandas as pd


def importData(data_dir, findex, il, drange):
    ftr = []
    ftr_array = pd.read_csv(os.path.join(data_dir, 'train'+str(findex)+'_augmented.csv'), delimiter=' ',
                                header=None, usecols=drange)
        # append each data point to ftr and lbl
    ftr = ftr_array.iloc[il]
    ftr = np.array(ftr, dtype='float32')
    return ftr

ftr = pd.DataFrame()
for i in range(1,11):
    ftr_single_array = pd.read_csv('D:/Yang_MM_Absorber_ML/dataIn/train'+str(i)+'_augmented.csv',delimiter=' ', header=None, usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    local_iloc = pd.DataFrame(data=np.arange(len(ftr_single_array)))
    findex = pd.DataFrame(data=np.full((len(ftr_single_array)), i))
    combined = pd.concat([ftr_single_array, local_iloc, findex], axis=1)
    ftr = ftr.append(combined, ignore_index=True)

ftr.to_csv(path_or_buf='D:/Yang_MM_Absorber_ML/helper.csv', sep=' ',  index=False, header=None)

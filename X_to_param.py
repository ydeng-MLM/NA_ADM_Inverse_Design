import pandas as pd
import os
import numpy as np

import time
import datetime



def X_to_param(col_names, geoboundary = [0.3, 0.6, 1, 1.5, 0.1, 0.2, -0.7854, 0.7854], out_dir='D:/Yang_MM_Absorber_ML/NA/meta_material', rand_seed=1234):
    print('Transferring normalized X to param data...')

    # generate random parameter samples for each element
    xdata = pd.read_csv(os.path.join('D:/Yang_MM_Absorber_ML/NA/meta_material/test_Xpred_test_updated_data.csv'), delimiter=' ', header=None)
    xdata = np.array(xdata, dtype='float32')
    xdata[:, 0:1] = xdata[:, 0:1]*((geoboundary[1] - geoboundary[0])/2)+((geoboundary[0] + geoboundary[1])/2)
    xdata[:, 1:2] = xdata[:, 1:2]*((geoboundary[3] - geoboundary[2])/2)+((geoboundary[2]+geoboundary[3])/2)
    xdata[:, 2:10] = xdata[:, 2:10]*((geoboundary[5] - geoboundary[4])/2)+((geoboundary[4]+geoboundary[5])/2)
    xdata[:, 10:] = xdata[:, 10:]*((geoboundary[7] - geoboundary[6])/2)+((geoboundary[6]+geoboundary[7])/2)
    xdata[:, 10:] = xdata[:, 10:]/3.1415926*180

    # convert to dataframe and export
    param = pd.DataFrame(xdata, columns=col_names)
    param.to_csv(os.path.join(out_dir, 'param_pred.txt'), index=False, sep='\t')

if __name__=='__main__':
    # choices built-in for dlm project
    col_names = ['h', 'p', 'rma1', 'rma2', 'rma3', 'rma4', 'rmi1', 'rmi2', 'rmi3', 'rmi4', 'theta1', 'theta2', 'theta3', 'theta4']
    X_to_param(col_names)

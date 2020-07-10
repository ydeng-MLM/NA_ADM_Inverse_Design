#import plotsAnalysis
import os
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt

def InferenceAccuracyExamplePlot(model_name, save_name, title, sample_num=10,  fig_size=(15,5), random_seed=1,
                                 target_region=[0,300 ]):
    """
    The function to plot the Inference accuracy and compare with FFDS algorithm.
    It takes the Ypred and Ytruth file as input and plot the first <sample_num> of spectras.
    It also takes a random of 10 points to at as the target points.
    :param model_name: The model name as the postfix for the Ytruth file
    :param save_name:  The saving name of the figure
    :param title:  The saving title of the figure
    :param sample_num: The number of sample to plot for comparison
    :param fig_size: The size of the figure
    :param random_seed:  The random seed value
    :param target_region:  The region that the targets get
    :return:
    """
    # Get the prediction and truth file first
    Ytruth_file = os.path.join('data','test_Ytruth_{}.csv'.format(model_name))
    Ypred_file = os.path.join('data','test_Ypred_{}.csv'.format(model_name))
    Ytruth = pd.read_csv(Ytruth_file, header=None, delimiter=' ').values
    Ypred = pd.read_csv(Ypred_file, header=None, delimiter=' ').values

    # Draw uniform random distribution for the reference points
    np.random.seed(random_seed)     # To make sure each time we have same target points
    targets = target_region[0] + (target_region[1] - target_region[0]) * np.random.uniform(low=0, high=1, size=10) # Cap the random numbers within 0-299
    targets = targets.astype("int")
    # Make the frequency into real frequency in THz
    fre_low = 0.86
    fre_high = 1.5
    frequency = fre_low + (fre_high - fre_low)/len(Ytruth[0, :]) * np.arange(300)

    for i in range(sample_num):
        # Start the plotting
        f = plt.figure(figsize=fig_size)
        plt.title(title)
        plt.scatter(frequency[targets], Ytruth[i,targets], label='S*')
        plt.plot(frequency, Ytruth[i,:], label='FFDS')
        plt.plot(frequency, Ypred[i,:], label='Candidate')
        plt.legend()
        plt.ylim([0,1])
        plt.xlim([fre_low, fre_high])
        plt.grid()
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Transmittance")
        plt.savefig(os.path.join('data',save_name + str(i) + '.png'))

filename_VAE = '20190905_171115'
filename_BKP = '20190903_220309'
filename_TDS = '15e-4_3'
if __name__ == '__main__':
    InferenceAccuracyExamplePlot(filename_VAE, 'VAE', 'VAE spectra reconstruction result')
    InferenceAccuracyExamplePlot(filename_BKP, 'Backprop', 'VAE spectra reconstruction result')
    InferenceAccuracyExamplePlot(filename_TDS, 'Tantem', 'VAE spectra reconstruction result')

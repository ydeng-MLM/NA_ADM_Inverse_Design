"""
This file serves as a prediction interface for the network
"""
# Built in
import os
import sys
sys.path.append('../utils/')
# Torch

# Own
import flag_reader
from class_wrapper import Network
from model_maker import Backprop
from utils import data_reader
from utils.helper_functions import load_flags
from utils.evaluation_helper import plotMSELossDistrib
# Libs
import numpy as np
import matplotlib.pyplot as plt


def predict_from_model(pre_trained_model, Xpred_file, no_plot=True):
    """
    Predicting interface. 1. Retreive the flags 2. get data 3. initialize network 4. eval
    :param model_dir: The folder to retrieve the model
    :param Xpred_file: The Prediction file position
    :param no_plot: If True, do not plot (For multi_eval)
    :return: None
    """
    # Retrieve the flag object
    print("This is doing the prediction for file", Xpred_file)
    print("Retrieving flag object for parameters")
    if (pre_trained_model.startswith("models")):
        eval_model = pre_trained_model[7:]
        print("after removing prefix models/, now model_dir is:", eval_model)
    
    flags = load_flags(pre_trained_model)                       # Get the pre-trained model
    flags.eval_model = pre_trained_model                    # Reset the eval mode
    flags.test_ratio = 0.1              #useless number  

    # Get the data, this part is useless in prediction but just for simplicity
    train_loader, test_loader = data_reader.read_data(flags)
    print("Making network now")

    # Make Network
    ntwk = Network(Backprop, flags, train_loader, test_loader, inference_mode=True, saved_model=flags.eval_model)
    print("number of trainable parameters is :")
    pytorch_total_params = sum(p.numel() for p in ntwk.model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    # Evaluation process
    print("Start eval now:")
    
    if not no_plot:
        # Plot the MSE distribution
        pred_file, truth_file = ntwk.predict(Xpred_file, no_save=False)
        flags.eval_model = pred_file.replace('.','_') # To make the plot name different
        plotMSELossDistrib(pred_file, truth_file, flags)
    else:
        pred_file, truth_file = ntwk.predict(Xpred_file, no_save=True)
    
    print("Evaluation finished")

    return pred_file, truth_file, flags

def ensemble_predict(model_list, Xpred_file, model_dir=None, no_plot=True, remove_extra_files=True):
    """
    This predicts the output from an ensemble of models
    :param model_list: The list of model names to aggregate
    :param Xpred_file: The Xpred_file that you want to predict
    :param model_dir: The directory to plot the plot
    :param no_plot: If True, do not plot (For multi_eval)
    :param remove_extra_files: Remove all the files generated except for the ensemble one
    :return: The prediction Ypred_file
    """
    print("this is doing ensemble prediction for models :", model_list)
    pred_list = []
    # Get the predictions into a list of np array
    for pre_trained_model in model_list:
        pred_file, truth_file, flags = predict_from_model(pre_trained_model, Xpred_file)
        #pred = np.loadtxt(pred_file, delimiter=' ')
        #if remove_extra_files:          # Remove the generated files
        #    os.remove(pred_file)
        pred_list.append(np.copy(np.expand_dims(pred_file, axis=2)))
    # Take the mean of the predictions
    pred_all = np.concatenate(pred_list, axis=2)
    pred_mean = np.mean(pred_all, axis=2)
    save_name = Xpred_file.replace('Xpred', 'Ypred')
    np.savetxt(save_name, pred_mean)
    
    # If no_plot, then return
    if no_plot:
        return

    # saving the plot down
    flags.eval_model = 'ensemble_plot' + Xpred_file.replace('/', '')
    if model_dir is None:
        plotMSELossDistrib(save_name, truth_file, flags)
    else:
        plotMSELossDistrib(save_name, truth_file, flags, save_dir=model_dir)




def predict_all(models_dir="data"):
    """
    This function predict all the models in the models/. directory
    :return: None
    """
    for file in os.listdir(models_dir):
        if 'Xpred' in file and 'meta_material' in file:                     # Only meta material has this need currently
            print("predicting for file", file)
            predict_from_model("models/meta_materialreg0.0005trail_2_complexity_swipe_layer1000_num6", 
            os.path.join(models_dir,file))
    return None


def ensemble_predict_master(model_dir, Xpred_file, plot_dir=None):
    print("entering folder to predict:", model_dir)
    model_list = []
    for model in os.listdir(model_dir):
        print("entering:", model)
        if 'skip' in model:             # For skipping certain folders
            continue;
        if os.path.isdir(os.path.join(model_dir,model)):
            model_list.append(os.path.join(model_dir, model))
    if plot_dir is None:
        ensemble_predict(model_list, Xpred_file, model_dir)
    else:
        ensemble_predict(model_list, Xpred_file, plot_dir)
        


def predict_ensemble_for_all(model_dir, Xpred_file_dirs):
    for files in os.listdir(Xpred_file_dirs):
        if 'Xpred' in files:
            ensemble_predict_master(model_dir, os.path.join(Xpred_file_dirs, files), Xpred_file_dirs)


if __name__ == '__main__':
    """
    #predict_all('/work/sr365/multi_eval/Random/meta_material')
    k_list = [5,10,15,19]
    for k in k_list:
        ensemble_predict_master('/work/sr365/new_data_investigation/MM_augmented/top{}/'.format(k), 
                                '/work/sr365/new_data_investigation/MM_augmented/top{}/Xpred.csv'.format(k))
                                """
    #ensemble_predict_master('/work/sr365/new_data_investigation/MM_both_augmented_ensemble/', 
    #                        '/work/sr365/new_data_investigation/MM_both_augmented_ensemble/Xpred.csv')
    predict_from_model("models/20200603_123559/","data/Xpred.csv",no_plot=False)
   
    #ensemble_predict_master('/work/sr365/ensemble_forward/models', '/work/sr365/ensemble_forward/Xpred.csv')
    #predict_ensemble_for_all('/work/sr365/new_data_investigation/MM_both_augmented_ensemble/', '/hpc/home/sr365/Pytorch/VAE/data/')  
    #predict_ensemble_for_all('/work/sr365/new_data_investigation/MM_both_augmented_ensemble/', '/work/sr365/multi_eval/Backprop/MMcombined')  

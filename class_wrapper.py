"""
The class wrapper for the networks
"""
# Built-in
import os
import time
import sys
sys.path.append('../utils/')

# Torch
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
from torch.optim import lr_scheduler

# Libs
import numpy as np
import random
from math import inf
import matplotlib.pyplot as plt
import pandas as pd
# Own module
from utils.time_recorder import time_keeper
from utils.helper_functions import simulator

class Network(object):
    def __init__(self, model_fn, flags, train_loader, test_loader,
                 ckpt_dir=os.path.join(os.path.abspath(''), 'models'),
                 inference_mode=False, saved_model=None):
        self.model_fn = model_fn                                # The model maker function
        self.flags = flags                                      # The Flags containing the specs
        if inference_mode:                                      # If inference mode, use saved model
            if saved_model.startswith('models/'):
                saved_model = saved_model.replace('models/','')
            self.ckpt_dir = os.path.join(ckpt_dir, saved_model)
            self.saved_model = saved_model
            print("This is inference mode, the ckpt is", self.ckpt_dir)
        else:                                                   # training mode, create a new ckpt folder
            if flags.model_name is None:                    # leave custume name if possible
                self.ckpt_dir = os.path.join(ckpt_dir, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
            else:
                self.ckpt_dir = os.path.join(ckpt_dir, flags.model_name)
        self.model = self.create_model()                        # The model itself
        self.loss = self.make_loss()                            # The loss function
        self.optm = None                                        # The optimizer: Initialized at train() due to GPU
        self.optm_eval = None                                   # The eval_optimizer: Initialized at eva() due to GPU
        self.lr_scheduler = None                                # The lr scheduler: Initialized at train() due to GPU
        self.train_loader = train_loader                        # The train data loader
        self.test_loader = test_loader                          # The test data loader
        self.log = SummaryWriter(self.ckpt_dir)     # Create a summary writer for keeping the summary to the tensor board
        self.best_validation_loss = float('inf')    # Set the BVL to large number

    def make_optimizer_eval(self, geometry_eval):
        """
        The function to make the optimizer during evaluation time.
        The difference between optm is that it does not have regularization and it only optmize the self.geometr_eval tensor
        :return: the optimizer_eval
        """
        if self.flags.optim == 'Adam':
            op = torch.optim.Adam([geometry_eval], lr=self.flags.lr)
        elif self.flags.optim == 'RMSprop':
            op = torch.optim.RMSprop([geometry_eval], lr=self.flags.lr)
        elif self.flags.optim == 'SGD':
            op = torch.optim.SGD([geometry_eval], lr=self.flags.lr)
        else:
            raise Exception("Your Optimizer is neither Adam, RMSprop or SGD, please change in param or contact Ben")
        return op

    def create_model(self):
        """
        Function to create the network module from provided model fn and flags
        :return: the created nn module
        """
        model = self.model_fn(self.flags)
        # summary(model, input_size=(128, 8))
        print(model)
        return model

    def make_loss(self, logit=None, labels=None, G=None, W=False):
        """
        Create a tensor that represents the loss. This is consistant both at training time \
        and inference time for Backward model
        :param logit: The output of the network
        :param labels: The ground truth labels
        :return: the total loss
        """
        if logit is None:
            return None
        #if self.flags.data_set != 'gaussian_mixture':
        MSE_loss = nn.functional.mse_loss(logit, labels)          # The MSE Loss
        if W:
            weight = torch.zeros(logit.shape, requires_grad=False, device='cuda', dtype=torch.float)
            weight[:, 0:1000] = 1
            MSE_loss = torch.mean(weight * ((logit - labels) ** 2))
        BDY_loss = 0
        if G is not None:
            if self.flags.data_set != 'ballistics':                 # For non-ballisitcs dataset
                X_range, X_lower_bound, X_upper_bound = self.get_boundary_lower_bound_uper_bound()
                X_mean = (X_lower_bound + X_upper_bound) / 2        # Get the mean
            else:                                                   # For ballistics dataset
                X_mean = [0, 1.5, np.radians(40.5), 18]
                X_range = [1.5, 1.5, 1.1, 32]
            relu = torch.nn.ReLU()
            BDY_loss_all = 10 * relu(torch.abs(G - self.build_tensor(X_mean)) - 0.5 * self.build_tensor(X_range))
            BDY_loss = torch.mean(BDY_loss_all)
        self.MSE_loss = MSE_loss
        self.Boundary_loss = BDY_loss
        return torch.add(MSE_loss, BDY_loss)
        #else:                           # This is cross entropy loss where data is categorical
        #    criterion = nn.CrossEntropyLoss()
        #    return criterion(logit, labels.long())


    def build_tensor(self, nparray, requires_grad=False):
        return torch.tensor(nparray, requires_grad=requires_grad, device='cuda', dtype=torch.float)


    def make_optimizer(self):
        """
        Make the corresponding optimizer from the flags. Only below optimizers are allowed. Welcome to add more
        :return:
        """
        if self.flags.optim == 'Adam':
            op = torch.optim.Adam(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'RMSprop':
            op = torch.optim.RMSprop(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'SGD':
            op = torch.optim.SGD(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        else:
            raise Exception("Your Optimizer is neither Adam, RMSprop or SGD, please change in param or contact Ben")
        return op

    def make_lr_scheduler(self, optm):
        """
        Make the learning rate scheduler as instructed. More modes can be added to this, current supported ones:
        1. ReduceLROnPlateau (decrease lr when validation error stops improving
        :return:
        """
        return lr_scheduler.ReduceLROnPlateau(optimizer=optm, mode='min',
                                              factor=self.flags.lr_decay_rate,
                                              patience=10, verbose=True, threshold=1e-4)

    def save(self):
        """
        Saving the model to the current check point folder with name best_model_forward.pt
        :return: None
        """
        # torch.save(self.model.state_dict, os.path.join(self.ckpt_dir, 'best_model_state_dict.pt'))
        torch.save(self.model, os.path.join(self.ckpt_dir, 'best_model_forward.pt'))

    def load(self):
        """
        Loading the model from the check point folder with name best_model_forward.pt
        :return:
        """
        # self.model.load_state_dict(torch.load(os.path.join(self.ckpt_dir, 'best_model_state_dict.pt')))
        if torch.cuda.is_available():
            self.model = torch.load(os.path.join(self.ckpt_dir, 'best_model_forward.pt'))
        else:
            self.model = torch.load(os.path.join(self.ckpt_dir, 'best_model_forward.pt'), map_location=torch.device('cpu'))

    def train(self):
        """
        The major training function. This would start the training using information given in the flags
        :return: None
        """


        cuda = True if torch.cuda.is_available() else False

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 2:
            print("Using", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)

        if cuda:
            self.model.cuda()

        # Construct optimizer after the model moved to GPU
        self.optm = self.make_optimizer()
        self.lr_scheduler = self.make_lr_scheduler(self.optm)

        # Time keeping
        tk = time_keeper(time_keeping_file=os.path.join(self.ckpt_dir, 'training time.txt'))

        for epoch in range(self.flags.train_step):
            # Set to Training Mode
            train_loss = 0
            # boundary_loss = 0                 # Unnecessary during training since we provide geometries
            self.model.train()
            for j, (geometry, spectra) in enumerate(self.train_loader):
                if self.flags.data_set == 'gaussian_mixture':
                    spectra = torch.nn.functional.one_hot(spectra.to(torch.int64), 4).to(torch.float) # Change the gaussian labels into one-hot
                if cuda:
                    geometry = geometry.cuda()                       # Put data onto GPU
                    spectra = spectra.cuda()                            # Put data onto GPU
                self.optm.zero_grad()                               # Zero the gradient first
                logit = self.model(geometry)                        # Get the output
                loss = self.make_loss(logit, spectra)               # Get the loss tensor
                loss.backward()                                     # Calculate the backward gradients
                self.optm.step()                                    # Move one step the optimizer
                train_loss += loss                                  # Aggregate the loss
                # boundary_loss += self.Boundary_loss                 # Aggregate the BDY loss

            # Calculate the avg loss of training
            train_avg_loss = train_loss.cpu().data.numpy() / (j + 1)
            # boundary_avg_loss = boundary_loss.cpu().data.numpy() / (j + 1)

            if epoch % self.flags.eval_step == 0:                      # For eval steps, do the evaluations and tensor board
                # Record the training loss to the tensorboard
                self.log.add_scalar('Loss/train', train_avg_loss, epoch)
                # self.log.add_scalar('Loss/BDY_train', boundary_avg_loss, epoch)

                # Set to Evaluation Mode
                self.model.eval()
                print("Doing Evaluation on the model now")
                test_loss = []
                for j, (geometry, spectra) in enumerate(self.test_loader):  # Loop through the eval set
                    if self.flags.data_set == 'gaussian_mixture':
                        spectra = torch.nn.functional.one_hot(spectra.to(torch.int64), 4).to(torch.float) # Change the gaussian labels into one-hot
                    if cuda:
                        geometry = geometry.cuda()
                        spectra = spectra.cuda()
                    logit = self.model(geometry)
                    loss = self.make_loss(logit, spectra)                   # compute the loss
                    #test_loss += loss                                       # Aggregate the loss
                    test_loss.append(np.copy(loss.cpu().data.numpy()))

                # Record the testing loss to the tensorboard
                #test_avg_loss = test_loss.cpu().data.numpy() / (j+1)
                test_avg_loss = np.mean(test_loss)
                self.log.add_scalar('Loss/test', test_avg_loss, epoch)

                print("This is Epoch %d, training loss %.5f, validation loss %.5f" \
                      % (epoch, train_avg_loss, test_avg_loss ))
                # Plotting the first spectra prediction for validation
                # f = self.compare_spectra(Ypred=logit[0,:].cpu().data.numpy(), Ytruth=spectra[0,:].cpu().data.numpy())
                # self.log.add_figure(tag='spectra compare',figure=f,global_step=epoch)

                # Model improving, save the model down
                if test_avg_loss < self.best_validation_loss:
                    self.best_validation_loss = test_avg_loss
                    self.save()
                    print("Saving the model down...")

                    if self.best_validation_loss < self.flags.stop_threshold:
                        print("Training finished EARLIER at epoch %d, reaching loss of %.5f" %\
                              (epoch, self.best_validation_loss))
                        break

            # Learning rate decay upon plateau
            self.lr_scheduler.step(train_avg_loss)
        self.log.close()
        tk.record(1)                    # Record at the end of the training

    def evaluate(self, save_dir='data/', save_all=False, MSE_Simulator=False, save_misc=False, save_Simulator_Ypred=False):
        self.load()                             # load the model as constructed
        try:
            bs = self.flags.backprop_step         # for previous code that did not incorporate this
        except AttributeError:
            print("There is no attribute backprop_step, catched error and adding this now")
            self.flags.backprop_step = 300
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()
        self.model.eval()
        saved_model_str = self.saved_model.replace('/','_')
        # Get the file names
        Ypred_file = os.path.join(save_dir, 'test_Ypred_{}.csv'.format(saved_model_str))
        Xtruth_file = os.path.join(save_dir, 'test_Xtruth_{}.csv'.format(saved_model_str))
        Ytruth_file = os.path.join(save_dir, 'test_Ytruth_{}.csv'.format(saved_model_str))
        Xpred_file = os.path.join(save_dir, 'test_Xpred_{}.csv'.format(saved_model_str))
        print("evalution output pattern:", Ypred_file)

        # Time keeping
        tk = time_keeper(time_keeping_file=os.path.join(save_dir, 'evaluation_time.txt'))

        # Open those files to append
        with open(Xtruth_file, 'a') as fxt,open(Ytruth_file, 'a') as fyt,\
                open(Ypred_file, 'a') as fyp, open(Xpred_file, 'a') as fxp:
            # Loop through the eval data and evaluate
            for ind, (geometry, spectra) in enumerate(self.test_loader):
                if self.flags.data_set == 'gaussian_mixture':
                    spectra = torch.nn.functional.one_hot(spectra.to(torch.int64), 4).to(torch.float) # Change the gaussian labels into one-hot
                if cuda:
                    geometry = geometry.cuda()
                    spectra = spectra.cuda()
                # Initialize the geometry first
                Xpred, Ypred, loss = self.evaluate_one(spectra, save_dir=save_dir, save_all=save_all, ind=ind,
                                                        MSE_Simulator=MSE_Simulator, save_misc=save_misc, save_Simulator_Ypred=save_Simulator_Ypred)
                tk.record(ind)                          # Keep the time after each evaluation for backprop
                # self.plot_histogram(loss, ind)                                # Debugging purposes
                if save_misc:
                    np.savetxt('visualize_final/point{}_Xtruth.csv'.format(ind), geometry.cpu().data.numpy())
                    np.savetxt('visualize_final/point{}_Ytruth.csv'.format(ind), spectra.cpu().data.numpy())

                # suppress printing to evaluate time
                np.savetxt(fxt, geometry.cpu().data.numpy())
                np.savetxt(fyt, spectra.cpu().data.numpy())
                np.savetxt(fyp, Ypred)
                np.savetxt(fxp, Xpred)

        return Ypred_file, Ytruth_file

    def evaluate_one(self, target_spectra, save_dir='data/', MSE_Simulator=False ,save_all=False, ind=None, save_misc=False, save_Simulator_Ypred=False):
        """
        The function which being called during evaluation and evaluates one target y using # different trails
        :param target_spectra: The target spectra/y to backprop to 
        :param save_dir: The directory to save to when save_all flag is true
        :param MSE_Simulator: Use Simulator Loss to get the best instead of the default NN output logit
        :param save_all: The multi_evaluation where each trail is monitored (instad of the best) during backpropagation
        :param ind: The index of this target_spectra in the batch
        :param save_misc: The flag to print misc information for degbugging purposes, usually printed to best_mse
        :return: Xpred_best: The 1 single best Xpred corresponds to the best Ypred that is being backproped 
        :return: Ypred_best: The 1 singe best Ypred that is reached by backprop
        :return: MSE_list: The list of MSE at the last stage
        """

        # Initialize the geometry_eval or the initial guess xs
        geometry_eval = self.initialize_geometry_eval()
        # Set up the learning schedule and optimizer
        self.optm_eval = self.make_optimizer_eval(geometry_eval)
        self.lr_scheduler = self.make_lr_scheduler(self.optm_eval)
        
        # expand the target spectra to eval batch size
        target_spectra_expand = target_spectra.expand([self.flags.eval_batch_size, -1])
        
        # If saving misc files, initialize them
        if save_misc:
            Best_MSE_list = []
            Avg_MSE_list = []
            Xpred_best = None
            Best_MSE = 999
            # Define the best MSE list and place holder for Best Xpred Ypreds
            save_all_Best_MSE_list = np.ones([self.flags.eval_batch_size, 1]) * 999
            save_all_Xpred_best = np.zeros_like(geometry_eval.cpu().data.numpy())
            save_all_Ypred_best = None
            # Define the full loss matrix, real means simulator loss, fake means NN loss
            Full_loss_matrix_real = np.zeros([self.flags.eval_batch_size, self.flags.backprop_step])
            Full_loss_matrix_fake = np.zeros([self.flags.eval_batch_size, self.flags.backprop_step])
        
        # Begin Backprop
        for i in range(self.flags.backprop_step):
            # Make the initialization from [-1, 1], can only be in loop due to gradient calculator constraint
            geometry_eval_input = self.initialize_from_uniform_to_dataset_distrib(geometry_eval)
            if save_misc and ind == 0 and i == 0:                       # save the modified initial guess
                np.savetxt('geometry_initialization.csv',geometry_eval_input.cpu().data.numpy())
            self.optm_eval.zero_grad()                                  # Zero the gradient first
            logit = self.model(geometry_eval_input)                     # Get the output
            loss = self.make_loss(logit, target_spectra_expand, G=geometry_eval_input, W=True)         # Get the loss
            loss.backward()                                             # Calculate the Gradient

            if save_misc:
                ###################################
                # evaluate through simulator part #
                ###################################
                Ypred = simulator(self.flags.data_set, geometry_eval_input.cpu().data.numpy())
                if len(np.shape(Ypred)) == 1:                           # If this is the ballistics dataset where it only has 1d y'
                    Ypred = np.reshape(Ypred, [-1, 1])
                # Get the MSE list of these
                MSE_list = np.mean(np.square(Ypred - target_spectra_expand.cpu().data.numpy()), axis=1)
                # Get the best and the index of it
                best_MSE_in_batch = np.min(MSE_list)
                avg_MSE_in_batch = np.mean(MSE_list)
                Best_MSE_list.append(best_MSE_in_batch)
                Avg_MSE_list.append(avg_MSE_in_batch)
                best_estimate_index = np.argmin(MSE_list)
                if best_MSE_in_batch < Best_MSE:
                    # Update the best one
                    Best_MSE = best_MSE_in_batch
                    # Get the best Xpred
                    Xpred_best = np.reshape(np.copy(geometry_eval_input.cpu().data.numpy()[best_estimate_index, :]), [1, -1])
                    Ypred_best = np.reshape(np.copy(Ypred[best_estimate_index, :]), [1, -1])

                # record the full loss matrix
                Full_loss_matrix_real[:, i] = np.squeeze(MSE_list)
                Real_MSE_list = np.mean(np.square(logit.cpu().data.numpy() - target_spectra_expand.cpu().data.numpy()), axis=1)
                Full_loss_matrix_fake[:, i] = np.copy(Real_MSE_list)
                
                """
            if save_all and  chose_middle_value:
                    save_all_Ypred_best = Ypred
                # Record the trails that gets better
                better_index = save_all_Best_MSE_list > MSE_list
                # Update those MSE that is better now
                save_all_Best_MSE_list = np.where(better_index, MSE_list, save_all_Best_MSE_list)
                save_all_Xpred_best = np.where(better_index, geometry_eval_input.cpu().data.numpy(), save_all_Xpred_best)
                save_all_Ypred_best = np.where(better_index, Ypred, save_all_Ypred_best)
                """
            # update weights and learning rate scheduler
            if i != self.flags.backprop_step - 1:
                self.optm_eval.step()  # Move one step the optimizer
                self.lr_scheduler.step(loss.data)

        # Save the Best_MSE list for first few to sample
        if save_misc and ind < 20:
            np.savetxt('best_mse/best_mse_list{}.csv'.format(ind), Best_MSE_list)
            np.savetxt('best_mse/avg_mse_list{}.csv'.format(ind), Avg_MSE_list)
            #np.savetxt('best_mse/full_loss_mat_real{}.csv'.format(ind), Full_loss_matrix_real)
            #np.savetxt('best_mse/full_loss_mat_fake{}.csv'.format(ind), Full_loss_matrix_fake)

        
        if save_all:
            #######################################################
            # Choose the top 1,000 points from Backprop solutions #
            #######################################################
            mse_loss = np.reshape(np.sum(np.square(logit.cpu().data.numpy() - target_spectra_expand.cpu().data.numpy()), axis=1), [-1, 1])
            #print("shape of mse_loss", np.shape(mse_loss))
            mse_loss = np.concatenate((mse_loss, np.reshape(np.arange(self.flags.eval_batch_size), [-1, 1])), axis=1)
            #print("shape of mse_loss", np.shape(mse_loss))
            loss_sort = mse_loss[mse_loss[:, 0].argsort(kind='mergesort')]                         # Sort the loss list
            #print("shape of loss_sort is:", np.shape(loss_sort))
            #print("print loss_srt", loss_sort)
            #print(loss_sort)
            exclude_top = 0
            trail_nums = 1000
            good_index = loss_sort[exclude_top:trail_nums+exclude_top, 1].astype('int')                        # Get the indexs
            #print("good index", good_index)
            saved_model_str = self.saved_model.replace('/', '_') + 'inference' + str(ind)
            Ypred_file = os.path.join(save_dir, 'test_Ypred_point{}.csv'.format(saved_model_str))
            Xpred_file = os.path.join(save_dir, 'test_Xpred_point{}.csv'.format(saved_model_str))
            if self.flags.data_set != 'meta_material':
                # 2 options: simulator/logit
                Ypred = simulator(self.flags.data_set, geometry_eval_input.cpu().data.numpy())
                #print("shape of Ypred is", np.shape(Ypred))
                #print("shape of good index is", np.shape(good_index))
                if not save_Simulator_Ypred:            # The default is the simulator Ypred output
                    Ypred = logit.cpu().data.numpy()
                if len(np.shape(Ypred)) == 1:           # If this is the ballistics dataset where it only has 1d y'
                    Ypred = np.reshape(Ypred, [-1, 1])
                with open(Xpred_file, 'a') as fxp, open(Ypred_file, 'a') as fyp:
                    np.savetxt(fyp, Ypred[good_index, :])
                    np.savetxt(fxp, geometry_eval_input.cpu().data.numpy()[good_index, :])
            else:                       # This is meta-meterial dataset, handle with special
                with open(Xpred_file, 'a') as fxp:
                    np.savetxt(fxp, geometry_eval_input.cpu().data.numpy()[good_index, :])
                
       
        #############################
        # After BP, choose the best #
        #############################
        print("Your MSE_Simulator status is :", MSE_Simulator)
        if MSE_Simulator:                               # If we are using Simulator as Ypred standard
            Ypred = simulator(self.flags.data_set, geometry_eval_input.cpu().data.numpy())
        else:
            Ypred = logit.cpu().data.numpy()

        if len(np.shape(Ypred)) == 1:           # If this is the ballistics dataset where it only has 1d y'
            Ypred = np.reshape(Ypred, [-1, 1])
        
        # calculate the MSE list and get the best one
        MSE_list = np.mean(np.square(Ypred - target_spectra_expand.cpu().data.numpy()), axis=1)
        best_estimate_index = np.argmin(MSE_list)
        Xpred_best = np.reshape(np.copy(geometry_eval_input.cpu().data.numpy()[best_estimate_index, :]), [1, -1])
        if save_Simulator_Ypred:
            Ypred = simulator(self.flags.data_set, geometry_eval_input.cpu().data.numpy())
            if len(np.shape(Ypred)) == 1:           # If this is the ballistics dataset where it only has 1d y'
                Ypred = np.reshape(Ypred, [-1, 1])
        Ypred_best = np.reshape(np.copy(Ypred[best_estimate_index, :]), [1, -1])

        ######################
        # Test code on 04.23 #
        ######################
        """
        As one of the attempts to make Backprop better (or maybe as before), this saves all the Xpred and Ypred made 
        after the backpropagation and then try to visualize them afterwards
        """
        if save_misc:
            # Save the Xpred, Ypred, Ypred_Simulator, Ytruth, Xtruth
            np.savetxt('visualize_final/point{}_Xpred.csv'.format(ind), geometry_eval_input.cpu().data.numpy())
            Ypred = logit.cpu().data.numpy()
            if len(np.shape(Ypred)) == 1:           # If this is the ballistics dataset where it only has 1d y'
                Ypred = np.reshape(Ypred, [-1, 1])
            np.savetxt('visualize_final/point{}_Ypred.csv'.format(ind), Ypred)
            Ypred_Simulator = simulator(self.flags.data_set, geometry_eval_input.cpu().data.numpy())
            if len(np.shape(Ypred_Simulator)) == 1:           # If this is the ballistics dataset where it only has 1d y'
                Ypred = np.reshape(Ypred_Simulator, [-1, 1])
            np.savetxt('visualize_final/point{}_Ypred_Simulator.csv'.format(ind), Ypred_Simulator)

        return Xpred_best, Ypred_best, MSE_list


    def initialize_geometry_eval(self):
        """
        Initialize the geometry eval according to different dataset
        :return: The initialized geometry eval
        """
        if self.flags.data_set == 'ballistics':
            bs = self.flags.eval_batch_size
            numpy_geometry = np.zeros([bs, self.flags.linear[0]])
            numpy_geometry[:, 0] = np.random.normal(0, 0.25, size=[bs,])
            numpy_geometry[:, 1] = np.random.normal(1.5, 0.25, size=[bs,])
            numpy_geometry[:, 2] = np.radians(np.random.uniform(9, 72, size=[bs,]))
            numpy_geometry[:, 3] = np.random.poisson(15, size=[bs,])
            geomtry_eval = self.build_tensor(numpy_geometry, requires_grad=True)
        elif self.flags.data_set == 'robotic_arm':
            bs = self.flags.eval_batch_size
            numpy_geometry = np.random.normal(0, 0.5, size=[bs, 4])
            numpy_geometry[:, 0] /= 2
            geomtry_eval = self.build_tensor(numpy_geometry, requires_grad=True)
            print("robotic_arm specific initialization")
        else:
            geomtry_eval = torch.rand([self.flags.eval_batch_size, self.flags.linear[0]], requires_grad=True, device='cuda')
        #geomtry_eval = torch.randn([self.flags.eval_batch_size, self.flags.linear[0]], requires_grad=True, device='cuda')
        return geomtry_eval

    def initialize_from_uniform_to_dataset_distrib(self, geometry_eval):
        """
        since the initialization of the backprop is uniform from [0,1], this function transforms that distribution
        to suitable prior distribution for each dataset. The numbers are accquired from statistics of min and max
        of the X prior given in the training set and data generation process
        :param geometry_eval: The input uniform distribution from [0,1]
        :return: The transformed initial guess from prior distribution
        """
        X_range, X_lower_bound, X_upper_bound = self.get_boundary_lower_bound_uper_bound()
        geometry_eval_input = geometry_eval * self.build_tensor(X_range) + self.build_tensor(X_lower_bound)
        if self.flags.data_set == 'robotic_arm':
            return geometry_eval
        return geometry_eval_input
        #return geometry_eval

    
    def get_boundary_lower_bound_uper_bound(self):
        if self.flags.data_set == 'sine_wave': 
            return np.array([2, 2]), np.array([-1, -1]), np.array([1, 1])
        elif self.flags.data_set == 'meta_material':
            return np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), np.array([-1,-1,-1,-1,-1,-1,-1,-1, -1,-1, -1,-1, -1,-1]), np.array([1,1,1,1, 1,1,1,1, 1,1,1,1, 1, 1])
        elif self.flags.data_set == 'ballistics':
            return np.array([1, 1, 1, 1]), np.array([0, 0, 0, 0]), None
        elif self.flags.data_set == 'robotic_arm':
            return np.array([1.5, 3.0, 3.0, 3.0]), np.array([-0.75, -1.5, -1.5, -1.5]), np.array([0.75, 1.5, 1.5, 1.5])
        else:
            sys.exit("In Backprop, during initialization from uniform to dataset distrib: Your data_set entry is not correct, check again!")


    def predict(self, Xpred_file, no_save=False):
        """
        The prediction function, takes Xpred file and write Ypred file using trained model
        :param Xpred_file: Xpred file by (usually VAE) for meta-material
        :param no_save: do not save the txt file but return the np array
        :return: pred_file, truth_file to compare
        """
        self.load()         # load the model
        Ypred_file = Xpred_file.replace('Xpred', 'Ypred')
        Ytruth_file = Ypred_file.replace('Ypred', 'Ytruth')
        Xpred = pd.read_csv(Xpred_file, header=None, delimiter=',')     # Read the input
        Xpred.info()
        Xpred_tensor = torch.from_numpy(Xpred.values).to(torch.float)

        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()
            Xpred_tensor = Xpred_tensor.cuda()
        Ypred = self.model(Xpred_tensor)
        if self.flags.model_name is not None:
                Ypred_file = Ypred_file.replace('Ypred', 'Ypred' + self.flags.model_name)
        if no_save:                             # If instructed dont save the file and return the array
             return Ypred.cpu().data.numpy(), Ytruth_file
        np.savetxt(Ypred_file, Ypred.cpu().data.numpy())

        return Ypred_file, Ytruth_file



    def plot_histogram(self, loss, ind):
        """
        Plot the loss histogram to see the loss distribution
        """
        f = plt.figure()
        plt.hist(loss, bins=100)
        plt.xlabel('MSE loss')
        plt.ylabel('cnt')
        plt.suptitle('(Avg MSE={:4e})'.format(np.mean(loss)))
        plt.savefig(os.path.join('data','loss{}.png'.format(ind)))
        return None
        
    def compare_spectra(self, Ypred, Ytruth, T=None, title=None, figsize=[15, 5],
                        T_num=10, E1=None, E2=None, N=None, K=None, eps_inf=None):
        """
        Function to plot the comparison for predicted spectra and truth spectra
        :param Ypred:  Predicted spectra, this should be a list of number of dimension 300, numpy
        :param Ytruth:  Truth spectra, this should be a list of number of dimension 300, numpy
        :param title: The title of the plot, usually it comes with the time
        :param figsize: The figure size of the plot
        :return: The identifier of the figure
        """
        # Make the frequency into real frequency in THz
        fre_low = 0.8
        fre_high = 1.5
        frequency = fre_low + (fre_high - fre_low) / len(Ytruth) * np.arange(300)
        f = plt.figure(figsize=figsize)
        plt.plot(frequency, Ypred, label='Pred')
        plt.plot(frequency, Ytruth, label='Truth')
        if T is not None:
            plt.plot(frequency, T, linewidth=1, linestyle='--')
        if E2 is not None:
            for i in range(np.shape(E2)[0]):
                plt.plot(frequency, E2[i, :], linewidth=1, linestyle=':', label="E2" + str(i))
        if E1 is not None:
            for i in range(np.shape(E1)[0]):
                plt.plot(frequency, E1[i, :], linewidth=1, linestyle='-', label="E1" + str(i))
        if N is not None:
            plt.plot(frequency, N, linewidth=1, linestyle=':', label="N")
        if K is not None:
            plt.plot(frequency, K, linewidth=1, linestyle='-', label="K")
        if eps_inf is not None:
            plt.plot(frequency, np.ones(np.shape(frequency)) * eps_inf, label="eps_inf")
        # plt.ylim([0, 1])
        plt.legend()
        #plt.xlim([fre_low, fre_high])
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Transmittance")
        if title is not None:
            plt.title(title)
        return f


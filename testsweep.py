"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion.
"""
import train
import  numpy as np
import flag_reader

if __name__ == '__main__':
    flags = flag_reader.read_flag()  	#setting the base case
    lr_list = [5e-5, 8e-5, 1e-4, 2e-4, 5e-4]
    for lr in lr_list:
        flags.lr = lr
        for j in range(1):
            flags.model_name = flags.data_set + "_lr_"+ str(flags.lr)
            train.training_from_flag(flags)
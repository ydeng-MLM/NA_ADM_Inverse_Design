"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion.
"""
import train
import  numpy as np
import flag_reader

if __name__ == '__main__':
    flags = flag_reader.read_flag()  	#setting the base case
    linear = [linear_unit for j in range(i)]        #Set the linear units
    linear[0] = 2                   # The start of linear
    linear[-1] = 1                # The end of linear
    flags.linear = linear
    reg_scale = reg_scale_list = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    for reg_scale in reg_scale_list:
        flags.reg_scale = reg_scale
        for j in range(1):
            flags.model_name = flags.data_set + "reg"+ str(flags.reg_scale)
             train.training_from_flag(flags)
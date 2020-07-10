"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion.
"""
import train
#os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import  numpy as np
import flag_reader
if __name__ == '__main__':
    linear_unit_list = [100,  500,  1000]
    #linear_unit_list = [1000]
    #linear_unit_list = [1000, 500, 300, 150]
    #reg_scale_list = [3e-5, 4e-5, 5e-5]
    reg_scale_list = [2e-5]
    #reg_scale_list = [5e-4]
    for linear_unit in linear_unit_list:
        # Setting the loop for setting the parameter
        for i in range(6, 10):
            flags = flag_reader.read_flag()  	#setting the base case
            linear = [linear_unit for j in range(i)]        #Set the linear units
            linear[0] = 2                   # The start of linear
            linear[-1] = 1                # The end of linear
            flags.linear = linear
            for reg_scale in reg_scale_list:
                flags.reg_scale = reg_scale
                for j in range(1):
                        flags.model_name = flags.data_set + "reg"+ str(flags.reg_scale) + "trail_"+str(j) + "_forward_swipe" +  str(i)
                        train.training_from_flag(flags)


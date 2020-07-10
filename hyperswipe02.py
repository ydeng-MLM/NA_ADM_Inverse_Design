"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion.
"""
import train
#os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import  numpy as np
import flag_reader
if __name__ == '__main__':
    #linear_unit_list = [ 50, 100, 200, 500]
    #linear_unit_list = [1000, 500]
    conv_kernel_size_first_list = [2, 4, 8, 16, 32]
    conv_kernel_size_second_list = [3, 5, 9, 15, 33]
    #linear_unit_list = [1000, 500, 300, 150, 50]
    # reg_scale_list = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    reg_scale_list = [5e-4]
    #for linear_unit in linear_unit_list:
    for kernel_first in conv_kernel_size_first_list:
        for kernel_second in conv_kernel_size_second_list:
            # Setting the loop for setting the parameter
            #for i in range(3, 10):
            flags = flag_reader.read_flag()  	#setting the base case
            #linear = [linear_unit for j in range(i)]        #Set the linear units
            #linear[0] = 4                   # The start of linear
            #linear[-1] = 1                # The end of linear
            #flags.linear = linear
            flags.conv_kernel_size[0] = kernel_first
            flags.conv_kernel_size[1] = kernel_second
            flags.conv_kernel_size[2] = kernel_second
            for reg_scale in reg_scale_list:
                flags.reg_scale = reg_scale
                for j in range(2,5):
                        flags.model_name = flags.data_set + "reg"+ str(flags.reg_scale) + "trail_"+str(j) + "_conv_kernel_swipe[" + str(kernel_first)+ "," + str(kernel_second) + "," + str(kernel_second) + "]"
                        train.training_from_flag(flags)


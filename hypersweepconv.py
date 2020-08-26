"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion for conv layers.
"""
import train
#os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import  numpy as np
import flag_reader
if __name__ == '__main__':
    # Setting the loop for setting the parameter
    for i in range(28, 54, 20):
        for j in range(3, 7, 2):
            flags = flag_reader.read_flag()  	#setting the base case
            conv_kernel_size = [i, i, j, j]
            flags.conv_kernel_size = conv_kernel_size
            for k in range(1, 2):
                flags.model_name = "trial_"+str(k)+"_convsize_"+str(i)+"_"+str(j)+"_"+str(j)
                train.training_from_flag(flags)
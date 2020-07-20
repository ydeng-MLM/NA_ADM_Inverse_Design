"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion.
"""
import evaluate
#os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import  numpy as np
import flag_reader
if __name__ == '__main__':
    for linear_unit in linear_unit_list:
        # Setting the loop for setting the parameter
        for i in range(6, 10):
            flags.model_name = flags.data_set + "reg"+ str(flags.reg_scale) + "trail_"+str(j) + "_forward_swipe" +  str(i)
            eval_flags = flag_reader.read_flag()
            evaluate.evaluate_from_model(eval_flags.eval_model, save_misc=False, multi_flag=False, save_Simulator_Ypred=False, MSE_Simulator=False)
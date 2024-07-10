import numpy as np
def shout_dif(array1, array2):
    
    meandiff = np.mean(np.abs(array1-array2))
    
    print(f"The average difference between observations is {meandiff}")
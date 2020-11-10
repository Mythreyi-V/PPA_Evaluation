#from google.colab import drive
#drive.mount('/content/drive')
import sys
import os
#PATH = '/content/drive/My Drive/PPM_Stability/'
#PATH = "C:/Users/velmurug/Documents/Stability Experiments/benchmark_interpretability/PPM_Stability/"
#PATH = "PPM_Evaluation\Stability-Experiments\\benchmark_interpretability\PPM_Stability\\"
PATH = "/home/n9455647/PPM_Evaluation/Stability-Experiments/benchmark_interpretability/PPM_Stability/"
#PATH = "C:/Users/Mythreyi/Documents/GitHub/Stability-Experiments/benchmark_interpretability/PPM_Stability/"
sys.path.append(PATH)
#print(os.getcwd())
#print(os.listdir(os.path.join(PATH)))

import stability as st

lst = [[0, 1, 0, 1],[1,0,1,0]]

stab = st.getStability(lst)

print(stab)

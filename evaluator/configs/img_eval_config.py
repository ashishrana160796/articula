import sys
import os

current_directory = os.path.dirname(os.path.realpath(__file__))

sys.path.append(current_directory + '/../../common')

from img_utils import mkdir


# root to the testsets
dataroot = '../data'

# list of synthesis algorithms
vals = ['subset']
# 'progan', 'stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan',
#         'stylegan2', 'whichfaceisreal',
#         'ADM','Glide','Midjourney','stable_diffusion_v_1_4','stable_diffusion_v_1_5','VQDM','wukong',
'''
python image_evaluator.py --model_path ./weights/{}.pth --detect_method {CNNSpot,Gram,Fusing,FreDect,LGrad,LNP,DIRE}  --noise_type {blur,jpg,resize}
'''
import os
import csv
import torch
import sys

from configs.img_validate import validate,validate_single
from img_options import TestOptions

current_directory = os.path.dirname(os.path.realpath(__file__))

sys.path.append(current_directory)
sys.path.append(current_directory + '/configs')
sys.path.append('../common')

from configs.img_eval_config import *
from PIL import ImageFile
from img_utils import create_argparser,get_model, set_random_seed

ImageFile.LOAD_TRUNCATED_IMAGES = True

set_random_seed()

opt = TestOptions().parse(print_options=True) 

for i, model_path in enumerate(model_paths): 
    model_name = os.path.basename(model_path).replace('.pth', '')
    opt.detect_method = detect_methods[i]
    opt.noise_type = noise_types[i]
    results_dir=f"../../results/{opt.detect_method}"
    mkdir(results_dir)
    
    rows = [["{} model testing on...".format(model_name)],
            ['testset', 'accuracy', 'avg precision']]
    
    print("{} model testing on...".format(model_name))
    for v_id, val in enumerate(vals):
        opt.dataroot = '{}/{}'.format(dataroot, val)
    
        # model = resnet50(num_classes=1)
        model = get_model(opt)
        state_dict = torch.load(model_path, map_location='cpu')
        try:
            if opt.detect_method in ["FreDect","Gram"]:
                model.load_state_dict(state_dict['netC'],strict=True)
            elif opt.detect_method == "UnivFD":
                model.fc.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict['model'],strict=True)
        except:
            print("[ERROR] model.load_state_dict() error")
        model.cuda()
        model.eval()
    
        opt.process_device=torch.device("cpu")
        acc, ap, _, _, _, _ = validate(model, opt)
        rows.append([val, acc, ap])
        print("({}) acc: {}; ap: {}".format(val, acc, ap))
    
    csv_name = results_dir + '/{}_{}.csv'.format(opt.detect_method,opt.noise_type)
    with open(csv_name, 'a+') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerows(rows)

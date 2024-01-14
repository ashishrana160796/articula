import os
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score
from PIL import ImageFile
import torch.nn.functional as F 

current_directory = os.path.dirname(os.path.realpath(__file__))

sys.path.append(current_directory)
sys.path.append(current_directory + '/../evaluator/configs')
sys.path.append(current_directory + '/../models')
sys.path.append(current_directory + '/../common')
sys.path.append(current_directory + '/../data')

from img_options import TestOptions
from img_eval_config import *
from img_validate import validate
from data import create_dataloader_new
from img_utils import mkdir, create_argparser, get_model, set_random_seed
from process import get_processing_model

ImageFile.LOAD_TRUNCATED_IMAGES = True

opt = TestOptions().parse(print_options=True)

# Define a simple neural network for meta-model training
# class MetaModel(nn.Module):
#     def __init__(self, input_size):
#         super(MetaModel, self).__init__()
#         self.fc = nn.Linear(input_size, 1)

#     def forward(self, x):
#         return torch.sigmoid(self.fc(x))

class MetaModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, dropout_rate):
        super(MetaModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(hidden_size2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return torch.sigmoid(self.output_layer(x))


def load_and_test_model(model_path, detect_method, noise_type, val, dataroot):
    opt.detect_method = detect_method
    opt.noise_type = noise_type

    print("{} model testing on...".format(model_path))
    opt.dataroot = '{}/{}'.format(dataroot, val)

    model = get_model(opt)
    state_dict = torch.load(model_path, map_location='cpu')
    try:
        if detect_method in ["FreDect", "Gram"]:
            model.load_state_dict(state_dict['netC'], strict=True)
        elif detect_method == "UnivFD":
            model.fc.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict['model'], strict=True)
    except:
        print("[ERROR] model.load_state_dict() error")
    model.cuda()
    model.eval()

    opt.process_device = torch.device("cuda")
    _, _, _, _, y_true, y_pred = validate(model, opt)
    
    return y_pred, y_true


def main():

    
    # Set random seed
    set_random_seed()
    
    # Ensemble configuration
    model_paths = ['../weights/classifier/CNNSpot.pth', '../weights/classifier/Gram.pth', '../weights/classifier/PSM.pth', '../weights/classifier/DCTAnalysis.pth']
    detect_methods = ['CNNSpot', 'Gram', 'Fusing', 'FreDect']
    noise_types = ['jpg', 'jpg', 'jpg', 'jpg']
    
    # Running tests
    results_dir = "../results/ensemble"
    mkdir(results_dir)
    
    vals = ['subset']
    
    rows = [["Ensemble model testing on..."],
            ['testset', 'accuracy', 'avg precision']]
    
    # Testing and storing individual model results
    ensemble_predictions = []
    
    for i, model_path in enumerate(model_paths):
        model_name = os.path.basename(model_path).replace('.pth', '')
        detect_method = detect_methods[i]
        noise_type = noise_types[i]
    
        rows[0].append(f"{model_name}_{detect_method}_{noise_type}")
    
        for v_id, val in enumerate(vals):
            y_pred, y_true = load_and_test_model(model_path, detect_method, noise_type, val, dataroot)
            rows.append([val, accuracy_score(y_true, y_pred > 0.5), average_precision_score(y_true, y_pred)])
            ensemble_predictions.append(y_pred)
    
    # Training the meta-model
    ensemble_predictions = np.array(ensemble_predictions).T  # Transpose to have predictions in columns
    
    
    hidden_size1 = 64
    hidden_size2 = 32
    dropout_rate = 0.5
    
    meta_model = MetaModel(input_size=len(model_paths),
                                                      hidden_size1=hidden_size1,
                                                      hidden_size2=hidden_size2,
                                                      dropout_rate=dropout_rate)
    
    # meta_model = MetaModel(input_size=len(model_paths))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(meta_model.parameters(), lr=0.001)
    
    X_train = torch.from_numpy(ensemble_predictions).float()
    y_train = y_true.reshape(-1, 1)
    y_train = torch.from_numpy(y_train).float()
    
    for epoch in range(1000):
        optimizer.zero_grad()
        outputs = meta_model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    # Save the trained meta-model
    torch.save(meta_model.state_dict(), '../results/ensemble/meta_model.pth')
    
    # Get final ensemble predictions using the trained meta-model
    X_ensemble = torch.from_numpy(ensemble_predictions).float()
    final_ensemble_predictions = meta_model(X_ensemble).detach().numpy()
    
    # Evaluate ensemble accuracy and average precision
    ensemble_accuracy = accuracy_score(y_true, final_ensemble_predictions > 0.5)
    ensemble_avg_precision = average_precision_score(y_true, final_ensemble_predictions)
    
    print("Ensemble Accuracy: {}".format(ensemble_accuracy))
    print("Ensemble Average Precision: {}".format(ensemble_avg_precision))
    
    # Ensemble results
    ensemble_csv_name = results_dir + '/ensemble_results_1.csv'
    with open(ensemble_csv_name, 'a+') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerows(rows)


if __name__ == "__main__":
    main()
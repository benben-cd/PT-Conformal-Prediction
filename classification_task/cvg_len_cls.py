from importlib import reload
import utils
import conformal_procedure
reload(utils)
reload(conformal_procedure)
from utils import *
from conformal_procedure import *

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.backends.cudnn as cudnn
import random
import csv

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Normalization from torchvision repo
transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std= [0.229, 0.224, 0.225])
            ])

cudnn.benchmark = True
batch_size = 128

model_names = ['ResNet18', 'ResNet50', 'ResNet101', 'ResNet152', 'ResNeXt101', 'VGG16', 'ShuffleNet', 'Inception', 'DenseNet161']
model_name = 'ShuffleNet'
model = get_model(model_name)
model.eval()

# get data
num_calib = 10000
bias = 40
index_range = 300

cvg_list_VCP = []
size_list_VCP = []
cvg_list_PCP = []
size_list_PCP = []

for i in range(5):
    # Fix the random seed for reproducibility (you can change this, of course)
    seed=i
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    # Get the conformal calibcration dataset
    imagenet_calib_data, imagenet_val_data = torch.utils.data.random_split(torchvision.datasets.ImageFolder('./imagenet_val/', transform), [num_calib,50000-num_calib])

    # Initialize loaders
    calib_loader = torch.utils.data.DataLoader(imagenet_calib_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(imagenet_val_data, batch_size=batch_size, shuffle=True, pin_memory=True)

    # reset the seed to guarantee that we obtain same k and lamda
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    cmodel_VCP = ConformalModel(model, calib_loader, alpha=0.1, kreg=None, lamda=None, lamda_criterion='size',
                                CP_method='VCP', bias=bias, index=list(range(index_range)), allow_zero_sets=False)
    # reset the seed to guarantee that we obtain same k and lamda
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    cmodel_PCP = ConformalModel(model, calib_loader, alpha=0.1, kreg=None, lamda=None, lamda_criterion='size',
                                CP_method='PCP', PCP_prob=0.95, bias=bias, index=list(range(index_range)))

    top1_VCP, top5_VCP, coverage_VCP, size_VCP = validate(val_loader, cmodel_VCP, print_bool=True)
    cvg_list_VCP.append(coverage_VCP)
    size_list_VCP.append(size_VCP)
    top1_PCP, top5_PCP, coverage_PCP, size_PCP = validate(val_loader, cmodel_PCP, print_bool=True)
    cvg_list_PCP.append(coverage_PCP)
    size_list_PCP.append(size_PCP)

avg_cvg_VCP = np.mean(cvg_list_VCP)
std_cvg_VCP = np.std(cvg_list_VCP)
avg_size_VCP = np.mean(size_list_VCP)
std_size_VCP = np.std(size_list_VCP)
avg_cvg_PCP = np.mean(cvg_list_PCP)
std_cvg_PCP = np.std(cvg_list_PCP)
avg_size_PCP = np.mean(size_list_PCP)
std_size_PCP = np.std(size_list_PCP)

with open(f'{model_name}.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['avg_cvg_VCP', 'std_cvg_VCP', 'avg_size_VCP', 'std_size_VCP', 'avg_cvg_PCP', 'std_cvg_PCP', 'avg_size_PCP', 'std_size_PCP'])
    writer.writerow([avg_cvg_VCP, std_cvg_VCP, avg_size_VCP, std_size_VCP, avg_cvg_PCP, std_cvg_PCP, avg_size_PCP, std_size_PCP])

print(f'Results saved to {model_name}.csv')


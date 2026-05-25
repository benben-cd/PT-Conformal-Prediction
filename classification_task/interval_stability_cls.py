from importlib import reload
import utils
import conformal_procedure
reload(utils)
reload(conformal_procedure)
from utils import *
from conformal_procedure import *

# Import other standard packages
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
num_calib = 10000
bias = 40
index_range = 300

csv_file = 'std_results_cls.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Model', 'stability_RAPS', 'stability_RAPS_std', 'stability_PCP', 'stability_PCP_std'])

    model_names = ['ResNet18']  ##, 'ResNet50', 'ResNet101', 'ResNet152', 'ResNeXt101', 'VGG16', 'ShuffleNet', 'Inception', 'DenseNet161'

    for i in range(len(model_names)):
        # Load the model
        print('Loading model: {}'.format(model_names[i]))
        model_name = model_names[i]
        model = get_model(model_name)
        model.eval()

        interval_stability_VCP = np.zeros(5)
        interval_stability_PCP = np.zeros(5)

        for j in range(5):
            # Fix the random seed for reproducibility (you can change this, of course)
            seed = j
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



            m = 3
            size_array_VCP = np.zeros((m, 40000))
            size_array_PCP = np.zeros((m, 40000))
            for k in range(m):
                top1_VCP, top5_VCP, coverage_VCP, size_VCP, size_list_VCP = validate(val_loader, cmodel_VCP, print_bool=True)
                size_array_VCP[k] = np.array(size_list_VCP)
                top1_PCP, top5_PCP, coverage_PCP, size_PCP, size_list_PCP = validate(val_loader, cmodel_PCP, print_bool=True)
                size_array_PCP[k] = np.array(size_list_PCP)

            interval_stability_VCP[j] = np.mean(np.std(size_array_VCP, axis=0) / np.sqrt(m))
            interval_stability_PCP[j] = np.mean(np.std(size_array_PCP, axis=0) / np.sqrt(m))

        stability_RAPS = np.mean(interval_stability_VCP)
        stability_RAPS_std = np.std(interval_stability_VCP) / np.sqrt(5)
        stability_PCP = np.mean(interval_stability_PCP)
        stability_PCP_std = np.std(interval_stability_PCP) / np.sqrt(5)
        writer.writerow([model_name, stability_RAPS, stability_RAPS_std, stability_PCP, stability_PCP_std])
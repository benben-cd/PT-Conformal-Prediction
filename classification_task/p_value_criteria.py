import os
import csv
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from PIL import ImageFile

from utils import get_model
from conformal import ConformalModel, evaluate_p_value_criteria

ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_NAMES = [
	'ResNet18', 'ResNet50', 'ResNet101', 'ResNet152',
	'ResNeXt101', 'VGG16', 'ShuffleNet', 'Inception', 'DenseNet161'
]

METRICS = ['S', 'N', 'U', 'F', 'M', 'E', 'OU', 'OF', 'OM', 'OE']


def set_seed(seed):
	np.random.seed(seed=seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	random.seed(seed)


def resolve_dataset_dir(dataset_dir=None):
	if dataset_dir is not None:
		if not os.path.isdir(dataset_dir):
			raise FileNotFoundError(f'dataset_dir does not exist: {dataset_dir}')
		return dataset_dir

	script_dir = os.path.dirname(os.path.abspath(__file__))
	candidates = [
		os.path.join(script_dir, 'imagenet_val'),
		os.path.join(os.getcwd(), 'imagenet_val'),
	]
	for candidate in candidates:
		if os.path.isdir(candidate):
			return candidate

	raise FileNotFoundError(
		'imagenet_val not found. Checked: ' + ', '.join(candidates) +
		". Please pass dataset_dir='.../classification_task/imagenet_val'."
	)


def run_experiment(
    model_name='ShuffleNet',
    alpha=0.1,
    num_calib=10000,
    num_val=None,
    batch_size=128,
    num_seeds=5,
    pcp_prob=0.95,
    pt_bias=40,
    pt_index_range=300,
    smoothed_p_values=True,
    randomized_tau=False,
    max_eval_samples=None,
    dataset_dir=None,
    output_dir='./p_value_result',
    live_line_progress=True,
):
    dataset_dir = resolve_dataset_dir(dataset_dir)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    cudnn.benchmark = True

    model = get_model(model_name)
    model.eval()

    stats = {}
    for method in ['RAPS', 'PT_RAPS']:
        for metric in METRICS:
            stats[f'{method}_{metric}'] = []
    stats['RAPS_true_p'] = []
    stats['PT_RAPS_true_p'] = []

    print(f'\n### Model: {model_name}')
    print(f'Using dataset: {dataset_dir}')

    seed_iter = tqdm(range(num_seeds), desc=f'{model_name} seeds', leave=False)
    for seed in seed_iter:
        seed_iter.set_postfix({'seed': seed})
        set_seed(seed)

        dataset = torchvision.datasets.ImageFolder(dataset_dir, transform)
        imagenet_calib_data, imagenet_val_data = torch.utils.data.random_split(
            dataset,
            [num_calib, len(dataset) - num_calib],
        )
        if num_val is not None and num_val < len(imagenet_val_data):
            imagenet_val_data, _ = torch.utils.data.random_split(
                imagenet_val_data,
                [num_val, len(imagenet_val_data) - num_val],
            )

        calib_loader = torch.utils.data.DataLoader(
            imagenet_calib_data,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            imagenet_val_data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )

        # RAPS with the same misspecification as PT-RAPS for fair comparison.
        set_seed(seed)
        raps_model = ConformalModel(
            model,
            calib_loader,
            alpha=alpha,
            kreg=None,
            lamda=None,
            lamda_criterion='size',
            CP_method='VCP',
            allow_zero_sets=False,
            bias=pt_bias,
            index=list(range(pt_index_range)),
        )

        set_seed(seed)
        pt_raps_model = ConformalModel(
            model,
            calib_loader,
            alpha=alpha,
            kreg=None,
            lamda=None,
            lamda_criterion='size',
            CP_method='PCP',
            PCP_prob=pcp_prob,
            allow_zero_sets=False,
            bias=pt_bias,
            index=list(range(pt_index_range)),
        )

        raps_result = evaluate_p_value_criteria(
            raps_model,
            val_loader,
            smoothed=smoothed_p_values,
            randomized_tau=randomized_tau,
            max_samples=max_eval_samples,
            print_bool=live_line_progress,
            progress_prefix=f'{model_name} | seed {seed} | RAPS',
        )
        pt_raps_result = evaluate_p_value_criteria(
            pt_raps_model,
            val_loader,
            smoothed=smoothed_p_values,
            randomized_tau=randomized_tau,
            max_samples=max_eval_samples,
            print_bool=live_line_progress,
            progress_prefix=f'{model_name} | seed {seed} | PT',
        )

        print(
            f"RAPS  S={raps_result['S']:.4f} N={raps_result['N']:.4f} "
            f"OF={raps_result['OF']:.4f} OE={raps_result['OE']:.4f}"
        )
        print(
            f"PT    S={pt_raps_result['S']:.4f} N={pt_raps_result['N']:.4f} "
            f"OF={pt_raps_result['OF']:.4f} OE={pt_raps_result['OE']:.4f}"
        )
        seed_iter.set_postfix({
            'seed': seed,
            'RAPS_N': f"{raps_result['N']:.2f}",
            'PT_N': f"{pt_raps_result['N']:.2f}",
        })

        for metric in METRICS:
            stats[f'RAPS_{metric}'].append(raps_result[metric])
            stats[f'PT_RAPS_{metric}'].append(pt_raps_result[metric])
        stats['RAPS_true_p'].append(raps_result['mean_true_p'])
        stats['PT_RAPS_true_p'].append(pt_raps_result['mean_true_p'])

    summary = {
        'model_name': model_name,
        'alpha': alpha,
        'num_calib': num_calib,
        'num_val': num_val,
        'num_seeds': num_seeds,
        'pcp_prob': pcp_prob,
        'pt_bias': pt_bias,
        'pt_index_range': pt_index_range,
        'smoothed_p_values': smoothed_p_values,
        'randomized_tau': randomized_tau,
        'max_eval_samples': max_eval_samples,
        'live_line_progress': live_line_progress,
        'dataset_dir': dataset_dir,
    }
    for method in ['RAPS', 'PT_RAPS']:
        for metric in METRICS:
            key = f'{method}_{metric}'
            summary[f'avg_{key}'] = float(np.mean(stats[key]))
            summary[f'std_{key}'] = float(np.std(stats[key]))
    summary['avg_RAPS_true_p'] = float(np.mean(stats['RAPS_true_p']))
    summary['avg_PT_RAPS_true_p'] = float(np.mean(stats['PT_RAPS_true_p']))

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{model_name}_p_value_criteria.csv')
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(list(summary.keys()))
        writer.writerow(list(summary.values()))

    print(f'\nSaved model summary to: {output_path}')
    return summary


def run_all_models(
    model_names=None,
    alpha=0.1,
    num_calib=10000,
    num_val=None,
    batch_size=128,
    num_seeds=5,
    pcp_prob=0.95,
    pt_bias=40,
    pt_index_range=300,
    smoothed_p_values=True,
    randomized_tau=False,
    max_eval_samples=None,
    dataset_dir=None,
    output_root='./p_value_result_full',
    live_line_progress=True,
):
    if model_names is None:
        model_names = MODEL_NAMES

    all_summaries = []
    model_iter = tqdm(model_names, desc='All models', leave=True)
    for model_name in model_iter:
        model_iter.set_postfix({'model': model_name})
        summary = run_experiment(
            model_name=model_name,
            alpha=alpha,
            num_calib=num_calib,
            num_val=num_val,
            batch_size=batch_size,
            num_seeds=num_seeds,
            pcp_prob=pcp_prob,
            pt_bias=pt_bias,
            pt_index_range=pt_index_range,
            smoothed_p_values=smoothed_p_values,
            randomized_tau=randomized_tau,
            max_eval_samples=max_eval_samples,
            dataset_dir=dataset_dir,
            output_dir=output_root,
            live_line_progress=live_line_progress,
        )
        all_summaries.append(summary)
        model_iter.set_postfix({
            'model': model_name,
            'RAPS_N': f"{summary['avg_RAPS_N']:.2f}",
            'PT_N': f"{summary['avg_PT_RAPS_N']:.2f}",
        })

    summary_path = os.path.join(output_root, 'all_models_p_value_criteria_summary.csv')
    os.makedirs(output_root, exist_ok=True)
    with open(summary_path, mode='w', newline='') as file:
        fieldnames = list(all_summaries[0].keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_summaries)

    print(f'\nSaved all-model summary to: {summary_path}')
    return all_summaries


if __name__ == '__main__':
	# Full run for all models (server-ready).
	run_all_models(
		model_names=MODEL_NAMES,
		alpha=0.1,
		num_calib=10000,
		num_val=None,
		batch_size=128,
		num_seeds=5,
		pcp_prob=0.95,
		pt_bias=40,
		pt_index_range=300,
		smoothed_p_values=True,
		randomized_tau=False,
		max_eval_samples=None,
		dataset_dir=None,
		output_root='./p_value_result_full',
        live_line_progress=True,
	)

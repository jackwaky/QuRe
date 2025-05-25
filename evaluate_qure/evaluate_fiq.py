import os
import sys
import json
from statistics import mean
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(parent_dir)

from options import get_experiment_config
from set_up import setup_experiment
from transforms import image_transform_factory
from data import create_dataloaders
from models import create_qure_models
import torch

def main():
    configs = get_experiment_config()
    export_root, configs = setup_experiment(configs)
    device = torch.device(f"cuda:{configs['device_idx']}") if torch.cuda.is_available() else "cpu"
    print(f"Experiment: {configs['experiment_description']}")

    image_transform = image_transform_factory(config=configs)
    train_dataloader, test_dataloaders = create_dataloaders(image_transform, None, configs)

    print(len(train_dataloader), len(test_dataloaders))

    MS_pretrained_path = configs["pretrained_path"]
    print(f"Pretrained Model Path : {MS_pretrained_path}")

    model, txt_processors = create_qure_models(configs, device)
    msg = model.load_state_dict(torch.load(f'{MS_pretrained_path}/model.pth', map_location=device), strict=False)
    model.to(device)
    model.eval()

    print(f"Loaded Finetuned QuRe models : {msg}")

    recalls_at5 = []
    recalls_at10 = []
    recalls_at50 = []
    results_dict = dict()

    for cloth_type, cur_test_dataloader in test_dataloaders.items():
        cur_test_samples_dataloader = cur_test_dataloader['samples']
        cur_test_query_dataloader = cur_test_dataloader['query']

        index_features, index_names = model.extract_target_features(cur_test_samples_dataloader, configs['use_temp'], device)
        predicted_features, target_names = model.extract_query_features_fiq(
            cur_test_query_dataloader, configs['use_temp'], txt_processors, device)

        scores = model.score(predicted_features, index_features)

        sorted_indices = torch.argsort(scores, dim=-1, descending=True).cpu()
        sorted_index_names = np.array(index_names)[sorted_indices]

        # Compute the ground-truth labels wrt the predictions
        labels = torch.tensor(
            sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
        assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

        # Compute the metrics
        recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
        recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
        recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

        recalls_at5.append(recall_at5)
        recalls_at10.append(recall_at10)
        recalls_at50.append(recall_at50)

        results_dict[f"{cloth_type}_recall@5"] = recall_at5
        results_dict[f"{cloth_type}_recall@10"] = recall_at10
        results_dict[f"{cloth_type}_recall@50"] = recall_at50

    results_dict.update({
        f'average_recall_at5': mean(recalls_at5),
        f'average_recall_at10': mean(recalls_at10),
        f'average_recall_at50': mean(recalls_at50),
        f'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2
    })

    print(json.dumps(results_dict, indent=4))
    save_path = f"./cir_eval/fiq"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pretrained_note = configs['pretrained_path'].split('/')[-1]
    with open(f'{save_path}/{pretrained_note}.json', 'w') as json_file:
        json.dump(results_dict, json_file, indent=4)


if __name__ == '__main__':
    main()
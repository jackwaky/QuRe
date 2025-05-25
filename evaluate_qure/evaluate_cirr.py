import os
import sys
import json

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

    submission = {
        'version': 'rc2',
        'metric': 'recall'
    }
    group_submission = {
        'version': 'rc2',
        'metric': 'recall_subset'
    }

    for cloth_type, cur_test_dataloader in test_dataloaders.items():
        cur_test_samples_dataloader = cur_test_dataloader['samples']
        cur_test_query_dataloader = cur_test_dataloader['query']

        index_features, index_names = model.extract_target_features(cur_test_samples_dataloader, configs['use_temp'], device)
        predicted_features, reference_names, group_members, pairs_id = model.extract_query_features_cirr(cur_test_query_dataloader, configs['use_temp'], txt_processors, device)

        scores = model.score(predicted_features, index_features)

        sorted_indices = torch.argsort(scores, dim=-1, descending=True).cpu()
        sorted_index_names = np.array(index_names)[sorted_indices]

        # Delete the reference image from the results
        reference_mask = torch.tensor(sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names), -1))
        sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0], sorted_index_names.shape[1] - 1)

        # Compute the subset predictions
        group_members = np.array(group_members)
        group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
        sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)

        # Generate prediction dicts
        pairid_to_predictions = {str(int(pair_id)): prediction[:50].tolist() for (pair_id, prediction) in
                                 zip(pairs_id, sorted_index_names)}

        pairid_to_group_predictions = {str(int(pair_id)): prediction[:3].tolist() for (pair_id, prediction) in
                                       zip(pairs_id, sorted_group_names)}

        submission.update(pairid_to_predictions)
        group_submission.update(pairid_to_group_predictions)

        submission_path = './cir_eval/cirr/submission'
        pretrained_path_description = configs["pretrained_path"].split('/')[-1]
        if not os.path.exists(submission_path):
            os.makedirs(submission_path)

        print(f"Saving CIRR test predictions")
        with open(os.path.join(submission_path, f"{pretrained_path_description}_recall_submission_test.json"), 'w+') as file:
            json.dump(submission, file, sort_keys=True)

        with open(os.path.join(submission_path, f"{pretrained_path_description}_recall_subset_submission_test.json"), 'w+') as file:
            json.dump(group_submission, file, sort_keys=True)


if __name__ == '__main__':
    main()
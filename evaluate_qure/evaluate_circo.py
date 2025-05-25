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
    _, test_dataloaders = create_dataloaders(image_transform, None, configs)

    print(len(test_dataloaders))

    MS_pretrained_path = configs["pretrained_path"]
    print(f"Pretrained Model Path : {MS_pretrained_path}")

    model, txt_processors = create_qure_models(configs, device)
    msg = model.load_state_dict(torch.load(f'{MS_pretrained_path}/model.pth', map_location=device), strict=False)
    model.to(device)
    model.eval()

    print(f"Loaded Finetuned QuRe models : {msg}")

    for cloth_type, cur_test_dataloader in test_dataloaders.items():
        cur_test_samples_dataloader = cur_test_dataloader['samples']
        cur_test_query_dataloader = cur_test_dataloader['query']

        index_features, index_names = model.extract_target_features(cur_test_samples_dataloader, configs['use_temp'], device)
        predicted_features, query_ids = model.extract_query_features_circo(
            cur_test_query_dataloader, configs['use_temp'], txt_processors, device)


        scores = model.score(predicted_features, index_features)

        # sorted_indices = torch.argsort(scores, dim=-1, descending=True).cpu()
        sorted_indices = torch.topk(scores, dim=-1, k=50).indices.cpu()
        sorted_index_names = np.array(index_names)[sorted_indices]

        queryid_to_retrieved_images = {query_id: query_sorted_names[:50].tolist() for
                                       (query_id, query_sorted_names) in zip(query_ids, sorted_index_names)}


        submission_path = './cir_eval/circo/submission'
        pretrained_path_description = configs["pretrained_path"].split('/')[-1]
        if not os.path.exists(submission_path):
            os.makedirs(submission_path)

        print(f"Saving CIRCO test predictions")
        with open(os.path.join(submission_path, f"{pretrained_path_description}_fiq_circo_recall_submission_test.json"), 'w+') as file:
            json.dump(queryid_to_retrieved_images, file, sort_keys=True)



if __name__ == '__main__':
    main()
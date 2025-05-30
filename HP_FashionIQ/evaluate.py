import os
import sys
import json
import torch
import PIL
import PIL.Image
from torch.utils.data import DataLoader

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(parent_dir)

from options import get_experiment_config
from set_up import setup_experiment
from transforms import image_transform_factory
from models import create_qure_models
from data.utils import targetpad_transform

def load_model():
    # Load configuration
    configs = get_experiment_config()
    export_root, configs = setup_experiment(configs)
    device = torch.device(f"cuda:{configs['device_idx']}") if torch.cuda.is_available() else "cpu"
    
    # Create model
    model, txt_processors = create_qure_models(configs, device)
    
    # Load pretrained weights
    MS_pretrained_path = configs["pretrained_path"]
    msg = model.load_state_dict(torch.load(f'{MS_pretrained_path}/model.pth', map_location=device), strict=False)
    model.to(device)
    model.eval()
    
    print(f"Loaded Finetuned QuRe models : {msg}")
    return model, txt_processors, device, configs



def get_score_of_image(model, txt_processors, sentence, query_image_path, retrieved_image_paths, config, device):
    target_ratio = 1.25
    preprocess = targetpad_transform(target_ratio, config['img_size'])

    score = []
    query_image = preprocess(PIL.Image.open(query_image_path))

    # query_image = query_image.to(device)
    query_dataloader = DataLoader([(query_image, sentence, 'None')], batch_size=1, shuffle=False)

    # Query Feature
    predicted_features, target_names = model.extract_query_features_fiq(
        query_dataloader, config['use_temp'], txt_processors, device)

    retrieved_image_list = []
    for retrieve_image_path in retrieved_image_paths:
        retrieve_image = preprocess(PIL.Image.open(retrieve_image_path))
        # retrieve_image = retrieve_image.to(device)
        retrieved_image_list.append((retrieve_image, 'None'))
    #todo: get scores with batch size 5 and average it
    image_dataloader = DataLoader(retrieved_image_list, batch_size=5, shuffle=False)
    # Target Feature
    index_features, index_names = model.extract_target_features(image_dataloader,
                                                                config['use_temp'], device)


    cur_score = model.score(predicted_features, index_features)
    # score.append(cur_score.item())

    averaged_score = torch.mean(cur_score).item()

    return averaged_score

def evaluate_preference_ratio(model_list_1, model_list_2, preferred_set):
    assert len(model_list_1) == len(model_list_2) == len(preferred_set)

    conditioned_total = 0
    preferred_count = 0
    
    for m1, m2, p in zip(model_list_1, model_list_2, preferred_set):
        if m1 > m2:
            conditioned_total += 1
            if p == '1':
                preferred_count += 1
                
    if conditioned_total == 0:
        return 0.0  # or np.nan
    
    return preferred_count / conditioned_total
            

def main():
    # Load configuration
    
    # Load model
    model, txt_processors, device, configs = load_model()
    
    # Load hpfiq.json
    with open('hpfiq.json', 'r') as f:
        hpfiq_data = json.load(f)
    
    # Extract model scores and user scores
    model_list_1, model_list_2, preferred_list = [], [], []
    for user_id, cur_user_data in hpfiq_data.items():
        for question_id, cur_question_data in cur_user_data.items():
            ref_img_paths = cur_question_data['ref_img_paths']
            sentences = cur_question_data['sentences']
            retrieved_set1 = cur_question_data['retrieved_set1']
            retrieved_set2 = cur_question_data['retrieved_set2']
            preferred_set = cur_question_data['preferred set']
            for cur_ref_img_path, cur_ref_text, cur_retrieved_set1, cur_retrieved_set2, cur_preferred_set in zip(ref_img_paths, sentences, retrieved_set1, retrieved_set2, preferred_set):

                _cur_retrieved_set1 = cur_retrieved_set1['img_path']
                _cur_retrieved_set2 = cur_retrieved_set2['img_path']

                set1_model_score = get_score_of_image(model, txt_processors, cur_ref_text, cur_ref_img_path, _cur_retrieved_set1, configs, device)
                set2_model_score = get_score_of_image(model, txt_processors, cur_ref_text, cur_ref_img_path, _cur_retrieved_set2, configs, device)
    

                model_list_1.append(set1_model_score)
                model_list_2.append(set2_model_score)
                preferred_list.append(cur_preferred_set)

                
    # todo: you can save the scores for efficiency

    # Evaluation
    # Extract Preference Ratio
    evaluate_preference_ratio(model_list_1, model_list_2, preferred_list)

if __name__ == "__main__":
    main()

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(parent_dir)

from lavis.models import load_model_and_preprocess

def create_qure_models(configs, device):
    print(f"{configs['model']} {configs['backbone']} training")

    model_name = f"{configs['model']}_qure"
    backbone = configs["backbone"]
    model, _, txt_processors = load_model_and_preprocess(name=model_name, model_type=backbone, is_eval=False,
                                                         device=device)

    return model, txt_processors
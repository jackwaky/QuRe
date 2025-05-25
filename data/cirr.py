import json
import os
import random
from tqdm import tqdm
import numpy as np
import wandb

from data.utils import targetpad_transform
from torch.utils.data import Dataset, DataLoader
from data.collate_fns import BLIPPaddingCollateFunctionTest4CIRR
import torch

import PIL
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None

base_path = '-'
target_ratio = 1.25

class CIRRDataset(Dataset):
    """
       CIRR dataset class which manage CIRR data
       The dataset can be used in 'relative' or 'classic' mode:
           - In 'classic' mode the dataset yield tuples made of (image_name, image)
           - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, rel_caption) when split == train
                - (reference_name, target_name, rel_caption, group_members) when split == val
                - (pair_id, reference_name, rel_caption, group_members) when split == test1
    """

    def __init__(self, split: str, mode: str, preprocess: callable, config):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param mode: dataset mode, should be in ['relative', 'classic']:
                  - In 'classic' mode the dataset yield tuples made of (image_name, image)
                  - In 'relative' mode the dataset yield tuples made of:
                        - (reference_image, target_image, rel_caption) when split == train
                        - (reference_name, target_name, rel_caption, group_members) when split == val
                        - (pair_id, reference_name, rel_caption, group_members) when split == test1
        :param preprocess: function which preprocesses the image
        """
        self.preprocess = targetpad_transform(target_ratio, config['img_size'])
        self.mode = mode
        self.split = split
        self.config = config
        self.negative = "random"

        if split not in ['test1', 'train', 'val']:
            raise ValueError("split should be in ['test1', 'train', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        # get triplets made by (reference_image, target_image, relative caption)
        with open(os.path.join(base_path , 'cirr' , 'captions' , f'cap.rc2.{split}.json')) as f:
            self.triplets = json.load(f)

        # get a mapping from image name to relative path
        with open(os.path.join(base_path , 'cirr' , 'image_splits' , f'split.rc2.{split}.json')) as f:
            self.name_to_relpath = json.load(f)

        self.image_names = list(self.name_to_relpath.keys())
        self.hard_images = []
        print(f"CIRR {split} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                group_members = self.triplets[index]['img_set']['members']
                reference_name = self.triplets[index]['reference']
                rel_caption = self.triplets[index]['caption']

                if self.split == 'train':
                    reference_image_path = os.path.join(base_path  , self.name_to_relpath[reference_name])
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_hard_name = self.triplets[index]['target_hard']
                    target_image_path = os.path.join(base_path  , self.name_to_relpath[target_hard_name])
                    target_image = self.preprocess(PIL.Image.open(target_image_path))

                    if self.negative == 'random': # random negative sampling
                        cur_non_target_pool = [name for name in self.image_names if name != target_hard_name]
                        negative_name = random.choice(cur_non_target_pool)
                        negative_targ_img_path = os.path.join(base_path  , self.name_to_relpath[negative_name])
                        negative_target_img = self.preprocess(PIL.Image.open(negative_targ_img_path))
                    elif self.negative == 'random_rerank': # hard negative sampling after reranking
                        # negative_name = random.choice(self.hard_images[index])
                        if index not in self.negative_selection_history:
                            self.negative_selection_history[index] = []

                        if len(self.hard_images[index]) == len(self.negative_selection_history[index]):
                            self.negative_selection_history[index] = []

                        while True:
                            negative_name = random.choice(self.hard_images[index])
                            if negative_name not in self.negative_selection_history[index]:
                                self.negative_selection_history[index].append(negative_name)
                                break

                        negative_targ_img_path = os.path.join(base_path  , self.name_to_relpath[negative_name])
                        negative_target_img = self.preprocess(PIL.Image.open(negative_targ_img_path))
                    else:
                        raise ValueError("Undefined Negative Sampling Method")
                    # captions = [txt_processors["eval"](caption) for caption in rel_caption]
                    return reference_image, target_image, negative_target_img, target_hard_name, rel_caption

                elif self.split == 'val':
                    reference_image_path = os.path.join(base_path, self.name_to_relpath[reference_name])
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_hard_name = self.triplets[index]['target_hard']
                    target_image_path = os.path.join(base_path, self.name_to_relpath[target_hard_name])
                    target_image = self.preprocess(PIL.Image.open(target_image_path))

                    pair_id = self.triplets[index]['pairid']
                    return reference_name, reference_image, target_hard_name, target_image, rel_caption, pair_id, group_members
                    # target_hard_name = self.triplets[index]['target_hard']
                    # return reference_name, target_hard_name, rel_caption, group_members

                elif self.split == 'test1':
                    reference_image_path = os.path.join(base_path, self.name_to_relpath[reference_name])
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))

                    pair_id = self.triplets[index]['pairid']
                    return reference_name, reference_image, rel_caption, pair_id, group_members

            elif self.mode == 'classic':
                image_name = list(self.name_to_relpath.keys())[index]
                image_path = os.path.join(base_path, self.name_to_relpath[image_name])
                im = PIL.Image.open(image_path)
                image = self.preprocess(im)

                # # This is only used for User survey generation
                # if self.split == 'val':
                #     return image, image_path, image_name
                return image, image_name

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.name_to_relpath)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
        
    @torch.no_grad()
    def rerank_score(self, model, device, txt_processors, configs, epoch):
        self.negative = configs["negative"]
        self.negative_selection_history = dict()
        self.hard_images = []
        model.eval()

        sample_dataloader = self.get_sample_loader()
        query_dataloader = self.get_query_loader()

        index_features, index_names = model.extract_target_features(sample_dataloader, configs['use_temp'], device)
        predicted_features, _, _, _ = model.extract_query_features_cirr(query_dataloader, configs['use_temp'], txt_processors, device)

        scores = model.score(predicted_features, index_features)
        score_sorted_indices = torch.argsort(scores, dim=-1, descending=True).cpu()

        for index in tqdm(range(len(self.triplets)), desc=f"Reranking with score and image similarity"):
            target_hard_name = self.triplets[index]['target_hard']
            # target_index = self.image_names.index(target_hard_name)

            # Compute Score (it consider not only the image, but also text)
            cur_query_score_sorted_indices = score_sorted_indices[index]
            cur_query_score = scores[index]
            
            target_index = self.image_names.index(target_hard_name)
            sorted_indices = [i for i in cur_query_score_sorted_indices.tolist()]
            target_sorted_index = sorted_indices.index(target_index)

            score_list = cur_query_score[cur_query_score_sorted_indices]
            y = np.array(score_list)

            start_index = target_sorted_index + 1
            scores_after_target = y[start_index:]

            # Step 1: Compute differences
            differences = np.diff(scores_after_target)  # Differences between consecutive elements
            # Sort indices by difference values (ascending order)
            diff_sorted_indices = np.argsort(differences)

            idx = 0
            while idx < len(diff_sorted_indices) - 1:
                idx1 = diff_sorted_indices[idx]
                idx2 = diff_sorted_indices[idx + 1]

                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                if idx2 - idx1 > configs["consecutive_threshold"]:
                    # Get the largest and second-largest drop indices
                    start_of_hard_negatives = idx1 + start_index
                    end_of_hard_negatives = idx2 + start_index
                    break
                else:
                    idx += 1


            top_k_score_indices = [i for i in cur_query_score_sorted_indices.tolist()][
                                    start_of_hard_negatives + 1:end_of_hard_negatives + 1]

            if len(top_k_score_indices) == 0:
                assert 0, "Empty negative set is defined"

            # wandb.log({"query_idx" : index, "number_of_hard_images" : len(top_k_score_indices)})
            topk_image_names = list(set([self.image_names[i] for i in top_k_score_indices if self.image_names[i] != target_hard_name]))
            self.hard_images.append(topk_image_names)


        if configs["experiment_description"] != 'debug':
            average_size_negative_set = sum(len(sublist) for sublist in self.hard_images) / len(self.hard_images)
            wandb.log({'average size of negative set': average_size_negative_set, "epoch": epoch})

        model.train()
        print("Done\n")

    def get_sample_loader(self):
        sample_dataset = CIRRSampleDataset(base_path, self.image_names, self.name_to_relpath, self.preprocess)
        collate_fn = BLIPPaddingCollateFunctionTest4CIRR()
        sample_dataloader = DataLoader(sample_dataset, batch_size=64, shuffle=False, num_workers=16, pin_memory=True, collate_fn=collate_fn)

        return sample_dataloader
    def get_query_loader(self):
        query_dataset = CIRRQueryDataset(base_path, self.triplets, self.name_to_relpath, self.preprocess)
        collate_fn = BLIPPaddingCollateFunctionTest4CIRR()
        query_dataloader = DataLoader(query_dataset, batch_size=64, shuffle=False, num_workers=16, pin_memory=True, collate_fn=collate_fn)

        return query_dataloader
    @classmethod
    def code(cls):
        return 'cirr'

    @classmethod
    def all_codes(cls):
        return ['cirr']

    @classmethod
    def vocab_path(cls):
        return None

class CIRRSampleDataset(Dataset):
    def __init__(self, base_path, image_names, name_to_relpath, preprocess):
        self.base_path = base_path
        self._image_names = image_names
        self._name_to_relpath = name_to_relpath
        self.preprocess = preprocess

    def __len__(self):
        return len(self._image_names)

    def __getitem__(self, idx):
        image_name = self._image_names[idx]
        image_path = os.path.join(self.base_path, self._name_to_relpath[image_name])
        im = PIL.Image.open(image_path)
        image = self.preprocess(im)

        return image, image_name


class CIRRQueryDataset(Dataset):
    def __init__(self, base_path, triplets, name_to_relpath, preprocess):
        self.base_path = base_path
        self._triplets = triplets
        self._name_to_relpath = name_to_relpath
        self.preprocess = preprocess

    def __len__(self):
        return len(self._triplets)

    def __getitem__(self, index):
        group_members = self._triplets[index]['img_set']['members']
        reference_name = self._triplets[index]['reference']
        rel_caption = self._triplets[index]['caption']
        reference_image_path = os.path.join(base_path, self._name_to_relpath[reference_name])
        reference_image = self.preprocess(PIL.Image.open(reference_image_path))
        pair_id = self._triplets[index]['pairid']

        return reference_name, reference_image, rel_caption, pair_id, group_members

if __name__ == '__main__':

    # cirr_train_dataset = CIRRDataset('test1', 'classic', None, None)
    #
    # train_loader = DataLoader(dataset=cirr_train_dataset, batch_size=32,
    #                           pin_memory=False, collate_fn=collate_fn,
    #                           drop_last=True, shuffle=True)
    #
    # for idx, (reference_image, target_image, negative_target_img, rel_caption) in enumerate(train_loader):
    #     print('hello')
    #
    # cirr_val_dataset = CIRRDataset('val', 'relative', None)
    # cirr_test1_dataset = CIRRDataset('test1', 'relative', None)
    #
    # print(len(cirr_train_dataset), len(cirr_val_dataset), len(cirr_test1_dataset))
    pass

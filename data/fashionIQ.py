import copy
import json
import numpy as np
import os
import random
from tqdm import tqdm
from typing import List
from pathlib import Path
import wandb
import math

from torch.utils.data import Dataset, DataLoader
from data.utils import targetpad_transform, _get_img_from_path
from data.abc import AbstractBaseDataset
from data.collate_fns import BLIPPaddingCollateFunctionTest4FIQ
import torch
from torch.nn import functional as F

import PIL
import PIL.Image


base_path = '-'
target_ratio = 1.25

class FashionIQDataset(Dataset):
    """
    FashionIQ dataset class which manage FashionIQ data.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield tuples made of (image_name, image)
        - In 'relative' mode the dataset yield tuples made of:
            - (reference_image, target_image, image_captions) when split == train
            - (reference_name, target_name, image_captions) when split == val
            - (reference_name, reference_image, image_captions) when split == test
    The dataset manage an arbitrary numbers of FashionIQ category, e.g. only dress, dress+toptee+shirt, dress+shirt...
    """

    def __init__(self, split, dress_types, mode, preprocess: callable, config):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param dress_types: list of fashionIQ category
        :param mode: dataset mode, should be in ['relative', 'classic']:
            - In 'classic' mode the dataset yield tuples made of (image_name, image)
            - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, image_captions) when split == train
                - (reference_name, target_name, image_captions) when split == val
                - (reference_name, reference_image, image_captions) when split == test
        :param preprocess: function which preprocesses the image
        """
        self.preprocess = targetpad_transform(target_ratio, config['img_size'])
        # self.preprocess = image_transform(config, split)
        self.mode = mode
        self.split = split
        self.config = config
        self.dress_types = dress_types
        self.negative = 'random'

        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")

        # get triplets made by (reference_image, target_image, a pair of relative captions)
        self.triplets: List[dict] = []
        self.domain_triplets = dict()
        self.dress_type_indices_relative = dict()
        for dress_type in dress_types:
            with open(base_path /  'captions' / f'cap.{dress_type}.{split}.json') as f:
                triplets = json.load(f)

                start_idx = len(self.triplets)
                for triplet in triplets:
                    triplet['dress_type'] = dress_type
                self.triplets.extend(triplets)
                end_idx = len(self.triplets)
                self.dress_type_indices_relative[dress_type] = list(range(start_idx, end_idx))

                self.domain_triplets[dress_type] = triplets

        # get the image names
        self.domain_image_names = dict()
        self.image_names: list = []
        self.dress_type_indices_classic = dict()
        for dress_type in dress_types:
            with open(base_path / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
                images = json.load(f)
                start_idx = len(self.image_names)
                self.image_names.extend(images)
                end_idx = len(self.image_names)
                self.dress_type_indices_classic[dress_type] = list(range(start_idx, end_idx))
                self.domain_image_names[dress_type] = images

        self.hard_images = []

        print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                image_captions = self.triplets[index]['captions']
                reference_name = self.triplets[index]['candidate']
                rel_caption = image_captions[0].strip('.?, ').capitalize() + " and " + image_captions[1].strip('.?, ')
                # rel_caption = caption_post_process(rel_caption)

                if self.split == 'train':
                    reference_image_path = base_path / 'images' / f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    # reference_image = _get_img_from_path(reference_image_path, self.preprocess)
                    target_name = self.triplets[index]['target']
                    target_image_path = base_path / 'images' / f"{target_name}.png"
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    # target_image = _get_img_from_path(target_image_path, self.preprocess)

                    cur_dress_type = self.triplets[index]['dress_type']
                    # self.non_target_pool = self.domain_image_names[cur_dress_type]
                    self.non_target_pool = [triplet['target'] for triplet in self.domain_triplets[cur_dress_type]]

                    if self.negative == 'random': # random negative sampling
                        cur_non_target_pool = [name for name in self.non_target_pool if name != target_name]
                        negative_name = random.choice(cur_non_target_pool) # Sample from Target pool (!= target)
                        negative_targ_img_path = base_path / 'images' / f"{negative_name}.png"
                        negative_target_img = self.preprocess(PIL.Image.open(negative_targ_img_path))
                        # negative_target_img = _get_img_from_path(negative_targ_img_path, self.preprocess)
                    elif self.negative == 'random_rerank': # hard negative sampling after reranking
                        if index not in self.negative_selection_history:
                            self.negative_selection_history[index] = []

                        if len(self.hard_images[index]) == len(self.negative_selection_history[index]):
                            self.negative_selection_history[index] = []

                        while True:
                            negative_name = random.choice(self.hard_images[index])
                            if negative_name not in self.negative_selection_history[index]:
                                break  # Stop resampling once a unique negative is found
                        # negative_name = random.choice(self.hard_images[index])
                        negative_targ_img_path = base_path / 'images' / f"{negative_name}.png"
                        negative_target_img = self.preprocess(PIL.Image.open(negative_targ_img_path))

                        self.negative_selection_history[index].append(negative_name)

                    # elif self.negative == 'rerank':
                    #     negative_target_img = []
                    #     for negative_name in self.hard_images[index]:
                    #         negative_targ_img_path = base_path / 'images' / f"{negative_name}.png"
                    #         negative_target_img_ = self.preprocess(PIL.Image.open(negative_targ_img_path))
                    #         negative_target_img.append(negative_target_img_)
                    #     # negative_target_img = torch.stack(negative_target_img, dim=0)
                    else:
                        raise ValueError("Undefined Negative Sampling Method")

                    return reference_image, target_image, negative_target_img, target_name, rel_caption

                elif self.split == 'val':
                    reference_image_path = base_path / 'images' / f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    # reference_image = _get_img_from_path(reference_image_path, self.preprocess)
                    target_name = self.triplets[index]['target']
                    return reference_image, rel_caption, target_name

                elif self.split == 'test':
                    reference_image_path = base_path / 'images' / f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    return reference_name, reference_image, rel_caption

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image_path = base_path / 'images' / f"{image_name}.png"
                image = self.preprocess(PIL.Image.open(image_path))
                # image = _get_img_from_path(image_path, self.preprocess)
                return image, image_name

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")

    @torch.no_grad()
    def rerank_score(self, model, device, txt_processors, configs, epoch):
        self.negative_selection_history = dict()
        self.negative = configs['negative']
        self.hard_images = []
        model.eval()

        sample_dataloader = self.get_sample_loader()
        query_dataloader = self.get_query_loader()

        predicted_features, _ = model.extract_query_features_fiq(query_dataloader, configs['use_temp'], txt_processors, device)
        index_features, index_names = model.extract_target_features(sample_dataloader, configs['use_temp'], device)

        scores = model.score(predicted_features, index_features)

        for index in tqdm(range(len(self.triplets)), desc=f"Reranking with score and image similarity"):
            image_captions = self.triplets[index]['captions']
            reference_name = self.triplets[index]['candidate']
            target_name = self.triplets[index]['target']
            cur_dress_type = self.triplets[index]['dress_type']

            cur_query_scores = scores[index]
            cur_query_cur_dress_type_scores = cur_query_scores[self.dress_type_indices_relative[cur_dress_type]]
            cur_query_score_sorted_indices = torch.argsort(cur_query_cur_dress_type_scores, dim=-1,
                                                           descending=True).cpu()
            
            cur_dress_type_target_images = [triplet['target'] for triplet in self.domain_triplets[cur_dress_type]]
            # cur_dress_type_target_images = [image for image in self.domain_image_names[cur_dress_type]]
            target_index = cur_dress_type_target_images.index(target_name)
            sorted_indices = [i for i in cur_query_score_sorted_indices.tolist()]
            target_sorted_index = sorted_indices.index(target_index)

            score_list = cur_query_cur_dress_type_scores[cur_query_score_sorted_indices]
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
                    # diff_sorted_indices = [idx_ for idx_ in diff_sorted_indices if idx_ > idx2]

            # Negative set : start_of_false_negatives + 1 ~ end_of_false_negatives
            top_k_score_indices = [i for i in cur_query_score_sorted_indices.tolist()][start_of_hard_negatives + 1:end_of_hard_negatives + 1]

            if len(top_k_score_indices) == 0:
                assert 0, "Empty negative set is defined"


            if len(top_k_score_indices) != 0:
                cur_dress_type_target_images = [triplet['target'] for triplet in self.domain_triplets[cur_dress_type]]
                # cur_dress_type_target_images = [image for image in self.domain_image_names[cur_dress_type]]
                topk_image_names = list(set([cur_dress_type_target_images[i] for i in top_k_score_indices if cur_dress_type_target_images[i] != target_name]))
            else:
                assert 0, "Empty negative set is defined"
            self.hard_images.append(topk_image_names)

        if configs["experiment_description"] != 'debug':
            sizes = [len(sublist) for sublist in self.hard_images]
            average_size_negative_set = sum(sizes) / len(sizes)
            variance = sum((size - average_size_negative_set) ** 2 for size in sizes) / len(sizes)
            std_dev = math.sqrt(variance)

            wandb.log({'average size of negative set': average_size_negative_set, "epoch": epoch})
            wandb.log({'std of negative set size': std_dev, "epoch": epoch})

        model.train()
        print("Done\n")

    def get_sample_loader(self):
        image_corpus = [triplet['target'] for triplet in self.triplets]
        # image_corpus = [image for image in self.image_names]
        sample_dataset = FashionIQSampleDataset(base_path, image_corpus, self.preprocess)
        collate_fn = BLIPPaddingCollateFunctionTest4FIQ()
        sample_dataloader = DataLoader(sample_dataset, batch_size=64, shuffle=False, num_workers=16, pin_memory=True, collate_fn=collate_fn)

        return sample_dataloader
    def get_query_loader(self):
        query_dataset = FashionIQQueryDataset(base_path, self.triplets, self.preprocess)
        collate_fn = BLIPPaddingCollateFunctionTest4FIQ()
        query_dataloader = DataLoader(query_dataset, batch_size=64, shuffle=False, num_workers=16, pin_memory=True, collate_fn=collate_fn)

        return query_dataloader

    @classmethod
    def code(cls):
        return 'fashionIQ'

    @classmethod
    def all_codes(cls):
        return ['fashionIQ']
    
    @classmethod
    def all_subset_codes(cls):
        return ['dress', 'shirt', 'toptee']

    @classmethod
    def vocab_path(cls):
        return None
        
class FashionIQSampleDataset(Dataset):
    def __init__(self, base_path, image_names, preprocess):
        self.base_path = base_path
        self._image_names = image_names
        self.preprocess = preprocess

    def __len__(self):
        return len(self._image_names)

    def __getitem__(self, idx):
        image_name = self._image_names[idx]
        image_path = base_path / 'images' / f"{image_name}.png"
        im = PIL.Image.open(image_path)
        image = self.preprocess(im)

        return image, image_name


class FashionIQQueryDataset(Dataset):
    def __init__(self, base_path, triplets, preprocess):
        self.base_path = base_path
        self._triplets = triplets
        self.preprocess = preprocess

    def __len__(self):
        return len(self._triplets)

    def __getitem__(self, index):
        reference_name = self._triplets[index]['candidate']
        target_name = self._triplets[index]['target']
        image_captions = self._triplets[index]['captions']

        reference_image_path = base_path / 'images' / f"{reference_name}.png"
        reference_image = self.preprocess(PIL.Image.open(reference_image_path))
        # rel_caption = image_captions[0] + " and " + image_captions[1]
        rel_caption = f"{image_captions[0].strip('.?, ').capitalize()} and {image_captions[1].strip('.?, ')}"

        return target_name, reference_image, rel_caption





##############################################################
############## For User Suvery Generation ####################
##############################################################

def _get_img_caption_json(dataset_root, clothing_type, split):
    with open(os.path.join(dataset_root, 'captions', 'cap.{}.{}.json'.format(clothing_type, split))) as json_file:
        img_caption_data = json.load(json_file)
    return img_caption_data

def _get_img_caption_txt(dataset_root, clothing_type, split, config):
    with open(os.path.join(dataset_root, 'captions_pairs', 'fashion_iq-{}-cap-{}.txt'.format(split, clothing_type))) as f:
        file_content = f.readlines()
    return file_content

def _get_img_split_json_as_list(dataset_root, clothing_type, split):
    with open(os.path.join(dataset_root, 'image_splits', 'split.{}.{}.json'.format(clothing_type, split))) as json_file:
        img_split_list = json.load(json_file)
    return img_split_list

def _create_img_path_from_id(root, id):
    return os.path.join(root, '{}.jpg'.format(id))

def _get_img_path_using_idx(img_caption_data, img_root, idx, is_ref=True):
    img_caption_pair = img_caption_data[idx]
    key = 'candidate' if is_ref else 'target'

    img = _create_img_path_from_id(img_root, img_caption_pair[key])
    id = img_caption_pair[key]
    return img, id

def _get_modifier(img_caption_data, idx, reverse=False):
    img_caption_pair = img_caption_data[idx]
    cap1, cap2 = img_caption_pair['captions']
    return _create_modifier_from_attributes(cap1, cap2) if not reverse else _create_modifier_from_attributes(cap2, cap1)

def _cat_captions(caps):
    I = []
    for i in range(len(caps)):
        if i % 2 == 0:
            I.append(_create_modifier_from_attributes(caps[i], caps[i+1]))
        else:
            I.append(_create_modifier_from_attributes(caps[i], caps[i-1]))
    return I

def _create_modifier_from_attributes(ref_attribute, targ_attribute):
    return ref_attribute + " and " + targ_attribute

def caption_post_process(s):
    return s.strip().replace('.', 'dotmark').replace('?', 'questionmark').replace('&', 'andmark').replace('*', 'starmark')

class AbstractBaseFashionIQDataset(AbstractBaseDataset):

    @classmethod
    def code(cls):
        return 'fashionIQ'

    @classmethod
    def all_codes(cls):
        return ['fashionIQ']
    
    @classmethod
    def all_subset_codes(cls):
        return ['dress', 'shirt', 'toptee']
    
    @classmethod
    def vocab_path(cls):
        return _DEFAULT_FASHION_IQ_VOCAB_PATH
    

class FashionIQUserSurveyDataset(AbstractBaseFashionIQDataset):
    """
    Fashion200K dataset.
    Image pairs in {root_path}/image_pairs/{split}_pairs.pkl

    """

    def __init__(self, root_path=base_path, clothing_type='dress', split='train',
                 img_transform=None, id_transform=None, config=None):
        super().__init__(root_path, split, img_transform)
        self.root_path = root_path
        self.img_root_path = os.path.join(self.root_path, 'image_data')
        self.clothing_type = clothing_type
        self.split = split
        self.img_transform = img_transform
        self.id_transform = id_transform
        self.img_caption_data = _get_img_caption_txt(root_path, clothing_type, split, config)
        self.ref_img_path = np.array([ff.strip().split(';')[0] for ff in self.img_caption_data])
        self.targ_img_path = np.array([ff.strip().split(';')[1] for ff in self.img_caption_data])
        self.caps = [ff.strip('\n').split(';')[-1] for ff in self.img_caption_data]
        self.caps_cat = _cat_captions(self.caps)
        self.config = config


    def __getitem__(self, idx):

        ref_img_path = os.path.join(self.img_root_path, self.ref_img_path[idx])
        targ_img_path = os.path.join(self.img_root_path, self.targ_img_path[idx])
        reference_img = _get_img_from_path(ref_img_path, self.img_transform)
        target_img = _get_img_from_path(targ_img_path, self.img_transform)


        sentences = self.caps_cat[idx]
        sentences = caption_post_process(sentences)
        # modifier = self.text_transform(sentences) if self.text_transform else sentences

        ref_id = self.ref_img_path[idx].split('/')[-1].split('.')[0]
        targ_id = self.targ_img_path[idx].split('/')[-1].split('.')[0]

        if self.id_transform:
            ref_id = self.id_transform(ref_id)
            targ_id = self.id_transform(targ_id)


        return reference_img, ref_img_path, target_img, targ_img_path, targ_id, sentences

    def __len__(self):
        return len(self.img_caption_data)# * 2


class FashionIQUserSurveyTestSampleDataset(AbstractBaseFashionIQDataset):
    """
    FashionIQ Test (Samples) dataset.
    indexing returns target samples and their unique ID
    """

    def __init__(self, root_path=base_path, clothing_type='dress', split='val',
                 img_transform=None, config=None):
        super().__init__(root_path, split, img_transform)
        self.root_path = root_path
        self.img_root_path = os.path.join(self.root_path, 'image_data')
        self.clothing_type = clothing_type
        self.img_transform = img_transform

        #self.img_list = _get_img_split_json_as_list(root_path, clothing_type, split)

        ''' Uncomment below for VAL Evaluation method '''
        self.img_caption_data = _get_img_caption_txt(root_path, clothing_type, split, config)
        self.img_list = []
        for d in self.img_caption_data:
            ref = d.split(';')[0].split('/')[-1].split('.')[0] #dress/xxx.jpg --> xxx
            targ = d.split(';')[1].split('/')[-1].split('.')[0] #dress/xxx.jpg --> xxx
            self.img_list.append(ref)
            self.img_list.append(targ)
        self.img_list = list(set(self.img_list))

    def __getitem__(self, idx, use_transform=True):
        img_transform = self.img_transform if use_transform else None
        img_id = self.img_list[idx]
        img_path = _create_img_path_from_id(os.path.join(self.img_root_path,self.clothing_type), img_id)

        target_img = _get_img_from_path(img_path, img_transform)

        return target_img, img_path, img_id

    def __len__(self):
        return len(self.img_list)
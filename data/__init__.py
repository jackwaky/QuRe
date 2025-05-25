from torch.utils.data import DataLoader
from data.fashionIQ import FashionIQDataset, FashionIQUserSurveyDataset, FashionIQUserSurveyTestSampleDataset
from data.cirr import CIRRDataset
from data.collate_fns import BLIPPaddingCollateFunction, BLIPPaddingCollateFunctionTest, BLIPPaddingCollateFunctionTest4CIRR, base_collate_fn
from data.circo import CIRCODataset

def train_dataset_factory(transforms, config):
    dataset_code = config['dataset']
    dataset = None

    if dataset_code == 'fashionIQ':
        dataset = FashionIQDataset(split='train', dress_types=FashionIQDataset.all_subset_codes(), mode='relative', preprocess=None, config=config)

    elif dataset_code == 'cirr':
        dataset = CIRRDataset('train', 'relative', None, config)
    elif dataset_code == 'circo':
        pass
    else:
        raise ValueError("There's no {} dataset".format(dataset_code))

    return dataset



def test_dataset_factory(transforms, config, split='val'):
    image_transform = transforms['image_transform']
    dataset_code = config['dataset']
    test_datasets = {}

    if dataset_code == 'fashionIQ':
        if config['mode'] == 'train' or config['mode'] == 'eval':
            for dress_type in FashionIQDataset.all_subset_codes():
                test_datasets['fashionIQ_' + dress_type] = {
                    "samples": FashionIQDataset(split="val", dress_types=[dress_type],
                                                    mode="classic", preprocess=None,
                                                    config=config),
                    "query": FashionIQDataset(split="val", dress_types=[dress_type],
                                                    mode="relative", preprocess=None,
                                                    config=config)
                }

        elif config['mode'] == 'user_survey':
            for clothing_type in FashionIQDataset.all_subset_codes():
                test_datasets['fashionIQ_' + clothing_type] = {
                    "query" : FashionIQUserSurveyDataset(split='val', clothing_type=clothing_type,
                                                                               img_transform=image_transform,
                                                                               config=config),
                    "samples" :  FashionIQUserSurveyTestSampleDataset(split=split, clothing_type=clothing_type,
                                                    img_transform=image_transform,
                                                    config=config)
                }

        else:
            raise ValueError("Not implemented")

    elif dataset_code == 'cirr':
        if config['mode'] == 'train':
            test_datasets[CIRRDataset.code()] = {
                "samples": CIRRDataset('val', 'classic', None, config),
                "query": CIRRDataset('val', 'relative', None, config)
            }

        elif config['mode'] == 'eval':
            test_datasets[CIRRDataset.code()] = {
                "samples": CIRRDataset('test1', 'classic', None, config),
                "query": CIRRDataset('test1', 'relative', None, config)
            }

        else:
            raise ValueError("Not Implemented")

    elif dataset_code == 'circo':
        if config['mode'] == 'val':
            test_datasets[dataset_code] = {
                "samples": CIRCODataset('val', 'classic', None, config),
                "query": CIRCODataset('val', 'relative', None, config)
            }
        elif config['mode'] == 'test':
            test_datasets[dataset_code] = {
                "samples": CIRCODataset('test', 'classic', None, config),
                "query": CIRCODataset('test', 'relative', None, config)
            }
        else:
            raise ValueError("Not Implemented")

    if len(test_datasets) == 0:
        raise ValueError("There's no {} dataset".format(dataset_code))

    return test_datasets


def train_dataloader_factory(dataset, config, collate_fn=None):
    batch_size = config['batch_size']
    num_workers = config.get('num_workers', 16)
    shuffle = config.get('shuffle', True)
    # # TODO: remove this
    # drop_last = batch_size == 32

    return DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
                      collate_fn=collate_fn)

def test_dataloader_factory(datasets, config, collate_fn=None):
    batch_size = config['batch_size']
    num_workers = config.get('num_workers', 16)
    shuffle = False

    return {
        'query': DataLoader(datasets['query'], batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                            collate_fn=collate_fn),
        'samples': DataLoader(datasets['samples'], batch_size, shuffle=False, num_workers=num_workers,
                              pin_memory=True,
                              collate_fn=collate_fn)
    }

def create_dataloaders(image_transform, text_transform, configs):
    train_dataloader, test_dataloaders = None, None
    train_dataset = train_dataset_factory(
        transforms={'image_transform': image_transform['train'], 'text_transform': text_transform},
        config=configs)
    test_datasets = test_dataset_factory(
        transforms={'image_transform': image_transform['val'], 'text_transform': text_transform},
        config=configs)

    # collate_fn for train
    collate_fn = BLIPPaddingCollateFunction()

    # collate_fn for test
    if configs['dataset'] == 'fashionIQ':
        collate_fn_test = BLIPPaddingCollateFunctionTest()
    elif configs['dataset'] == 'cirr':
        collate_fn_test = BLIPPaddingCollateFunctionTest4CIRR()
    elif configs['dataset'] == 'circo':
        collate_fn_test = base_collate_fn

    if train_dataset != None:
        train_dataloader = train_dataloader_factory(dataset=train_dataset, config=configs, collate_fn=collate_fn)
    test_dataloaders = {key: test_dataloader_factory(datasets=value, config=configs, collate_fn=collate_fn_test) for
                        key, value in test_datasets.items()}


    return train_dataloader, test_dataloaders

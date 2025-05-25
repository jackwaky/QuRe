from typing import List

import torch


def base_collate_fn(batch):
    '''
    function which discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    '''
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
class BLIPPaddingCollateFunction(object):
    def __init__(self):
        pass
    def __call__(self, batch: List[tuple]):
        reference_images, target_images, negative_target_img, targ_id, sentences = zip(*batch)

        reference_images = torch.stack(reference_images, dim=0)
        target_images = torch.stack(target_images, dim=0)
        negative_target_img = torch.stack(negative_target_img, dim=0)

        return reference_images, target_images, negative_target_img, sentences


class BLIPPaddingCollateFunctionTest(object):
    def __init__(self):
        pass

    @staticmethod
    def _collate_test_dataset(batch):
        reference_images, ids = zip(*batch)
        reference_images = torch.stack(reference_images, dim=0)
        return reference_images, ids

    def _collate_test_query_dataset(self, batch):
        reference_image, rel_caption, target_name = zip(*batch)
        reference_images = torch.stack(reference_image, dim=0)

        target_name = list(target_name)

        return reference_images, rel_caption, target_name
        
    def __call__(self, batch: List[tuple]):
        num_items = len(batch[0])
        if num_items > 2:
            return self._collate_test_query_dataset(batch)
        else:
            return self._collate_test_dataset(batch)


class BLIPPaddingCollateFunctionTest4CIRR(object):
    def __init__(self):
        pass

    @staticmethod
    def _collate_test_dataset(batch):
        reference_images, ids = zip(*batch)
        reference_images = torch.stack(reference_images, dim=0)
        return reference_images, ids

    def _collate_val_query_dataset(self, batch):
        # reference_images, ref_attrs, modifiers, target_attrs, lengths, sentences = zip(*batch)
        reference_name, reference_image, target_hard_name, target_image, rel_caption, pair_id, group_members = zip(*batch)
        reference_images = torch.stack(reference_image, dim=0)
        target_images = torch.stack(target_image, dim=0)

        reference_name = list(reference_name)
        target_hard_name = list(target_hard_name)

        return reference_name, reference_images, target_hard_name, rel_caption, pair_id, group_members

    def _collate_test_query_dataset(self, batch):
        # reference_images, ref_attrs, modifiers, target_attrs, lengths, sentences = zip(*batch)
        reference_name, reference_image, rel_caption, pair_id, group_members = zip(*batch)
        reference_images = torch.stack(reference_image, dim=0)
        reference_name = list(reference_name)

        return reference_name, reference_images, rel_caption, pair_id, group_members

    def __call__(self, batch: List[tuple]):
        num_items = len(batch[0])
        if num_items > 2 and num_items != 7:
            return self._collate_test_query_dataset(batch)
        elif num_items == 7:
            return self._collate_val_query_dataset(batch)
        else:
            return self._collate_test_dataset(batch)
        
class BLIPPaddingCollateFunctionTest4FIQ(object):
    def __init__(self):
        pass

    @staticmethod
    def _collate_test_dataset(batch):
        reference_images, ids = zip(*batch)
        reference_images = torch.stack(reference_images, dim=0)
        return reference_images, ids

    def _collate_test_query_dataset(self, batch):
        # reference_images, ref_attrs, modifiers, target_attrs, lengths, sentences = zip(*batch)
        target_name, reference_image, rel_caption = zip(*batch)
        reference_images = torch.stack(reference_image, dim=0)
        target_name = list(target_name)

        return reference_images, rel_caption, target_name

    def __call__(self, batch: List[tuple]):
        num_items = len(batch[0])
        if num_items > 2:
            return self._collate_test_query_dataset(batch)
        else:
            return self._collate_test_dataset(batch)





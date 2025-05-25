"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from tqdm import tqdm
import PIL
import PIL.Image
import numpy as np

import torch
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.models.blip_models.blip import BlipBase
from torch import nn
from lavis.models.med import XBertEncoder

from lavis.models.vit import VisionTransformerEncoder

from data.utils import targetpad_transform
import contextlib

preprocess = targetpad_transform(1.25, 224)


@registry.register_model("blip_qure")
class BlipQuRe(BlipBase):
    """
    BLIP Image-Text Matching (ITM) model.

    Supported model types:
        - base: fine-tuned BLIP retrieval weights on COCO dataset (Karpathy split).
        - large: fine-tuned BLIP retrieval weights on COCO dataset (Karpathy split).

    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip_image_text_matching", "base")
        >>> model = load_model("blip_image_text_matching", "large")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/blip_itm_base.yaml",
        "large": "configs/models/blip_pretrain_large.yaml",
        "coco": "configs/models/blip_itm_large.yaml",
    }

    def __init__(self, image_encoder, text_encoder, embed_dim=256, max_txt_len=35):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.text_encoder = text_encoder

        self.visual_encoder = image_encoder

        self.max_txt_len = max_txt_len

        # creating projection layers for ITC
        text_width = text_encoder.config.hidden_size
        vision_width = image_encoder.vision_width

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2)
        self.temp = nn.Parameter(0.07 * torch.ones([]))

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def forward(self, ref_images, tar_images, sentences, negative_images, use_temp):
        device = ref_images.device

        with self.maybe_autocast():
            ref_image_embedding = self.visual_encoder.forward_features(ref_images)

        ref_image_atts = torch.ones(ref_image_embedding.size()[:-1], dtype=torch.long).to(device)

        text = self.tokenizer(
            sentences,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id

        text_output = self.text_encoder(encoder_input_ids,
                                        attention_mask=text.attention_mask,
                                        encoder_hidden_states=ref_image_embedding,
                                        encoder_attention_mask=ref_image_atts,
                                        return_dict=True
                                        )

        projected_text = self.text_proj(text_output.last_hidden_state[:, 0, :])

        with self.maybe_autocast():
            tar_image_embedding = self.visual_encoder.forward_features(tar_images)
            negative_image_embedding = self.visual_encoder.forward_features(negative_images)
        tar_projected_image = self.vision_proj(tar_image_embedding[:, 0, :])
        negative_projected_image = self.vision_proj(negative_image_embedding[:, 0, :])

        if use_temp:
            projected_text = F.normalize(projected_text, p=2, dim=-1)
            tar_projected_image = F.normalize(tar_projected_image, p=2, dim=-1)
            negative_projected_image = F.normalize(negative_projected_image, p=2, dim=-1)

            score_1 = torch.sum(projected_text * tar_projected_image, dim=1) / self.temp
            score_2 = torch.sum(projected_text * negative_projected_image, dim=1) / self.temp

        else:
            score_1 = torch.sum(projected_text * tar_projected_image, dim=1)
            score_2 = torch.sum(projected_text * negative_projected_image, dim=1)

        scores = torch.stack([score_1, score_2], dim=1)
        return scores

    @torch.no_grad()
    def extract_target_features(self, dataloader, use_temp, device):
        feature_dim = 256
        index_features = torch.empty((0, feature_dim)).to(device, non_blocking=True)
        index_names = []

        for images, names in tqdm(dataloader, desc="Extract Sample Features"):
            images = images.to(device, non_blocking=True)

            with self.maybe_autocast():
                img_embedding = self.visual_encoder.forward_features(images)

            batch_features = self.vision_proj(img_embedding[:, 0, :])
            batch_features = F.normalize(batch_features, dim=-1)
            index_features = torch.vstack((index_features, batch_features))
            index_names.extend(names)
        return index_features.cpu(), index_names

    @torch.no_grad()
    def extract_query_features_fiq(self, dataloader, use_temp, txt_processors, device):

        feature_dim = 256

        # Initialize predicted features, target_names, group_members and reference_names
        predicted_features = torch.empty((0, feature_dim)).to(device, non_blocking=True)
        target_names = []

        for ref_images, sentences, target_name in tqdm(dataloader, desc="Extract Query Features"):  # Load data
            sentences = [txt_processors["eval"](caption) for caption in sentences]

            text_tokens = self.tokenizer(
                sentences,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)

            ref_images = ref_images.to(device, non_blocking=True)

            with self.maybe_autocast():
                ref_image_embedding = self.visual_encoder.forward_features(ref_images)
            ref_image_embedding_atts = torch.ones(ref_image_embedding.size()[:-1], dtype=torch.long).to(device)

            encoder_input_ids = text_tokens.input_ids.clone()
            encoder_input_ids[:, 0] = self.tokenizer.enc_token_id

            query_output = self.text_encoder(encoder_input_ids,
                                             attention_mask=text_tokens.attention_mask,
                                             encoder_hidden_states=ref_image_embedding,
                                             encoder_attention_mask=ref_image_embedding_atts,
                                             return_dict=True
                                             )

            query_features = self.text_proj(query_output.last_hidden_state[:, 0, :])
            query_features = F.normalize(query_features, dim=-1)

            predicted_features = torch.vstack((predicted_features, query_features))
            target_names.extend(target_name)

        return predicted_features.cpu(), target_names

    @torch.no_grad()
    def extract_query_features_cirr(self, dataloader, use_temp, txt_processors, device):

        feature_dim = 256

        # Initialize predicted features, target_names, group_members and reference_names
        predicted_features = torch.empty((0, feature_dim)).to(device, non_blocking=True)
        group_members = []
        reference_names = []
        pairs_ids = []

        for ref_names, ref_images, sentences, pair_id, batch_group_members in tqdm(dataloader,
                                                                                   desc="Extract Query Features"):  # Load data
            # text_inputs = clip.tokenize(captions).to(device, non_blocking=True)
            sentences = [txt_processors["eval"](caption) for caption in sentences]
            text_tokens = self.tokenizer(
                sentences,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)

            batch_group_members = np.array(batch_group_members).tolist()
            ref_images = ref_images.to(device, non_blocking=True)

            with self.maybe_autocast():
                ref_image_embedding = self.visual_encoder.forward_features(ref_images)
            ref_image_embedding_atts = torch.ones(ref_image_embedding.size()[:-1], dtype=torch.long).to(device)

            encoder_input_ids = text_tokens.input_ids.clone()
            encoder_input_ids[:, 0] = self.tokenizer.enc_token_id

            query_output = self.text_encoder(encoder_input_ids,
                                             attention_mask=text_tokens.attention_mask,
                                             encoder_hidden_states=ref_image_embedding,
                                             encoder_attention_mask=ref_image_embedding_atts,
                                             return_dict=True
                                             )

            query_features = self.text_proj(query_output.last_hidden_state[:, 0, :])
            query_features = F.normalize(query_features, dim=-1)

            predicted_features = torch.vstack((predicted_features, query_features))
            reference_names.extend(ref_names)
            group_members.extend(batch_group_members)
            pairs_ids.extend(pair_id)

        return predicted_features.cpu(), reference_names, group_members, pairs_ids

    @torch.no_grad()
    def extract_query_features_circo(self, dataloader, use_temp, txt_processors, device):

        feature_dim = 256

        # Initialize predicted features, target_names, group_members and reference_names
        predicted_features = torch.empty((0, feature_dim)).to(device, non_blocking=True)
        query_ids_list = []

        for batch in tqdm(dataloader, desc="Extract Query Features"):  # Load data
            ref_images = batch['reference_img']
            relative_captions = batch['relative_caption']
            query_ids = batch['query_id']

            sentences = [f"a photo of $ that {caption}" for caption in relative_captions]

            sentences = [txt_processors["eval"](caption) for caption in sentences]
            text_tokens = self.tokenizer(
                sentences,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)

            ref_images = ref_images.to(device, non_blocking=True)

            with self.maybe_autocast():
                ref_image_embedding = self.visual_encoder.forward_features(ref_images)
            ref_image_embedding_atts = torch.ones(ref_image_embedding.size()[:-1], dtype=torch.long).to(device)

            encoder_input_ids = text_tokens.input_ids.clone()
            encoder_input_ids[:, 0] = self.tokenizer.enc_token_id

            query_output = self.text_encoder(encoder_input_ids,
                                             attention_mask=text_tokens.attention_mask,
                                             encoder_hidden_states=ref_image_embedding,
                                             encoder_attention_mask=ref_image_embedding_atts,
                                             return_dict=True
                                             )

            query_features = self.text_proj(query_output.last_hidden_state[:, 0, :])
            query_features = F.normalize(query_features, dim=-1)

            predicted_features = torch.vstack((predicted_features, query_features))
            query_ids_list.extend(query_ids)

        return predicted_features.cpu(), query_ids_list

    @torch.no_grad()
    def extract_query_features_cirr_val(self, dataloader, use_temp, txt_processors, device):

        feature_dim = 256

        # Initialize predicted features, target_names, group_members and reference_names
        predicted_features = torch.empty((0, feature_dim)).to(device, non_blocking=True)
        group_members = []
        reference_names = []
        pairs_ids = []
        target_names = []

        for ref_names, ref_images, targ_names, sentences, pair_id, batch_group_members in tqdm(dataloader,
                                                                                               desc="Extract Query Features"):  # Load data
            # text_inputs = clip.tokenize(captions).to(device, non_blocking=True)
            sentences = [txt_processors["eval"](caption) for caption in sentences]
            text_tokens = self.tokenizer(
                sentences,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)

            batch_group_members = np.array(batch_group_members).tolist()
            ref_images = ref_images.to(device, non_blocking=True)

            with self.maybe_autocast():
                ref_image_embedding = self.visual_encoder.forward_features(ref_images)
            ref_image_embedding_atts = torch.ones(ref_image_embedding.size()[:-1], dtype=torch.long).to(device)

            encoder_input_ids = text_tokens.input_ids.clone()
            encoder_input_ids[:, 0] = self.tokenizer.enc_token_id

            query_output = self.text_encoder(encoder_input_ids,
                                             attention_mask=text_tokens.attention_mask,
                                             encoder_hidden_states=ref_image_embedding,
                                             encoder_attention_mask=ref_image_embedding_atts,
                                             return_dict=True
                                             )

            query_features = self.text_proj(query_output.last_hidden_state[:, 0, :])
            query_features = F.normalize(query_features, dim=-1)

            predicted_features = torch.vstack((predicted_features, query_features))
            reference_names.extend(ref_names)
            target_names.extend(targ_names)
            group_members.extend(batch_group_members)

        return predicted_features.cpu(), reference_names, target_names, group_members, pairs_ids

    @torch.no_grad()
    def score(self, query_features, target_features):
        # print(query_features.shape, target_features.shape)
        scores = torch.matmul(query_features, target_features.T)
        return scores

    @torch.no_grad()
    def extract_target_image_features_fiq(self, dataloader, device):

        feature_dim = 256
        index_features = torch.empty((0, feature_dim)).to(device, non_blocking=True)
        target_name_list = []
        for ref_images, sentences, target_names in dataloader:  # Load data
            target_images = []
            for target_name in target_names:
                target_image_path = '-' + '/images' + f"/{target_name}.png"
                target_image = preprocess(PIL.Image.open(target_image_path)).unsqueeze(0)
                target_images.append(target_image)
            target_image = torch.cat(target_images, dim=0)
            target_image = target_image.to(device, non_blocking=True)

            with self.maybe_autocast():
                target_image_embedding = self.visual_encoder.forward_features(target_image)

            batch_features = self.vision_proj(target_image_embedding[:, 0, :])
            batch_features = F.normalize(batch_features, dim=-1)
            index_features = torch.vstack((index_features, batch_features))
            target_name_list.extend(target_names)

        return index_features.cpu(), target_name_list

    def itm_rank(self, image_embeds, image_atts, encoder_input_ids, match_head='itm'):
        # breakpoint()
        encoder_input_ids = encoder_input_ids.clone()
        encoder_input_ids = encoder_input_ids[:, 3:]
        text_attention_mask = (encoder_input_ids != self.tokenizer.pad_token_id).long()

        if match_head == 'itm':
            # encoder_input_ids = encoder_input_ids.clone()
            encoder_input_ids[:, 0] = self.tokenizer.enc_token_id
            output = self.text_encoder(encoder_input_ids,
                                       attention_mask=text_attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       )
            # print(output.last_hidden_state.shape)
            itm_output = self.itm_head(output.last_hidden_state[:, 0, :])
            itm_output = F.softmax(itm_output, dim=1)[:, 1]
            return itm_output  # , mask, token_length

        elif match_head == 'itc':
            encoder_input_ids[:, 0] = self.tokenizer.cls_token_id
            text_output = self.text_encoder(encoder_input_ids, attention_mask=text_attention_mask,
                                            return_dict=True, mode='text')
            image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
            text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

            sim = image_feat @ text_feat.t()
            return sim

    @classmethod
    def from_config(cls, cfg=None):
        image_encoder = VisionTransformerEncoder.from_config(cfg)
        text_encoder = XBertEncoder.from_config(cfg)

        embed_dim = cfg.get("embed_dim", 256)
        max_txt_len = cfg.get("max_txt_len", 35)

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            embed_dim=embed_dim,
            max_txt_len=max_txt_len,
        )

        model.load_checkpoint_from_config(cfg)

        return model


def compute_gradcam(model, visual_input, text_input, tokenized_text, block_num=6):
    model.text_encoder.base_model.base_model.encoder.layer[
        block_num
    ].crossattention.self.save_attention = True

    output = model({"image": visual_input, "text_input": text_input}, match_head="itm")
    loss = output[:, 1].sum()

    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        mask = tokenized_text.attention_mask.view(
            tokenized_text.attention_mask.size(0), 1, -1, 1, 1
        )  # (bsz,1,token_len, 1,1)
        token_length = tokenized_text.attention_mask.sum(dim=-1) - 2
        token_length = token_length.cpu()
        # grads and cams [bsz, num_head, seq_len, image_patch]
        grads = model.text_encoder.base_model.base_model.encoder.layer[
            block_num
        ].crossattention.self.get_attn_gradients()
        cams = model.text_encoder.base_model.base_model.encoder.layer[
            block_num
        ].crossattention.self.get_attention_map()

        # assume using vit with 576 num image patch
        cams = cams[:, :, :, 1:].reshape(visual_input.size(0), 12, -1, 24, 24) * mask
        grads = (
                grads[:, :, :, 1:].clamp(0).reshape(visual_input.size(0), 12, -1, 24, 24)
                * mask
        )

        gradcams = cams * grads
        gradcam_list = []

        for ind in range(visual_input.size(0)):
            token_length_ = token_length[ind]
            gradcam = gradcams[ind].mean(0).cpu().detach()
            # [enc token gradcam, average gradcam across token, gradcam for individual token]
            gradcam = torch.cat(
                (
                    gradcam[0:1, :],
                    gradcam[1: token_length_ + 1, :].sum(dim=0, keepdim=True)
                    / token_length_,
                    gradcam[1:, :],
                )
            )
            gradcam_list.append(gradcam)

    return gradcam_list, output

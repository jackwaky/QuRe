"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
import numpy as np
import logging
from tqdm import tqdm
import PIL
import PIL.Image
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from data.utils import targetpad_transform

preprocess = targetpad_transform(1.25, 224)

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures


@registry.register_model("blip2_qure")
class Blip2QuRe(Blip2Base):
    """
    BLIP Image-Text Matching (ITM) model.
    Supported model types:
        - pretrained: pretrained model
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_qure", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            print("freeze vision encoder")
        else:
            print("train vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len

    def forward(self, ref_images, tar_images, sentences, negative_images, use_temp):
        device = ref_images.device
        batch_size = ref_images.shape[0]

        text_tokens = self.tokenizer(
            sentences,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

        # Query Embeddings
        with self.maybe_autocast():
            ref_image_embedding = self.ln_vision(self.visual_encoder(ref_images))
        ref_image_embedding = ref_image_embedding.float()
        ref_image_embedding_atts = torch.ones(ref_image_embedding.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.query_tokens.expand(ref_image_embedding.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)

        query_output = self.Qformer.bert(
                text_tokens.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=ref_image_embedding,
                encoder_attention_mask=ref_image_embedding_atts,
                return_dict=True,
            )

        query_embeds = query_output.last_hidden_state[:, : query_tokens.size(1), :]
        projected_query = self.text_proj(query_embeds).mean(dim=1)
        # projected_query = projected_query.mean(dim=1)

        # Target & Negative Embeddings
        with self.maybe_autocast():
            tar_image_embedding = self.ln_vision(self.visual_encoder(tar_images))
            neg_image_embedding = self.ln_vision(self.visual_encoder(negative_images))

        tar_image_embedding = tar_image_embedding.float()
        tar_image_atts = torch.ones(
            tar_image_embedding.size()[:-1], dtype=torch.long
        ).to(self.device)

        tar_query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=tar_image_embedding,
            encoder_attention_mask=tar_image_atts,
            return_dict=True,
        )
        tar_image_embeds = tar_query_output.last_hidden_state

        neg_image_embedding = neg_image_embedding.float()
        neg_image_atts = torch.ones(
            neg_image_embedding.size()[:-1], dtype=torch.long
        ).to(self.device)
        # neg_query_tokens = self.query_tokens.expand(
        #     neg_image_embedding.shape[0], -1, -1
        # )

        neg_query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=neg_image_embedding,
            encoder_attention_mask=neg_image_atts,
            return_dict=True,
        )
        neg_image_embeds = neg_query_output.last_hidden_state

        tar_projected_image = self.vision_proj(tar_image_embeds)
        neg_projected_image = self.vision_proj(neg_image_embeds)

        projected_query = F.normalize(projected_query, p=2, dim=-1)
        tar_projected_image = F.normalize(tar_projected_image, p=2, dim=-1)
        neg_projected_image = F.normalize(neg_projected_image, p=2, dim=-1)

        scores_1 = torch.bmm(tar_projected_image, projected_query.unsqueeze(-1))
        score_1, _ = torch.max(scores_1, dim=1)

        scores_2 = torch.bmm(neg_projected_image, projected_query.unsqueeze(-1))
        score_2, _ = torch.max(scores_2, dim=1)

        if use_temp:
            score_1 = score_1 / self.temp
            score_2 = score_2 / self.temp

        scores = torch.stack([score_1, score_2], dim=1).squeeze(-1)
        return scores

    @torch.no_grad()
    def extract_target_features(self, dataloader, use_temp, device):
        feature_dim = 256
        index_features = torch.empty((0, 32, feature_dim)).to(device, non_blocking=True)
        index_names = []

        for images, names in tqdm(dataloader, desc="Extract Sample Features"):
            images = images.to(device, non_blocking=True)

            with self.maybe_autocast():
                img_embedding = self.ln_vision(self.visual_encoder(images))

            img_embedding = img_embedding.float()
            img_atts = torch.ones(img_embedding.size()[:-1], dtype=torch.long).to(device)
            query_tokens = self.query_tokens.expand(img_embedding.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=img_embedding,
                encoder_attention_mask=img_atts,
                return_dict=True,
            )
            batch_features = self.vision_proj(query_output.last_hidden_state)
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
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)

            ref_images = ref_images.to(device, non_blocking=True)
            # modifiers = modifiers.to(device, non_blocking=True)
            # attn_mask = attn_mask.to(device, non_blocking=True)

            with self.maybe_autocast():
                ref_image_embedding = self.ln_vision(self.visual_encoder(ref_images))
            ref_image_embedding = ref_image_embedding.float()
            ref_image_embedding_atts = torch.ones(ref_image_embedding.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(ref_image_embedding.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
            attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)

            query_output = self.Qformer.bert(
                text_tokens.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=ref_image_embedding,
                encoder_attention_mask=ref_image_embedding_atts,
                return_dict=True,
            )

            query_embeds = query_output.last_hidden_state[:, : query_tokens.size(1), :]
            query_features = self.text_proj(query_embeds).mean(dim=1)
            # query_features = query_features.mean(dim=1)
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

        for ref_names, ref_images, sentences, pair_id, batch_group_members in tqdm(dataloader, desc="Extract Query Features"):  # Load data
            # text_inputs = clip.tokenize(captions).to(device, non_blocking=True)
            sentences = [txt_processors["eval"](caption) for caption in sentences]
            text_tokens = self.tokenizer(
                sentences,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)

            batch_group_members = np.array(batch_group_members).tolist()
            ref_images = ref_images.to(device, non_blocking=True)

            with self.maybe_autocast():
                ref_image_embedding = self.ln_vision(self.visual_encoder(ref_images))
            ref_image_embedding = ref_image_embedding.float()
            ref_image_embedding_atts = torch.ones(ref_image_embedding.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(ref_image_embedding.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
            attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)

            query_output = self.Qformer.bert(
                text_tokens.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=ref_image_embedding,
                encoder_attention_mask=ref_image_embedding_atts,
                return_dict=True,
            )

            query_embeds = query_output.last_hidden_state[:, : query_tokens.size(1), :]
            query_features = self.text_proj(query_embeds).mean(dim=1)
            # query_features = query_features.mean(dim=1)
            if use_temp:
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
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)

            ref_images = ref_images.to(device, non_blocking=True)

            with self.maybe_autocast():
                ref_image_embedding = self.ln_vision(self.visual_encoder(ref_images))
            ref_image_embedding = ref_image_embedding.float()
            ref_image_embedding_atts = torch.ones(ref_image_embedding.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(ref_image_embedding.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
            attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)

            query_output = self.Qformer.bert(
                text_tokens.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=ref_image_embedding,
                encoder_attention_mask=ref_image_embedding_atts,
                return_dict=True,
            )

            query_embeds = query_output.last_hidden_state[:, : query_tokens.size(1), :]
            query_features = self.text_proj(query_embeds).mean(dim=1)
            # query_features = query_features.mean(dim=1)
            if use_temp:
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
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)

            batch_group_members = np.array(batch_group_members).tolist()
            ref_images = ref_images.to(device, non_blocking=True)

            with self.maybe_autocast():
                ref_image_embedding = self.ln_vision(self.visual_encoder(ref_images))
            ref_image_embedding = ref_image_embedding.float()
            ref_image_embedding_atts = torch.ones(ref_image_embedding.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(ref_image_embedding.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
            attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)

            query_output = self.Qformer.bert(
                text_tokens.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=ref_image_embedding,
                encoder_attention_mask=ref_image_embedding_atts,
                return_dict=True,
            )

            query_embeds = query_output.last_hidden_state[:, : query_tokens.size(1), :]
            query_features = self.text_proj(query_embeds).mean(dim=1)
            # query_features = query_features.mean(dim=1)
            if use_temp:
                query_features = F.normalize(query_features, dim=-1)

            predicted_features = torch.vstack((predicted_features, query_features))
            reference_names.extend(ref_names)
            target_names.extend(targ_names)
            group_members.extend(batch_group_members)

        return predicted_features.cpu(), reference_names, target_names, group_members, pairs_ids

    @torch.no_grad()
    def score(self, query_features, target_features):
        scoress = torch.einsum('ik,jlk->ijl', query_features, target_features)
        scores, _ = scoress.max(-1)
        # scores = scores / self.temp.cpu()
        return scores

    @torch.no_grad()
    def extract_target_image_features_fiq(self, dataloader, device):

        feature_dim = 256
        index_features = torch.empty((0, 32, feature_dim)).to(device, non_blocking=True)
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
                target_image_embedding = self.ln_vision(self.visual_encoder(target_image))

            target_image_embedding = target_image_embedding.float()
            target_image_atts = torch.ones(target_image_embedding.size()[:-1], dtype=torch.long).to(device)
            query_tokens = self.query_tokens.expand(target_image_embedding.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=target_image_embedding,
                encoder_attention_mask=target_image_atts,
                return_dict=True,
            )
            batch_features = self.vision_proj(query_output.last_hidden_state)
            batch_features = F.normalize(batch_features, dim=-1)
            index_features = torch.vstack((index_features, batch_features))
            target_name_list.extend(target_names)

        return index_features.cpu(), target_name_list

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model









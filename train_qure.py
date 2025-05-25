from options import get_experiment_config
from set_up import setup_experiment
from transforms import image_transform_factory
from data import create_dataloaders
from models import create_qure_models
import wandb


import math
from tqdm import tqdm
import torch
import torch.nn.functional as F
import os

from evaluate_qure.eval_utils import eval_fiq, eval_cirr


def main():
    configs = get_experiment_config()
    export_root, configs = setup_experiment(configs)
    device = torch.device(f"cuda:{configs['device_idx']}") if torch.cuda.is_available() else "cpu"

    model, txt_processors = create_qure_models(configs, device)

    image_transform = image_transform_factory(config=configs)
    train_dataloader, test_dataloaders = create_dataloaders(image_transform, None, configs)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=configs['init_lr'],
                                  weight_decay=configs['weight_decay'])

    max_epochs = configs['epoch']

    model.train()
    scaler =  torch.cuda.amp.GradScaler()
    for epoch in range(max_epochs):
        model.train()
        if configs["use_rerank"]:
            if (epoch % (max_epochs // configs["negative_definition_epoch_num"])) == 0:
                if not configs['rerank_warmup'] or (epoch != 0 and configs['rerank_warmup']):
                    train_dataloader.dataset.rerank_score(model, device, txt_processors, configs, epoch)


        epoch_running_loss = 0.0
        cosine_lr_schedule(optimizer, epoch, max_epochs, configs["init_lr"], configs["init_lr"]/100)
        train_dataloader_tqdm = tqdm(train_dataloader, desc="Epoch {}".format(epoch+1))
        for batch_idx, (ref_images, tar_images, negative_images, sentences) in enumerate(train_dataloader_tqdm):
            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                ref_images, tar_images, negative_images = ref_images.to(device), tar_images.to(
                    device), negative_images.to(device)

                sentences = [txt_processors["eval"](caption) for caption in sentences]
                scores = model(ref_images, tar_images, sentences, negative_images, configs['use_temp'])

                log_softmax_scores = F.log_softmax(scores, dim=1)  

                target_probs = torch.zeros_like(log_softmax_scores)
                target_probs[:, 0] = 1.0  # p0 = 1 for each data point

                loss = F.kl_div(log_softmax_scores, target_probs, reduction='batchmean')

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_running_loss += loss.item()
            cur_loss = epoch_running_loss / (batch_idx + 1)
            train_dataloader_tqdm.set_postfix({'loss': cur_loss})

        train_results = {'loss' : cur_loss, 'lr' : get_current_lr(optimizer)}
        print(f"[EPOCH {epoch+1}/{max_epochs}] Loss : {train_results['loss']} lr : {train_results['lr']}")
        if configs["experiment_description"] != 'debug':
            wandb.log({"loss" : cur_loss, "epoch" : epoch})
            wandb.log({'lr' : get_current_lr(optimizer), "epoch": epoch})
            wandb.log({'k': topk, "epoch": epoch})
            if hasattr(model, 'temp'):
                wandb.log({'temp': model.temp.item(), "epoch": epoch})

            if hasattr(model, 'logit_scale'):
                wandb.log({'temp': model.logit_scale.item(), "epoch": epoch})


        # Save the model for every 10 epochs
        # todo: adding evaluation
        if (epoch + 1) % 1 == 0:
            #Save the model
            saving_path = f""
            os.makedirs(saving_path, exist_ok=True)
            torch.save(model.state_dict(), f'{saving_path}/model.pth')
            append_log(f'{saving_path}/log.txt', f"Epoch : {epoch + 1}\n")

        eval_frequency = 1
        if (epoch + 1) % eval_frequency == 0:
            if configs['dataset'] == 'fashionIQ':
                eval_fiq(model, txt_processors, test_dataloaders, epoch, configs)

            elif configs['dataset'] == 'cirr':
                eval_cirr(model, txt_processors, test_dataloaders, epoch, configs)

def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def append_log(file_path, log_message):
    with open(file_path, 'a') as file:
        file.write(log_message + '\n')

if __name__ == '__main__':
    main()
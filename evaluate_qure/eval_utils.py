from statistics import mean
import numpy as np
import torch
import wandb

def eval_fiq(model, txt_processors, test_dataloaders, epoch, configs):
    model.eval()
    device = model.device
    recalls_at10 = []
    recalls_at50 = []
    results_dict = dict()

    for cloth_type, cur_test_dataloader in test_dataloaders.items():
        cur_test_samples_dataloader = cur_test_dataloader['samples']
        cur_test_query_dataloader = cur_test_dataloader['query']

        index_features, index_names = model.extract_target_features(cur_test_samples_dataloader,
                                                                    configs['use_temp'], device)
        predicted_features, target_names = model.extract_query_features_fiq(
            cur_test_query_dataloader, configs['use_temp'], txt_processors, device)

        scores = model.score(predicted_features, index_features)

        sorted_indices = torch.argsort(scores, dim=-1, descending=True).cpu()
        sorted_index_names = np.array(index_names)[sorted_indices]

        # Compute the ground-truth labels wrt the predictions
        labels = torch.tensor(
            sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names),
                                                                                              -1))
        assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

        # Compute the metrics
        recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
        recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

        recalls_at10.append(recall_at10)
        recalls_at50.append(recall_at50)

        results_dict[f"{cloth_type}_recall@10"] = recall_at10
        results_dict[f"{cloth_type}_recall@50"] = recall_at50

        wandb.log({f"{cloth_type}_recall@10": recall_at10, "epoch": epoch})
        wandb.log({f"{cloth_type}_recall@50": recall_at50, "epoch": epoch})
        # torch.cuda.empty_cache()

    average_recall_at10 = mean(recalls_at10)
    average_recall_at50 = mean(recalls_at50)
    average_recall = (mean(recalls_at50) + mean(recalls_at10)) / 2

    wandb.log({f"average_recall_at10": average_recall_at10, "epoch": epoch})
    wandb.log({f"average_recall_at50": average_recall_at50, "epoch": epoch})
    wandb.log({f"average_recall": average_recall, "epoch": epoch})


def eval_cirr(model, txt_processors, test_dataloaders, epoch, configs):
    model.eval()
    device = model.device

    for cloth_type, cur_test_dataloader in test_dataloaders.items():
        cur_test_samples_dataloader = cur_test_dataloader['samples']
        cur_test_query_dataloader = cur_test_dataloader['query']

        index_features, index_names = model.extract_target_features(cur_test_samples_dataloader,
                                                                    configs['use_temp'], device)
        predicted_features, reference_names, target_names, group_members, pairs_id = model.extract_query_features_cirr_val(
            cur_test_query_dataloader, configs['use_temp'], txt_processors, device)

        scores = model.score(predicted_features, index_features)

        sorted_indices = torch.argsort(scores, dim=-1, descending=True).cpu()
        sorted_index_names = np.array(index_names)[sorted_indices]

        # Delete the reference image from the results
        reference_mask = torch.tensor(
            sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(
                len(sorted_index_names), -1))
        sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                        sorted_index_names.shape[1] - 1)

        labels = torch.tensor(
            sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(
                len(target_names), -1))

        # Compute the subset predictions and ground-truth labels
        group_members = np.array(group_members)
        group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
        group_labels = labels[group_mask].reshape(labels.shape[0], -1)

        assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
        assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

        # Compute the metrics
        recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
        recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
        recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
        recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
        group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
        group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
        group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

        wandb.log({f"recall_at1": recall_at1, "epoch": epoch})
        wandb.log({f"recall_at5": recall_at5, "epoch": epoch})
        wandb.log({f"recall_at10": recall_at10, "epoch": epoch})
        wandb.log({f"recall_at50": recall_at50, "epoch": epoch})
        wandb.log({f"group_recall_at1": group_recall_at1, "epoch": epoch})
        wandb.log({f"group_recall_at2": group_recall_at2, "epoch": epoch})
        wandb.log({f"group_recall_at3": group_recall_at3, "epoch": epoch})
        wandb.log({f"mean(R@5+R_s@1)": (group_recall_at1 + recall_at5) / 2, "epoch": epoch})
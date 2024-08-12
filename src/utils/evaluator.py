from tqdm import tqdm
from src.utils.metrics import metric, precision_recall_f1score, det_error_metric
import torch
import numpy as np


@torch.no_grad()
def evaluator(contact_labels_3d_pred, contact_labels_3d_gt, normalize=True, return_dict=False):
    #total_epochs = hparams.TRAINING.NUM_EPOCHS

    contact_labels_3d_pred = torch.stack(contact_labels_3d_pred).view(-1, 6890)
    contact_labels_3d_gt = torch.stack(contact_labels_3d_gt).view(-1, 6890).cuda()
    cont_pre, cont_rec, cont_f1 = precision_recall_f1score(contact_labels_3d_gt, contact_labels_3d_pred)
    fp_geo_err, fn_geo_err = det_error_metric(contact_labels_3d_pred, contact_labels_3d_gt)

    return cont_pre.mean(), cont_rec.mean(), cont_f1.mean(), fp_geo_err.mean(), fn_geo_err.mean()
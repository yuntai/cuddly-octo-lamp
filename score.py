import math
import pandas as pd
import torch
import torch.nn.functional as F
from pyts.metrics import dtw
import numpy as np
from tqdm import tqdm
from scipy import stats

def unroll(vals):
    f = vals.fliplr()
    for i in range(f.shape[1]-1, -f.shape[0], -1):
        yield f.diagonal(i)

def _dtw_error(y, y_hat, score_window=10):
    length_dtw = (score_window // 2) * 2 + 1
    half_length_dtw = length_dtw // 2

    # add padding
    y_pad = F.pad(y, (half_length_dtw, half_length_dtw),  mode='constant', value=0)
    y_hat_pad = F.pad(y_hat, (half_length_dtw, half_length_dtw),  mode='constant', value=0)

    similarity_dtw = []
    for i in tqdm(range(len(y) - length_dtw)):
        gt = y_pad[i:i+length_dtw].flatten()
        pred = y_hat_pad[i:i+length_dtw].flatten()
        similarity_dtw.append(dtw(gt, pred))

    errors = ([0] * half_length_dtw + similarity_dtw + [0] * (len(y) - len(similarity_dtw) - half_length_dtw))

    return torch.tensor(errors)

def reconstruction_errors(y, y_hat, step_size=1, score_window=10, smoothing_window=0.01,
                          smooth=True, rec_error_type='point'):
    if isinstance(smoothing_window, float):
        smoothing_window = min(math.trunc(len(y) * smoothing_window), 200)

    predictions = []
    predictions_vs = []
    for entries in unroll(y_hat):
        predictions.append(entries.median())
        predictions_vs.append(entries.quantile(torch.tensor([0., 0.25, 0.5, 0.75, 1.])))

    predictions = torch.tensor(predictions)
    predictions_vs = torch.stack(predictions_vs)

    gt = torch.cat([y[:, 0, 0], y[-1,1:,0]])

    # Compute reconstruction errors
    if rec_error_type.lower() == "point":
        errors = _point_wise_error(gt, predictions)

    elif rec_error_type.lower() == "area":
        errors = _area_error(gt, predictions, score_window)

    elif rec_error_type.lower() == "dtw":
        errors = _dtw_error(gt, predictions, score_window)

    # Apply smoothing
    if smooth:
        errors = pd.Series(errors).rolling(
            smoothing_window, center=True, min_periods=smoothing_window // 2).mean().values

    return errors, predictions_vs

def unroll_kde_max(vals):
    kde_max = []
    for entries in unroll(vals):
        if len(entries) > 1:
            try:
                kernel = stats.gaussian_kde(entries)
                ix = torch.argmax(torch.tensor(kernel(entries)))
                kde_max.append(entries[ix])
            except np.linalg.LinAlgError:
                kde_max.append(entries.quantile(0.5))
        else:
            kde_max.append(entries.quantile(0.5))
    return torch.stack(kde_max)

def _compute_critic_score(critics, smooth_window):
    l_quantile = critics.quantile(0.25)
    u_quantile = critics.quantile(0.75)
    in_range = torch.logical_and(critics >= l_quantile, critics <= u_quantile)
    critic_mean = critics[in_range].mean()
    critic_std = critics.std()

    z_scores = (critics - critic_mean).abs() / critic_std + 1
    #z_scores = np.absolute(np.asarray(critics) - critic_mean) / critic_std) + 1
    z_scores = pd.Series(z_scores).rolling(
        smooth_window, center=True, min_periods=smooth_window // 2).mean().values

    return z_scores

def score_anomalies(y, y_hat, critic, score_window=10, critic_smooth_window=None,
                    error_smooth_window=None, smooth=True, rec_error_type="point", comb="mult",
                    lambda_rec=0.5):

    critic_smooth_window = critic_smooth_window or math.trunc(y.shape[0] * 0.01)
    error_smooth_window = error_smooth_window or math.trunc(y.shape[0] * 0.01)

    pred_length = y_hat.shape[1]
    step_size = 1  # expected to be 1

    gt = np.concatenate([y[:, 0, 0], y[-1,1:,0]])
    critic_extended = np.repeat(critic, pred_length, axis=-1)

    critic_kde_max = unroll_kde_max(critic_extended)

    # Compute critic scores
    critic_scores = _compute_critic_score(critic_kde_max, critic_smooth_window)

    # Compute reconstruction scores
    rec_scores, predictions = reconstruction_errors(
        y, y_hat, step_size, score_window, error_smooth_window, smooth, rec_error_type)

    rec_scores = stats.zscore(rec_scores)
    rec_scores = np.clip(rec_scores, a_min=0, a_max=None) + 1

    # Combine the two scores
    if comb == "mult":
        final_scores = np.multiply(critic_scores, rec_scores)

    elif comb == "sum":
        final_scores = (1 - lambda_rec) * (critic_scores - 1) + lambda_rec * (rec_scores - 1)

    elif comb == "rec":
        final_scores = rec_scores

    else:
        raise ValueError(
            'Unknown combination specified {}, use "mult", "sum", or "rec" instead.'.format(comb))

    return final_scores, gt, predictions

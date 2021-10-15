import math
import pandas as pd
import torch
import torch.nn.functional as F
from pyts.metrics import dtw
import numpy as np
from tqdm import tqdm
from scipy import stats
from scipy import integrate

def _area_error(y, y_hat, score_window=10):
    smooth_y = pd.DataFrame(y).rolling(
        score_window, center=True, min_periods=score_window // 2).apply(integrate.trapz)
    smooth_y_hat = pd.DataFrame(y_hat).rolling(
        score_window, center=True, min_periods=score_window // 2).apply(integrate.trapz)
    e = torch.tensor((smooth_y - smooth_y_hat).values)

    errors = e.abs().mean(dim=-1)

    return errors

def _point_wise_error(y, y_hat):
    return (y - y_hat).abs().mean(dim=-1)

def find_attack_intervals(attacks):
	ints = []
	st = 0
	for ix, a in attacks.iteritems():
		if st == 0 and a == 1:
			assert len(ints) == 0 or len(ints[-1]) == 2
			ints.append([ix])
			st = 1
		elif st == 1 and a == 0:
			assert len(ints) > 0 and len(ints[-1]) == 1
			ints[-1].append(ix-1)
			st = 0
		elif st == 1 and a == 1:
			assert len(ints) > 0 and len(ints[-1]) == 1
		elif st == 0 and a == 0:
			assert len(ints) == 0 or len(ints[-1]) == 2
		else:
			1/0
	return [(s,e) for s, e in ints]

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

def reconstruction_errors(gt, preds, step_size=1, score_window=10, smoothing_window=0.01,
                          smooth=True, rec_error_type='point'):
    if isinstance(smoothing_window, float):
        smoothing_window = min(math.trunc(gt.shape[0] * smoothing_window), 200)

    # compute reconstruction errors
    if rec_error_type.lower() == "point":
        errors = _point_wise_error(gt, preds)

    elif rec_error_type.lower() == "area":
        errors = _area_error(gt, preds, score_window)

    elif rec_error_type.lower() == "dtw":
        errors = _dtw_error(gt, preds, score_window)

    # apply smoothing
    if smooth:
        errors = pd.Series(errors).rolling(
            smoothing_window, center=True, min_periods=smoothing_window // 2).mean().values

    return errors

def unroll_kde_max(vals):
    kde_max = []
    for entries in unroll(vals):
        if len(entries) > 1:
            try:
                _kernel = stats.gaussian_kde(entries)
                ix = torch.argmax(torch.tensor(_kernel(entries)))
                kde_max.append(entries[ix])
            except np.linalg.LinAlgError:
                kde_max.append(entries.quantile(0.5))
        else:
            kde_max.append(entries.quantile(0.5))
    return torch.stack(kde_max)

def compute_critic_score(wnd_size, critic_score, smooth_window):
    critic_extended = critic_score.repeat(1, wnd_size)
    critic_kde_max = unroll_kde_max(critic_extended)

    l_q = critic_score.quantile(0.25)
    u_q = critic_score.quantile(0.75)
    in_range = torch.logical_and(critic_kde_max >= l_q, critic_kde_max <= u_q)
    critic_mean = critic_kde_max[in_range].mean()
    critic_std = critic_kde_max.std()

    z_scores = (critic_kde_max - critic_mean).abs() / critic_std + 1
    z_scores = pd.Series(z_scores).rolling(
        smooth_window, center=True, min_periods=smooth_window // 2).mean().values

    return torch.tensor(z_scores)

def unroll_predictions(x_hat):
    preds = []
    for entries in unroll(x_hat):
        preds.append(entries.quantile(torch.tensor([0., 0.25, 0.5, 0.75, 1.]), dim=-1).transpose(0,1))

    preds = torch.stack(preds)

    return preds

def score_anomalies(gt, pred, critic_score, wnd_size, score_window=10, critic_smooth_window=None,
                    error_smooth_window=None, smooth=True, rec_error_type="point", comb="mult",
                    lambda_rec=0.5):

    critic_smooth_window = critic_smooth_window or math.trunc(gt.shape[0] * 0.01)
    error_smooth_window = error_smooth_window or math.trunc(gt.shape[0] * 0.01)

    step_size = 1  # expected to be 1

    # Compute critic scores
    critic_zscore = compute_critic_score(wnd_size, critic_score, critic_smooth_window)

    # Compute reconstruction scores
    rec_errors = reconstruction_errors(
        gt, pred, step_size, score_window, error_smooth_window, smooth, rec_error_type)

    rec_score = stats.zscore(rec_errors)
    rec_zscore = np.clip(rec_score, a_min=0, a_max=None) + 1

    assert comb in ["mult", "sum", "rec"]
    # combine the two scores
    if comb == "mult":
        final_score = np.multiply(critic_zscore, rec_zscore)

    elif comb == "sum":
        final_score = (1 - lambda_rec) * (critic_zscore - 1) + lambda_rec * (rec_zscore - 1)

    elif comb == "rec":
        final_score = rec_zscore

    return final_score, critic_zscore, rec_zscore

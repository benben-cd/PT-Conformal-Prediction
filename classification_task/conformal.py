import numpy as np
from scipy.special import softmax
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
import pandas as pd
import random
import time
from utils import validate, get_logits_targets, sort_sum



class ConformalModel(nn.Module):
    def __init__(self, model, calib_loader, alpha, CP_method, PCP_prob=None, kreg=None, lamda=None, randomized=True,
                 allow_zero_sets=False, pct_paramtune=0.3, batch_size=32, lamda_criterion='size', index=0, bias=0):
        super(ConformalModel, self).__init__()
        self.model = model
        self.alpha = alpha
        self.T = torch.Tensor([1.3])  # initialize (1.3 is usually a good value)
        self.T, calib_logits = platt(self, calib_loader)
        self.randomized = randomized
        self.allow_zero_sets = allow_zero_sets
        self.num_classes = len(calib_loader.dataset.dataset.classes)
        self.CP_method = CP_method
        self.PCP_prob = PCP_prob
        self.index = index   # index is the position of the class to be added a bias
        self.bias = bias  # bias is the value to be added to the logits of the class at position index

        if kreg == None or lamda == None:
            kreg, lamda, calib_logits = pick_parameters(model, calib_logits, alpha, kreg, lamda, randomized,
                                                        allow_zero_sets, pct_paramtune, batch_size, lamda_criterion)

        self.penalties = np.zeros((1, self.num_classes))
        self.penalties[:, kreg:] += lamda
        self.kreg = kreg
        self.lamda = lamda

        calib_loader = tdata.DataLoader(calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True)

        self.Qhat, self.E = conformal_calibration(self, calib_loader)

    def forward(self, *args, randomized=None, allow_zero_sets=None, **kwargs):
        if randomized == None:
            randomized = self.randomized
        if allow_zero_sets == None:
            allow_zero_sets = self.allow_zero_sets
        logits = self.model(*args, **kwargs)

        with torch.no_grad():
            logits_numpy = logits.detach().cpu().numpy()
            Adjust_array = np.zeros(len(logits_numpy[1]))
            Adjust_array[self.index] = self.bias
            logits_numpy = logits_numpy + Adjust_array
            scores = softmax(logits_numpy / self.T.item(), axis=1)

            I, ordered, cumsum = sort_sum(scores)

            S = gcq(scores, self.Qhat, I=I, ordered=ordered, cumsum=cumsum, penalties=self.penalties,
                    randomized=randomized, allow_zero_sets=allow_zero_sets, CP_method=self.CP_method,
                    PCP_prob=self.PCP_prob)

        return torch.from_numpy(logits_numpy).to('cuda'), S


def conformal_calibration(cmodel, calib_loader):
    with torch.no_grad():
        E = np.array([])
        for logits, targets in calib_loader:
            Adjust_array = np.zeros(len(logits[1]))
            Adjust_array[cmodel.index] = cmodel.bias
            logits = logits.detach().cpu().numpy() + Adjust_array

            scores = softmax(logits / cmodel.T.item(), axis=1)

            I, ordered, cumsum = sort_sum(scores)

            E = np.concatenate((E, giq(scores, targets, I=I, ordered=ordered, cumsum=cumsum, penalties=cmodel.penalties,
                                       randomized=True, allow_zero_sets=False)))

        if cmodel.CP_method == 'VCP':
            Qhat = np.quantile(E, 1 - cmodel.alpha, interpolation='higher')
        elif cmodel.CP_method == 'PCP':
            Qhat = np.quantile(E, (1 - cmodel.alpha) / cmodel.PCP_prob, interpolation='higher')

        return Qhat, E


def conformal_calibration_logits(cmodel, calib_loader):
    with torch.no_grad():
        E = np.array([])
        for logits, targets in calib_loader:
            Adjust_array = np.zeros(len(logits[1]))
            Adjust_array[cmodel.index] = cmodel.bias
            logits = logits.detach().cpu().numpy() + Adjust_array

            scores = softmax(logits / cmodel.T.item(), axis=1)

            I, ordered, cumsum = sort_sum(scores)

            E = np.concatenate((E, giq(scores, targets, I=I, ordered=ordered, cumsum=cumsum, penalties=cmodel.penalties,
                                       randomized=True, allow_zero_sets=True)))
            Qhat = np.quantile(E, 1 - cmodel.alpha, interpolation='higher')

        return Qhat


def platt(cmodel, calib_loader, max_iters=10, lr=0.01, epsilon=0.01):
    print("Begin Platt scaling.")
    # Save logits so don't need to double compute them
    logits_dataset = get_logits_targets(cmodel.model, calib_loader)
    logits_loader = torch.utils.data.DataLoader(logits_dataset, batch_size=calib_loader.batch_size, shuffle=False,
                                                pin_memory=True)

    T = platt_logits(cmodel, logits_loader, max_iters=max_iters, lr=lr, epsilon=epsilon)

    print(f"Optimal T={T.item()}")
    return T, logits_dataset


def platt_logits(cmodel, calib_loader, max_iters=10, lr=0.01, epsilon=0.01):
    nll_criterion = nn.CrossEntropyLoss().cuda()

    T = nn.Parameter(torch.Tensor([1.3]).cuda())

    optimizer = optim.SGD([T], lr=lr)
    for iter in range(max_iters):
        T_old = T.item()
        for x, targets in calib_loader:
            optimizer.zero_grad()
            x = x.cuda()
            x.requires_grad = True
            out = x / T
            loss = nll_criterion(out, targets.long().cuda())
            loss.backward()
            optimizer.step()
        if abs(T_old - T.item()) < epsilon:
            break
    return T


def get_tau(score, target, I, ordered, cumsum, penalty, randomized, allow_zero_sets):  # For one example
    idx = np.where(I == target)
    tau_nonrandom = cumsum[idx]

    if not randomized:
        return tau_nonrandom + penalty[0]

    U = np.random.random()

    if idx == (0, 0):
        if not allow_zero_sets:
            return tau_nonrandom + penalty[0]
        else:
            return U * tau_nonrandom + penalty[0]
    else:
        return U * ordered[idx] + cumsum[(idx[0], idx[1] - 1)] + (penalty[0:(idx[1][0] + 1)]).sum()


def giq(scores, targets, I, ordered, cumsum, penalties, randomized, allow_zero_sets):
    """
        Generalized inverse quantile conformity score function.
        E from equation (7) in Romano, Sesia, Candes.  Find the minimum tau in [0, 1] such that the correct label enters.
    """
    E = -np.ones((scores.shape[0],))
    for i in range(scores.shape[0]):
        E[i] = get_tau(scores[i:i + 1, :], targets[i].item(), I[i:i + 1, :], ordered[i:i + 1, :], cumsum[i:i + 1, :],
                       penalties[0, :], randomized=randomized, allow_zero_sets=allow_zero_sets)

    return E


def gcq(scores, tau, I, ordered, cumsum, penalties, randomized, allow_zero_sets, CP_method, PCP_prob):
    penalties_cumsum = np.cumsum(penalties, axis=1)
    sizes_base = ((cumsum + penalties_cumsum) <= tau).sum(axis=1) + 1  # 1 - 1001
    sizes_base = np.minimum(sizes_base, scores.shape[1])  # 1-1000  size_base is an array of length equal to the number of samples in scores

    if randomized:
        V = np.zeros(sizes_base.shape)
        for i in range(sizes_base.shape[0]):
            V[i] = 1 / ordered[i, sizes_base[i] - 1] * \
                   (tau - (cumsum[i, sizes_base[i] - 1] - ordered[i, sizes_base[i] - 1]) - penalties_cumsum[
                       0, sizes_base[i] - 1])  # -1 since sizes_base \in {1,...,1000}.

        sizes = sizes_base - (np.random.random(V.shape) >= V).astype(int)
    else:
        sizes = sizes_base

    if not allow_zero_sets:
        sizes[
            sizes == 0] = 1  # allow the user the option to never have empty sets (will lead to incorrect coverage if 1-alpha < model's top-1 accuracy

    S = list()

    # Construct S from equation (5)

    for i in range(I.shape[0]):
        if CP_method == 'VCP':
            S = S + [I[i, 0:sizes[i]], ]
        elif CP_method == 'PCP':
            flag = random.random()
            if flag <= PCP_prob:
                S = S + [I[i, 0:sizes[i]], ]
            else:
                S = S + [np.array([])]
    return S


def pick_parameters(model, calib_logits, alpha, kreg, lamda, randomized, allow_zero_sets, pct_paramtune, batch_size,
                    lamda_criterion):
    num_paramtune = int(np.ceil(pct_paramtune * len(calib_logits)))
    paramtune_logits, calib_logits = tdata.random_split(calib_logits,
                                                        [num_paramtune, len(calib_logits) - num_paramtune])
    calib_loader = tdata.DataLoader(calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True)
    paramtune_loader = tdata.DataLoader(paramtune_logits, batch_size=batch_size, shuffle=False, pin_memory=True)

    if kreg == None:
        kreg = pick_kreg(paramtune_logits, alpha)
    if lamda == None:
        if lamda_criterion == "size":
            lamda = pick_lamda_size(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets)
        elif lamda_criterion == "adaptiveness":
            lamda = pick_lamda_adaptiveness(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets)
    return kreg, lamda, calib_logits


def pick_kreg(paramtune_logits, alpha):
    gt_locs_kstar = np.array([np.where(np.argsort(x[0]).flip(dims=(0,)) == x[1])[0][0] for x in paramtune_logits])
    kstar = np.quantile(gt_locs_kstar, 1 - alpha, interpolation='higher') + 1
    return kstar


def pick_lamda_size(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets):
    # Calculate lamda_star
    best_size = iter(paramtune_loader).__next__()[0][1].shape[0]  # number of classes
    # Use the paramtune data to pick lamda.  Does not violate exchangeability.
    for temp_lam in [0.001, 0.01, 0.1, 0.2, 0.5]:  # predefined grid, change if more precision desired.
        conformal_model = ConformalModelLogits(model, paramtune_loader, alpha=alpha, kreg=kreg, lamda=temp_lam,
                                               randomized=randomized, allow_zero_sets=allow_zero_sets, naive=False)
        top1_avg, top5_avg, cvg_avg, sz_avg, size_list = validate(paramtune_loader, conformal_model, print_bool=False)
        if sz_avg < best_size:
            best_size = sz_avg
            lamda_star = temp_lam
    return lamda_star


def pick_lamda_adaptiveness(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets,
                            strata=[[0, 1], [2, 3], [4, 6], [7, 10], [11, 100], [101, 1000]]):
    # Calculate lamda_star
    lamda_star = 0
    best_violation = 1
    # Use the paramtune data to pick lamda.  Does not violate exchangeability.
    for temp_lam in [0, 1e-5, 1e-4, 8e-4, 9e-4, 1e-3, 1.5e-3,
                     2e-3]:  # predefined grid, change if more precision desired.
        conformal_model = ConformalModelLogits(model, paramtune_loader, alpha=alpha, kreg=kreg, lamda=temp_lam,
                                               randomized=randomized, allow_zero_sets=allow_zero_sets, naive=False)
        curr_violation = get_violation(conformal_model, paramtune_loader, strata, alpha)
        if curr_violation < best_violation:
            best_violation = curr_violation
            lamda_star = temp_lam
    return lamda_star


class ConformalModelLogits(nn.Module):
    def __init__(self, model, calib_loader, alpha, kreg=None, lamda=None, randomized=True, allow_zero_sets=False,
                 naive=False, LAC=False, pct_paramtune=0.3, batch_size=32, lamda_criterion='size', index=0, bias=0):
        super(ConformalModelLogits, self).__init__()
        self.model = model
        self.alpha = alpha
        self.randomized = randomized
        self.LAC = LAC
        self.allow_zero_sets = allow_zero_sets
        self.T = platt_logits(self, calib_loader)
        self.index = index
        self.bias = bias

        if (kreg == None or lamda == None) and not naive and not LAC:
            kreg, lamda, calib_logits = pick_parameters(model, calib_loader.dataset, alpha, kreg, lamda, randomized,
                                                        allow_zero_sets, pct_paramtune, batch_size, lamda_criterion)
            calib_loader = tdata.DataLoader(calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True)

        self.penalties = np.zeros((1, calib_loader.dataset[0][0].shape[0]))
        if not (kreg == None) and not naive and not LAC:
            self.penalties[:, kreg:] += lamda
        self.Qhat = 1 - alpha
        if not naive and not LAC:
            self.Qhat = conformal_calibration_logits(self, calib_loader)
        elif not naive and LAC:
            gt_locs_cal = np.array(
                [np.where(np.argsort(x[0]).flip(dims=(0,)) == x[1])[0][0] for x in calib_loader.dataset])
            scores_cal = 1 - np.array(
                [np.sort(torch.softmax(calib_loader.dataset[i][0] / self.T.item(), dim=0))[::-1][gt_locs_cal[i]] for i
                 in range(len(calib_loader.dataset))])
            self.Qhat = np.quantile(scores_cal, np.ceil((scores_cal.shape[0] + 1) * (1 - alpha)) / scores_cal.shape[0])

    def forward(self, logits, randomized=None, allow_zero_sets=None):
        if randomized == None:
            randomized = self.randomized
        if allow_zero_sets == None:
            allow_zero_sets = self.allow_zero_sets

        with torch.no_grad():
            logits_numpy = logits.detach().cpu().numpy()
            Adjust_array = np.zeros(len(logits[1]))
            Adjust_array[self.index] = self.bias
            logits = logits.detach().cpu().numpy() + Adjust_array
            scores = softmax(logits_numpy / self.T.item(), axis=1)

            if not self.LAC:
                I, ordered, cumsum = sort_sum(scores)

                S = gcq_logits(scores, self.Qhat, I=I, ordered=ordered, cumsum=cumsum, penalties=self.penalties,
                               randomized=randomized, allow_zero_sets=allow_zero_sets)
            else:
                S = [np.where((1 - scores[i, :]) < self.Qhat)[0] for i in range(scores.shape[0])]

        return torch.from_numpy(logits_numpy).to('cuda'), S


def gcq_logits(scores, tau, I, ordered, cumsum, penalties, randomized, allow_zero_sets):
    penalties_cumsum = np.cumsum(penalties, axis=1)
    sizes_base = ((cumsum + penalties_cumsum) <= tau).sum(axis=1) + 1  # 1 - 1001
    sizes_base = np.minimum(sizes_base, scores.shape[1])  # 1-1000  size_base is an array of length equal to the number of samples in scores

    if randomized:
        V = np.zeros(sizes_base.shape)
        for i in range(sizes_base.shape[0]):
            V[i] = 1 / ordered[i, sizes_base[i] - 1] * \
                   (tau - (cumsum[i, sizes_base[i] - 1] - ordered[i, sizes_base[i] - 1]) -
                    penalties_cumsum[0, sizes_base[i] - 1])  # -1 since sizes_base \in {1,...,1000}.

        sizes = sizes_base - (np.random.random(V.shape) >= V).astype(int)
    else:
        sizes = sizes_base

    if not allow_zero_sets:
        sizes[
            sizes == 0] = 1  # allow the user the option to never have empty sets (will lead to incorrect coverage if 1-alpha < model's top-1 accuracy

    S = list()

    # Construct S from equation (5)

    for i in range(I.shape[0]):
        S = S + [I[i, 0:sizes[i]], ]
    return S


def get_violation(cmodel, loader_paramtune, strata, alpha):
    df = pd.DataFrame(columns=['size', 'correct'])
    for logit, target in loader_paramtune:
        # compute output
        output, S = cmodel(logit)  # This is a 'dummy model' which takes logits, for efficiency.
        # measure accuracy and record loss
        size = np.array([x.size for x in S])
        I, _, _ = sort_sum(logit.numpy())
        correct = np.zeros_like(size)
        for j in range(correct.shape[0]):
            correct[j] = int(target[j] in list(S[j]))
        batch_df = pd.DataFrame({'size': size, 'correct': correct})
        df = df.append(batch_df, ignore_index=True)
    wc_violation = 0
    for stratum in strata:
        temp_df = df[(df['size'] >= stratum[0]) & (df['size'] <= stratum[1])]
        if len(temp_df) == 0:
            continue
        stratum_violation = abs(temp_df.correct.mean() - (1 - alpha))
        wc_violation = max(wc_violation, stratum_violation)
    return wc_violation  # the violation


def _compute_scores_with_calibration(cmodel, logits_numpy):
    adjust_array = np.zeros(logits_numpy.shape[1])
    adjust_array[cmodel.index] = cmodel.bias
    adjusted_logits = logits_numpy + adjust_array
    scores = softmax(adjusted_logits / cmodel.T.item(), axis=1)
    return adjusted_logits, scores


def compute_label_nonconformity(scores, penalties, randomized=False, allow_zero_sets=False):
    """
    Compute RAPS-style nonconformity scores for every sample-label pair.
    Returned matrix has shape [num_samples, num_classes].
    """
    I, ordered, cumsum = sort_sum(scores)
    penalties_cumsum = np.cumsum(penalties, axis=1)[0]

    num_samples, num_classes = scores.shape
    tau = np.zeros((num_samples, num_classes), dtype=np.float64)

    for i in range(num_samples):
        for rank in range(num_classes):
            cls = I[i, rank]
            if randomized:
                if rank == 0 and not allow_zero_sets:
                    tau[i, cls] = cumsum[i, rank] + penalties_cumsum[rank]
                else:
                    u = np.random.random()
                    prev_cumsum = 0.0 if rank == 0 else cumsum[i, rank - 1]
                    tau[i, cls] = u * ordered[i, rank] + prev_cumsum + penalties_cumsum[rank]
            else:
                tau[i, cls] = cumsum[i, rank] + penalties_cumsum[rank]
    return tau


def compute_conformal_p_values(calibration_scores, label_nonconformity, smoothed=True):
    """
    Compute conformal p-values for all sample-label pairs using calibration scores.
    """
    calibration_scores = np.asarray(calibration_scores).reshape(-1)
    n_calib = calibration_scores.shape[0]

    flat_tau = label_nonconformity.reshape(-1)
    flat_pvals = np.zeros_like(flat_tau, dtype=np.float64)

    for i, tau in enumerate(flat_tau):
        num_greater = np.sum(calibration_scores > tau)
        num_equal = np.sum(calibration_scores == tau)

        if smoothed:
            u = np.random.random()
            flat_pvals[i] = (num_greater + u * (num_equal + 1.0)) / (n_calib + 1.0)
        else:
            flat_pvals[i] = (num_greater + num_equal + 1.0) / (n_calib + 1.0)

    return flat_pvals.reshape(label_nonconformity.shape)


def apply_pt_to_p_values(p_values, pcp_prob):
    """
    Apply PT (called PCP in this codebase) to p-values sample-wise.

    For each validation sample i:
    - with probability (1 - pcp_prob), set all p-values to 0
    - with probability pcp_prob, set p_y to 1 - pcp_prob * (1 - p_y) for every label y
    """
    if pcp_prob is None:
        raise ValueError('PCP_prob must be provided when applying PT to p-values.')
    if not (0.0 <= pcp_prob <= 1.0):
        raise ValueError('PCP_prob must be in [0, 1].')

    transformed = np.zeros_like(p_values, dtype=np.float64)
    sample_keep_mask = np.random.random(size=(p_values.shape[0],)) <= pcp_prob
    transformed[sample_keep_mask, :] = 1.0 - pcp_prob * (1.0 - p_values[sample_keep_mask, :])
    return transformed


def compute_all_p_value_criteria(p_values, targets, epsilon):
    """
    Compute all criteria from 1603.04416v2 used in this project:
    S, N, U, F, M, E, OU, OF, OM, OE.

    p_values: [num_samples, num_classes]
    targets: [num_samples]
    epsilon: significance level for set-based criteria (N/M/E/OM/OE)
    """
    targets = np.asarray(targets).astype(int)
    num_samples, num_classes = p_values.shape

    # Set-valued prediction region from p-values.
    included = p_values > epsilon
    set_size = np.sum(included, axis=1).astype(np.float64)

    row_sum = np.sum(p_values, axis=1)
    top_idx = np.argmax(p_values, axis=1)
    top_p = p_values[np.arange(num_samples), top_idx]
    true_p = p_values[np.arange(num_samples), targets]

    # Wrong-label mask for observed criteria.
    wrong_mask = np.ones((num_samples, num_classes), dtype=bool)
    wrong_mask[np.arange(num_samples), targets] = False
    wrong_p_values = np.where(wrong_mask, p_values, -np.inf)
    wrong_included = np.logical_and(included, wrong_mask)
    wrong_set_size = np.sum(wrong_included, axis=1).astype(np.float64)

    return {
        # Sum criterion
        'S': np.mean(row_sum),
        # Number criterion
        'N': np.mean(set_size),
        # Unconfidence criterion (largest non-top p-value)
        'U': np.mean(np.max(np.where(np.eye(num_classes, dtype=bool)[top_idx], -np.inf, p_values), axis=1)),
        # Fuzziness criterion (sum excluding top label)
        'F': np.mean(row_sum - top_p),
        # Multiple criterion
        'M': np.mean((set_size > 1).astype(np.float64)),
        # Excess criterion
        'E': np.mean(np.maximum(set_size - 1.0, 0.0)),
        # Observed unconfidence (largest p-value among incorrect labels)
        'OU': np.mean(np.max(wrong_p_values, axis=1)),
        # Observed fuzziness
        'OF': np.mean(row_sum - true_p),
        # Observed multiple criterion
        'OM': np.mean((wrong_set_size > 0).astype(np.float64)),
        # Observed excess criterion
        'OE': np.mean(wrong_set_size),
        # Kept for diagnostics.
        'mean_true_p': np.mean(true_p),
    }


def evaluate_p_value_criteria(cmodel, data_loader, smoothed=True, randomized_tau=False, apply_pt=True,
                              epsilon=None, max_samples=None, print_bool=False, progress_prefix=''):
    """
    Evaluate p-value-based criteria for a fitted ConformalModel.
    """
    if epsilon is None:
        epsilon = cmodel.alpha

    metric_keys = ['S', 'N', 'U', 'F', 'M', 'E', 'OU', 'OF', 'OM', 'OE', 'mean_true_p']
    metric_totals = {k: 0.0 for k in metric_keys}
    total_n = 0
    end = time.time()

    with torch.no_grad():
        for x, target in data_loader:
            logits = cmodel.model(x.cuda())
            logits_numpy = logits.detach().cpu().numpy()

            _, scores = _compute_scores_with_calibration(cmodel, logits_numpy)
            tau = compute_label_nonconformity(
                scores,
                cmodel.penalties,
                randomized=randomized_tau,
                allow_zero_sets=cmodel.allow_zero_sets,
            )
            p_values = compute_conformal_p_values(cmodel.E, tau, smoothed=smoothed)
            if apply_pt and getattr(cmodel, 'CP_method', None) == 'PCP':
                p_values = apply_pt_to_p_values(p_values, cmodel.PCP_prob)

            target_np = target.detach().cpu().numpy()
            batch_stats = compute_all_p_value_criteria(p_values, target_np, epsilon=epsilon)
            bsz = target_np.shape[0]

            for key in metric_keys:
                metric_totals[key] += batch_stats[key] * bsz
            total_n += bsz

            if print_bool:
                elapsed = time.time() - end
                end = time.time()
                avg_s = metric_totals['S'] / total_n
                avg_n = metric_totals['N'] / total_n
                avg_of = metric_totals['OF'] / total_n
                avg_oe = metric_totals['OE'] / total_n
                prefix = f'{progress_prefix} | ' if progress_prefix else ''
                print(
                    f"\r{prefix}N: {total_n} | Time: {elapsed:.3f} | "
                    f"S: {batch_stats['S']:.3f} ({avg_s:.3f}) | "
                    f"N-set: {batch_stats['N']:.3f} ({avg_n:.3f}) | "
                    f"OF: {batch_stats['OF']:.3f} ({avg_of:.3f}) | "
                    f"OE: {batch_stats['OE']:.3f} ({avg_oe:.3f})",
                    end=''
                )
            else:
                end = time.time()

            if max_samples is not None and total_n >= max_samples:
                break

    if total_n == 0:
        raise ValueError('No samples were evaluated in evaluate_p_value_criteria.')

    if print_bool:
        print('')

    result = {key: metric_totals[key] / total_n for key in metric_keys}
    result['num_eval_samples'] = int(total_n)
    result['epsilon'] = float(epsilon)
    return result
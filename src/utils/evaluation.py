import numpy as np
import pandas as pd
import textwrap
import itertools
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import normalized_mutual_info_score


def assign_cluster_label(X, Y):
    """
    Assigns the label to from cluster of X to classes present in Y
    Args:
        X: Encoded data
        Y: Ccorresponding Labels

    Returns:
        Prediction of cluster from input data
    """
    kmeans = KMeans(n_clusters=len(set(Y))).fit(X)
    Y_pred = np.zeros(Y.shape)
    for i in set(kmeans.labels_):
        ind = kmeans.labels_ == i
        Y_pred[ind] = Counter(Y[ind]).most_common(1)[0][0] # assign label.
    return Y_pred


def make_grid_alg_kwargs(alg, **kwargs):
    """
    Creates a grid of arguments to feed benchmark with the given algorithm

    Args:
        alg: the algorithm to use
        **kwargs: Dictionary of arrays of parameters to do grid test on

    Returns:
        list to feed into benchmark
    """
    keys = []
    values_list = []
    for key, values in kwargs.items():
        keys.append(key)
        values_list.append(values)

    grid_kwargs = []
    for value_product in list(itertools.product(*values_list)):
        kwargs = {}
        grid_kwargs.append(kwargs)
        for key, value in zip(keys, value_product):
            kwargs[key] = value
    alg_kwargs_pairs = [(alg, kwargs) for kwargs in grid_kwargs]
    return alg_kwargs_pairs


def indent(text, amount, ch=' '):
    """
    Indents a given text
    """
    return textwrap.indent(text, amount * ch)


def rre_score(model, X, Y, Y_pred, W, H):
    """
    Relative reconstruction error
    """
    return np.linalg.norm(X - W.dot(H)) / np.linalg.norm(X)


def acc_score(model, X, Y, Y_pred, W, H):
    """
    Accuracy
    """
    return accuracy_score(Y, Y_pred)


def nmi_score(model, X, Y, Y_pred, W, H):
    """
    Normalized mutual information
    """
    return normalized_mutual_info_score(Y, Y_pred)


def benchmark(X, Y, scaler,
              alg_kwargs_pairs, all_n_components, # algs configs
              noise_kwargs_pairs, # noise configs
              metrics, metrics_names=None, # evaluations
              n_trials=5, pc_sample=0.9): # sampling configs
    """
    Complete grid style benchmark of algs and output long form evaluation results

    Args:
        X: Input array
        Y: Labels
        scaler: Scler to use
        alg_kwargs_pairs: Pairs of nmf algorithms and parameters to test
        all_n_components: list of component number to use
        noise_kwargs_pairs: Pairs of noise algorithms and parameters to test
        metrics: metrics to evaluate the model with
        metrics_names: names of these metrics
        n_trials: Number of trials to run
        pc_sample: percent of sample to sample each trial

    Returns:
        A long form dataframe with all the experiment results recorded
    """
    # Prepare column names in data frame
    if metrics_names is None:
        metrics_names = [m.__name__ for m in metrics]

    evaluations = pd.DataFrame(columns =
                               ["alg", "n_components", "kwargs", "noise_id", "trial_id"] 
                               + metrics_names)
    if isinstance(all_n_components, int):
        all_n_components = [all_n_components]

    # Prepare salt and pepper
    noises = []
    for i, (noise_alg, noise_kwargs) in enumerate(noise_kwargs_pairs):
        noise = noise_alg(**noise_kwargs)
        noises.append([noise, {"noise_id": i, **noise_kwargs}])

    # Prepare subseting
    subset_idxs = []
    for n in range(n_trials):
        subset_idxs.append(np.random.choice(range(X.shape[1]), size=int(pc_sample * X.shape[1]), replace=False))

    for noise, noise_kwargs in noises:
        # Noise outer loop to keep it consistent between runs
        X_noise = noise.fit_transform(X)
        if scaler is not None:
            # preprocess data
            X_noise = scaler.fit_transform(X_noise)
        print(indent("Noise: " + str(noise_kwargs), 0))
        for i, subset_idx in enumerate(subset_idxs):
            print(indent("Trail: " + str(i), 4))
            X_subset = X_noise[:, subset_idx]
            Y_subset = Y[subset_idx]
            for alg in alg_kwargs_pairs:
                # Separate kwargs if additional kwargs are provided
                try:
                    if len(alg) == 2:
                        alg, alg_kwargs = alg
                except:
                    alg_kwargs = {}

                print(indent("Alg: " + alg.__name__ + " " + str(alg_kwargs), 8))
                for k in all_n_components:
                    row = {**noise_kwargs}
                    row.update(
                        {"alg": alg.__name__, "n_components": k, "kwargs": alg_kwargs, "trial_id": i})
                    model = alg(n_components=k, **alg_kwargs)
                    H = model.fit_transform(X_subset)
                    W = model.components_
                    # print(model.__class__, "W", W.shape, "H", H.shape)
                    # Y_pred = assign_cluster_label(H.T, Y_subset)
                    Y_pred = assign_cluster_label(H.T, Y_subset)

                    for metric, name in zip(metrics, metrics_names):
                        row[name] = metric(model, X_subset, Y_subset, Y_pred, W, H)
                    evaluations = evaluations.append(row, ignore_index=True)
    return evaluations

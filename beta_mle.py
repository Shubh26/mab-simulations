import re, sys, os, numpy as np
from scipy.stats import beta
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()


def mle_beta(bin_boundaries=None, ctr_confs=None):
    """
    User provides bin boundaries and corresponding CTR confidences.
    Sample code - probably needs more checks
    :param bin_boundaries:
    :param ctr_confs:
    :return:
    """
    if bin_boundaries is None or ctr_confs is None:
        # if any of these are None, lets demo the functionality with values of our choice
        num_bins = 5
        a, b = 2, 5
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        print(f"Bin boundaries: {bin_boundaries}")
        bin_centres = [(bin_boundaries[i] + bin_boundaries[i + 1]) / 2 for i in range(num_bins)]
        ctr_confs = beta.pdf(bin_centres, a, b)
    else:
        num_bins = len(bin_boundaries) - 1
        bin_centres = [(bin_boundaries[i] + bin_boundaries[i + 1]) / 2 for i in range(num_bins)]
        ctr_confs = np.array(ctr_confs)
        bin_boundaries = np.array(bin_boundaries)
    ctr_confs = ctr_confs / sum(ctr_confs)  # be sure to normalize this to add to 1
    assert len(bin_boundaries) == len(ctr_confs) + 1, f"Number of bin boundary entries should exceed CTR entries by 1."
    unequal_bins = False
    if len(set(bin_boundaries[:-1] - bin_boundaries[1:])) > 1:
        print(set(bin_boundaries[:-1] - bin_boundaries[1:]))
        unequal_bins = True
    bin_confs = dict([(i + 1, j) for i, j in enumerate(ctr_confs)])
    print(f"Bin centres: {bin_centres}")
    print(f"CTR confs: {ctr_confs}")

    fig = plt.figure()
    ax = fig.add_subplot(121)
    sns.barplot(bin_centres, np.array(ctr_confs)*100, ax=ax)
    additional_info = '' if unequal_bins is False else 'NOTE: unequal bins.'
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([f"{bin_boundaries[i]:.2f}-{bin_boundaries[i+1]:.2f}" for i in range(num_bins)], minor=False)
    ax.set_title(f'User-provided CTR confidences. {additional_info}')
    ax.set_xlabel('CTR bins')
    ax.set_ylabel('confidence of CTR bucket occurring')

    num_samples = 50000  # we will sample data to fit a Beta
    # sample bin nos. - need to add 1 since bin nos. start with 1
    sampled_bins = np.random.choice(num_bins, size=num_samples, p=ctr_confs) + 1

    ax = fig.add_subplot(122)
    # for a specific sampled bin, sample uniformly within the bin
    sampled_ctrs = [np.random.uniform(bin_boundaries[b-1], bin_boundaries[b], 1)[0] for b in sampled_bins]
    sns.kdeplot(sampled_ctrs, ax=ax, label='generated dist.')#, fill=True
    A, B, _, _ = beta.fit(sampled_ctrs)
    print(f"Results of MLE fit, a={A}, b={B}.")
    X = np.linspace(0, 1, 100)
    y = beta.pdf(X, A, B)
    ax.set_title(f'num_samples={num_samples}, prior shape params: ({A:.1f}, {B:.1f})')
    ax.plot(X, y, label=f'Beta MLE fit ({A:.1f}, {B:.1f})', c='k')
    ax.set_xlabel('CTR')
    ax.set_ylabel('Beta pdf')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    #mle_beta([0, 0.25, 0.5, 0.75, 1.0], [0.1, 0.2, 0.5, 0.2])
    mle_beta([0, 0.01, 0.1, 0.75, 1.0], [0.9, 0.1, 0.0, 0.0])

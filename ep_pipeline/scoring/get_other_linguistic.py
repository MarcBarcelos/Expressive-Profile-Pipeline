import math
from collections import Counter
import numpy as np

### TO BE INCLUDED: VAD, Sentment stuff maybe?, ...

# calculate the moving average type-token ratio (MATTR) for a list of tokens
def mattr(tokens, mattr_window): 
    n = len(tokens) # number of tokens
    if n == 0: 
        return float("nan")                         # if there are no tokens, return NaN to indicate that MATTR cannot be calculated
    if n < mattr_window:                            # if the number of tokens is less than the specified window size, calculate the type-token ratio for the entire list of tokens
        return len(set(tokens)) / n
    counts = Counter(tokens[:mattr_window])         # initialize a counter to keep track of the frequency of each token in the current window
    ratios = [len(counts) / mattr_window]           # calculate the initial type-token ratio for the first window and store it in a list
    for i in range(mattr_window, n):                # iterate through the tokens starting from the index equal to the window size
        out_tok = tokens[i - mattr_window]          # token that is sliding out of the window
        counts[out_tok] -= 1                        # decrement the count of the outgoing token in the counter
        if counts[out_tok] == 0:                    # if the count of the outgoing token drops to zero, remove it from the counter to maintain an accurate count of unique tokens in the current window
            del counts[out_tok]                     # remove the outgoing token from the counter if its count drops to zero
        counts[tokens[i]] += 1                      # increment the count of the incoming token in the counter
        ratios.append(len(counts) / mattr_window)   # calculate the type-token ratio for the current window and append it to the list of ratios
    return float(np.mean(ratios))                   # return the average type-token ratio across all windows as the final MATTR value

# calculate the windowed unigram entropy for a list of tokens
def windowed_unigram_entropy(tokens, entropy_window, step=None): 
    n = len(tokens) # number of tokens
    if n == 0:
        return float("nan"), float("nan"), float("nan") # mean, std, exp(mean)
    win = min(entropy_window, n)                                                        # window size
    step = step or max(1, win // 2)                                                     # step size for sliding window
    Hs = []                                                                             # list to store entropy values for each window
    for s in (list(range(0, max(1, n - win + 1), step)) or [0]):                        # start indices for sliding window
        chunk = tokens[s:s + win]                                                       # current window of tokens
        total = len(chunk)                                                              # total number of tokens in the current window
        H = -sum((c / total) * math.log(c / total) for c in Counter(chunk).values())    # calculate entropy for the current window
        Hs.append(H)                                                                    # append the calculated entropy to the list
    H_arr = np.array(Hs)                                                                # convert list of entropy values to a numpy array for easier calculation of mean and std
    return float(H_arr.mean()), float(H_arr.std()), float(np.exp(H_arr.mean()))         # return mean, std, and exp(mean) of the entropy values across all windows

# calculate the trigram entropy for a list of tokens
def trigram_entropy(tokens, trigram_test_frac, trigram_alpha, seed): 
    n = len(tokens) # number of tokens
    if n < 10:
        return float("nan"), float("nan")            # if there are fewer than 10 tokens, return NaN to indicate that trigram entropy cannot be calculated due to insufficient data for reliable estimation
    rng = np.random.default_rng(seed)                # initialize a random number generator with a fixed seed for reproducibility 
    idx = rng.permutation(n)                         # generate a random permutation of indices for the tokens to create a random split of the data into training and testing sets for calculating trigram entropy
    split = int(n * (1 - trigram_test_frac))         # calculate the index at which to split the data into training and testing sets based on the specified fraction for testing
    train = [tokens[i] for i in sorted(idx[:split])] # create the training set by selecting tokens corresponding to the first part of the randomly permuted indices, sorted to maintain the original order of tokens in the training set
    test = [tokens[i] for i in sorted(idx[split:])]  # create the testing set by selecting tokens corresponding to the second part of the randomly permuted indices, sorted to maintain the original order of tokens in the testing set
    V = len(set(train))                              # calculate the vocabulary size (number of unique tokens) in the training set, which is used for smoothing in the calculation of trigram probabilities and entropy to account for unseen trigrams in the test set.
    if V == 0: 
        return float("nan"), float("nan")                               # if there are no unique tokens in the training set, return NaN to indicate that trigram entropy cannot be calculated 
    bigram_counts, trigram_counts = Counter(), Counter()                # frequency of bigrams and trigrams in training , which are used to calculate probabilities for computing trigram entropy on test set
    for i in range(len(train) - 2):                                     # iterate through the training tokens to populate the bigram and trigram counts, which will be used to calculate the probabilities of                                                                    trigrams in the test set for computing trigram entropy.
        bigram_counts[(train[i], train[i + 1])] += 1                    # increment the count of the bigram consisting of the current token and the next token in the training set, which is used to calculate the probability of trigrams in the test set for computing trigram entropy.
        trigram_counts[(train[i], train[i + 1], train[i + 2])] += 1     # increment the count of the trigram consisting of the current token and the next two tokens in the training set, which is used to calculate the probability of trigrams in the test set for computing trigram entropy.
    log_probs = [] 
    for i in range(len(test) - 2):                                      # iterate through the test tokens to calculate the log probabilities of the trigrams, which are used to compute the trigram entropy.
        bg = (test[i], test[i + 1])                                     # create a bigram consisting of the current token and the next token in the test set, which is used to calculate the probability of the corresponding trigram for computing trigram entropy.
        tg = (test[i], test[i + 1], test[i + 2])                        # create a trigram consisting of the current token and the next two tokens in the test set, which is used to calculate the probability of the trigram for computing trigram entropy.
        log_probs.append(math.log((trigram_counts.get(tg, 0) + trigram_alpha) / (bigram_counts.get(bg, 0) + trigram_alpha * V))) # calculate the log probability of the trigram in the test set using add-alpha smoothing based on counts from the training set, append it to list of log probabilities for computing trigram entropy.
    if not log_probs:
        return float("nan"), float("nan")       # if there are no log probabilities calculated (which can happen if the test set has fewer than 3 tokens), return NaN to indicate that trigram entropy cannot be calculated due to insufficient data for computing probabilities.
    H = -float(np.mean(log_probs))              # calculate the average log probability of the trigrams in the test set and take the negative to compute the trigram entropy, which represents the average uncertainty or unpredictability of the next token given the previous two tokens in the test set 
    return H, float(np.exp(H))                  # return the calculated trigram entropy and its exponential (which can be interpreted as the effective number of choices for the next token given the previous two tokens in the test set 

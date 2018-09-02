
def ctmc_aggregateevents2(data, numstates, toltime=1e-8):
    import scipy.sparse
    import numpy as np
    import warnings

    transcount = scipy.sparse.lil_matrix((numstates, numstates), dtype=int)
    statetime = np.zeros(numstates, dtype=float)

    for exid, example in enumerate(data):
        states = example[0]
        times = example[1]

        if len(np.unique(states)) < 2:
            warnings.warn(
                ("The example id={:d} has only 1 or no distinct "
                 "state and will be ignored. Please remove incomplete "
                 "examples from the dataset").format(exid))
        else:
            for i, s in enumerate(states):
                if times[i] < toltime:
                    warnings.warn(
                        ("The example id={:d} has a state that "
                         "have not been active for longer than "
                         "toltime.").format(exid))
                else:
                    statetime[s] += times[i]
                    if i:
                        if states[i-1] == s:
                            warnings.warn(
                                ("The example id={:d} contains two entries "
                                 "for the same state in a row. Please  "
                                 "aggregate these obs.").format(exid))
                        else:
                            transcount[states[i-1], s] += 1

    return transcount.toarray(), statetime


def ctmc_generatormatrix2(transcount, statetime, toltime=1e-8):
    """Compute the Generator Matrix

    Parameters:
    -----------
    transcount : ndarray
        NxN matrix with the counted number of transitions
        from state i to j and state j to i (diagonals are
        naturally 0).

    statetime : ndarray
        The cumulated time passed at certain state i.

    Returns:
    --------
    genmat : ndarray
        Generator Matrix

    Correction 1:
    -------------
    The diagonal elements of transcount should be Zero.
    There should be any tr

    Correction 2:
    -------------
    The generator matrix is transition count matrix divided
    by the cumulated time in a certain state

        genmat[i,:] = transcount[i,:] / statetime[i]

    If a certain state is not used, i.e. statetime[i]==0.0,
    then genmat[i,:]=NaN because it's a division by zero.
    Thus, every time

        statetime[i] < toltime

    (e.g. toltime=1e-8) then the generator matrix is reset
    to zero genmat[i,:]=0
    """
    import numpy as np
    import warnings

    # deep copy
    tmp = transcount.copy()

    # get matrix dimension
    n = tmp.shape[0]

    # reset diagonal elements to zero
    # same as "tmp[np.eye(n) == 1] = 0"
    for i in range(n):
        if tmp[i, i] != 0:
            tmp[i, i] = 0
            warnings.warn(
                ("transcount({0:d},{0:d})={1:d} is not Zero. "
                 "There is bug in the way you count transitions"
                 ).format(i, tmp[i, i]))

    # subtract the row sum from the diagonal element
    # same as "tmp -= np.diag(np.sum(tmp, axis=1))"
    rowsum = np.sum(tmp, axis=1)
    for i in range(n):
        tmp[i, i] = -rowsum[i]

    # divide transitions counts by the state idle times
    genmat = np.zeros(shape=(n, n), dtype=float)
    for i in range(n):
        if statetime[i] >= toltime:
            genmat[i, :] = tmp[i, :] / statetime[i]
        else:
            warnings.warn(
                ("The state i={:d} has a cumulated state time "
                 "of less than toltime").format(i))

    # done
    return genmat


def ctmc_fit2(data, numstates, transintv=1.0, toltime=1e-8):
    """ Continous Time Markov Chain (with automatic error correction)

    Warning:
    --------
    - Try ctmc_fit2 if you cannot fix your data object to
        run with ctmc_fit
    - ctmc_aggregateevents2 will ignore all transitions
        with an i-th state that have not been active for
        longer than toltime
    - ctmc_aggregateevents2 will ignore all examples with
        less than 2 distinct states.
    - ctmc_aggregateevents2 will ignore cnt[i,i] transitions
        that does not exist by default.
    - ctmc_generatormatrix2 automagically resets diagonal
        elements of the transcount matrix to zero. There are
        no transitions from the i-th to the i-th state
        by definition.
    - ctmc_generatormatrix2 automagically corrects missing
        cumulated state times (toltime=0) or if measured cumulated
        state time are below a certain threshold (toltime=1e-8 is
        Default), i.e. one state happens to never occur.
        In ctmc_generatormatrix (ctmc_fit) the generator
        matrix would exhibit a NaN-row for the i-th state,
        that never happened.

    In both cases ctmc_fit2 will raise warning messages. If
    you don't like these warning then disable them

        import warnings
        warnings.filterwarnings('ignore')

    """
    import scipy.linalg
    transcount, statetime = ctmc_aggregateevents2(data, numstates, toltime)
    genmat = ctmc_generatormatrix2(transcount, statetime, toltime)
    transmat = scipy.linalg.expm(genmat * transintv)
    return transmat, genmat, transcount, statetime

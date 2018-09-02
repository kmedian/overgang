
def ctmc_datacheck(data, numstates, toltime):
    import numpy as np

    eligiblestates = range(numstates)

    for exid, example in enumerate(data):
        states = example[0]
        times = example[1]

        if not np.all(np.isin(states, eligiblestates)):
            raise Exception(
                ("The example id={:d} has faulty state "
                 "labels/encodings").format(exid))

        if len(np.unique(states)) < 2:
            raise Exception(
                ("The example id={:d} has only 1 distinct "
                 "state").format(exid))

        for i in range(1, len(states)):
            if states[i-1] == states[i]:
                raise Exception(
                    ("The example id={:d} has two consequtive entries "
                     "state[{:d}]==state[{:d}]").format(exid, i-1, i))

        for i, t in enumerate(times):
            if t < toltime:
                raise Exception(
                    ("The example id={:d} has a state[{:d}] that have not "
                     "been active for longer than toltime").format(exid, i))

    return False


def ctmc_errorcheck(transcount, statetime, toltime):
    import numpy as np
    # check transitions counting went wrong
    if np.any(np.diag(transcount) == 0):
        raise Exception(
            ("Transition Count Matrix have diagonal "
             "elements 'm[i,i] != 0'. There are no transition counts "
             "for the i-th state to itself by definition."))

    # check if statetime[i] is big enough to work as divisor
    if np.any(statetime < toltime):
        raise Exception(
            ("The states i="
             + ",".join([str(i) for i in np.where(statetime < toltime)[0]])
             + " have cumulated time period that is smaller than toltime."))

    return False


def ctmc_aggregateevents(data, numstates):
    import scipy.sparse
    import numpy as np

    transcount = scipy.sparse.lil_matrix((numstates, numstates), dtype=int)
    statetime = np.zeros(numstates, dtype=float)

    for _, example in enumerate(data):
        states = example[0]
        times = example[1]

        for i, s in enumerate(states):
            statetime[s] += times[i]
            if i:
                transcount[states[i-1], s] += 1

    return transcount.toarray(), statetime


def ctmc_generatormatrix(transcount, statetime):
    import numpy as np

    tmp = transcount.copy()
    n = tmp.shape[0]

    rowsum = np.sum(tmp, axis=1)
    for i in range(n):
        tmp[i, i] = -rowsum[i]

    genmat = np.zeros(shape=(n, n), dtype=float)
    for i in range(n):
        genmat[i, :] = tmp[i, :] / statetime[i]

    return genmat


def ctmc_fit(data, numstates, transintv=1.0, toltime=1e-8, debug=False):
    """ Continous Time Markov Chain

    Warning:
    --------
    - ctmc_fit assumes a clean data object and does not
        autocorrect any errors as result of it

    The main error sources are

    - transitions counting (e.g. two consequtive states
        has not been aggregated, only one distinct state
        reported) and
    - a state is modeled ore required that does not occur
        in the dataset (e.g. you a certain scale in mind
        and just assume it's in the data) or resp. involved
        in any transition (e.g. an example with just one
        state)

    You can disable any error checking and prevent exceptions
    by setting checks=False but I strongly advice against it.

    Example:
    --------
    Use `ctmc_datacheck` to check during preprocessing the
    dataset

        data = ...
        og.ctmc_datacheck(data, numstates, toltime)

    Disable checks in `ctmc_fit`

        transmat, genmat, transcount, statetime = og.ctmc_fit(
            data, numstates, toltime, checks=False)

    Check aftwards if there has been an error

        og.ctmc_errorcheck(transcount, statetime, toltime)

    """
    import scipy.linalg

    if debug:
        ctmc_datacheck(data, numstates, toltime)  # raises an exception

    transcount, statetime = ctmc_aggregateevents(data, numstates)

    if debug:
        ctmc_errorcheck(transcount, statetime, toltime)  # raises an exception

    genmat = ctmc_generatormatrix(transcount, statetime)

    transmat = scipy.linalg.expm(genmat * transintv)

    return transmat, genmat, transcount, statetime

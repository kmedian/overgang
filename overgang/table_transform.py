

def yearfrac(a, b):
    """date difference in years between date b and date a."""
    from datetime import date

    # days between date b and a, if a would be in the same year
    if a.day is 29 and a.month is 2:
        diff_days = (b.date() - date(b.year, a.month, 28)).days
    else:
        diff_days = (b.date() - date(b.year, a.month, a.day)).days

    # days of year b
    days_of_yr_b = (date(b.year, 12, 31) - date(b.year-1, 12, 31)).days

    # difference between year b and a, plus the fractional year
    return (b.year - a.year) + diff_days / days_of_yr_b


def table_transform(data):
    """Transforms array/list to internal data format

    Parameters:
    -----------
    data : ndarray
        Table/panel data with the following columns

            data[:,0]   Examples IDs
            data[:,1]   Event dates as datetime object
            data[:,2]
    """
    import numpy as np

    lastdate = np.max(data[:, 1])

    newdata = list()

    for _, exampleid in enumerate(np.unique(data[:, 0])):
        # read all entries for the example
        tmp = data[data[:, 0] == exampleid]

        # sort by date
        idxsorted = tmp[:, 1].argsort()
        dates = tmp[idxsorted, 1]
        states = tmp[idxsorted, 2]

        # filter state changes
        idxunique = np.append([True], states[1:] != states[:-1])

        # year fraction
        dates2 = np.append(dates[idxunique], [lastdate])
        times = np.vectorize(yearfrac)(dates2[:-1], dates2[1:])

        # store
        if len(times) > 1:
            newdata.append((list(states[idxunique]), list(times)))

    return newdata

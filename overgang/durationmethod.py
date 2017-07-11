
def label_encode_rating(x, labels=['AAA','AA','A','BBB','BB','B','CCC','D']):
    """"Label Ecnoder for Ratings"""
    import numpy as np
    y = np.zeros(shape=x.shape, dtype=int)
    for num, label in enumerate(labels):
        y[x==label] = num
    return y, len(labels), labels



def yearfrac(b, a):
    """date difference in years between date b and date a."""
    from datetime import date
    
    # days between date b and a, if a would be in the same year
    if a.day is 29 and a.month is 2:
        diff_days = (b.date() - date(b.year, a.month, 28)).days
    else:
        diff_days = (b.date() - date(b.year, a.month, a.day)).days
        
    # days of year b
    days_of_yr_b = (date(b.year, 12, 31) - date(b.year-1,12,31)).days

    # difference between year b and a, plus the fractional year
    return (b.year - a.year) + diff_days / days_of_yr_b

#def yearfrac_np(b, a):
#    """date difference in years between date vector b and date vector a."""
#    import numpy as np
#    return np.vectorize(yearfrac)(b,a)
#
#create time differences in years
#   f = np.vectorize(lambda x: x.days)  #convert timedelta to days as int
#   lagged = np.append([start_dt], dates[:-1])  #start date plus lagged dates
#   deltayears = f(dates - lagged) / 365.2425
#deltayears3 = np.vectorize(lambda x: x.days)( dates - np.append([start_dt], dates[:-1]) ) / 365.2425
#deltayears2 = np.vectorize(lambda x,y: (x-y).days)( dates, np.append([start_dt], dates[:-1]) ) / 365.2425



def state_idle_time(dates, start_dt, end_dt, values, num_states):
    """total time spent in each state"""
    from scipy.sparse import lil_matrix
    import numpy as np
    #create time differences in years
    #   f = np.vectorize(yearfrac)  #convert timedelta to days as int
    #   t1 = np.append(dates,[end_dt])
    #   t0 = np.append([start_dt],dates)
    #   deltayears = f(t1, t0)
    deltayears = np.vectorize(yearfrac)( np.append(dates,[end_dt]), np.append([start_dt],dates))
    #the corresponding states
    state_ids = np.append([num_states], values)
    #add the times/durations spent in the different states
    state_time = lil_matrix((1,num_states+1), dtype=float)
    for i,s in enumerate(state_ids):
        state_time[0,s] += deltayears[i]
    #return the result
    return state_time



def state_transition_count(values, num_states):
    """count state transitions"""
    from scipy.sparse import lil_matrix
    import numpy as np
    # create empty transition matrix
    trans_cnt = lil_matrix((num_states+1,num_states+1), dtype=int)
    # the first transition is from an artificial 'no state' to some state
    prev_state = num_states
    # the last transition is from some state to the artificial 'no state'
    for curr_state in np.append(values, [num_states]):
        #increment if a transition happened
        if curr_state is not prev_state:
            trans_cnt[prev_state, curr_state] += 1
        #replace previous state
        prev_state = curr_state
    #result
    return trans_cnt


def total_count_and_time(data, num_states):
    """total count of transitions and total idle time per state"""
    import numpy as np
    # extract unique IDs
    ids = np.unique(data[:,0])

    # other parameters
    start_dt = data[:,1].min()
    end_dt = data[:,1].max()
    #start_dt = datetime(1982, 12, 13, 0, 0)
    #end_dt   = datetime(1994, 6, 20, 0, 0)

    ### for each InstrID
    # initialize variables
    from scipy.sparse import lil_matrix
    state_time = lil_matrix((1, num_states+1), dtype=float)
    trans_cnt = lil_matrix((num_states+1, num_states+1), dtype=int)
    # a) measure the total duration waiting in a certain state => vector <1 x states>
    # b) count the number of transitions => matrix <states x states>
    # c) kumuliere je ID alle trans_cnt und state_time auf
    for id in ids:
        #extract all data for the instrument
        idx_instr = data[:,0]==id
        dates  = data[idx_instr,1]
        values = data[idx_instr,2]

        # sort by date (old first)
        idx_sorted = dates.argsort()
        dates  = dates[idx_sorted]
        values = values[idx_sorted]

        #create time differences in years
        state_time += state_idle_time(dates, start_dt, end_dt, values, num_states)
        #print(state_time.toarray())

        #count the number of transitions
        trans_cnt += state_transition_count(values, num_states)
        #print(trans_cnt.toarray())

    return trans_cnt[:-1,:-1].toarray(), state_time[:,:-1].toarray()
    #return trans_cnt.toarray(), state_time.toarray()


def create_generator_matrix(trans_cnt, state_time):
    """berechne die eigentliche Generator Matrix"""
    #load modules
    import numpy as np
    
    # copy matrix with counted transitions
    tmp = trans_cnt.copy()
    # remove diagonal elements / set diagonals to zero
    tmp -= np.diag(np.diag(trans_cnt))
    # subtract the row sum from the diagonal element
    tmp -= np.diag(np.sum(trans_cnt, axis=1)) 

    # exclude unused states
    idx_row = state_time[0,:] > 0.
    # create diagonal matrix from state idle times vector
    diagvec = np.diag(state_time[0,idx_row])
    # remove unused states from transition count matrix
    adjmat = tmp[idx_row,:]
    
    # divide transitions counts by the state idle times
    gen_mat = np.zeros(trans_cnt.shape, dtype=float);
    gen_mat[idx_row,:] = np.dot( np.linalg.inv(diagvec), adjmat)

    # return generator matrix
    return gen_mat


def preprocess(data_raw, labels=['AAA','AA','A','BBB','BB','B','CCC','D']):
    """Data Preprocessing"""
    #copy data
    data = data_raw.copy()
    
    # encode rating labels
    data[:,2], num_states, labels2 = label_encode_rating(data[:,2], labels)  
    
    # convert time
    import numpy as np
    data[:,1] = np.vectorize(lambda x: x.to_pydatetime())(data[:,1])
    
    # return new dataset
    return data, num_states, labels2


def estimate(data, num_states, trans_intv=1.0):
    # total count of transitions and total idle time per state
    trans_cnt, state_time = total_count_and_time(data, num_states)

    # compute the Generator Matrix
    gen_mat = create_generator_matrix(trans_cnt, state_time)
    
    # generate the transition matrix
    from scipy import linalg
    trans_mat = linalg.expm(gen_mat * trans_intv)

    # return results
    return trans_mat, gen_mat, trans_cnt, state_time


overgang -- Transition probability estimation
=============================================

The overgang package includes the Duration Method to estimate 
the transitions probability matrix (stochastic matrix).
A common application are Credit Rating migrations.


Installation
------------
It's a pip package
```
pip install overgang
```


Example 1
---------

Data Source "Fitch Sovereign Ratings":

- https://www.fitchratings.com/web_content/ratings/sovereign_ratings_history.xls
- https://www.dropbox.com/s/phoqbg7p8rpr3bz/sovereign_ratings_history.xls?dl=1



load the example data

.. code:: python

    import pandas as pd
    import numpy as np
    data_raw = np.array(pd.read_excel("https://www.dropbox.com/s/phoqbg7p8rpr3bz/sovereign_ratings_history.xls?dl=1", skiprows=4, skip_footer=6, parse_cols="A:C"))


estimate the transition matrix (stochastic matrix)

.. code:: python

    # load module
    import overgang.durationmethod as og

    # preprocessing
    data, num_states, labels = og.preprocess(data_raw)
    print(data); print(num_states); print(labels)

    # Estimate Transition matrix
    trans_intv = 1.0; #annual transitions
    trans_mat, gen_mat, trans_cnt, state_time = og.estimate(data, num_states, trans_intv)

check the results

.. code:: python

    # display results
    print("Idle time in a certin state\n", state_time.round(1))
    print("Number of transitions\n", trans_cnt.round(0))
    print("Generator Matrix\n", gen_mat.round(3))
    print("Transition Matrix\n", trans_mat.round(3))

    # check. the sum of all elements of a row must equal to one
    print(trans_mat.sum(axis=1, keepdims=True))

    # one transition
    import numpy as np
    s = np.zeros((1,num_states)); s[0,0] = 1.
    s2 = np.dot(s, trans_mat);
    print(np.r_[s, s2].round(3))

    




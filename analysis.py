import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")


def load_csv(directory, filename, **kwargs):
    """Load csv files from directory

    Parameters
    ----------
    directory : str, bytes, PathLike
        Directory to yield collect data from.
    filename : str
        CSV filename
    kwargs
        Kwargs for pandas.read_csv function

    Yields
    ------
    DataFrame
        DataFrames of the csv data inside directory.

    """
    for dirpath, dirnames, filenames in os.walk(directory):
        if filename in filenames:
            filepath = os.path.join(dirpath, filename)
            yield pd.read_csv(filepath, **kwargs)


# Load all the csv data and concatenate it into one dataframe
df = pd.concat(list(load_csv(
    directory='leader_follower_3',
    filename='data.csv',
    index_col=['time_tot'])), axis=1)
df.fillna(method='ffill', inplace=True)


titles = ['inactive'] + [f'target_{index}' for index in range(4)]

for title in titles:
    data = df[title]
    z = 1.96
    ci_min = data.mean(axis=1) - z * np.sqrt(data.var(axis=1) / data.shape[1])
    ci_max = data.mean(axis=1) + z * np.sqrt(data.var(axis=1) / data.shape[1])

    data.plot(legend=False, alpha=0.1, figsize=(12, 8), title=title)
    data.mean(axis=1).plot()
    ci_min.plot()
    ci_max.plot()

    plt.savefig(f'figures/{title}.png')

import pandas as pd
from fixes import *

def load(fn, test=False):
    columns = ("age", "workclass", "fnlwgt", "education",
            "education_num", "marital_status", "occupation",
            "relationship", "race", "sex", "capital_gain",
            "capital_loss", "hours_per_week", "native_country",
            "income")

    if test:
        columns = columns[:-1]
    data = pd.read_csv(fn,
            names=columns, skipinitialspace=True,
            na_values="?", index_col=False)
    fill_missing(data)
    return data

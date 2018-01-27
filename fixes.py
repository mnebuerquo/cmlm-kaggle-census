import pandas as pd

def fill_missing(df):
    """Modifies the input dataframe to fill missing data in the columns
    workclass, occupation, and native_country."""
    # Fill the NAs in workclass with the mode ('Private' outnumbers
    # all other categories combined):
    df.workclass.fillna("Private", inplace=True)
    # Do likewise for native_country (vast majority are from US):
    df.native_country.fillna("United-States", inplace=True)
    # NAs in occupation occur primarily where workclass is also NA,
    # but no particular value dominates all the others.  This is still
    # ~6% of our data - so for now, fill it with a new value and treat
    # it perhaps like it has information.
    df.occupation.fillna("Other", inplace=True)

def drop_income(df):
    cols = [c for c in df.columns if c != 'income']
    return df[cols]

def get_y(df):
    return (df.income == ">50K") * 1

def feature_xform(df):
    """Given raw data (as from 'read_data'), selects and transforms
    features, including turning categorical columns into numerical
    form.

    Returns (X, y) where 'X' is a DataFrame for features and 'y' is a
    Series for the corresponding labels (where 0 is <= 50K, and 1 is >
    50K).
    """
    # Extract just the features (everything but 'income'):
    cols = df.columns #[c for c in df.columns if c != 'income']
    X = df[cols]
    # One-hot encode everything in this tuple, join it to X, and
    # drop the original column:
    onehot_cols = ("workclass", "education", "marital_status", "occupation",
                   "relationship", "race", "native_country")
    for col in onehot_cols:
        feature = X[col]
        feature_onehot = pd.get_dummies(feature, col)
        X = X.join(feature_onehot).drop(col, axis=1)
    # Gender is binary (here at least):
    X = X.assign(male = (X.sex == "Male")*1).drop("sex", axis=1)
    # 'fnlwgt' appears to be meaningless here as it's relative to the state, which isn't given:
    X = X.drop("fnlwgt", axis=1)
    # 'capital_gain' and 'capital_loss' never appear together and can
    # probably be turned to one feature:
    X = X.assign(net_capital = X.capital_gain - X.capital_loss).\
          drop(["capital_gain", "capital_loss"], axis=1)
    return drop_income(X).astype(float)

def list_to_ints(s, cutoff):
    return [ int(x+1-cutoff) for x in s ]

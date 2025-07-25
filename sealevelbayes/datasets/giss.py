import numpy as np
import pandas as pd
from sealevelbayes.datasets.catalogue import require_giss

def load_giss():
    return pd.read_csv(require_giss(), skiprows=1, delimiter=',').set_index('Year').replace('***', np.nan).astype(float).dropna()
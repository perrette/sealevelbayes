import numpy as np
import pandas as pd
from sealevelbayes.datasets.manager import require_dataset

def load_giss():
    return pd.read_csv(require_dataset("gistemp/tabledata_v4/GLB.Ts+dSST.csv"), skiprows=1, delimiter=',').set_index('Year').replace('***', np.nan).astype(float).dropna()
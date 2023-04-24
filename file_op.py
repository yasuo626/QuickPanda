import os.path
import pandas as pd
import numpy as np
import polars as pla
import matplotlib.pyplot as plt
import pickle
from collections import deque
from typing import Optional

df=pd.DataFrame({'1':[1,2,3],
                 '2':[2,3,4]})

print((pd.notna(df.iloc[:,1])).all())










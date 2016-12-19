# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


with pd.HDFStore("train.h5", "r") as train:
    df = train.get("train")

print(df.head())
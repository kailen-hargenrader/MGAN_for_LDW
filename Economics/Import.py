import pyarrow.feather as feather
import pandas as pd

path = "C:\\Users\Kailen\MGAN\Economics\cps.feather"
imported_df = feather.read_feather(path)
print(imported_df.head(20))
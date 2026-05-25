import pandas as pd 
import numpy as np 
from src.auto_ml_engine import scale_and_reduce, build_features 
from src.preprocess import load_and_clean 
df = load_and_clean('data/events.csv') 
features = build_features(df) 
result = scale_and_reduce(features) 
print('type:', type(result)) 
print('len tuple:', len(result) if isinstance(result, tuple) else 'NOT A TUPLE') 

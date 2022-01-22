import os
from matplotlib import widgets
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tqdm
import cv2 as cv 
import tensorflow as tf 
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import  Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler


from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


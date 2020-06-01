import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns; sns.set()
from matplotlib.colors import LinearSegmentedColormap

#Set path to input data file
path = ''

#Set event type (sSSE or lSSE)
event_type = ''

# Read data
data=pd.read_csv(path, sep=" ")

# Define dataset used for model training

# Replace ETS NaN (i.e. missing values) with ?
data.fillna('?', inplace=True)

# Get data with known values
data= data[data['Dip'].map(lambda x: x != '?')]
data = data[data['Age'].map(lambda x: x != '?')]
data = data[data['Sed_Thick'].map(lambda x: x != '?')]
data = data[data['Vel'].map(lambda x: x != '?')]
data = data[data['Rough'].map(lambda x: x != '?')]
data = data.drop('Sub_Zone', axis=1)
data = data.drop('Segment', axis=1)
data = data.drop('Longitude', axis=1)
data = data.drop('Latitude', axis=1)

if event_type == 'sSSE':
    palette = {'Y': '#2b936c', 'N':'#390e41', 'U':'gray'}
elif event_type == 'lSSE':
    palette = {'Y': '#fcc00e', 'N':'#390e41', 'U':'gray'}

# Now plot feature space
sns.pairplot(data, hue='sSSE', palette=palette, height=2, aspect=1.25)
plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.show()
plt.close()

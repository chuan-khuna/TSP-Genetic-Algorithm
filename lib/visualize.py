import numpy as np
import  pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
color_palette = sns.color_palette("muted")
sns.set_palette(color_palette)


def visualize_route(city_df, route):

    for i in range(len(route)):
        if i != len(route) - 1:
            c1 = city_df.iloc[route[i]]
            c2 = city_df.iloc[route[i+1]]
        else:
            c1 = city_df.iloc[route[i]]
            c2 = city_df.iloc[route[0]]
        
        # plot currenct city c1
        sns.scatterplot([c1['x']], [c1['y']], color='k', s=50)
        
        # plot route from current c1 to next c2
        sns.lineplot([c1['x'], c2['x']], [c1['y'], c2['y']], color='blue', alpha=0.75)

def distance_heatmap(distance_df):

    mask = np.zeros_like(distance_df)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(distance_df, square=True, annot=True, fmt=".1f", cmap='viridis', mask=mask)
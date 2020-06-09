import numpy as np
import pandas as pd


def gen_city_df(num_city, min_coord, max_coord):
    """
        Generate city dataframe by random integer coordinate.

        it will drop duplicate city.
    """

    city_coord = np.random.randint(min_coord, max_coord, size=(num_city, 2))

    return pd.DataFrame(city_coord, columns=['x', 'y']).drop_duplicates()


def cal_distance_matrix(city_df):
    """
        return distance matrix dataframe.
    """
    distance = pd.DataFrame()
    for i, row in city_df.iterrows():
        distance[str(i)] = np.round(
            np.sqrt(((city_df - city_df.iloc[i, :])**2).sum(axis=1)), 2)
    return distance


def arrange_route(route):
    """
        rearrange route, start with min city code
    """
    min_city_index = np.where(route == np.min(route))[0][0]
    route_size = len(route)
    return np.roll(route, route_size - min_city_index)


def calculate_distance(distance_df, route):
    sum_distance = 0
    for i in range(len(route)):
        if i != len(route) - 1:
            d = distance_df.iloc[route[i], route[i + 1]]
        else:
            d = distance_df.iloc[route[i], route[0]]

        sum_distance += d
    return np.round(sum_distance, 2)

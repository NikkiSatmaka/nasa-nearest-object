#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Useful functions to handle missing values
"""

import numpy as np
import pandas as pd


def prepare_imputation(data, variable, *args):
    """
    Prepare data for imputation

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe to be prepared for imputation
    variable : list
        List of columns to be imputed
    *args :
        List of special keywords representing the missing values

    Returns
    -------
    pandas.DataFrame
        Dataframe prepared for imputation
    """

    if data is None or variable is None:
        raise ValueError('data and variable must be specified')

    # prepare output dataframe
    output_data = data.copy()

    # replace missval with nan for features in impute_cols
    for col in variable:
        for missval in args:
            output_data[col] = output_data[col].replace(missval, np.nan)

    return output_data


def impute_na(data, variable, mean_value, median_value):
    """
    Function to Fill Missing Values with Zeroes, Mean, and Median

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe to be imputed
    variable : str
        Column to be imputed
    mean_value : float
        Mean value to be used for imputation
    median_value : float
        Median value to be used for imputation

    Returns
    -------
    pandas.DataFrame
        Dataframe with imputed values
    """

    # prepare output dataframe
    output_data = data.copy()

    output_data[variable+'_mean'] = output_data[variable].fillna(mean_value)
    output_data[variable+'_median'] = output_data[variable].fillna(median_value)
    output_data[variable+'_zero'] = output_data[variable].fillna(0)

    return output_data


def drop_missing_news(data, date_col, variable, target=None):
    """
    Function to check for instances of features which contains only links in
    the dataset with missing dates and return a DataFrame with
    the number of instances and their percentage

    Parameters:
    -----------
    data (dataframe): dataframe to be checked
    target : pandas Series or DataFrame
        Target variable name
    date_col : str
        Date column name
    variable : list
        List of columns to be imputed

    Returns
    -------
    DataFrame
        Number of instances in dataset
    """

    if data is None or variable is None:
        raise ValueError('data, target, and variable must be specified')

    # define whether to adjust target
    adjust_target = False
    if target is not None:
        adjust_target = True
        output_target = target.copy()

    # prepare output dataframe
    output_data = data.copy()

    # convert to datetime object and pass errors as missing
    output_data[date_col] = pd.to_datetime(output_data[date_col], errors='coerce')

    # drop rows with missing dates
    output_data = output_data.dropna(subset=[date_col])

    # drop instances which contains only links for the specified features
    for col in variable:
        output_data = output_data[~output_data[col].str.contains(r'^https?:\S+$', regex=True)]

    # adjust output_target to match the features
    if adjust_target:
        output_target = output_target.drop(output_target.index.difference(output_data.index))
        return output_data, output_target

    else:
        return output_data

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

"""
Useful functions to check dataframe
"""


def check_unique(data, col_type='both'):
    """
    Count the number of unique values in each features for 'numeric', 'categorical', or 'both'

    Parameters
    ----------
    data : DataFrame

    col_type : str
        The type of the column to filter. Either 'number', 'object', or 'both'

    Returns
    -------
    DataFrame
        Number of unique values of each features
    """

    # check if the column type is valid
    if col_type not in ('number', 'object', 'both'):
        raise ValueError('col_type must be either "number", "object", or "both"')

    # create a list if the column type is 'both'
    if col_type == 'both':
        col_type = ['number', 'object']

    # get the number of unique values in each column
    data_unique_count = pd.DataFrame.from_records(
        [(col, data[col].nunique()) for col in data.select_dtypes(include=col_type).columns],
        columns=['feats', 'num_unique']
    )
    data_unique_count['pct_unique'] = data_unique_count['num_unique'] / data.shape[0] * 100

    return data_unique_count


def check_missing(data):
    """
    Check the missing values in dataset

    Parameters
    ----------
    data : DataFrame

    Returns
    -------
    DataFrame
        Missing values in dataset
    """

    # create a DataFrame to store the missing values
    data_missing = pd.DataFrame(data.isna().sum().sort_values(ascending=False), columns=['tot_missing']).reset_index()

    # reset the index and make the features columns
    data_missing = data_missing.rename(columns={'index': 'feats'})

    # drop the rows with no missing values
    data_missing = data_missing[data_missing['tot_missing'] > 0]

    # calculate the percentage of missing values for each features
    data_missing['tot_missing_pct'] = data_missing['tot_missing'] / len(data) * 100

    return data_missing


def check_missing_special(data, *args):
    """
    Function to check special missing values in dataset
    (e.g. missing values in categorical features)
    and return a DataFrame with the missing values and their percentage

    Parameters:
    -----------
    data (dataframe): dataframe to be checked
    *args : list of special keywords representing the missing values

    Returns
    -------
    DataFrame
        Missing values in dataset
    """

    # create dictionary to store missing values
    missing_values = {
        'feats': [],
        'tot_missing': [],
        'tot_missing_pct': []
    }

    # Loop through the columns
    for col in data.columns:
        # set total missing values to 0
        tot_missval = 0
        # Loop through special keywords representing missing values
        for missval in args:
            if missval not in data[col].unique():
                continue
            # count the number of missing values
            tot_missval += len(data[data[col] == missval])

        # add the missing values to the dictionary
        missing_values['feats'].append(col)
        missing_values['tot_missing'].append(tot_missval)
        missing_values['tot_missing_pct'].append(tot_missval / len(data) * 100)

    # create a dataframe with the missing values dictionary
    missing_values = pd.DataFrame(missing_values)

    # drop the rows with no missing values
    missing_values = missing_values[missing_values['tot_missing'] > 0].reset_index(drop=True)

    return missing_values


def check_links_only(data):
    """
    Function to check for instances of features which contains only links in
    the dataset and return a DataFrame with the number of instances and their
    percentage

    Parameters:
    -----------
    data (dataframe): dataframe to be checked

    Returns
    -------
    DataFrame
        Number of instances in dataset
    """

    # create dictionary to store missing values
    missing_values = {
        'feats': [],
        'tot_missing': [],
        'tot_missing_pct': []
    }

    # Loop through the columns
    for col in data.columns:
        tot_missval = len(data[data[col].str.contains(r'^https?:\S+$', regex=True)])

        # add the missing values to the dictionary
        missing_values['feats'].append(col)
        missing_values['tot_missing'].append(tot_missval)
        missing_values['tot_missing_pct'].append(tot_missval / len(data) * 100)

    # create a dataframe with the missing values dictionary
    missing_values = pd.DataFrame(missing_values)

    # drop the rows with no missing values
    missing_values = missing_values[missing_values['tot_missing'] > 0].reset_index(drop=True)

    return missing_values

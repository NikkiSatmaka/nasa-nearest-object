#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from feature_engine.outliers import OutlierTrimmer, Winsorizer

"""
Useful functions to handle outliers
"""


def find_normal_boundaries(data, variable):
    """
    Calculate the boundaries outside which sit the outliers for a Gaussian distribution

    Parameters
    ----------
    data : DataFrame

    variable : string
        The feature of the DataFrame in which to the calculation will be performed

    Returns
    -------
    upper_boundary : float
        The computed upper boundary of the data

    lower_boundary : float
        The computed lower boundary of the data
    """

    upper_boundary = data[variable].mean() + 3 * data[variable].std()
    lower_boundary = data[variable].mean() - 3 * data[variable].std()

    return upper_boundary, lower_boundary


def find_skewed_boundaries(data, variable, fold):
    """
    Calculate the boundaries outside which sit the outliers for skewed distribution

    Parameters
    ----------
    data : DataFrame

    variable : string
        The feature of the DataFrame in which to the calculation will be performed

    fold : float
        The multiplier of IQR to calculate the boundaries

    Returns
    -------
    upper_boundary : float
        The computed upper boundary of the data

    lower_boundary : float
        The computed lower boundary of the data
    """

    IQR = data[variable].quantile(0.75) - data[variable].quantile(0.25)

    upper_boundary = data[variable].quantile(0.75) + (IQR * fold)
    lower_boundary = data[variable].quantile(0.25) - (IQR * fold)

    return upper_boundary, lower_boundary


def check_dist(data):
    """
    Check the Skewness and Distribution for each features in a dataset

    Parameters
    ----------
    data : DataFrame

    Returns
    -------
    DataFrame
        Skewness and distribution types of each features
    """

    # create a DataFrame containing the features of the dataset and their respective skewness
    data_skewness = pd.DataFrame(data.skew(), columns=['skew']).reset_index()

    # reset the index and make the features columns
    data_skewness = data_skewness.rename(columns={'index': 'feats'})

    # create a new column to describe whether the feature in the dataset is normal or skewed
    data_skewness['dist'] = np.where(
        (data_skewness['skew'] > -0.5) & (data_skewness['skew'] < 0.5),
        'normal',
        'skewed'
    )

    return data_skewness


def check_outlier(data, fold=1.5):
    """
    Check the outlier info for each features in a dataset

    Parameters
    ----------
    data : DataFrame

    fold : float
        The multiplier of IQR to calculate the boundaries for skewed distributions. It's either 1.5 or 3

    Returns
    -------
    DataFrame
        Outlier infos such as upper and lower boundary, and also the number of outliers for each features
    """

    if fold not in (1.5, 3):
        raise ValueError('Parameter fold only accepts numeric value of either 1.5 or 3')

    data_skewness = check_dist(data)

    # create a dictionary to store the outlier infos
    data_outlier = {
        'feats': [],
        'upper_bound': [],
        'lower_bound': [],
        'tot_right_tail': [],
        'tot_left_tail': [],
        'tot_right_tail_pct': [],
        'tot_left_tail_pct': [],
        'tot_outlier': [],
        'tot_outlier_pct': [],
    }

    # loop over each row in the `skewness` DataFrame
    # calculate each features upper and lower boundaries and the outlier percentage
    for row in data_skewness.index:
        col = data_skewness.iloc[row]['feats']

        if data_skewness.iloc[row]['dist'] == 'normal':
            upper_bound, lower_bound = find_normal_boundaries(data, col)
        else:
            upper_bound, lower_bound = find_skewed_boundaries(data, col, fold)

        tot_right_tail = len(data[data[col] > upper_bound])
        tot_left_tail = len(data[data[col] < lower_bound])
        tot_right_tail_pct = tot_right_tail / len(data) * 100
        tot_left_tail_pct = tot_left_tail / len(data) * 100
        tot_outlier =  tot_right_tail + tot_left_tail
        tot_outlier_pct = tot_right_tail_pct + tot_left_tail_pct

        data_outlier['feats'].append(col)
        data_outlier['upper_bound'].append(upper_bound)
        data_outlier['lower_bound'].append(lower_bound)
        data_outlier['tot_right_tail'].append(tot_right_tail)
        data_outlier['tot_left_tail'].append(tot_left_tail)
        data_outlier['tot_right_tail_pct'].append(tot_right_tail_pct)
        data_outlier['tot_left_tail_pct'].append(tot_left_tail_pct)
        data_outlier['tot_outlier'].append(tot_outlier)
        data_outlier['tot_outlier_pct'].append(tot_outlier_pct)

    data_outlier = pd.DataFrame(data_outlier)

    return data_outlier


def outlier_summary(data, fold=1.5):
    """
    Check the summary for outlier data, such as distribution and number of outliers for each features

    Parameters
    ----------
    data : DataFrame

    fold : float
        The multiplier of IQR to calculate the boundaries for skewed distributions. It's either 1.5 or 3

    Returns
    -------
    DataFrame
        Summary of outlier such as distribution and number of outliers for each features
    """

    data_skewness = check_dist(data)
    data_outlier = check_outlier(data, fold)

    outlier_summary_cols = ['feats', 'skew', 'dist', 'tot_outlier', 'tot_outlier_pct']

    data_outlier_summary = pd.merge(data_skewness, data_outlier, on=['feats'])
    data_outlier_summary = data_outlier_summary[outlier_summary_cols]

    return data_outlier_summary


def trim_cap_outliers(data, exception_list=[], target=None, fold=1.5):
    """
    Function to trim outliers based on the cap outliers

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame to trim outliers
    exception_list : list
        List of features to be excluded from trimming
    target : pandas Series or DataFrame
        Target variable name
    fold : float
        The multiplier of IQR to calculate the boundaries for skewed distributions. It's either 1.5 or 3

    Returns
    -------
    output_data : pandas DataFrame
        Trimmed data
    if target is not None:
        target : pandas Series or DataFrame
            Trimmed target variable
    """

    # define whether to adjust target
    adjust_target = False
    if target is not None:
        adjust_target = True
        output_target = target.copy()

    # prepare output dataframe
    output_data = data.copy()

    # run outlier detection
    data_outlier = outlier_summary(output_data, fold)

    # create a list of columns to trim outliers for normal distribution
    norm_trim_cols = data_outlier[
        (data_outlier['dist'] == 'normal') &
        (data_outlier['tot_outlier_pct'] < 5)
    ]['feats'].to_list()

    # create a list of columns to cap outliers for normal distribution
    norm_cap_cols = data_outlier[
        (data_outlier['dist'] == 'normal') &
        (data_outlier['tot_outlier_pct'] >= 5) &
        (data_outlier['tot_outlier_pct'] < 15)
    ]['feats'].to_list()

    # create a list of columns to trim outliers for skew distribution
    skew_trim_cols = data_outlier[
        (data_outlier['dist'] == 'skewed') &
        (data_outlier['tot_outlier_pct'] < 5)
    ]['feats'].to_list()

    # create a list of columns to cap outliers for skew distribution
    skew_cap_cols = data_outlier[
        (data_outlier['dist'] == 'skewed') &
        (data_outlier['tot_outlier_pct'] >= 5) &
        (data_outlier['tot_outlier_pct'] < 15)
    ]['feats'].to_list()

    # remove exception columns from the list
    norm_trim_cols = [x for x in norm_trim_cols if x not in exception_list]
    norm_cap_cols = [x for x in norm_cap_cols if x not in exception_list]
    skew_trim_cols = [x for x in skew_trim_cols if x not in exception_list]
    skew_cap_cols = [x for x in skew_cap_cols if x not in exception_list]

    # outlier trimming for normal distribution
    if len(norm_trim_cols) > 0:
        trim_norm = OutlierTrimmer(
            capping_method='gaussian',
            tail='both',
            fold=3,
            variables=norm_trim_cols,
            missing_values='ignore'
        )

        # trim outliers for normal distribution
        output_data = trim_norm.fit_transform(output_data)

        # adjust output_target to match the features
        if adjust_target:
            output_target = output_target.drop(output_target.index.difference(output_data.index))

    # outlier capping for normal distribution
    if len(norm_cap_cols) > 0:
        cap_norm = Winsorizer(
            capping_method='gaussian',
            tail='both',
            fold=3,
            variables=norm_cap_cols,
            missing_values='ignore'
        )

        # cap outliers for normal distribution
        output_data = cap_norm.fit_transform(output_data)

    # outlier trimming for skewed distribution
    if len(skew_trim_cols) > 0:
        trim_skew = OutlierTrimmer(
            capping_method='iqr',
            tail='both',
            fold=1.5,
            variables=skew_trim_cols,
            missing_values='ignore'
        )

        # trim outliers for skewed distribution
        output_data = trim_skew.fit_transform(output_data)

        # adjust output_target to match the features
        if adjust_target:
            output_target = output_target.drop(output_target.index.difference(output_data.index))

    # outlier capping for skewed distribution
    if len(skew_cap_cols) > 0:
        cap_skew = Winsorizer(
            capping_method='iqr',
            tail='both',
            fold=1.5,
            variables=skew_cap_cols,
            missing_values='ignore'
        )

        # cap outliers for skewed distribution
        output_data = cap_skew.fit_transform(output_data)

    if adjust_target:
        return output_data, output_target
    else:
        return output_data

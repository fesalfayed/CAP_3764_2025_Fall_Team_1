"""
Data Collection Module for Amazon Sales Analysis

This module handles loading and initial processing of the Amazon product dataset.
Data was collected using Amazon SP-API (Selling Partner API) and enriched with
computer vision-derived image quality metrics.

Dataset includes:
- Product metadata from Amazon SP-API
- Sales performance indicators (BSR, units sold)
- Customer feedback (reviews, ratings)
- Image quality metrics computed via computer vision
"""

import pandas as pd
import numpy as np


def load_bsr_data(filepath, verbose=False):
    """
    Load the BSR visual data CSV file

    Parameters:
    -----------
    filepath : str
        Path to the bsr_visual_data.csv file
    verbose : bool, default False
        If True, print diagnostic information

    Returns:
    --------
    pd.DataFrame
        Loaded dataframe

    Examples:
    ---------
    >>> df = load_bsr_data('data/raw/bsr_visual_data.csv', verbose=True)
    Dataset loaded successfully!
    Shape: (18148, 34)
    """
    df = pd.read_csv(filepath)

    if verbose:
        print(f"Dataset loaded successfully!")
        print(f"Shape: {df.shape}")
        df.info()

    return df


def clean_data(df, verbose=False):
    """
    Perform initial data cleaning

    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataframe
    verbose : bool, default False
        If True, print cleaning statistics

    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    dict
        Cleaning statistics including duplicates_removed and missing_values

    Examples:
    ---------
    >>> df_clean, stats = clean_data(df, verbose=True)
    Removed 1855 duplicate rows
    Missing values: {...}
    """
    df_clean = df.copy()

    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)

    missing = df_clean.isnull().sum()
    missing_values = missing[missing > 0].to_dict()

    if verbose:
        print(f"Removed {duplicates_removed} duplicate rows")
        if missing_values:
            print("\nMissing values per column:")
            print(missing[missing > 0])

    stats = {
        'duplicates_removed': duplicates_removed,
        'missing_values': missing_values
    }

    return df_clean, stats


def get_numerical_columns(df):
    """
    Get list of numerical columns

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    list
        List of numerical column names
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df):
    """
    Get list of categorical columns

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    list
        List of categorical column names
    """
    return df.select_dtypes(include=['object']).columns.tolist()

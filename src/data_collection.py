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


def load_bsr_data(filepath):
    """
    Load the BSR visual data CSV file

    Parameters:
    -----------
    filepath : str
        Path to the bsr_visual_data.csv file

    Returns:
    --------
    pd.DataFrame
        Loaded dataframe with basic info displayed
    """
    # Read the CSV file
    df = pd.read_csv(filepath)

    # Display basic info
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    df.info()

    return df


def clean_data(df):
    """
    Perform initial data cleaning

    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataframe

    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    # Create a copy to avoid modifying original
    df_clean = df.copy()

    # Remove duplicate rows
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)
    print(f"Removed {duplicates_removed} duplicate rows")

    # Display missing values
    missing = df_clean.isnull().sum()
    print("\nMissing values per column:")
    print(missing[missing > 0])

    return df_clean


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

"""
    Read the processed Clarknet workload.
    Link: ftp://ita.ee.lbl.gov/traces/
"""
import pandas as pd
import os


def get_workload():
    df = pd.read_csv('~/dataset/clark_net.csv')
    return df['Counts'].values

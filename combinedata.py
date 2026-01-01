"""Module merging all csv data files into an Excel sheet."""

import glob
import pandas as pd

import config

columns_out = ['date', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'bonus']

MY_PATH = r'./data'
# MY_PATH = r'../data'
MY_PATTERN = "/[0-9][0-9][0-9][0-9].csv"

all_files = glob.glob(MY_PATH + MY_PATTERN)

lst = []

for filename in all_files:
    df = pd.read_csv(filename, names=columns_out, index_col=None, header=0)
    df['date'] = pd.to_datetime(df['date'])
    lst.append(df)

merged_df = pd.concat(lst, axis=0, ignore_index=True)

# set index and sort by 'date' then dump to excel
merged_df.set_index(columns_out[0], inplace=True)
merged_df.sort_index(inplace=True)

# print("Writing output to ../data/data_all_l649.xlsx")
# merged_df.to_excel(f"{MY_PATH}/data_all_l649.xlsx")
merged_df.to_csv(f"{config.MY_PATH}/{config.COMBINED_FILE}.csv")

print(merged_df.dtypes)
print(merged_df.info)
print('Combine Data Done.')

"""Module merging all csv data files into an Excel sheet."""

import glob
import pandas as pd

import logging
import config
import app_logging

# initialize logging and log file
app_logging.init_logging()
logger = logging.getLogger(config.APP_NAME_SHORT)
loggerChild = logger.getChild(f"sub{__name__}")
loggerChild.info("Multi-year data merge started")

all_files = glob.glob(config.MY_PATH + config.FILE_NAME_PATTERN)
loggerChild.info(f"Found {len(all_files)} files to merge.")

lst = []
for filename in all_files:
    df = pd.read_csv(filename, names=config.columns_out, index_col=None, header=0)
    df['date'] = pd.to_datetime(df['date'])
    lst.append(df)
loggerChild.info("All files read into dataframes.")

merged_df = pd.concat(lst, axis=0, ignore_index=True)
loggerChild.info("Dataframes concatenated. length of merged data: {}".format(len(merged_df)))

# set index and sort by 'date' then dump to excel
merged_df.set_index(config.columns_out[0], inplace=True)
merged_df.sort_index(inplace=True)

merged_df.to_csv(f"{config.MY_PATH}/{config.COMBINED_FILE}.csv")
loggerChild.info(f"Merged data written to file: {config.COMBINED_FILE}.csv")

loggerChild.info('Multi-year data merge completed.')

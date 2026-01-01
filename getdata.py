""" read data from the web and store it locally in a .csv files """

from datetime import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd

import logging
import config
import app_logging


# initialize logging and log file
app_logging.init_logging()
logger = logging.getLogger(config.APP_NAME_SHORT)
loggerChild = logger.getChild(f"sub{__name__}")
loggerChild.info("Application Started")

# initialize panda frame for results
columns_out = ['date', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'bonus']
results_df = pd.DataFrame(columns=columns_out)

TargetURL = f"{config.RESULTS_URL}{config.CURRENT_YEAR}"

htmlPage = requests.get(TargetURL, timeout=10)
soup = BeautifulSoup(htmlPage.content, 'html.parser')
loggerChild.info(f"Retrieved URL: {TargetURL}")
loggerChild.info(f"Page Title: {soup.title}")
# find results table, get table body and rows in the table body
resultsTableTag = soup.find('table', attrs={'class':'mobFormat past-results'})
resultsTag = resultsTableTag.find('tbody')
resultsRows = resultsTag.find_all('tr')

# in each row find href an dextract date and then the number from all list elements
for result in resultsRows:
    resultsCells = result.find_all('td')
    # if row has only one cell then ignore it
    if len(resultsCells) == 1:
        continue
    # get the date and remove teh day of week
    dateStrLength = len(resultsCells[0].find('strong').get_text())
    drawDateTxt = resultsCells[0].get_text()[dateStrLength:]
    drawDate = datetime.strptime(drawDateTxt, "%B %d %Y")
    ballsTag = result.find_all('li', attrs={'class':'ball ball'})
    b1 = ballsTag[0].get_text()
    b2 = ballsTag[1].get_text()
    b3 = ballsTag[2].get_text()
    b4 = ballsTag[3].get_text()
    b5 = ballsTag[4].get_text()
    b6 = ballsTag[5].get_text()
    bonusBallTag = result.find('li', attrs={'class':'ball bonus-ball'})
    drawBonus = bonusBallTag.get_text()
    df = pd.DataFrame([[drawDate, b1, b2, b3, b4, b5, b6, drawBonus]], columns=columns_out)
    results_df = pd.concat([results_df, df], ignore_index=True)

results_df.to_csv(f"{config.MY_PATH}/{config.CURRENT_YEAR}.csv")
loggerChild.info(f"Data written to file: {config.CURRENT_YEAR}.csv")
loggerChild.info(f"From: {results_df['date'].min().strftime('%Y-%m-%d')} To: {results_df['date'].max().strftime('%Y-%m-%d')}")
# loggerChild.info(f"From: {results_df['date'].min()} To: {results_df['date'].max()}")

loggerChild.info("Application Ended")

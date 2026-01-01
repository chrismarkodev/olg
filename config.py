""" application configuration settings """

# set and retrieve application wide settings
APP_NAME = 'Lotto649 Data'
APP_NAME_SHORT = 'L649Data'
# add year '2024' at the end of the RESULTS_URL string
RESULTS_URL = "https://ca.lottonumbers.com/lotto-649/numbers/"

# data path
MY_PATH = r'./data'
# MY_PATH = r'../data'
# combine the URL and a year for a given year results page
CURRENT_YEAR = "2025"
COMBINED_FILE = "data_all_l649"
FILE_NAME_PATTERN = "/[0-9][0-9][0-9][0-9].csv"

columns_out = ['date', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'bonus']


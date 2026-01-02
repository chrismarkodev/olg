""" This is the main application module. """

import logging
import config
import app_logging
import getdata
import combinedata

def main():
    """ Main application function. """
    app_logging.init_logging()
    logger = logging.getLogger(config.APP_NAME_SHORT)
    loggerChild = logger.getChild(f"sub{__name__}")
    loggerChild.info("Applkication started")


    # Step 1: Retrieve data
    getdata.retrieve_data()

    # Step 2: Combine data
    combinedata.combine_data()

    logger.info("Application completed")

if __name__ == "__main__":
    main()
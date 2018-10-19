from datetime import datetime # to get the current date and time


def log(log_message):
    """
    - DOES: adds a log message "log_message" and its time stamp to a log file.
    """

    # open the log file and make sure that it's closed properly at the end of the
    # block, even if an exception occurs:
    with open("log.txt", "a") as log_file:
        # write the current time stamp and log message to logfile:
        log_file.write(datetime.strftime(datetime.today(),
                    "%Y-%m-%d %H:%M:%S") + ": " + log_message)
        log_file.write("\n") # (so the next message is put on a new line)


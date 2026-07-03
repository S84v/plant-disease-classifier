from pathlib import Path
import logging

import config


def configure_logging():
    logger = logging.getLogger()

    if logger.handlers:
        return
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file = config.LOGS_DIR / "app.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(filename=log_file, mode="a", encoding="utf-8")

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

if __name__ == "__main__":
    configure_logging()

    logger = logging.getLogger(__name__)

    logger.debug("This is a DEBUG message.")
    logger.info("This is an INFO message.")
    logger.warning("This is a WARNING message.")
    logger.error("This is an ERROR message.")

    try:
        1/0
    except ZeroDivisionError:
        logger.exception("An exception occured.")
    print("Logging test complete.")

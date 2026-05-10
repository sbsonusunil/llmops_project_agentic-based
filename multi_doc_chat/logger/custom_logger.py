import os
import logging
from datetime import datetime
import structlog



class CustomLogger:
    def __init__(self,log_dir="logs"):
        self.logs_dir=os.path.join(os.cwd(),log_dir)
        os.makedirs(self.logs_dir,exist_ok=True)
        log_file=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
        self.log_file_path=os.path.join(self.logs_dir,log_file)
        
    def get_logger(self,name=__file__):
        logger_name=os.path.basename(name)
        
        file_handler=logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        
        console_handler=logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        
        logging.basicConfig(level="logging.INFO",handlers=[console_handler,file_handler],format="%(message)s")
        structlog.configure(processors=[structlog.processors.TimeStamper(fmt="iso",utc=True,key="timestamp"),structlog.processor.add_log_level,structlog.processors.StackInfoRenderer(),structlog.processors.format_exc_info,structlog.processors.UnicodeDecoder(),structlog.processors.JSONRenderer(),structlog.processors.EventRenamer(to="event")],logger_factory=structlog.stdlib.LoggerFactory(),wrapper_class=structlog.stdlib.BoundLogger,cache_logger_on_first_use=True)
        
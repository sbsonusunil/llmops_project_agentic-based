from .custom_logger import CustomLogger

GLOBAL_LOGGER = CustomLogger().get_logger(__name__)

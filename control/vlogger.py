import logging

# 日志内容特征分析
# 自动删除超15天旧文件

class VLogger():
    def __init__(self, file_path) -> None:
        self.file_path = file_path

    @property    
    def logger(self) -> logging.Logger:
        logger = logging.getLogger('wolfsion')
        log_format = '%(asctime)s | %(message)s'
        formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
        file_handler = logging.FileHandler(self.file_path)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)
        return logger
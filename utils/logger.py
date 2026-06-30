import logging
import os
import sys
class Logger:
    _instance = None
    def __new__(cls, log_file="app.log", console_output = False):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)

            # ✅ Logger 설정
            cls._instance.logger = logging.getLogger("DQN_Logger")
            cls._instance.logger.setLevel(logging.DEBUG)

            if console_output :
                # ✅ 콘솔 핸들러 추가 (StreamHandler)
                # 🚨 수정: encoding='utf-8'을 추가하여 콘솔 출력 인코딩을 통일합니다.
                console_handler = logging.StreamHandler(sys.stdout) # sys.stdout을 명시
                console_handler.setLevel(logging.INFO)
                console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
                console_handler.setFormatter(console_formatter)
                cls._instance.logger.addHandler(console_handler)

            # ✅ 파일 핸들러 추가
            # 🚨 수정: encoding='utf-8'을 추가하여 파일 기록 인코딩을 통일합니다.
            file_handler = logging.FileHandler(log_file, mode="a", encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(file_formatter)
            cls._instance.logger.addHandler(file_handler)

        return cls._instance

    def get_logger(self):
        return self.logger  # ✅ Logger 객체 반환

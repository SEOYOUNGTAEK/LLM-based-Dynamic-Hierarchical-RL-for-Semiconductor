# utils.py
import json
import numpy as np

class NumpyJSONEncoder(json.JSONEncoder):
    """
    NumPy의 int64, float64 등을 Python 기본 타입으로 변환하여
    JSON 직렬화가 가능하도록 만듭니다.
    """
    def default(self, obj):
        if isinstance(obj, np.int64) or isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.float64) or isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)
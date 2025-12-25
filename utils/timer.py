# Timer utility for performance measurement
import time
from functools import wraps

class TimeCounter:
    @staticmethod
    def count_time(warmup_interval=4):
        def decorator(func):
            call_count = [0]
            total_time = [0.0]
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                call_count[0] += 1
                start = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                
                if call_count[0] > warmup_interval:
                    total_time[0] += elapsed
                    avg_time = total_time[0] / (call_count[0] - warmup_interval)
                    if call_count[0] % 100 == 0:
                        print(f"{func.__name__} avg time: {avg_time:.4f}s")
                
                return result
            return wrapper
        return decorator


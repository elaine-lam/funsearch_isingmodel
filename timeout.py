class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out")
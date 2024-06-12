import signal

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out")

def execute_code_with_timeout(code_str, timeout_seconds):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    state = 0

    try:
        # Execute the code string
        exec(code_str, globals())
        print("execution finished")
    except TimeoutError:
        print("Code execution timed out")
        state = 1
    except Exception as e:
        print("An error occurred during code execution:", e)
        state = 2
    finally:
        signal.alarm(0)  # Cancel the alarm
        return state


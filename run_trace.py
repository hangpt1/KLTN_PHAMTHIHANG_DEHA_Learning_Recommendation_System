import sys
import time

def trace_calls(frame, event, arg):
    if event == 'call':
        func_name = frame.f_code.co_name
        file_name = frame.f_code.co_filename
        if 'E-Learning-Recommendation-System' in file_name or 'pandas' in file_name or 'numpy' in file_name:
            print(f"Call to {func_name} in {file_name}")
    return trace_calls

sys.settrace(trace_calls)
print("Starting...")
import app
print("Done loading app!")

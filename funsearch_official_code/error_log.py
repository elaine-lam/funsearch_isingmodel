from datetime import date, datetime
import traceback
"""
This is a self defined error log class. It will log the error in a txt file inside the errorlog directory.
"""


def _getfilename() -> str:
    return "./errorlog/" + date.today().strftime("%Y-%m-%d") + ".txt"



def log(e:Exception)->str:
    if isinstance(e, IndentationError):
        with open(_getfilename(), 'a') as file: 
            file.writelines( "Log Time:" + datetime.now().strftime("%H:%M:%S") + "\nError:" + str(e) + '\n\n')
    else:
        with open(_getfilename(), 'a') as file: 
            file.writelines( "Log Time:" + datetime.now().strftime("%H:%M:%S") + "\nError:\n" + str(e) + "\nlocation:\n" + traceback.format_exc() + '\n\n')

print(log(TypeError))
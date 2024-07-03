from datetime import date, datetime
import traceback

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
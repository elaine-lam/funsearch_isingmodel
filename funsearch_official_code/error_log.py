from datetime import date, datetime
import logging
import os
import sys
import traceback

def _getfilename() -> str:
    return "./errorlog/" + date.today().strftime("%Y-%m-%d") + ".txt"



def log(e:Exception)->str:
    with open(_getfilename(), 'a') as file: 
        file.writelines( "Log Time:" + datetime.now().strftime("%H:%M:%S") + "\nError:\n" + str(e) + "\nlocation:\n" + traceback.format_exc() + '\n\n')

print(log(TypeError))
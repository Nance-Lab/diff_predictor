import warnings
import functools
import inspect
import time
import sys
import pathlib
from datetime import datetime 
from os import listdir, getcwd, chdir # Added by Nels...



string_types = (type(b''), type(u''), type(f''))


class tcolor:
    ResetAll = "\033[0m"

    Bold       = "\033[1m"
    Dim        = "\033[2m"
    Underlined = "\033[4m"
    Blink      = "\033[5m"
    Reverse    = "\033[7m"
    Hidden     = "\033[8m"

    ResetBold       = "\033[21m"
    ResetDim        = "\033[22m"
    ResetUnderlined = "\033[24m"
    ResetBlink      = "\033[25m"
    ResetReverse    = "\033[27m"
    ResetHidden     = "\033[28m"

    Default      = "\033[39m"
    Black        = "\033[30m"
    Red          = "\033[31m"
    Green        = "\033[32m"
    Yellow       = "\033[33m"
    Blue         = "\033[34m"
    Magenta      = "\033[35m"
    Cyan         = "\033[36m"
    LightGray    = "\033[37m"
    DarkGray     = "\033[90m"
    LightRed     = "\033[91m"
    LightGreen   = "\033[92m"
    LightYellow  = "\033[93m"
    LightBlue    = "\033[94m"
    LightMagenta = "\033[95m"
    LightCyan    = "\033[96m"
    White        = "\033[97m"

    BackgroundDefault      = "\033[49m"
    BackgroundBlack        = "\033[40m"
    BackgroundRed          = "\033[41m"
    BackgroundGreen        = "\033[42m"
    BackgroundYellow       = "\033[43m"
    BackgroundBlue         = "\033[44m"
    BackgroundMagenta      = "\033[45m"
    BackgroundCyan         = "\033[46m"
    BackgroundLightGray    = "\033[47m"
    BackgroundDarkGray     = "\033[100m"
    BackgroundLightRed     = "\033[101m"
    BackgroundLightGreen   = "\033[102m"
    BackgroundLightYellow  = "\033[103m"
    BackgroundLightBlue    = "\033[104m"
    BackgroundLightMagenta = "\033[105m"
    BackgroundLightCyan    = "\033[106m"
    BackgroundWhite        = "\033[107m"
    
        
        
def change_dir(directory = '.'):
    '''
    Simple funciton to change current directory.
    '''
    chdir(directory)
    workbook = getcwd()
    print(f'Using current directory for loading/saving: ' + 
          tcolor.Blue + tcolor.Bold + f'{workbook}' + tcolor.ResetAll)
    print(f'To change current directory, call change_dir(...)')

    
change_dir('.')
        
    
def deprecated(reason):
    '''
    Decorator which can be used to mark functions
    as deprecated. This will result in a warning being
    emitted when the decorated funciton is called.
    '''
    if isinstance(reason, string_types):
        # @deprecated is used with a reason attached
        # Only works for functions
        # ex: @deprecated("please use another function")
        def decorator(func1):
            fmt1 = "Call to deprecated function {name} ({reason})."
            @functools.wraps(func1)
            def new_func1(*args, **kwargs):
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn(fmt1.format(name=repr(func1.__name__), reason=reason),
                              category=DeprecationWarning,
                              stacklevel=2)
                warnings.simplefilter('default', DeprecationWarning)
                return func1(*args, **kwargs)
            return new_func1
        return decorator
    elif inspect.isfunction(reason):
        func2 = reason
        fmt2 = "Call to deprecated function {name}."
        @functools.wraps(func2)
        def new_func2(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(fmt2.format(name=repr(func2.__name__)),
                          category=DeprecationWarning,
                          stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)
            return func2(*args, **kwargs)
        return new_func2
    else:
        raise TypeError(repr(type(reason)))
    
        
def timer(func):
    '''
    Decorator which will time funciton from start to
    end.
    '''
    @functools.wraps(func)
    def decorator_timer(*args, **kwargs):
        print(f"Call to funciton {repr(func.__name__)} with timer set on")
        start_time = time.perf_counter()
        ret_val = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {repr(func.__name__)} in {run_time:.4f} secs")
        return ret_val
    return decorator_timer


def print_log(func, outputfile = './out.txt', access = 'w'):
    '''
    Decorator which saves all print statement to a log file. Note
    that this method temporarily changes sys.stdout which may not 
    be desired.
    '''
    @functools.wraps(func)
    def decorator_log(*args, **kwargs):
        print(f"Call to function {repr(func.__name__)} " +
              f"with print output saved to {outputfile}.")
        orig_stdout = sys.stdout
        f = open(outputfile, access)
        sys.stdout = f 
        header = (f'RUNNING FUNCTION {repr(func.__name__)} | ' +
                  f'MONTH-DAY-YEAR ' +
                  datetime.today().strftime("%b-%d-%Y | ") +
                  f'HOUR:MINUTE:SECOND ' +
                  datetime.today().strftime("%H:%M:%S"))
        print('-'*90)
        print("{:<90}".format(header))
        print()
        ret_val = func(*args, **kwargs)
        f.close()
        sys.stdout = orig_stdout
        return ret_val
    return decorator_log
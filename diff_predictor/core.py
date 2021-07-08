from os import chdir, getcwd
import warnings
import functools
import inspect
import time
import sys
from datetime import datetime


string_types = (type(b''), type(u''), type(f''))


def change_dir(directory='.'):
    '''
    Simple funciton to change current directory.
    '''
    chdir(directory)
    workbook = getcwd()
    print(f'Using current directory for loading/saving: ' +\
          "\033[34m" + "\033[1m" + f'{workbook}' + "\033[0m")
    print(f'To change current directory, call diff_predictor.core.change_dir(...)')


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
                warnings.warn(fmt1.format(name=repr(func1.__name__),
                                          reason=reason),
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


def print_log(func, outputfile='./out.txt', access='w'):
    '''
    Decorator which saves all print statement to a log file. Note
    that this method temporarily changes sys.stdout which may not
    be desired.
    '''
    @functools.wraps(func)
    def decorator_log(*args, **kwargs):
        print(f"Call to function {repr(func.__name__)} " +\
              f"with print output saved to {outputfile}.")
        orig_stdout = sys.stdout
        f = open(outputfile, access)
        sys.stdout = f
        header = (f'RUNNING FUNCTION {repr(func.__name__)} | ' +\
                  f'MONTH-DAY-YEAR ' +\
                  datetime.today().strftime("%b-%d-%Y | ") +\
                  f'HOUR:MINUTE:SECOND ' +\
                  datetime.today().strftime("%H:%M:%S"))
        print('-'*90)
        print("{:<90}".format(header))
        print()
        ret_val = func(*args, **kwargs)
        f.close()
        sys.stdout = orig_stdout
        return ret_val
    return decorator_log


def search_nested_dict(value, keyword):
    '''
    Function which searches a nested dictionary
    recurrently for a keyword.
    Parameters
    ----------
    value : dict
        dictionary to search through
    keyword : str
        keyword to find withing dictionary
    Returns
    -------
    result
        resulting value of search
    '''
    result = None
    if keyword in value.keys() and not isinstance(value[keyword], dict):
        return(value[keyword])
    else:
        for key in value.keys():
            if isinstance(value[key], dict):
                result = search_nested_dict(value[key], keyword)
            if result is not None:
                break
    return result


def is_numeric(value):
    '''
    Function to check if a value is a numeric value or not
    Parameters
    ----------
    value : any value
        value to check if it is numeric or not
    Returns
    -------
    Will return a float of tht value if it is numeric or false otherwise
    '''
    try:
        return float(value)
    except Exception:
        return False
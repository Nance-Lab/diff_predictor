import warnings
import functools
import inspect
import time


string_types = (type(b''), type(u''), type(f''))


def deprecated(reason):
    """
    Decorator which can be sused to mark functions
    as deprecated. This will result in a warning being
    emitted when the decorated funciton is called.
    """
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
        fmt2 = "Call to deprecated funciton {name}."
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
    """
    Decorator which will time funciton from start to
    end.
    """
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
    Parameters
    ----------
    value : str
        String that will be checked if it can be converted to a number
    Returns
    -------
    Either the float value if string can be turned into float or 
    boolean false
    '''
    try:
        return float(value)
    except Exception:
        return False
    
    

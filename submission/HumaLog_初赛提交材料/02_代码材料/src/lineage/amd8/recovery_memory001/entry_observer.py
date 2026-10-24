"""Install observers before the original guarded main, preserving spawn semantics."""
import os
import sys


def run_observed(run_path,path,install):
    target=os.path.abspath(os.fspath(path))
    previous=sys.gettrace()
    assert previous is None,'Do not replace an existing external debugger'
    observed=False
    def trace(frame,event,arg):
        nonlocal observed
        if event=='call' and frame.f_code.co_filename==target and frame.f_code.co_name=='main':
            assert frame.f_globals['__name__']=='__main__'
            sys.settrace(previous)
            install(frame.f_globals)
            observed=True
        return None
    sys.settrace(trace)
    try:
        result=run_path(target,run_name='__main__')
    finally:
        sys.settrace(previous)
    assert observed,'Original guarded main was not observed'
    return result

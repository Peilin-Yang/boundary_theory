import os
import subprocess
import time
from inspect import currentframe, getframeinfo


def IndriRunQuery(query_file_path, output_path, method=None):
    """
    This function should be outside the class so that it can be executed by 
    children process.
    """
    frameinfo = getframeinfo(currentframe())
    print frameinfo.filename+':'+str(frameinfo.lineno),
    print query_file_path, method, output_path
    with open(output_path, 'wb') as f:
        if method:
            subprocess.Popen(['IndriRunQuery', query_file_path, method], bufsize=-1, stdout=f)
        else:
            subprocess.Popen(['IndriRunQuery', query_file_path], bufsize=-1, stdout=f)
        f.flush()
        os.fsync(f.fileno())
    time.sleep(5)

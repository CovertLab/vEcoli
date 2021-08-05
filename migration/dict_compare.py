import numpy as np
from unum import Unum
from migration.migration_utils import percent_error
import numbers

def recursive_compare(d1, d2, level='root'):
    if isinstance(d1, dict) and isinstance(d2, dict):
        if d1.keys() != d2.keys():
            s1 = set(d1.keys())
            s2 = set(d2.keys())
            print('{:<20} + {} - {}'.format(level, s1-s2, s2-s1))
            common_keys = s1 & s2
        else:
            common_keys = set(d1.keys())

        for k in common_keys:
            recursive_compare(d1[k], d2[k], level='{}.{}'.format(level, k))

    elif isinstance(d1, list) and isinstance(d2, list):
        if len(d1) != len(d2):
            print('{:<20} len1={}; len2={}'.format(level, len(d1), len(d2)))
        common_len = min(len(d1), len(d2))

        for i in range(common_len):
            recursive_compare(d1[i], d2[i], level='{}[{}]'.format(level, i))
            
    elif isinstance(d1, np.ndarray) or isinstance(d2, np.ndarray):
        recursive_compare(list(d1), list(d2), level='{}'.format(level))
    
    elif isinstance(d1, Unum) and isinstance(d2, Unum):
        recursive_compare(d1.asNumber(), d2.asNumber(), level='{}'.format(level))
    
    elif isinstance(d1, numbers.Number) and isinstance(d2, numbers.Number):
        if not np.isclose(d1, d2, rtol=0.01, atol=1):
            print('{:<20} {} != {}'.format(level, d1, d2))
            
    elif isinstance(d1, set) and isinstance(d2, set):
        if len(d1 - d2) != 0:
            print('{:<20} {} in 1st but not 2nd set'.format(level, d1 - d2))
    
    else:
        try:
            if d1 != d2:
                print('{:<20} {} != {}'.format(level, d1, d2))
        except ValueError:
            print('{:<20} {} != {}'.format(level, d1, d2))

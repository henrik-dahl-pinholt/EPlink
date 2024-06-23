
import os
import sys

import EPlink.Compute
sys.path.insert(0, os.path.abspath("../"))
import EPlink



def test_full_iteration():
    """Tests that the runner class goes through all elements of the iterable."""
    iterable = range(100)
    func = lambda x: x
    runner = EPlink.Compute.Runner(iterable,func)
    result = runner.Run(print_stats=False)
    assert len(result) == 100
    assert result == list(iterable)
    
    # Check that the result works while printing stats
    iterable = range(4)
    func = lambda x: x
    runner = EPlink.Compute.Runner(iterable,func)
    result = runner.Run(print_stats=True)
    assert len(result) == 4
    assert result == list(iterable)

    
    

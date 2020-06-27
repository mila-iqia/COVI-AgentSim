import simpy
import datetime
from ._native import BaseEnvironment


class Environment(BaseEnvironment, simpy.Environment):
    """
    This class serves to inherit from simpy.Environment important logic, while
    also mixing in the data layout specified at the C level by _native.BaseEnvironment.
    """
    
    def __init__(self, initial_time=0):
        """
        Initialize a new, optimized Environment.
        
        Invokes explicitly the constructors for both superclasses.
        """
        
        if isinstance(initial_time, (int, float)):
            BaseEnvironment  .__init__(self, initial_time)
            simpy.Environment.__init__(self, self.now)
            self.initial_timestamp = datetime.datetime.fromtimestamp(self.now)
        else:
            BaseEnvironment  .__init__(self, initial_time.timestamp())
            simpy.Environment.__init__(self, self.now)
            self.initial_timestamp = initial_time
    
    @property
    def timestamp(self):
        """
        Returns: datetime.datetime: Current date.
        
        Currently, the date is computed without any compensation for Daylight
        Savings Time.
        """
        
        # return datetime.datetime.fromtimestamp(self.now)
        return self.initial_timestamp + datetime.timedelta(
            seconds=self.now-self.ts_initial)
    
    def time_of_day(self):
        """
        Time of day in iso format
        datetime(2020, 2, 28, 0, 0) => '2020-02-28T00:00:00'

        Returns:
            str: iso string representing current timestamp
        """
        return self.timestamp.isoformat()


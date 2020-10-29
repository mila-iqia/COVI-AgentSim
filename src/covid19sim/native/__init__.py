import simpy
import datetime
from ._native import BaseEnvironment
from covid19sim.utils.lmdb import LMDBSortedMap


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

        self.sorted_map = LMDBSortedMap()
    
    def init_timed_barrier(self, district_id, allowed_drift, sleep_interval):
        self.district_id = district_id
        self.allowed_drift = allowed_drift
        self.sleep_interval = sleep_interval

    def step(self) -> None:
        """
        Take environment step while waiting for the slowest district to catch up
        """
        last_timestamp = self._now
        super(simpy.Environment, self).step()
        if last_timestamp < self._now:
            self.sorted_map.replace(last_timestamp, self._now, self.pid)
            # if the slowest district's drift is large, sleep
            while (self._now - self.sorted_map.first()) >= self.allowed_drift:
                sleep(self.sleep_interval)

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

    def __reduce__(self):
        """
        Helper function for pickling
        """
        args = (self.initial_timestamp, )
        state = {'_now': self._now}
        return (self.__class__, args, state)

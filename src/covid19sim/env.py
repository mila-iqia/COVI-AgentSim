
import simpy
import datetime

class Env(simpy.Environment):
    """
    Custom simpy.Environment
    """

    def __init__(self, initial_timestamp):
        """
        Args:
            initial_timestamp (datetime.datetime): The environment's initial timestamp
        """
        self.initial_timestamp = datetime.datetime.combine(initial_timestamp.date(),
                                                           datetime.time())
        self.ts_initial = int(self.initial_timestamp.timestamp())
        super().__init__(self.ts_initial)

    @property
    def timestamp(self):
        """
        Returns:
            datetime.datetime: Current date.
        """
        #
        ##
        ## The following is preferable, but not equivalent to the initial
        ## version, because timedelta ignores Daylight Saving Time.
        ##
        #
        #return datetime.datetime.fromtimestamp(int(self.now))
        #
        return self.initial_timestamp + datetime.timedelta(
            seconds=self.now-self.ts_initial)

    def minutes(self):
        """
        Returns:
            int: Current timestamp minute
        """
        return self.timestamp.minute

    def hour_of_day(self):
        """
        Returns:
            int: Current timestamp hour
        """
        return self.timestamp.hour

    def day_of_week(self):
        """
        Returns:
            int: Current timestamp day of the week
        """
        return self.timestamp.weekday()

    def is_weekend(self):
        """
        Returns:
            bool: Current timestamp day is a weekend day
        """
        return self.day_of_week() >= 5

    def time_of_day(self):
        """
        Time of day in iso format
        datetime(2020, 2, 28, 0, 0) => '2020-02-28T00:00:00'

        Returns:
            str: iso string representing current timestamp
        """
        return self.timestamp.isoformat()


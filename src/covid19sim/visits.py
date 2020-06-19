from collections import defaultdict

class Visits(object):
    """
    [summary]
    """

    def __init__(self):
        """
        [summary]
        """
        self.parks = defaultdict(int)
        self.stores = defaultdict(int)
        self.hospitals = defaultdict(int)
        self.miscs = defaultdict(int)

    @property
    def n_parks(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        return len(self.parks)

    @property
    def n_stores(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        return len(self.stores)

    @property
    def n_hospitals(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        return len(self.hospitals)

    @property
    def n_miscs(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        return len(self.miscs)


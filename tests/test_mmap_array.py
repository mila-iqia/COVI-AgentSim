import os
import unittest
from covid19sim.utils.mmap import MMAPArray
from time import sleep
from itertools import zip_longest

class TestSharedMemory(unittest.TestCase):

    def setUp(self):
        """
        Sets up a shared memory
        """
        self.arr = MMAPArray(num_items=6, item_size=100)

    def test_len_functional(self):
        """
        Tests if the len of the object returns correct number of bytes filled
        """
        self.assertEqual(0, len(self.arr))
        self.arr[0] = dummy_object(12)
        self.arr[2] = dummy_object(21)
        self.assertEqual(3, len(self.arr))

    def test_get_set_roundtrip_functional(self):
        """
        Tests if the appened objects can be fetched by the forked processes
        """
        test_objects = [ [1,2], {1:2}, {1,2}, '12', b'12', dummy_object(12) ]

        pid = os.fork()
        # in the child process check if the returned object from memory is
        # equal to the original test object, further check that the returned
        # objects are rebuilt from scratch, and they posses different ids
        if pid == 0:
            # append all test objects to shared memory and store their positions
            for indx, test_object in enumerate(test_objects):
                self.arr[indx] = test_object
                self.assertEqual(self.arr[indx], test_object)
                self.assertNotEqual(id(self.arr[indx]), id(test_object))
        else:
            sleep(1)
            for indx, test_object in enumerate(test_objects):
                self.assertEqual(self.arr[indx], test_object)
                self.assertNotEqual(id(self.arr[indx]), id(test_object))

    def test_iterate_all_pairs_functional(self):
        """
        Tests if iterating through all stored items return original items
        """
        d1 = dummy_object(12)
        d2 = dummy_object(21)
        self.arr[0] = d1
        self.arr[2] = d2
        expected_items = [d1, None, d2, None, None, None]

        for stored_item, expected_item in zip_longest(self.arr, expected_items):
            self.assertEqual(stored_item, expected_item)

                
    def tearDown(self):
        """
        Dereference shared memory object
        """
        del self.arr

class dummy_object():
    """
    Dummy object used in tests above to showcase pickling into shared memory
    """
    def __init__(self, value):
        """
        initialize value attribute of the dummy object
        """
        self.value = value
        
    def __eq__(self, other):
        """
        Sets the equality criteria based on the equality of the dummy values
        """
        return self.value == other.value

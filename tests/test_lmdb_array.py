import os
import unittest
from covid19sim.utils.lmdb import LMDBArray
from itertools import zip_longest
from time import sleep

class TestLMDBArray(unittest.TestCase):

    def setUp(self):
        """
        Sets up a shared memory
        """
        self.arr = LMDBArray()

    def test_len_functional(self):
        """
        Tests if the len of the object returns correct number of bytes filled
        """
        self.assertEqual(len(self.arr), 0) # empty array
        self.arr.append(888)
        self.arr.append((777, 999))
        self.assertEqual(len(self.arr), 2)
        with self.assertRaises(AttributeError):
            del self.arr[0]

    def test_access_functional(self):
        """
        Tests if the appened objects can be fetched by the forked processes
        """
        pid = os.fork()
        if pid == 0:
            self.assertTrue(self.arr.append(  "test"  ))
            self.assertTrue(self.arr.append( (6,7,8) ))
            self.assertTrue(self.arr.append(    9    ))
            self.assertTrue(self.arr.append(  [10]  ))
            self.assertEqual(self.arr[0], "test" )
            self.assertEqual(self.arr[1], (6,7,8) )
            self.assertEqual(self.arr[2], 9 )
            self.assertEqual(self.arr[3], [10] )

            # get IndexError
            with self.assertRaises(IndexError):
                self.arr[4]

            # set IndexError
            with self.assertRaises(IndexError):
                self.arr[4] = 11

            self.arr[0] = "test2"
            for val, i in zip_longest(self.arr, ("test2", (6,7,8), 9, [10])):
                self.assertEqual(val, i)

            sleep(0.5)
            self.arr.reset()
            self.assertEqual(len(self.arr), 0)
            with self.assertRaises(IndexError):
                self.arr[0]
        else:
            sleep(0.5)
            for val, i in zip_longest(self.arr, ("test2", (6,7,8), 9, [10])):
                self.assertEqual(val, i)
            sleep(0.5)
                
    def tearDown(self):
        """
        Dereference lmdb database
        """
        del self.arr

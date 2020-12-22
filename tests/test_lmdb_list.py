import os
import unittest
from covid19sim.utils.lmdb import LMDBList
from itertools import zip_longest
from time import sleep

class TestLMDBList(unittest.TestCase):

    def setUp(self):
        """
        Sets up a shared memory
        """
        self.lmdb_list = LMDBList()

    def test_len_functional(self):
        """
        Tests if the len of the object returns correct number of bytes filled
        """
        self.assertEqual(len(self.lmdb_list), 0) # empty list
        self.lmdb_list.append(888)
        self.lmdb_list.append((777, 999))
        self.assertEqual(len(self.lmdb_list), 2)
        with self.assertRaises(AttributeError):
            del self.lmdb_list[0]

    def test_access_functional(self):
        """
        Tests if the appened objects can be fetched by the forked processes
        """
        pid = os.fork()
        if pid == 0:
            self.assertTrue(self.lmdb_list.append(  "test"  ))
            self.assertTrue(self.lmdb_list.append( (6,7,8) ))
            self.assertTrue(self.lmdb_list.append(    9    ))
            self.assertTrue(self.lmdb_list.append(  [10]  ))
            self.assertEqual(self.lmdb_list[0], "test" )
            self.assertEqual(self.lmdb_list[1], (6,7,8) )
            self.assertEqual(self.lmdb_list[2], 9 )
            self.assertEqual(self.lmdb_list[3], [10] )

            # get IndexError
            with self.assertRaises(IndexError):
                self.lmdb_list[4]

            # set IndexError
            with self.assertRaises(IndexError):
                self.lmdb_list[4] = 11

            self.lmdb_list[0] = "test2"
            for val, i in zip_longest(self.lmdb_list, ("test2", (6,7,8), 9, [10])):
                self.assertEqual(val, i)

            sleep(0.5)
            self.lmdb_list.reset()
            self.assertEqual(len(self.lmdb_list), 0)
            with self.assertRaises(IndexError):
                self.lmdb_list[0]
        else:
            sleep(0.5)
            for val, i in zip_longest(self.lmdb_list, ("test2", (6,7,8), 9, [10])):
                self.assertEqual(val, i)
            sleep(0.5)
                
    def tearDown(self):
        """
        Dereference lmdb database
        """
        del self.lmdb_list

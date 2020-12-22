import os
import unittest
from covid19sim.utils.lmdb import LMDBSortedDict
from itertools import zip_longest
from time import sleep

class TestLMDBSortedDict(unittest.TestCase):

    def setUp(self):
        """
        Sets up a shared memory
        """
        self.map = LMDBSortedDict()

    def test_len_functional(self):
        """
        Tests if the len of the object returns correct number of bytes filled
        """
        self.assertEqual(len(self.map), 0) # empty map
        self.map[5]= 888
        self.map[4]= (777, 999)
        self.map[3]= (777, 888, 999)
        self.assertEqual(len(self.map), 3)
        del self.map[3]
        self.assertEqual(len(self.map), 2)
        self.assertEqual(self.map.pop_all(), [(4, [(777, 999)]), (5, [888])])
        self.assertEqual(len(self.map), 0)

    def test_access_functional(self):
        """
        Tests if the appened objects can be fetched by the forked processes
        """
        pid = os.fork()
        if pid == 0:
            self.map[777] = 9
            self.map[3] = "test"
            self.map[999] = [10,11]
            self.map[55] = (6,7,8)
            self.map[999] = [10,11]
            self.assertEqual(self.map[3], ["test"] )
            self.assertEqual(self.map[55], [(6,7,8)] )
            self.assertEqual(self.map[777], [9] )
            self.assertEqual(self.map[999], [[10,11]] )

            # get KeyError
            with self.assertRaises(KeyError):
                self.map[0]

            self.map.append(index=0, value=4)
            self.map.append(index=0, value=5)

            self.map[55] = ()

            for (key, val), (i, j) in zip_longest(self.map.items(), ((0,4), (0,5), (3,"test"), (55,()), (777,9), (999,[10,11]))):
                self.assertEqual(key, i)
                self.assertEqual(val, j)

            for val, i in zip_longest(self.map.values(), (4, 5, "test", (), 9, [10,11])):
                self.assertEqual(val, i)

            for key, i in zip_longest(self.map, (0,0,3,55,777,999)):
                self.assertEqual(key, i)                

            for key, i in zip_longest(self.map.keys(), (0,0,3,55,777,999)):
                self.assertEqual(key, i)

            sleep(0.5)
            self.map.reset()
            self.assertEqual(len(self.map), 0)
            with self.assertRaises(KeyError):
                self.map[0]
        else:
            sleep(0.5)
            for (key, val), (i, j) in zip_longest(self.map.items(), ((0,4), (0,5), (3,"test"), (55,()), (777,9), (999,[10,11]))):
                self.assertEqual(key, i)
                self.assertEqual(val, j)
            sleep(0.5)
                
    def tearDown(self):
        """
        Dereference lmdb database
        """
        del self.map

class TestLMDBSortedDictNoDup(unittest.TestCase):

    def setUp(self):
        """
        Sets up a shared memory without duplicate keys
        """
        self.map = LMDBSortedDict(duplicate_key_allowed=False)

    def test_len_functional(self):
        """
        Tests if the len of the object returns correct number of bytes filled
        """
        self.assertEqual(len(self.map), 0) # empty map
        self.map[5]= 888
        self.map[4]= (777, 999)
        self.map[3]= (777, 888, 999)
        self.assertEqual(len(self.map), 3)
        del self.map[3]
        self.assertEqual(len(self.map), 2)
        self.assertEqual(self.map.pop_all(), [(4, (777, 999)), (5, 888)])
        self.assertEqual(len(self.map), 0)

    def test_access_functional(self):
        """
        Tests if the appened objects can be fetched by the forked processes
        """
        pid = os.fork()
        if pid == 0:
            self.map[777] = 9
            self.map[3] = "test"
            self.map[999] = [10,11]
            self.map[55] = (6,7,8)
            self.map[999] = [10,11]
            self.assertEqual(self.map[3], "test" )
            self.assertEqual(self.map[55], (6,7,8) )
            self.assertEqual(self.map[777], 9 )
            self.assertEqual(self.map[999], [10,11] )

            # get KeyError
            with self.assertRaises(KeyError):
                self.map[0]

            with self.assertRaises(AttributeError):
                self.map.append(index=0, value=4)

            self.map[55] = ()

            for (key, val), (i, j) in zip_longest(self.map.items(), ((3,"test"), (55,()), (777,9), (999,[10,11]))):
                self.assertEqual(key, i)
                self.assertEqual(val, j)

            for val, i in zip_longest(self.map.values(), ("test", (), 9, [10,11])):
                self.assertEqual(val, i)

            for key, i in zip_longest(self.map, (3,55,777,999)):
                self.assertEqual(key, i)                

            for key, i in zip_longest(self.map.keys(), (3,55,777,999)):
                self.assertEqual(key, i)

            sleep(0.5)
            self.map.reset()
            self.assertEqual(len(self.map), 0)
            with self.assertRaises(KeyError):
                self.map[0]
        else:
            sleep(0.5)
            for (key, val), (i, j) in zip_longest(self.map.items(), ((3,"test"), (55,()), (777,9), (999,[10,11]))):
                self.assertEqual(key, i)
                self.assertEqual(val, j)
            sleep(0.5)
                
    def tearDown(self):
        """
        Dereference lmdb database
        """
        del self.map

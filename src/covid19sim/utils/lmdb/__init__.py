import os
import sys
import lmdb
import shutil
import struct
import typing
import tempfile
import dill as pickle
from itertools import islice

class LMDBBase():
    """
    Base class that lmdb data structure inherit from
    LMDB is not as performant as MMAP, but it is the appropriate data structure when
    multiple processes write to the same index as the data is protected against corrupted
    read/write using its locking mechanism in place.
    :def: type code describes list values' data type
    :def: byte order describes if string format should be big/little endian
    :def: the format string is the format by which indices in lmdb are serialized to bytes
    please refer to https://docs.python.org/3.8/library/struct.html#format-characters
    for integerkey db of LMDB, one is required to use either 'I' or 'N', equivalent to
    unsigned int and size_t. Also, since we use big endian in order to preserve order
    when using append, we can't rely on native size_t's 'N' to preserve the order for us
    If we are interested in other types of data types, the considerations above need to be
    taken into account when planning
    """
    index_format_string: str='>I' # corresponding to unsigned int in big endian
    parent_temp_dir = os.path.join( tempfile.gettempdir(), 'covi-simulation-lmdb' )
    os.makedirs(parent_temp_dir, exist_ok=True)

    def __init__(self, memory_size: int = 20*1024*1024, **kwargs):
        """
        Initializes a lmdb database of given memory size and string format
        for items to be stored in the database
        :param memory_size: the allocated size of the shared memory file in bytes
        """
        self.path = tempfile.mkdtemp(dir=LMDBBase.parent_temp_dir)
        self._init_db(path=self.path, memory_size=memory_size, **kwargs)

    def _init_db(self, **kwargs):
        """
        initialize db based on data structure
        """
        raise NotImplemented

    def reset(self):
        """
        Removes all items in the database
        :returns: True if the transaction was successful, False otherwise
        """
        with self.env.begin(db=self.db, write=True) as txn:
            return txn.drop(db=self.db, delete=False)

    def __del__(self):
        """
        Removes the temporary directory holding the repository (at destruction)
        """
        try:
            shutil.rmtree(self.path)
        except FileNotFoundError:
            pass

    def __len__(self) -> int:
        """
        Returns the length of list
        :returns: the number of items stored in the lmdb database
        """
        with self.env.begin(db=self.db, buffers=True) as txn:
            return txn.stat(db=self.db)['entries']

    def iload_iter(self, buf: memoryview) -> int:
        """
        Iteratively unpacks index from the memoryview buffer into int
        :param buf: the memoryview buffer to unpack into index
        :yields: an int object contained in the buffer
        """
        for _item in struct.iter_unpack(self.index_format_string, buf):
            yield _item[0]

    def iloads(self, index: bytes) -> int:
        """
        Unpacks integer index from the bytes object into int
        :param index: the bytes format of index
        :returns: the int index
        """
        return int.from_bytes(index, byteorder='big')

    def idumps(self, index: int) -> bytes:
        """
        Packs index into a bytes value
        :param index: the index to be packed into bytes
        :returns: a bytes object equivalent to the bytes according to the format string
        """
        return struct.pack(self.index_format_string, index)

    def vloads(self, value: bytes) -> typing.Any:
        """
        Unpacks object from the bytes value into value object
        :param value: the bytes format of value
        :returns: the value object
        """
        return pickle.loads(value)

    def vdumps(self, value: typing.Any) -> bytes:
        """
        Packs value into a bytes value
        :param value: the value to be packed into bytes
        :returns: pickled bytes object
        """
        return pickle.dumps(value)

class LMDBList(LMDBBase):
    """
    Representing an LMDB database holding a pythonic list interface. Each LMDBList
    is associated with an integer index. Practically speaking, the index could easily be
    promoted to any other key type other than int by overriding idumps and iloads.
    """

    def _init_db(self, path: str, memory_size: int):
        """
        opens a db in the env and returns env, the db
        """
        USE_SPARSE_FILES = sys.platform != 'darwin'
        self.env = lmdb.Environment(path=self.path, map_size=memory_size, writemap=USE_SPARSE_FILES)
        self.db = env.open_db()

    def __getitem__(self, index: int) -> typing.Tuple[int]:
        """
        Returns all of the item stored at the given index
        :param index: the index whose item we want to get
        :returns: the item at index
        """
        if index >= self.env.stat()['entries']:
            raise IndexError(f"Index {index} is out of range")
        with self.env.begin(db=self.db) as txn:
            cur = txn.cursor(db=self.db)
            iterator = cur.iternext(keys=False, values=True)
            value: bytes = next( islice(iterator, index, None) )
            return self.vloads(value)

    def __delitem__(self, index: int):
        """
        Raises an AttributeError
        """
        raise AttributeError("Direct item deletion is prohibited; see reset")

    def __setitem__(self, index: int, values: typing.Any):
        """
        Modifies the item located at given index. This method overwrites the item
        located at the given index
        :param index: the index whose item we want to set
        :param values: the item to be overwriten at index, can be either int or a tuple
        :returns (if not suppressed): True if the transaction was successful, False otherwise
        """
        if index >= self.env.stat()['entries']:
            raise IndexError(f"Index {index} is out of range")
        with self.env.begin(db=self.db, write=True) as txn:
            return txn.put(db=self.db, key=self.idumps(index), value=self.vdumps(values), overwrite=True)

    def __iter__(self) -> typing.Any:
        """
        Iterates through all of the lmdb db items
        """
        with self.env.begin(db=self.db) as txn:
            cur = txn.cursor(db=self.db)
            for value in cur.iternext(keys=False, values=True):
                yield self.vloads(value)

    def __contains__(self, value: typing.Any):
        """
        checks if list contains value
        :returns: True is value is contained and False otherwise
        """
        for values_tuple in self:
            if value in values_tuple:
                return True
        else:
            return False

    def append(self, value: typing.Any):
        """
        Appends value to the database
        :param value: value to be appended, can be either int or a tuple
        :returns: True if the transaction was successful, False otherwise
        """
        with self.env.begin(db=self.db, write=True) as txn:
            index = txn.stat(db=self.db)['entries']
            return txn.put(db=self.db, key=self.idumps(index), value=self.vdumps(value), append=True, overwrite=True)

class LMDBSortedDict(LMDBBase):
    """
    Representing an LMDB database holding a pythonic dictionary interface
    it allows both multiple index values if duplicate_key_allowed=True is
    passed, i.e., each index can have multiple values (in such case dusport=True)
    """

    def _init_db(self, path: str, memory_size: int, duplicate_key_allowed=True):
        """
        opens a db in the env and returns env, the db 
        """
        USE_SPARSE_FILES = sys.platform != 'darwin'
        self.dupsort = duplicate_key_allowed
        if duplicate_key_allowed:
            self.env = lmdb.Environment(path=self.path, map_size=memory_size, writemap=USE_SPARSE_FILES, max_dbs=1)
            self.db = self.env.open_db(b"covi-simulation-lmdb-subdb", dupsort=True)
        else:
            self.env = lmdb.Environment(path=self.path, map_size=memory_size, writemap=USE_SPARSE_FILES)
            self.db = self.env.open_db()

    def __getitem__(self, index: int) -> typing.Tuple[int]:
        """
        Returns all of the item stored at the given index
        :param index: the index whose item we want to get
        :returns: the item at index
        """
        with self.env.begin(db=self.db) as txn:
            cur = txn.cursor(self.db)
            if not cur.set_key(self.idumps(index)):
                raise KeyError(index)
            else:
                return self._get_values(cur)

    def __setitem__(self, index: int, value: typing.Any):
        """
        Modifies the item located at given index. This method overwrites the item
        located at the given index
        :param index: the index whose item we want to set
        :param value: the item to be overwriten at index, can be either int or a tuple
        :returns (if not suppressed): True if the transaction was successful, False otherwise
        """
        with self.env.begin(db=self.db, write=True) as txn:
            return txn.replace(db=self.db, key=self.idumps(index), value=self.vdumps(value))

    def __delitem__(self, index: int):
        """
        deletes sorted index from db
        """
        with self.env.begin(db=self.db, write=True) as txn:
            txn.delete(db=self.db, key = self.idumps( index ) )

    def items(self) -> typing.Tuple[int, typing.Any]:
        """
        Iterates through all of the lmdb db items
        """
        with self.env.begin(db=self.db) as txn:
            cur = txn.cursor(db=self.db)
            for index, value in cur.iternext(keys=True, values=True):
                yield self.iloads(index), self.vloads(value)

    def __iter__(self) -> int:
        """
        Iterates through all of the lmdb db keys
        """
        for key in self.keys():
            yield key

    def __contains__(self, index: int):
        """
        checks if map contains index as key
        :returns: True is index exists and False otherwise
        """
        for key in self:
            if key == index:
                return True
        else:
            return False

    def keys(self) -> int:
        """
        Iterates through all of the lmdb db keys
        """
        with self.env.begin(db=self.db) as txn:
            cur = txn.cursor(db=self.db)
            for index in cur.iternext(keys=True, values=False):
                yield self.iloads(index)

    def values(self) -> typing.Any:
        """
        Iterates through all of the lmdb db values
        """
        with self.env.begin(db=self.db) as txn:
            cur = txn.cursor(db=self.db)
            for value in cur.iternext(keys=False, values=True):
                yield self.vloads(value)

    def _get_values(self, cur) -> typing.List:
        """
        Get values at certain cursor
        """
        if self.dupsort:
            values: List[typing.Any] = []
            for value in cur.iternext_dup(keys=False, values=True):
                values.append(self.vloads(value))
            return values
        else:
            return self.vloads(cur.value())

    def replace(self, old_index: int, new_index: int, value: typing.Any):
        """
        Removes given key, value pair and inserts it as a new index, value
        """
        values_bytes = self.vdumps(value)
        with self.env.begin(db=self.db, write=True) as txn:
            txn.delete(self.idumps(old_index), values_bytes, db=self.db)
            return txn.put(db=self.db, key=self.idumps(new_index), value=values_bytes, overwrite=True, dupdata=True)

    def first(self, return_values: bool=False) -> typing.Tuple[int, typing.List]:
        """
        Gets the first/smallest key in the sorted dictionary
        :param return_values: toggles if the returning tuple should include the values as well
        :returns: smallest key (and its corresponding values) or raises a KeyError if sorted dictionary is empty
        """
        # read transaction
        with self.env.begin(db=self.db) as txn:
            cur = txn.cursor(self.db)
            if not cur.first():
                raise KeyError
            else:
                index: int = self.iloads(cur.key())
                if return_values:
                    values = self._get_values(cur)
                    return index, values
                else:
                    return index

    def pop(self, index: int) -> typing.Tuple[int, typing.List]:
        """
        Pops the values of an index from the sorted dictionary
        :returns: list of all the values if any exists else returns empty list
        """
        # read and write transaction
        with self.env.begin(db=self.db, write=True) as txn:
            cur = txn.cursor(self.db)
            if not cur.set_key(self.idumps(index)):
                return [] # raise KeyError(index)
            else:
                values = self._get_values(cur)
                cur.prev()
                cur.delete(dupdata=True)
                return values

    def pop_all(self) -> typing.List[typing.Tuple]:
        """
        Pops all pairs of key-values from the sorted dictionary and empties the main db
        :returns: all key-values stored in the sorted dictionary
        """
        items: typing.List[typing.Tuple] = []
        with self.env.begin(db=self.db, write=True) as txn:
            cur = txn.cursor(self.db)
            while cur.next(): # or next_nodup()
                index: int = self.iloads(cur.key())
                items.append((index, self._get_values(cur)))
            txn.drop(db=self.db, delete=False)
            return items

    def append(self, index: int, value: typing.Any):
        """
        Appends a value to the index. In case that db is opened without duplicate key raises AttributeError
        :param index: the index whose item we want to append
        :param value: the item to be appended at index, can be either int or a tuple
        :returns (if not suppressed): True if the transaction was successful, False otherwise
        """
        if self.dupsort:
            with self.env.begin(db=self.db, write=True) as txn:
                return txn.put(db=self.db, key=self.idumps(index), value=self.vdumps(value), overwrite=True, dupdata=True)
        else:
            raise AttributeError("Append for duplicate key is prohibited")

    def get(self, index: int, default: typing.Any=None):
        """
        Safe get item returning default value if index doesn't hold a value in the map
        :param index: the index whose item we want to get
        :param default: the default item return if index doesn't exist
        :returns: the item at given index if exists, otherwise the default value
        """
        try:
            return self[index]
        except KeyError:
            return default

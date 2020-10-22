import os
import mmap
import dill as pickle
import typing

class MMAPArray():
    """
    Representing an anonymous shared memory file's utilities as a contiguous array
    This type of array is fast to read and write, but is not appropriate for
    multiple processes writing to the same index.
    """

    def __init__(self, num_items: int, item_size: int):
        """
        Opens an anonymous shared memory file with given capacity at
        construction
        :param num_items: number of items to be stored in the mmap
        :param item_size: maximum bytes of each item in the mmap
        """
        self._mm: mmap.mmap = mmap.mmap(-1, num_items * item_size)
        self.num_items = num_items
        self.item_size = item_size

    def __getitem__(self, index: int) -> typing.Any:
        """
        Reads the value whose pickled bytes is located at given index,
        throws an error in case of invalid index
        :param index: the line number of the value to be read from
        """
        pos: int = index * self.item_size
        self._mm.seek(pos)
        return pickle.load(self._mm)

    def __setitem__(self, index: int, value: typing.Any):
        """
        Sets an object at a line in the shared file and returns the
        :param index: line number in the mmap to store the object
        :param value: object to be pickled and stored in shared memory
        """
        pos: int = index * self.item_size
        self._mm.seek(pos)
        pickle.dump(value, self._mm)
        self._mm.write_byte(10) # b"\n"
        obj_size: int = self._mm.tell() - pos + 1
        assert self.item_size >= obj_size, \
            "The length of object ({} bytes) exceeds " \
            "the provisioned max bytes per item ({})".format(
                obj_size,
                self.item_size
                )

    def __delitem__(self, index: int):
        """
        Sets an object at a line in the shared file and returns the
        :param index: line number in the mmap to store the object
        :param value: object to be pickled and stored in shared memory
        """
        pos: int = index * self.item_size
        self._mm.seek(pos)
        self._mm.write(b'\x00' * self.item_size)

    def __iter__(self) -> typing.Any:
        """
        Iterates through the mmap items
        """
        # in the line below self.num_items may be replaced by len(self) at the
        # cost of an overhead of computing the exact length of the mmap so far
        for pos in range(0, self.item_size * self.num_items, self.item_size):
            self._mm.seek(pos)
            if self._mm.read_byte() != 0:
                # return to pos, i.e., 1 byte before the read_byte() happens
                self._mm.seek(-1, os.SEEK_CUR)
                yield pickle.load(self._mm)
            else:
                yield None

    def __len__(self) -> int:
        """
        Returns the number of items that use fill some bytes in mmap so far
        warning: likely to be expensive in large shared memories
        :returns: the index of the last item + 1 in the mmap
        """
        return ( self._mm.rfind(b"\n", 0) // self.item_size ) + 1

    def __del__(self):
        """
        Closes the underlying shared memory file (at destruction)
        """
        self._mm.close()

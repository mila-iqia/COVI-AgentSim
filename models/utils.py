import json
import numpy as np
from bitarray import bitarray
from collections import namedtuple

Message = namedtuple('message', 'uid risk day unobs_id')
UpdateMessage = namedtuple('update_message', 'uid new_risk risk day unobs_id')

def encode_message(message):
	# encode a contact message as a string
	return str(np.array(message.uid.tolist()).astype(int).tolist()) + "_" + str(message.risk) + "_" + str(message.day) + "_" + str(message.unobs_id)

def decode_message(message):
	# decode a string-encoded message into a tuple
	# TODO: make this a namedtuple
	uid, risk, day, unobs_id = message.split("_")
	obs_uid = bitarray(json.loads(uid))
	risk = int(risk)
	day = int(day)
	try:
		unobs_uid = int(unobs_id)
	except Exception:
		unobs_uid = int(unobs_id.split(":")[1])
	return obs_uid, risk, day, unobs_uid

# https://stackoverflow.com/questions/51843297/convert-real-numbers-to-binary-and-vice-versa-in-python
def float_to_binary(x, m, n):
    """Convert the float value `x` to a binary string of length `m + n`
    where the first `m` binary digits are the integer part and the last
    'n' binary digits are the fractional part of `x`.
    """
    x_scaled = round(x * 2 ** n)
    return '{:0{}b}'.format(x_scaled, m + n)

def binary_to_float(bstr, m, n):
    """Convert a binary string in the format '00101010100' to its float value."""
    return int(bstr, 2) / 2 ** n

def create_new_uid(rng):
	_uid = bitarray()
	_uid.extend(rng.choice([True, False], 4))  # generate a random 4-bit code
	return _uid

def update_uid(_uid, rng):
	_uid.pop()
	_uid.extend([rng.choice([True, False])])
	return _uid

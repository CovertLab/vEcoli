from uuid import uuid4


def create_unique_indexes(n_indexes):
	"""
	Creates a list by concatenating two strings formed using uuid4()
	to avoid generating the same index twice.
	"""
	return [str(int(str(uuid4())[0:7], 16)) + str(int(str(uuid4())[0:7], 16)) for i in range(n_indexes)]

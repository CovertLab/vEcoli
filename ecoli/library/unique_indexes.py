from uuid import uuid4


def create_unique_indexes(n_indexes):
	"""
	Creates a list of unique indexes by using uuid4() to generate each index.
	"""
	return [str(uuid4().int)[-14:] for i in range(n_indexes)]


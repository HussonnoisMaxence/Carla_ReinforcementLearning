class Server(object):
	def __init__(self):
		self.params = None

	def update_params(self, params):
		self.params = params
	
	def get_params(self):
		return self.params

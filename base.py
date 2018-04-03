from keras.models import load_model


class BaseModel(object):
    def __init__(self, name, model):
        self.name = name
        self.model = model

    def fit(self, x, y):
        raise NotImplementedError('to be implemented')

    def save_model(self, file_path):
        self.model.save(file_path)

    def load_model(self, file_path):
        self.model = load_model(file_path)

    def load_weights(self, file_path):
    	self.model.load_weights(file_path)
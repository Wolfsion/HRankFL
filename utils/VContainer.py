from collections import OrderedDict
from env.runtimeEnv import file_repo
from utils.pathHandler import store


class VContainer:
    ERROR_MESS1 = "The element is not fit."

    def __init__(self):
        self.container = OrderedDict()

    def flash(self, key: str, element):
        if key not in self.container.keys():
            self.container[key] = []
            self.container[key].append(element)
        else:
            assert type(self.container[key][0]) == type(element), self.ERROR_MESS1
            self.container[key].append(element)

    def store(self, key: str):
        path = file_repo.visual(name=key)
        store(path, self.container[key])

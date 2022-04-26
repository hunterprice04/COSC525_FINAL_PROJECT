import functools


class Wrapper:
    def __init__(self, verbosity=1):
        self.verbosity = verbosity
        self.print_len = 0

    def get_verbosity(self, verbosity=None):
        return self.verbosity if verbosity is None else verbosity

    def print_3(self, *args, sep=' ', end='\n', verbosity=None):
        if self.get_verbosity(verbosity) >= 3:
            print(*args, sep=sep, end=end)

    def print_2(self, *args, sep=' ', end='\n', verbosity=None):
        if self.get_verbosity(verbosity) >= 2:
            print(*args, sep=sep, end=end)

    def print_1(self, *args, sep=' ', end='\n', verbosity=None):
        if self.get_verbosity(verbosity) >= 1:
            print(*args, sep=sep, end=end)

    def print_0(self, *args, sep=' ', end='\n', verbosity=None):
        if self.get_verbosity(verbosity) >= 0:
            print(*args, sep=sep, end=end)


class AffineFunction:
    def __init__(self):
        self.a = None
        self.b = None

    def set_function(self, kwargs):
        self.a, self.b = kwargs

    def predict(self, x, y):
        return self.a * x + self.b * x * y

    def get_gradient(self, x):
        # gradient with respect to y
        return self.b * x

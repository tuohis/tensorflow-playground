from curve import SecondDegreeCurve
from pprint import pprint

# Training data
data = {'x': [0, 1, 2],
        'y': [1, 6, 17]}
pprint(SecondDegreeCurve().fit(data, iterations=1000))

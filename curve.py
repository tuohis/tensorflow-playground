import tensorflow as tf

class Curve(object):
    def __init__(self):
        raise NotImplementedError("Abstract class Curve not instantiable")

    def fit(self, data, **kwargs):
        """
        Fits the curve to the provided data.

        Data is accepted as a dict with array fields 'x' and 'y'.

        A keyword parameter 'iterations' can be given to specify the amount of
        training iterations. Defaults to 1000.

        Returns the function coefficients and loss function value.
        """
        iterations = kwargs.get("iterations", 1000)

        y = tf.placeholder(tf.float32)
        loss = tf.reduce_sum(tf.square(self.curve - y))

        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = optimizer.minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for _ in range(iterations):
            sess.run(train, {self.x:data['x'], y:data['y']})

        output = sess.run(self.vars + [loss], {self.x:data['x'], y:data['y']})
        labels = self.var_names + ['loss']
        return { labels[i]: output[i] for i in range(len(output)) }

class SecondDegreeCurve(Curve):
    """
    A second-degree curve of the form y = ax^2 + bx + c
    """
    def __init__(self):
        a = tf.Variable(1.0, tf.float32)
        b = tf.Variable(1.0, tf.float32)
        c = tf.Variable(0.0, tf.float32)
        self.x = tf.placeholder(tf.float32)
        self.curve = a * self.x**2 + b * self.x + c
        self.vars = [a, b, c]
        self.var_names = ['a', 'b', 'c']

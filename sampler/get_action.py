"""
speed test for different loop settings
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation

import gtimer as gt

input_size = 300
output_size = 10
layer_size = 3


model = Sequential([
    Dense(layer_size, input_dim=input_size, init='uniform'),
    Activation('tanh'),
    # Dense(output_size, init='uniform'),
    # Activation('softmax')
])

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

n_sample = 100000
batch_size = 100000
n_batch = n_sample // batch_size
# x = []
# y = []
# for i in range(n_batch):
#     x.append(np.random.rand(batch_size, input_size))
#     y.append(np.zeros([batch_size, output_size]))

x = np.random.rand(batch_size, input_size)

t0 = gt.start()
for i in range(n_batch):
    model.predict(x)
t1 = gt.stamp('predict')

# print("Predictions per second: {:,.0f}".format(n_sample / (t1 - t0)))


y_ints = np.random.randint(0, output_size, batch_size)

y = np.zeros([batch_size, output_size])

for i, y_int in enumerate(y_ints):
    y[i, y_int] = 1

# t0 = gt.blank_stamp()
# for i in range(n_batch):
#     model.fit(x, y, batch_size=batch_size, nb_epoch=1, verbose=0)
# t1 = gt.stamp('fit')
# print("Sample gradients per second: {:,.0f}".format(n_sample / (t1 - t0)))


# print(gt.report())


from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('./data/',one_hot=True)
print(mnist.train.labels[0])
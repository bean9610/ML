import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tmp/data",one_hot=True)

learning_rate=0.1
num_steps=500
batch_size=128
display_step=100

n_hidden_1=256
n_hidden_2=256
num_input=784
num_classes=10

X=tf.placeholder("float32",[None,784])
Y=tf.placeholder("float",[None,10])

def get_weights(shape):
    w=tf.Variable(tf.random_normal(shape))
    return w
def get_bias(shape):
    b=tf.Variable(tf.zeros(shape))
    return b
weights={
    'h1':get_weights([num_input,n_hidden_1]),
    'h2':get_weights([n_hidden_1,n_hidden_2]),
    'out':get_weights([n_hidden_2,num_classes])
}
bias={
    'b1':get_bias([n_hidden_1]),
    'b2':get_bias([n_hidden_2]),
    'out':get_bias([num_classes])
}
layer_1=tf.add(tf.matmul(X,weights['h1']),bias['b1'])
layer_1=tf.nn.relu(layer_1)
layer_2=tf.add(tf.matmul(layer_1,weights['h2']),bias['b2'])
layer_3=tf.add(tf.matmul(layer_2,weights['out']),bias['out'])
predictions=tf.nn.softmax(layer_3)

loss_op=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_3,labels=Y))
train_op=tf.train.AdadeltaOptimizer(learning_rate).minimize(loss_op)

correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(1,num_steps+1):
        batch_x,batch_y=mnist.train.next_batch(batch_size)
        sess.run(train_op,feed_dict={X:batch_x,Y:batch_y})
        if step % 100 == 0 :
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images,
                                            Y: mnist.test.labels}))

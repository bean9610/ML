import tensorflow as tf
a=tf.constant(2)
b=tf.constant(3)
with tf.Session() as sess:
    print("%i" %sess.run(a+b))
a=tf.placeholder(tf.int16)
b=tf.placeholder(tf.int16)
add=tf.add(a,b)
mul=tf.multiply(a,b)
with tf.Session() as sess:
    print("%i" %sess.run(add,feed_dict={a:2,b:3}))
    print("%i" %sess.run(mul,feed_dict={a:2,b:3}))
m1=tf.constant([[3,3]])
m2=tf.constant([[2],[2]])
mul1=tf.matmul(m1,m2)
with tf.Session() as sess:
    result=sess.run(mul1)
    print(result)
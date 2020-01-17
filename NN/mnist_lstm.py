import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

#设置标志，is_train变量默认为1
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("is_train", 1, "指定程序是预测还是训练")

# 获得真实的数据。 如果./data/minst/input_data目录不存在该数据集，便从网络上下载。如果已经存在，便不会再下载
mnist = input_data.read_data_sets("./data/minst/input_data", one_hot=True)
# mnist.train.images大小为[55000, 28*28]，即55000个训练数据的特征集, mnist.train.labels大小为[55000，10],
# mnist.test.images大小为[1000, 28*28], mnist.test.labels的大小为[1000,10]
# 所以，将28*28的一个数据看成28个1*28的序列组成的数据。所以一个数据包含的序列数目为28，单个序列的长度是1*28。
# 假设隐藏层有128个神经元，输出层有10个单元（输出的神经单元的个数应该和数据的类别数相同）
# 所以，网络的参数设置如下：
diminput = 28
nstep = 28
dimhidden = 128
dimoutput = mnist.train.labels.shape[1]


# 定义RNN模型
def _RNN(diminput, nsteps, dimhidden, dimoutput):
    """
    定义RNN模型
    :param diminput: 每个数据里的序列的长度
    :param nsteps: 每个数据里包含的序列的个数
    :param dimhidden: 隐藏层的神经元个数
    :param dimoutput: 输出层的神经元个数
    :return: x, y_true, y_pred
    """
    # 1.准备数据，处理数据
    with tf.variable_scope("data"):
        # 准备数据的占位符
        x = tf.placeholder("float", [None, nsteps, diminput])
        y = tf.placeholder("float", [None, dimoutput])
        # 将数据的第一维和第二维转换一下，第三维不变 [batchsize, nstep, diminput] =>[nstep batchsize, diminput]
        _x = tf.transpose(x, [1, 0 , 2])
        # 将数据=>[nstep* batchsize, diminput]
        _x = tf.reshape(_x, [-1, diminput])

    # 2.初始化权重和偏差
    # 初始化隐藏层和输出层的权值
    with tf.variable_scope("weights"):
        Wh = tf.Variable(tf.random_normal([diminput, dimhidden]))
        Wo = tf.Variable(tf.random_normal([dimhidden, dimoutput]))
    # 初始化隐藏层和输出层的偏置
    with tf.variable_scope("bias"):
        bh = tf.Variable(tf.random_normal([dimhidden]))
        bo = tf.Variable(tf.random_normal([dimoutput]))

    # 3.计算隐层
    with tf.variable_scope("hidden"):
        # 计算整个数据的隐藏层状态
        _H = tf.matmul(_x, Wh) + bh
        # 切分成nstep个隐藏层状态
        _Hsplit = tf.split(_H, nstep, 0)  #_Hsplit是个列表，_Hsplit[-1]代表最后一步的隐藏层输出

    # 4.数据经过LSTM单元
    with tf.variable_scope("LSTM"):
        # 实例化一个LSTM单元
        lstm_cell = rnn.BasicLSTMCell(dimhidden, forget_bias=1.0)
        _LSTM_O, _LSTM_S = rnn.static_rnn(lstm_cell, _Hsplit, dtype=tf.float32)

    # 5.计算最后的输出层
    with tf.variable_scope("output"):
        _O = tf.matmul(_LSTM_O[-1], Wo) + bo

    # #  高纬度变量收集】
    # tf.summary.histogram("weight_hidden", Wh)
    # tf.summary.histogram("biasses_hidden", bh)
    # tf.summary.histogram("weight_out", Wo)
    # tf.summary.histogram("biasses_out", bo)

    # 6.返回
    return x, y, _O


# 训练RNN网络
def _RNN_Mnist(diminput, nstep, dimhidden, dimoutput):
    # 1. 调用RNN模型
    x, y_true, y_pred = _RNN(diminput, nstep, dimhidden, dimoutput)

    # 2. 求出所有样本的损失，然后求平均值
    with tf.variable_scope("soft_cross"):
        # 计算样本的交叉熵损失
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

    # 3.梯度下降法求出损失
    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    # 4.计算准确度
    with tf.variable_scope("acc"):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_pred, 1))
        acc = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 收集变量，单个数字值收集
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("acc", acc)
    #  定义一个合并变量
    merged = tf.summary.merge_all()

    #  5. 定义一个初始化变量的op
    init_op = tf.global_variables_initializer()

    #  创建一个saver
    saver = tf.train.Saver()

    # 6.开启一个会话去训练
    with tf.Session() as sess:
        #  初始化变量
        sess.run(init_op)
        #  建立events文件，然后导入
        filewriter = tf.summary.FileWriter("D:/data/RNN/", graph=sess.graph)
        # 如果is_train为1，训练网络.否则测试网络
        if FLAGS.is_train == 1:
            # 设置训练网络的参数
            train_epoch = 1
            batch_size = 50
            total_batch = mnist.train.images.shape[0]//batch_size
            for epoch in range(train_epoch):  # 对所有的数据进行5次计算
                avg_cost = 0  # 初始化平均的损失
                for i in range(total_batch):  # 每batch_size个数据进行一次梯度下降运算
                    #  取出真实的数据
                    batch_x, batch_y = mnist.train.next_batch(batch_size)
                    # batch_x大小为[batch_size, 28*28],需转化成[batch_size, 28, 28]大小的
                    batch_x = batch_x.reshape((batch_size, nstep, diminput))
                    #  运行train_op训练
                    sess.run(train_op, feed_dict={x: batch_x, y_true: batch_y})
                    #  计算损失
                    avg_cost = sess.run(loss, feed_dict={x: batch_x, y_true: batch_y})/total_batch
                    # 写入每部训练的值
                    summary = sess.run(merged, feed_dict={x: batch_x, y_true: batch_y})
                    filewriter.add_summary(summary, i)
                print("Epoch:%03d/%03d, cost:%.9f" % (epoch, train_epoch, avg_cost))
                print("训练集的准确率为%f" % (sess.run(acc, feed_dict={x: batch_x, y_true: batch_y})))
            # 保存模型
            saver.save(sess, "./tem/ckpt/RNN_model")
        else:  # 否则测试数据集
            # 加载模型
            saver.restore(sess, "./tem/ckpt/RNN_model")
            testimags = mnist.test.images.reshape((mnist.test.images.shape[0], nstep, diminput))
            feeds = {x: testimags, y_true: mnist.test.labels}
            test_acc = sess.run(acc, feed_dict=feeds)
            print("Test accuracy: %.3f" % test_acc)
    return None


if __name__ == "__main__":
    _RNN_Mnist(diminput, nstep, dimhidden, dimoutput)



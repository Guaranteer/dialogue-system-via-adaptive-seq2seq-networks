import tensorflow as tf

def tensor_matmul(ts, mat):
    # ts is a 3d batched tensor, while mat is a 2d parameter matrix
    #matmul = lambda x: tf.matmul(x, mat)
    #output = tf.map_fn(matmul, ts)
    ts_s1 = int(ts.get_shape()[1])
    ts_s2 = int(ts.get_shape()[2])
    mat_s1 = int(mat.get_shape()[1])
    reshaped = tf.reshape(ts, [-1, ts_s2])
    matmul = tf.matmul(reshaped, mat)
    output = tf.reshape(matmul, [-1, ts_s1, mat_s1])
    return output

def count_total_variables():
    n_ts = 0
    n_var = 0
    for ts in tf.trainable_variables():
        n_ts += 1
        ts_size = 1
        for dim in ts.get_shape():
            ts_size *= dim.value
        n_var += ts_size
    print ('Total tensors:', n_ts)
    print ('Total variables:', n_var)

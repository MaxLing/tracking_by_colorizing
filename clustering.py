import tensorflow as tf
from tensorflow.contrib.factorization import KMeans

# creates operations in the default graph
class Clustering:
    def __init__(self, inputs, clusters, mini_batch_steps_per_iteration=100):
        self.clusters = tf.convert_to_tensor(clusters)
        self.kmeans = KMeans(inputs, self.clusters,
                             use_mini_batch=True,
                             mini_batch_steps_per_iteration=mini_batch_steps_per_iteration)
        out = self.kmeans.training_graph()
        self.cluster_centers = tf.get_default_graph().get_tensor_by_name('clusters:0')
        self.all_scores = out[0][0]
        self.cluster_index = out[1][0]
        self.scores = out[2][0]
        self.cluster_centers_initialized = out[3]
        self.init_op = out[4]
        self.train_op = out[5]

    def lab_to_labels(self, images, name='lab_to_labels'):
        a = tf.reshape(self.cluster_centers[:,0], [1,1,1,-1])
        b = tf.reshape(self.cluster_centers[:,1], [1,1,1,-1])
        da = tf.expand_dims(images[:,:,:,1],3) - a
        db = tf.expand_dims(images[:,:,:,2],3) - b
        d = tf.square(da) + tf.square(db)
        return tf.argmin(d, 3, name=name)

    def labels_to_lab(self, labels, name='labels_to_lab'):
        if labels.dtype in [tf.float16, tf.float32, tf.float64]: # soft label
            l = tf.cast(tf.expand_dims(labels,-1), tf.float32)
            c = tf.reshape(self.cluster_centers, [1,1,1,-1,2])
            ab = tf.reduce_sum(l * c, 3)
        else:
            ab = tf.gather(self.cluster_centers, labels) # hard label
        l = tf.ones(tf.shape(ab)[:-1], tf.float32) * 75
        return tf.concat([tf.expand_dims(l,-1), ab], 3, name=name)
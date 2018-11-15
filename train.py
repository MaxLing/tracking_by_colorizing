import os, cv2, pickle, argparse
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
from clustering import Clustering
from dataset import Dataset
from nets import feature_extractor, colorizer

# Parameters, use smaller ref_frame and batch_size if GPU OOM
parser = argparse.ArgumentParser(description='Train colorization model')
parser.add_argument('--ref_frame', type=int, default=3,
                    help='number of reference frame to track from')
parser.add_argument('--clusters',type=int, default=16,
                    help='number of clusters for color space K-means')
parser.add_argument('--learn_rate', '-lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch size')
parser.add_argument('--max_iter', type=int, default=100000,
                    help='max iteration')
parser.add_argument('--image_size', type=str, default='185,360',
                    help='downsized image size of input videos, seperated by comma')
parser.add_argument('--embed_size', type=str, default='47,90',
                    help='downsampled embedding size of input images, seperated by comma')
parser.add_argument('--embed_dim', type=int, default=64,
                    help='dimension of embeddings')
parser.add_argument('--data_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'data'),
                    help='directory of training data')
parser.add_argument('--window', type=int, default=4,
                    help='neighboring window for similairity computation')
parser.add_argument('--data_type', type=str, choices=['surgical','kinetics'], 
                    help='dataset type')
args = parser.parse_args()

'''
ref_frame = 3
color_clusters = 10
lr = 1e-4
batch_size = 4
max_iter = 1000

#image_size = [92, 180] # [480, 720]-crop->[370,720] downsize/4
#embed_size = [12, 23]  # image_size/8
image_size = [185, 360] # downsize/2
#embed_size = [24,45]
embed_size = [47,90] # image_size/4
embed_dim = 64
window = 5

data_dir = os.path.join(os.path.dirname(__file__), 'data')
'''
ref_frame = args.ref_frame
color_clusters = args.clusters
lr = args.learn_rate
batch_size = args.batch_size
max_iter = args.max_iter
image_size = [int(x) for x in args.image_size.split(',')]
embed_size = [int(x) for x in args.embed_size.split(',')]
embed_dim = args.embed_dim
window = args.window
data_dir = args.data_dir
data_type = args.data_type

model_spec = 'model_'+ 'insize' + str(image_size[0]) +'x' + str(image_size[1]) + '_emsize' + str(embed_size[0])+'x'+str(embed_size[1]) +'_lr' + str(args.learn_rate) + '_cluster' + str(args.clusters) + '_win' + str(args.window) + '_ref' + str(args.ref_frame) + '_batch' + str(args.batch_size)
model_dir = os.path.join(os.path.dirname(__file__), model_spec)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

'''load data'''
with tf.variable_scope("data_loader", reuse=tf.AUTO_REUSE):
    data = Dataset(data_dir, batch_size, ref_frame, image_size, data_type)
    data_loader = data.load_data_batch().repeat().batch(batch_size) # repeat(epoch) or indefinitely
    image_batch = data_loader.make_one_shot_iterator().get_next()
    image_batch = tf.concat([image_batch[...,0:1]*2-1, image_batch[...,1:]], axis=-1) # scale intensity to [-1,1]

'''build graph'''
with tf.variable_scope("input", reuse=tf.AUTO_REUSE):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    images = tf.placeholder(tf.float32, [None, ref_frame+1] + image_size + [3], name='images')
    is_training = tf.placeholder(tf.bool, name='is_training')

# color clustering
with tf.variable_scope("clustering", reuse=tf.AUTO_REUSE):
    KMeans = Clustering(tf.reshape(images[...,1:], [-1,2]), color_clusters)
    images_flat = tf.reshape(images, [-1]+image_size+[3])
    images_for_label = tf.image.resize_images(images_flat, embed_size)
    labels = KMeans.lab_to_labels(images_for_label)
    labels = tf.reshape(labels, [-1, ref_frame+1]+embed_size, name='labels')

# embeddings extraction from intensity
with tf.variable_scope("feature_extraction", reuse=tf.AUTO_REUSE):
    embeddings = feature_extractor(images[...,0:1], is_training = is_training)
    embeddings = tf.identity(embeddings, name='embeddings')

# predict color based on similarity
with tf.variable_scope("colorization", reuse=tf.AUTO_REUSE):
    losses = tf.zeros([0,1], dtype=tf.float32)
    predictions = tf.zeros([0,1]+embed_size+[color_clusters])
    predictions_lab = tf.zeros([0,1]+embed_size+[3])

    #pq = tf.reduce_mean(tf.reduce_mean(tf.one_hot(labels, color_clusters), 2), 2)
    #q = tf.reduce_mean(pq[:,:ref_frame,:], 1, keepdims=True)
    #p = pq[:,ref_frame:,:]
    #loss_weights = tf.reduce_sum((p**0.5)*(q**0.5), axis=-1, name = 'loss_weights')

    for i in range(batch_size):
        embedding = embeddings[i]
        label = labels[i]

        results = colorizer(embedding[:ref_frame], tf.one_hot(label[:ref_frame], color_clusters),
                            embedding[ref_frame:], label[ref_frame:], window=window)
        mean_losses = tf.reduce_mean(tf.reduce_mean(results['losses'], 2), 1)
        pred = results['predictions']

        losses = tf.concat([losses, tf.expand_dims(mean_losses, 0)], 0)
        predictions = tf.concat([predictions, tf.expand_dims(pred, 0)], 0)
        predictions_lab = tf.concat([predictions_lab, tf.expand_dims(KMeans.labels_to_lab(pred), 0)], 0)

    predictions = tf.identity(predictions, name='predictions')
    predictions_lab = tf.identity(predictions_lab, name='predictions_lab')
    losses = tf.identity(losses, name='losses')

'''training'''
with tf.variable_scope("training", reuse=tf.AUTO_REUSE):
    global_step = tf.get_default_graph().get_tensor_by_name('input/global_step:0')
    loss = tf.reduce_mean(losses)
    #loss = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(losses, -1), -1) * loss_weights)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss, global_step=global_step)

'''summary'''
with tf.variable_scope("summary", reuse=tf.AUTO_REUSE):
    loss_summary = tf.summary.scalar('loss', loss)
    ph_tag_img = tf.placeholder(tf.float32, shape=[None,None,None,3])
    ph_pred_img = tf.placeholder(tf.float32, shape=[None,None,None,3])
    ph_tag_embed = tf.placeholder(tf.float32, shape=[None,None,None,3])
    image_summary = tf.summary.merge([tf.summary.image('target_image', ph_tag_img),
                                      tf.summary.image('visualized_prediction', ph_pred_img),
                                      tf.summary.image('visualized_feature', ph_tag_embed)])
    writer = tf.summary.FileWriter(model_dir)

'''session'''
# use GPU memory based on runtime allocation and visible device
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "3" # TODO: currently no support on multiple GPU
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)
    saver = tf.train.Saver()
    latest_ckpt = tf.train.latest_checkpoint(model_dir)

    if latest_ckpt is not None:
        print('Restoring from ' + latest_ckpt)
        saver.restore(sess, latest_ckpt)
        KMeans_initialized = True
    else:
        print('Starting with a new model')
        tf.train.export_meta_graph(os.path.join(model_dir, 'model.meta'))
        KMeans_initialized = False

    pca = PCA(n_components=3) # for embedding visualization

    for j in range(max_iter):
        i = tf.train.global_step(sess, global_step)
        print ('iteration: ' + str(j) + ' global step: ' + str(i))
        # load image batch
        images_batch = sess.run(image_batch)

        if not KMeans_initialized:
            # init kmeans
            sess.run(KMeans.init_op, {images: images_batch})
        if i % 50 == 0:
            # update kmeans using minibatch
            sess.run(KMeans.train_op, {images: images_batch})
        
        if i % 500 != 0:
            # normal training
            _, summary = sess.run([train_op, loss_summary], {images: images_batch, is_training: True})
            writer.add_summary(summary, i)
        else:
            # training with visualize
            _, feats, preds, summary = sess.run([train_op, embeddings, predictions_lab, loss_summary],
                                                {images: images_batch, is_training: True})
            writer.add_summary(summary, i)

            tag_img = np.zeros([batch_size] + image_size + [3])
            tag_embed = np.zeros([batch_size] + embed_size + [3])
            pred_img = np.zeros([batch_size] + image_size + [3])

            pca.fit(feats[:,ref_frame].reshape(-1, embed_dim))
            for j in range(batch_size):
                img = images_batch[j, ref_frame]
                img[...,0] = (img[...,0]+1)/2 # scale back to [0,1]
                tag_img[j] = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

                pred = preds[j, 0] # [N, 1, H, W, 3]
                pred = np.dstack([img[:, :, 0:1], cv2.resize(pred[:, :, 1:], tuple(image_size)[::-1])])
                pred_img[j] = cv2.cvtColor(pred, cv2.COLOR_LAB2RGB)

                feat_flat = feats[j, ref_frame].reshape(-1, embed_dim)
                feat_flat = pca.transform(feat_flat)
                feat_flat -= np.min(feat_flat)
                feat_flat /= np.max(feat_flat)
                tag_embed[j] = feat_flat.reshape(embed_size + [3])
            summary = sess.run(image_summary, {ph_tag_img: tag_img,
                                               ph_pred_img: pred_img,
                                               ph_tag_embed: tag_embed})
            writer.add_summary(summary, i)

        if (i+1) % 1000 == 0:
            # save the model
            saver.save(sess, os.path.join(model_dir, 'model.ckpt'), global_step=i, write_meta_graph=False)
            # save pca (overwrite) for embedding visualization
            with open(os.path.join(model_dir, 'pca.pkl'), 'wb') as f:
                pickle.dump(pca, f)

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
slim = tf.contrib.slim

def conv3d_block(input, output_dim, kernel, dilation, name, activation=True, is_training=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # divide 3D conv to 2D+1D
        block = tf.layers.conv3d(input, output_dim, [1, kernel, kernel], dilation_rate=[1, dilation, dilation], padding='SAME')
        block = tf.layers.conv3d(block, output_dim, [kernel, 1, 1], padding='SAME')
        if activation:
            block = tf.layers.batch_normalization(block, training=is_training)
            block = tf.nn.relu(block)
        return block

def feature_extractor(images, is_training):
    # use tf slim to build ResNet18
    with tf.variable_scope('resnet', reuse=tf.AUTO_REUSE):
        # [N,T,H,W,C] -> [N*T,H,W,C]
        org_shape = tf.shape(images)
        images = tf.reshape(images, tf.concat([[-1],org_shape[2:]], axis=0))
        
        blocks = [
            resnet_v2.resnet_v2_block('block1', base_depth=64, num_units=2, stride=1),
            resnet_v2.resnet_v2_block('block2', base_depth=128, num_units=2, stride=2),
            resnet_v2.resnet_v2_block('block3', base_depth=256, num_units=2, stride=1),
            resnet_v2.resnet_v2_block('block4', base_depth=256, num_units=2, stride=1)
        ]
        _, end_points = resnet_v2.resnet_v2(images, blocks, is_training=is_training, include_root_block=True)
        # root 7*7 conv + 3*3 maxpool with stride 2
        # Note: modify depth in b4(512->256) and stride in b3b4(2->1)
    
    	net = end_points['feature_extraction/resnet/resnet_v2/block4'] 
        # because we don't want fc, pooling or softmax at end
    
    with tf.variable_scope('conv3d', reuse=tf.AUTO_REUSE):
    	# [N*T,H',W',C'] -> [N,T,H',W',C']
    	net = tf.reshape(net, tf.concat([org_shape[:2], tf.shape(net)[1:]], axis=0))
        
        '''
    	# concat spatial info
    	y = tf.lin_space(-1., 1., org_shape[2])
    	x = tf.lin_space(-1., 1., org_shape[3])
    	X, Y = tf.meshgrid(x, y)
        X = tf.broadcast_to()
        ''' 

        net = conv3d_block(net, 256, 3, 1, name='conv1', is_training=is_training)
        net = conv3d_block(net, 256, 3, 2, name='conv2', is_training=is_training)
        net = conv3d_block(net, 256, 3, 4, name='conv3', is_training=is_training)
        net = conv3d_block(net, 256, 3, 8, name='conv4', is_training=is_training)
        net = conv3d_block(net, 256, 3, 16, name='conv5', is_training=is_training)
        embeddings = tf.layers.conv3d(net, 64, [1, 1, 1], padding='SAME', name='conv6')
    return embeddings

def colorizer(ref_embed, ref_label, tag_embed, tag_label=None, temperature=1):
    # embed size [T, H, W, D], label size [T, H, W, C]
    # Note: ref_label is one-hot/probability while tag_label can be GT
    # during inference, tag_label is None and temperature can be 0.5
    with tf.variable_scope('colorizer', reuse=tf.AUTO_REUSE):
        dim = tf.shape(tag_embed)[-1]
        org_shape = tf.shape(tag_embed)[:-1]
        cat = tf.shape(ref_label)[-1]

        # TODO: inner product can be more efficient
        ref_embed = tf.reshape(ref_embed, [-1,1,dim]) # [ref*H*W, 1, dim]
        tag_embed = tf.reshape(tag_embed, [1,-1,dim]) # [1, H*W, dim]
        inner_product = tf.reduce_sum(ref_embed * tag_embed, -1) # [ref*H*W, H*W]
        similarity_matrix = tf.nn.softmax(inner_product/temperature, 0)

        ref_label = tf.reshape(ref_label, [-1, 1, cat]) # [ref*H*W, 1, d]
        prediction = tf.reduce_sum(tf.expand_dims(similarity_matrix, -1) * ref_label, 0) #[H*W, d]
        prediction = tf.reshape(prediction, tf.concat([org_shape, [-1]], axis=0))
        results = {'inner_product': inner_product,
                   'similarity_matrix': similarity_matrix,
                   'temperature': temperature,
                   'predictions': prediction}

        if tag_label is not None:
            tag_label = tf.convert_to_tensor(tag_label)
            if tag_label.dtype in [tf.float16, tf.float32, tf.float64]:
                fn = tf.nn.softmax_cross_entropy_with_logits
            else:
                fn = tf.nn.sparse_softmax_cross_entropy_with_logits
            results['losses'] = fn(logits=prediction, labels=tag_label)
        return results

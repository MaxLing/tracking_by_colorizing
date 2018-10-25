import os, cv2
import tensorflow as tf
import numpy as np
from nets import colorizer

# Parameters
data_dir = os.path.join(os.path.dirname(__file__), 'data')
model_dir = os.path.join(os.path.dirname(__file__), 'model')
image_size = (120, 180) # downsize/4
embed_size = (15, 23)  # image_size/8
ref_frame = 3
label_types = 6
temperature = 0.5

def image_preprocess(frames):
    # convert RGB to LAB
    frames = cv2.cvtColor(np.float32(frames/255.), cv2.COLOR_BGR2LAB)
    # scale L to [-1,1]
    frames[...,0] = 2*frames[...,0]-1
    return frames

def label_preprocess(masks):
    # convert to one_hot
    labels = np.zeros(masks.shape+(label_types,), dtype=np.float32)
    for type in range(label_types):
        labels[np.where(masks==type)+(label_types,)] = 1
    return labels

def apply_mask(frame, mask):
    # 1st label(no class) serves as intensity
    intensity = mask[...,0:1]*frame
    # rest labels as color [h,w,c-1,1]*[1,1, c-1, 3] = [h,w, c-1, 3]
    rest_mask = np.expand_dims(mask[...,1:], -1)
    rest_colors = np.broadcast_to(np.arange(label_types)[1:], (label_types-1, 3)).reshape(1, 1, -1, 3)
    color = np.sum(rest_mask*rest_colors, axis=2)
    return intensity+color

# read video and corresponding label dirs
with open(data_dir + '/video_dirs.txt', 'r') as f:
    video_dirs = f.read().splitlines()
mask_dirs = 'mask_' + video_dirs


'''load trained model'''
with tf.get_default_graph() as graph:
    # saver = tf.train.import_meta_graph(os.path.join(model_dir, 'model.meta')) # new model
    saver = tf.train.import_meta_graph(os.path.join(model_dir, 'model.ckpt-199.meta')) # old model

    # get tensors
    images = graph.get_tensor_by_name('input/images') # [N, T, H, W, 3]
    is_training = graph.get_tensor_by_name('input/is_training')
    embeddings = graph.get_tensor_by_name('feature_extraction/embeddings')  # [N, T, H', W', D]

    # labels input
    labels = tf.placeholder(tf.float32, [ref_frame, embed_size[0], embed_size[1], label_types])

    # get similarity matrix to track
    results = colorizer(embeddings[0,:ref_frame], labels, embeddings[0, ref_frame:], temperature=temperature)
    predictions = results['predictions']


'''session'''
with tf.Session(graph=graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    # TODO: now only support 1 video
    video_dir = video_dirs[0]
    mask_dir = mask_dirs[0]
    frame_list = sorted(os.listdir(video_dir))
    mask_list = sorted(os.listdir(mask_dir))

    frames = np.zeros([ref_frame+1, image_size[0], image_size[1], 3])
    masks = np.zeros([ref_frame, embed_size[0], embed_size[1]])

    # init
    for i in range(0, ref_frame + 1):
        frames[i] = cv2.resize(cv2.imread(video_dir+'/'+frame_list[i]), image_size[::-1])
        if i != ref_frame:
            mask_name = frame_list[i].split('.')
            mask_name = mask_name[0] + '_ordered.' + mask_name[-1]
            masks[i] = cv2.resize(cv2.imread(mask_dir + '/' + mask_name, cv2.IMREAD_GRAYSCALE), embed_size[::-1])

    # init video writer
    outfile = video_dir
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    video = cv2.VideoWriter(filename=outfile, fourcc=fourcc, fps=30.0, frameSize=image_size[::-1])

    for i in range(ref_frame, len(frame_list)):
        pred = sess.run(predictions, {images: image_preprocess(frames),
                                      labels: label_preprocess(masks),
                                      is_training: False})
        pred_mask = cv2.resize(pred[0], image_size[::-1])

        # write to video
        video.write(apply_mask(frames[i], pred_mask))

        # stop update at the end
        if i==len(frame_list)-1:
            break

        # update images
        frames[:-1] = frames[1:]
        frames[-1] = cv2.resize(cv2.imread(video_dir+'/'+frame_list[i+1]), image_size[::-1])

        # update labels
        masks[:-1] = masks[1:]
        mask_name = frame_list[i].split('.')
        mask_name = mask_name[0] + '_ordered.' + mask_name[-1]
        if mask_name in mask_list:
            # use ground truth if provided
            masks[-1] = cv2.resize(cv2.imread(mask_dir + '/' + mask_name, cv2.IMREAD_GRAYSCALE), embed_size[::-1])
        else:
            # use prediction as label
            masks[-1] = pred[0]

    # end capture
    video.release()







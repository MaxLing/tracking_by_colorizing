import os, cv2, pickle, argparse
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
from nets import colorizer

# Parameters
parser = argparse.ArgumentParser(description='Inference colorization model') 
parser.add_argument('--label_types', type=int, default=6,
                    help='number of label types')
parser.add_argument('--temperature', '-t', type=float, default=0.5,
                    help='softmax temperature [0,1]')
parser.add_argument('--alpha', '-a', type=float, default=0.4,
                    help='alpha for segmentation mask')
parser.add_argument('--data_dir', type=str,
                    help='directory of inference data')
parser.add_argument('--model_dir', type=str,
                    help='directory of inference model, dont include / at end')
parser.add_argument('--data_type', type=str, choices=['surgical','davis'],
                    help='dataset type')
args = parser.parse_args()

def read_model(model_dir):
    name = model_dir.split('/')[-1]
    specs = name.split('_')
    image_size = tuple([int(x) for x in specs[1][6:].split('x')])
    embed_size = tuple([int(x) for x in specs[2][6:].split('x')])
    clusters = int(specs[4][7:])
    window = int(specs[5][3:])
    ref = int(specs[6][3:])
    return image_size, embed_size, clusters, window, ref

label_types = args.label_types
temperature = args.temperature
alpha = args.alpha
data_dir = args.data_dir
model_dir = args.model_dir
data_type = args.data_type
image_size, embed_size, color_clusters, window, ref_frame = read_model(model_dir)

output_dir = os.path.join(data_dir, 'output')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# make colors for masking
colors = np.array([[0,0,0], [255,0,0],[0,255,0],[255,255,0],[255,0,255],[0,255,255],[255,255,255]])

def image_preprocess(frames):
    images = np.copy(frames)
    for i in range(images.shape[0]):
    	# convert RGB to LAB
    	images[i] = cv2.cvtColor(np.float32(images[i]/255.), cv2.COLOR_BGR2LAB)
    # scale L to [-1,1]
    images[...,0] = 2*images[...,0]-1
    return images

def label_preprocess(masks):
    # convert to one_hot 
    #print('class0: ' + str(np.sum(masks==0)))
    #print('class1: ' + str(np.sum(masks==1)))
    #print('class2: ' + str(np.sum(masks==2)))
    #print('class3: ' + str(np.sum(masks==3)))
    #print('class4: ' + str(np.sum(masks==4)))
    #print('class5: ' + str(np.sum(masks==5)))
    labels = np.zeros(masks.shape+(label_types,), dtype=np.float32)
    for type in range(label_types):
        labels[np.where(masks==type)+(type,)] = 1
    return labels

def apply_mask(frame, mask):
    # blend original frame with mask colors
    image = np.copy(frame)
    color = np.copy(frame)

    # use argmax to determine segmentation [h,w]  
    mask = np.argmax(mask, axis=2)
    for type in range(1,label_types): # no mask on no class
        color[np.where(mask==type)] = np.uint8(colors[type])

    # apply alpha transparency mask
    cv2.addWeighted(color, alpha, image, 1-alpha, 0, image)
    return np.uint8(image)

def lab_to_labels(images, cluster_centers):
    a = tf.reshape(cluster_centers[:,0], [1,1,1,-1])
    b = tf.reshape(cluster_centers[:,1], [1,1,1,-1])
    da = tf.expand_dims(images[:,:,:,1],3) - a
    db = tf.expand_dims(images[:,:,:,2],3) - b
    d = tf.square(da) + tf.square(db)
    return tf.argmin(d, 3)

def labels_to_lab(labels, cluster_centers):
    if labels.dtype in [tf.float16, tf.float32, tf.float64]: # soft label
        l = tf.cast(tf.expand_dims(labels,-1), tf.float32)
        c = tf.reshape(cluster_centers, [1,1,1,-1,2])
        ab = tf.reduce_sum(l * c, 3)
    else:
        ab = tf.gather(cluster_centers, labels) # hard label
    l = tf.ones(tf.shape(ab)[:-1], tf.float32) * 75
    return tf.concat([tf.expand_dims(l,-1), ab], 3)

# read video and corresponding label dirs
with open(data_dir + '/video_dirs.txt', 'r') as f:
    video_names = f.read().splitlines()
video_dirs = [data_dir+'/video/'+ name for name in video_names]
mask_dirs = [data_dir+'/mask/' + name for name in video_names]

'''load trained model'''
with tf.Graph().as_default() as graph:
    saver = tf.train.import_meta_graph(os.path.join(model_dir, 'model.meta')) # load model

    # get tensors
    images = graph.get_tensor_by_name('input/images:0') # [N, T, H, W, 3]
    embeddings = graph.get_tensor_by_name('feature_extraction/embeddings:0')  # [N, T, H', W', D]
    is_training = graph.get_tensor_by_name('input/is_training:0')
    cluster_centers = graph.get_tensor_by_name('clustering/clusters:0')

    # labels input
    labels = tf.placeholder(tf.float32, [ref_frame, embed_size[0], embed_size[1], label_types])
    labels_color = tf.reshape(lab_to_labels(tf.image.resize_images(tf.reshape(images, (-1,)+image_size+(3,)),embed_size), cluster_centers), (ref_frame+1,)+embed_size)

    # track segmentation and color
    results_seg = colorizer(embeddings[0,:ref_frame], labels, embeddings[0, ref_frame:], temperature=temperature, window=window)
    predictions_seg = results_seg['predictions']
    results_color = colorizer(embeddings[0,:ref_frame], tf.one_hot(labels_color[:ref_frame], color_clusters), embeddings[0, ref_frame:], temperature=temperature, window=window)
    predictions_color = tf.concat([(images[0,ref_frame:,:,:,0:1]+1)/2, tf.image.resize_images(labels_to_lab(results_color['predictions'], cluster_centers)[...,1:], image_size)], axis=-1)

'''session'''
# use GPU memory based on runtime allocation and visible device
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
with tf.Session(graph=graph,config=config) as sess:
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    frames = np.zeros([ref_frame+1, image_size[0], image_size[1], 3])
    masks = np.zeros([ref_frame, embed_size[0], embed_size[1], label_types])
    # read pca for embedding visualization
    with open(os.path.join(model_dir, 'pca.pkl'), 'rb') as f:
    	pca = pickle.load(f)    

    for video_id in range(len(video_dirs)):
        video_dir = video_dirs[video_id]
        mask_dir = mask_dirs[video_id]
        mask_list = set(os.listdir(mask_dir))
        print('inference on video: ' + video_dir)

        # init video writer
        outfile = output_dir + '/' + video_dir.split('/')[-1]
        outfile_embed = output_dir + '/embed_' + video_dir.split('/')[-1]
        outfile_color = output_dir + '/color_' + video_dir.split('/')[-1]
        fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        video = cv2.VideoWriter(filename=outfile, fourcc=fourcc, fps=30.0, frameSize=image_size[::-1])
        video_embed = cv2.VideoWriter(filename=outfile_embed, fourcc=fourcc, fps=30.0, frameSize=embed_size[::-1])
        video_color = cv2.VideoWriter(filename=outfile_color, fourcc=fourcc, fps=30.0, frameSize=image_size[::-1])

        # init video reader
        capture = cv2.VideoCapture(video_dir)
        capture.set(5, 30)
        count = 0
        while (capture.isOpened()):
            ret, frame = capture.read()
            if not ret:
                break
            else:
                if count < ref_frame+1:
                    frames[count] = cv2.resize(frame, image_size[::-1])
                    if count != ref_frame:
                        mask_name = 'Frame%04d_ordered.png' % (count+1)
                        masks[count] = label_preprocess(cv2.resize(cv2.imread(mask_dir + '/' + mask_name, cv2.IMREAD_GRAYSCALE), embed_size[::-1]))

                    if data_type=='davis': # only 1st frame mask
                        for i in range(1, ref_frame):
                            frames[i] = frames[0]
                            masks[i] = masks[0]
                        frames[ref_frame] = frames[0]
                        count = ref_frame
                else:
                    print('iteration ' + str(count))
                    pred, pred_color, pred_embed = sess.run([predictions_seg, predictions_color, embeddings], 
                                                {images: [image_preprocess(frames)],
                                                 labels: masks,
                                                 is_training: False})
                    pred_mask = cv2.resize(pred[0], image_size[::-1])
 
                    feat_flat = pred_embed[0,-1].reshape((-1, pred_embed.shape[-1]))
                    feat_flat = pca.transform(feat_flat)
                    feat_flat -= np.min(feat_flat)
                    feat_flat /= np.max(feat_flat)
  
                    # write to video
                    video.write(apply_mask(frames[-1], pred_mask))
                    video_embed.write(np.uint8(feat_flat.reshape(embed_size+(3,))*255.))
                    video_color.write(np.uint8(cv2.cvtColor(pred_color[0], cv2.COLOR_LAB2BGR)*255.))

                    # update images
                    frames[:-1] = frames[1:]
                    frames[-1] = cv2.resize(frame, image_size[::-1])

                    # update labels
                    masks[:-1] = masks[1:]
                    mask_name = 'Frame%04d_ordered.png' % (count+1)
                    if mask_name in mask_list and count%30==0:
                        # use ground truth if provided
                        masks[-1] = label_preprocess(cv2.resize(cv2.imread(mask_dir + '/' + mask_name, cv2.IMREAD_GRAYSCALE), embed_size[::-1]))
                    else:
                        # use prediction as label
                        masks[-1] = pred[0]

                # update counting
                count += 1

        # end all capture
        capture.release()
        video.release()
        video_embed.release()
        video_color.release()







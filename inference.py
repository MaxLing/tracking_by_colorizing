import os, cv2, pickle
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
from nets import colorizer

# Parameters
data_dir = os.path.join(os.path.dirname(__file__), 'data')
model_dir = os.path.join(os.path.dirname(__file__), 'model')
image_size = (92, 180) # crop downsize/4
embed_size = (12, 23)  # image_size/8
#image_size = (185,260) # downsize/2
#embed_size = (24,45)
ref_frame = 3
label_types = 6
temperature = 0.5
alpha = 0.4

output_dir = os.path.join(data_dir, 'output')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# make colors for masking
#colors = (np.arange(label_types)+1).reshape((-1,1))*(255./label_types)
#colors = np.broadcast_to(colors, (label_types, 3))
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

# read video and corresponding label dirs
with open(data_dir + '/video_dirs.txt', 'r') as f:
    video_dirs = f.read().splitlines()
mask_dirs = []
for video_dir in video_dirs:
    group = video_dir.split('/')
    mask_dir = group[0]+'/'+group[1]+'/mask_'+group[2]
    mask_dirs.append(mask_dir)


'''load trained model'''
with tf.Graph().as_default() as graph:
    saver = tf.train.import_meta_graph(os.path.join(model_dir, 'model.meta')) # new model
    # saver = tf.train.import_meta_graph(os.path.join(model_dir, 'model.ckpt-199.meta')) # old model

    # get tensors
    images = graph.get_tensor_by_name('input/images:0') # [N, T, H, W, 3]
    embeddings = graph.get_tensor_by_name('feature_extraction/embeddings:0')  # [N, T, H', W', D]
    is_training = graph.get_tensor_by_name('input/is_training:0') # new model
    # is_training = graph.get_tensor_by_name('input/Placeholder:0') # old model

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
    masks = np.zeros([ref_frame, embed_size[0], embed_size[1], label_types])
    # read pca for embedding visualization
    with open(os.path.join(model_dir, 'pca.pkl'), 'r') as f:
    	pca = pickle.load(f)    

    # init
    for i in range(0, ref_frame + 1):
        frames[i] = cv2.resize(cv2.imread(video_dir+'/'+frame_list[i]), image_size[::-1])
        if i != ref_frame:
            mask_name = frame_list[i].split('.')
            mask_name = mask_name[0] + '_ordered.' + mask_name[-1]
            masks[i] = label_preprocess(cv2.resize(cv2.imread(mask_dir + '/' + mask_name, cv2.IMREAD_GRAYSCALE), embed_size[::-1]))

    # init video writer
    outfile = output_dir + '/' + video_dir.split('/')[-1]
    outfile_embed = output_dir + '/embed_' + video_dir.split('/')[-1]
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    video = cv2.VideoWriter(filename=outfile, fourcc=fourcc, fps=30.0, frameSize=image_size[::-1])
    video_embed = cv2.VideoWriter(filename=outfile_embed, fourcc=fourcc, fps=30.0, frameSize=embed_size[::-1])
    for i in range(ref_frame, len(frame_list)):
        print('iteration ' + str(i))
        pred, pred_embed = sess.run([predictions,embeddings], {images: [image_preprocess(frames)],
                                      labels: masks,
                                      is_training: False})
        pred_mask = cv2.resize(pred[0], image_size[::-1])

        feat_flat = pred_embed[0,-1].reshape((-1, pred_embed.shape[-1]))
        feat_flat = pca.transform(feat_flat)
        feat_flat /= np.abs(feat_flat).max()
        feat_flat = (feat_flat+1)/2
  
        # write to video
        video.write(apply_mask(frames[-1], pred_mask))
        video_embed.write(np.uint8(feat_flat.reshape(embed_size+(3,))*255.))

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
            masks[-1] = label_preprocess(cv2.resize(cv2.imread(mask_dir + '/' + mask_name, cv2.IMREAD_GRAYSCALE), embed_size[::-1]))
        else:
            # use prediction as label
            masks[-1] = pred[0]

    # end capture
    video.release()
    video_embed.release()







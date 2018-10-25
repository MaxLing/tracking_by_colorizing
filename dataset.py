import os, cv2
import numpy as np
import tensorflow as tf
from itertools import cycle

class Dataset:
    def __init__(self, dir, batch_size, ref_frame, image_size):
        self.dir = dir
        self.vid_dirs = []
        self.batch_size = batch_size
        self.ref_frame = ref_frame
        self.image_size = tuple(image_size)
        self.frames = np.zeros([self.ref_frame + 1, self.image_size[0], self.image_size[1], 3])

    def split_video(self):
        vids = [vid for vid in os.listdir(self.dir+'/video')]
        for vid in vids:
            print('start to split video ' + vid)
            vid_dir = self.dir + '/' + vid
            if not os.path.exists(vid_dir):
                os.mkdir(vid_dir)
            self.vid_dirs.append(vid_dir)

            capture = cv2.VideoCapture(self.dir+'/video/'+vid)
            capture.set(5, 30) # set fps
            count = 1
            while (capture.isOpened()):
                ret, frame = capture.read()
                if not ret:
                    break
                else:
                    # crop black bound of frame, specific to video data!!!
                    frame = frame[55:425,...]

                    cv2.imwrite(vid_dir + '/Frame{:04d}.png'.format(count), frame)
                    print('frame {:04d} captured'.format(count))
                    count += 1
            capture.release()

        # write videos dir to file
        open(self.dir+'/video_dirs.txt', 'w').close()
        with open(self.dir+'/video_dirs.txt', 'w') as f:
            for vid_dir in self.vid_dirs:
                f.write("{:s}\n".format(vid_dir))

    def crop_mask(self):
        # crop mask also
	for vid_dir in self.vid_dirs:
	    group = vid_dir.split('/')
            mask_dir = group[0]+'/'+group[1]+'/mask_'+group[2]
            
            for mask in os.listdir(mask_dir):
	        frame = cv2.imread(mask_dir+'/'+mask, cv2.IMREAD_GRAYSCALE)
                frame = frame[55:425,...]
                cv2.imwrite(mask_dir+'/'+mask, frame)
                print('mask '+mask_dir+'/'+mask+' cropped')

    def load_data_batch(self):
        # load video dirs if necessary
        if not self.vid_dirs:
            with open(self.dir + '/video_dirs.txt', 'r') as f:
                self.vid_dirs = f.read().splitlines()

        # tf dataset from generator
        types = tf.float32
        shapes = tf.TensorShape([self.ref_frame+1, self.image_size[0], self.image_size[1], 3])
        return tf.data.Dataset.from_generator(self.frames_generator, types, shapes)

    def frames_generator(self):
        for vid_dir in cycle(self.vid_dirs):
            print('reading video ' + vid_dir)
            frame_list = sorted(os.listdir(vid_dir))
            length = len(frame_list)
            
            # init
            for i in range(0, self.ref_frame+1):
                org_frame = cv2.cvtColor(np.float32(cv2.imread(vid_dir+'/'+frame_list[i])/255.), cv2.COLOR_BGR2LAB)
                self.frames[i] = cv2.resize(org_frame, self.image_size[::-1]) # note cv2.resize w*h

            # update frames batch
            for i in range(self.ref_frame, length):
                if i == self.ref_frame:
                    yield self.frames
                else:
                    self.frames[:-1] = self.frames[1:]
                    org_frame = cv2.cvtColor(np.float32(cv2.imread(vid_dir+'/'+frame_list[i])/255.), cv2.COLOR_BGR2LAB)
                    self.frames[-1] = cv2.resize(org_frame, self.image_size[::-1])
                    yield self.frames

            # # load with stride (ref_frame+1)
            # frames = np.zeros([self.ref_frame + 1, self.image_size[0], self.image_size[1], 3])
            # for i in range(0, length-self.ref_frame+1, self.ref_frame+1):
            #     for j in range(0, self.ref_frame+1):
            #         org_frame = cv2.cvtColor(np.float32(cv2.imread(vid_dir+'/'+frame_list[i+j])/255.), cv2.COLOR_BGR2LAB)
            #         frames[j] = cv2.resize(org_frame, self.image_size[::-1])
            #     yield frames

if __name__ == '__main__':
    data = Dataset(dir='./data', batch_size=5, ref_frame=3, image_size=[240, 360])
    data.split_video()
    data.crop_mask()

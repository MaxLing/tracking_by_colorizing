import os, cv2, argparse
import numpy as np
import tensorflow as tf
import matplotlib._png as png
from itertools import cycle

class Dataset:
    def __init__(self, data_dir, batch_size, ref_frame, image_size, data_type):
        self.data_dir = data_dir
        self.vid_dirs = []
        self.batch_size = batch_size
        self.ref_frame = ref_frame
        self.image_size = tuple(image_size)
        self.frames = np.zeros([self.ref_frame + 1, self.image_size[0], self.image_size[1], 3])
        self.data_type = data_type
 
    def split_video(self):
        vids = [vid for vid in os.listdir(self.data_dir+'/video')]
        for vid in vids:
            print('start to split video ' + vid)
            self.vid_dirs.append(vid)
            
        # write videos dir to file
        open(self.data_dir+'/video_dirs.txt', 'w').close() # clear old file
        with open(self.data_dir+'/video_dirs.txt', 'w') as f:
            for vid_dir in self.vid_dirs:
                f.write("{:s}\n".format(vid_dir))
    
    def crop_mask(self):
        # crop mask, use matplotlib can read the mask(uint6)
        for vid_dir in self.vid_dirs:
            mask_dir = self.data_dir+'/mask/'+vid_dir
            print('start to crop mask ' + mask_dir)
            
            for mask in os.listdir(mask_dir):
                frame = np.uint8(png.read_png_int(mask_dir+'/'+mask))
                frame = frame[55:425,...]
                cv2.imwrite(mask_dir+'/'+mask, frame)
                # print('mask '+mask_dir+'/'+mask+' cropped')

    def load_data_batch(self):
        # load video dirs if necessary
        if not self.vid_dirs:
            with open(self.data_dir + '/video_dirs.txt', 'r') as f:
                self.vid_dirs = f.read().splitlines()
                np.random.shuffle(self.vid_dirs)
        
        # tf dataset from generator
        types = tf.float32
        shapes = tf.TensorShape([self.ref_frame+1, self.image_size[0], self.image_size[1], 3])
        return tf.data.Dataset.from_generator(self.frames_generator, types, shapes)

    def frames_generator(self):
        for vid_dir in cycle(self.vid_dirs):
            print('reading video ' + vid_dir)
              
            capture = cv2.VideoCapture(self.data_dir+'/video/'+vid_dir)
            capture.set(5, 30) # set fps
            count = 0
            while (capture.isOpened()):
                ret, frame = capture.read()
                if not ret:
                    break
                else:
                    # crop black bound of frame is sipecific to video data!!!
                    if self.data_type=='surgical':
                        frame = frame[55:425,...]

                    '''
                    # load stride = 1, faster
                    if count < self.ref_frame+1:
                        self.frames[count] = cv2.resize(cv2.cvtColor(np.float32(frame/255.), cv2.COLOR_BGR2LAB), self.image_size[::-1])
                        if count == self.ref_frame:
                            yield self.frames
                    else:    
                        self.frames[:-1] = self.frames[1:]
                        self.frames[-1] = cv2.resize(cv2.cvtColor(np.float32(frame/255.), cv2.COLOR_BGR2LAB), self.image_size[::-1])
                        yield self.frames
                    count += 1
                    '''
                    # load stride = ref_frame+1, use more data
                    if count < self.ref_frame+1:
                        self.frames[count] = cv2.resize(cv2.cvtColor(np.float32(frame/255.), cv2.COLOR_BGR2LAB), self.image_size[::-1])
                        count += 1
                    else:
                        count = 0
                        yield self.frames

            capture.release()

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Dataset preprocess')
    parser.add_argument('--dir', '-d', type=str, default='./kinetics',
                        help='dataset directory')
    parser.add_argument('--type', '-t', type=str, default='kinetics', 
                        choices=['surgical','kinetics'], help='dataset type')
    args = parser.parse_args()
    
    data = Dataset(data_dir=args.dir, batch_size=4, ref_frame=3, image_size=[185, 360], data_type=args.type)
    data.split_video()
    if args.type == 'surgical':
        data.crop_mask()

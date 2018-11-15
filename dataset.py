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
 
    def list_video(self):
        vids = [vid for vid in os.listdir(self.data_dir+'/video')]
        for vid in vids:
            print('start to list video ' + vid)
            self.vid_dirs.append(vid)
            
        # write videos dir to file
        open(self.data_dir+'/video_dirs.txt', 'w').close() # clear old file
        with open(self.data_dir+'/video_dirs.txt', 'w') as f:
            for vid_dir in self.vid_dirs:
                f.write("{:s}\n".format(vid_dir))
    
    def format_all(self):
        # read video names
        with open(self.data_dir + '/ImageSets/2017/test-dev.txt', 'r') as f:
                self.vid_dirs = f.read().splitlines()
        self.vid_dirs = [vid+'.avi' for vid in self.vid_dirs]

        # make dir
        video_dir = self.data_dir + '/video'
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)
        mask_dir = self.data_dir + '/mask'
        if not os.path.exists(mask_dir):
            os.mkdir(mask_dir)

        # make videos and check masks
        fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        for vid_dir in self.vid_dirs:
            print('start to format video: ' + vid_dir)

            # make video from frames
            vid_root = self.data_dir + '/JPEGImages/480p/' + vid_dir.split('.')[0]
            frame_list = sorted(os.listdir(vid_root))
            img = cv2.imread(vid_root+'/'+frame_list[0])
            h,w,_ = img.shape
            video = cv2.VideoWriter(filename=video_dir+'/'+vid_dir, fourcc=fourcc, fps=30.0, frameSize=(w, h))
            for frame in frame_list:
                video.write(cv2.imread(vid_root+'/'+frame))
            video.release()

            # order mask
            mask_subdir = mask_dir+'/'+vid_dir
            if not os.path.exists(mask_subdir):
                os.mkdir(mask_subdir)

            mask_frame = self.data_dir+'/Annotations/480p/'+vid_dir.split('.')[0]+'/00000.png'
            mask = cv2.imread(mask_frame, cv2.IMREAD_GRAYSCALE)
            labels = np.unique(mask).tolist()
            for i in range(len(labels)):
                mask[np.where(mask==labels[i])] = i
            cv2.imwrite(mask_subdir +'/Frame0001_ordered.png', mask)

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
                        choices=['surgical','kinetics', 'davis'], help='dataset type')
    args = parser.parse_args()
    
    data = Dataset(data_dir=args.dir, batch_size=4, ref_frame=3, image_size=[185, 360], data_type=args.type)

    if args.type == 'davis':
        data.format_all() # change to 1 file video_dirs.txt and 2 dirs video+mask
    else:
        data.list_video()

    if args.type == 'surgical':
        data.crop_mask()

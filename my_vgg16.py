import tensorflow.compat.v1 as tf
import numpy as np
import vgg16
import utils
import os,cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


data_path = '../../jaffe/'
data_dir_list = os.listdir(data_path)

img_rows=256
img_cols=256
num_channel=1

num_epoch=10

img_data_list=[]


for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
        input_img_resize=cv2.resize(input_img,(224,224))
        img_data_list.append(input_img_resize)
        
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data = img_data/255
img_data.shape

img_data=img_data[0:10]

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7,allow_growth=True)
features=[]
i=0
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    images = tf.placeholder("float", [10, 224, 224, 3])
    feed_dict = {images: img_data}
    vgg = vgg16.Vgg16()
    with tf.name_scope("content_vgg"):
        vgg.build(images)
    feature = sess.run(vgg.fc8, feed_dict=feed_dict)#提取fc8层的特征

    features.append(feature)

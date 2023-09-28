import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import trange
import os
from PIL import Image
import tensorflow as tf
np.random.seed(0)



DATA_DIR = "flowers"
x_paths = [DATA_DIR+"/"+i for i in sorted(os.listdir(DATA_DIR)) if not i.startswith(".")]
data = []
for image_path_i in trange(len(x_paths)):
    image_path = x_paths[image_path_i]
    data.append(tf.image.resize(np.array(Image.open(image_path)), (256,256)))

print("to np")
data = np.array(data)
print(data.shape)
print("train/test/val split pt1")
X_train, X_test, = train_test_split(data, test_size=0.2, random_state=1)
print("train/test/val split pt2")
X_train, X_val = train_test_split(X_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2


print("storing")
with open('cache/X_train.npy', 'wb+') as f:
    np.save(f, np.array(X_train))
with open('cache/X_val.npy', 'wb+') as f:
    np.save(f, np.array(X_val))
with open('cache/X_test.npy', 'wb+') as f:
    np.save(f, np.array(X_test))
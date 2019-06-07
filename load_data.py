import os
import pickle

import numpy as np
from scipy import misc


# Loading dataset
def load_datasets():
    
    X=[]
    y=[]
    for image_label in label:
        print(image_label)
        images = os.listdir("./images/train/"+image_label)
        for image in images:
            # print(image)
            img = misc.imread("./images/train/"+image_label+"/"+image)
            img = misc.imresize(img, (150, 150))
            X.append(img)
            # np.array(X)
            y.append(label.index(image_label))

    X = np.array(X)
    y=np.array(y)
    return X,y

# Save int2word dict
label = os.listdir("./images/train/")
save_label = open("int_to_word_out.pickle","wb")
pickle.dump(label, save_label)
save_label.close()

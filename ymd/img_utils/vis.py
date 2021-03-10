
import cv2
import matplotlib.pyplot as plt

def add_plt_image(img, name=None, title=None, flagBGR=False):
    if ( name is not None ):
        fig = plt.figure(num=name)
    else:
        fig = plt.figure()

    if ( flagBGR ):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ax = fig.add_subplot(111)
    ax.imshow(img)
    ax.axis('off')

    if ( title is not None ):
        ax.set_title(title)

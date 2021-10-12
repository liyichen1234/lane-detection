import os
import cv2
import matplotlib.pyplot as plt

for i in range(10,100):
    f, axs = plt.subplots(1, 2, figsize=(16, 9))
    f.tight_layout()
    path1 = 'approch/'+str(i)+'.jpg'
    path2 =  'hough/'+str(i)+'.jpg'
    axs[0].imshow(cv2.cvtColor(cv2.imread(path2),cv2.COLOR_BGR2RGB), cmap='gray')
    axs[0].set_title('Hough', fontsize=18)
    axs[1].imshow(cv2.cvtColor(cv2.imread(path1),cv2.COLOR_BGR2RGB), cmap='gray')
    axs[1].set_title('Ours', fontsize=18)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig('compare/'+str(i)+'.jpg')
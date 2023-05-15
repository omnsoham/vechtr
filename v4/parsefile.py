import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


def save_image(like_value, photo, img):
   filename = "C:/Users/Soham Kulkarni/OneDrive/Documents/GitHub/vechtor/v4/"

   # body like
   if like_value == 's':
       print('doorimages')
       filename += "doortestimages/" + photo
       plt.imsave(filename, img)

   # body dislike
   if like_value == 'd':
       print('notdoor')
       filename += 'trash/' + photo
       plt.imsave(filename, img)

def main():
   base_path = "C:/Users/Soham Kulkarni/OneDrive/Documents/GitHub/vechtor/v4/doorimages/"
   for photo in os.listdir(base_path):
       photo_path = base_path + photo
       img = mpimg.imread(photo_path)
       plt.imshow(img)
       plt.show()

       like_value = input()

       # save pic to new folder and delete it from 'to_label' folder
       save_image(like_value, photo, img)
       os.remove(photo_path)


if __name__ == "__main__":
   main()
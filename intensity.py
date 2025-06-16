import cv2
import matplotlib.pyplot as plt

img = cv2.imread('group-photo.jpg')
print(f'Image shape: {img.shape}')
new_size= (10,10)
resized_img = cv2.resize(img, new_size)
print(f'Resized image shape: {resized_img.shape}')
print(f'BGR format: \n{resized_img}')
img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB) 
print(f'RGB format: \n {img_rgb}')

pixel_values = [20/255, 233/255, 150/255]
'''
We need to divide each value by 255 to normalise the values 
because Matplotlip only works with colour values ranging from 0 to 1
'''
# Create a square patch with the chosen color
patch = plt.Rectangle((0, 0), 1, 1, color=pixel_values)

# Create a plot with the colored patch
fig, ax = plt.subplots()
ax.add_patch(patch)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Show the plot
plt.show()
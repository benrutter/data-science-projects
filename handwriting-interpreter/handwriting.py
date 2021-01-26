import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

digits = datasets.load_digits()

model = KMeans(n_clusters=10, random_state=7)

model.fit(digits.data)

fig = plt.figure(figsize=(8, 3))
fig.suptitle('Cluser Center Images', fontsize=14, fontweight='bold')
for i in range(10):

   # Initialize subplots in a grid of 2X5, at i+1th position
    ax = fig.add_subplot(2, 5, 1 + i)

   # Display images
    ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

plt.show()

new_samples = np.array([
[0.00,0.46,3.36,4.57,4.35,0.38,0.00,0.00,0.00,4.11,7.62,7.02,7.62,1.83,0.00,0.00,0.00,0.38,1.07,1.91,7.62,2.29,0.00,0.00,0.00,0.00,0.00,5.03,7.47,0.91,0.00,0.00,0.00,0.00,3.59,7.62,4.35,0.00,0.00,0.00,0.00,3.05,7.63,5.03,0.15,0.00,0.00,0.00,1.07,7.55,7.62,7.02,6.86,6.86,6.86,1.98,0.08,2.82,3.74,3.81,3.81,3.81,3.81,0.76],
[0.00,0.00,1.15,4.12,3.81,0.92,0.00,0.00,0.00,2.52,7.55,7.40,7.47,7.17,1.37,0.00,0.00,6.55,6.56,0.61,1.30,6.86,6.41,0.00,0.00,7.63,4.57,0.00,0.00,4.35,7.63,0.00,0.00,5.87,7.02,0.23,0.00,5.57,6.63,0.00,0.00,2.21,7.62,6.71,6.10,7.55,4.50,0.00,0.00,0.00,2.75,4.57,4.57,4.27,0.46,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.77,2.29,2.29,2.29,0.69,0.00,0.00,0.38,7.09,7.62,7.62,7.62,4.35,0.00,0.00,0.15,4.12,1.60,2.67,7.62,3.20,0.00,0.00,0.00,0.00,0.76,6.79,6.79,0.46,0.00,0.00,0.00,0.00,5.41,7.40,1.91,0.00,0.00,0.00,0.00,2.74,7.62,5.87,3.05,3.05,3.05,2.98,0.00,4.12,7.62,7.62,7.62,7.62,7.62,7.55,0.00,0.15,0.76,0.61,0.00,0.00,0.00,0.00],
[0.00,0.00,0.84,1.53,0.76,0.00,0.00,0.00,0.00,1.15,7.17,7.62,7.63,3.66,0.00,0.00,0.08,6.25,7.17,2.36,5.11,7.63,2.59,0.00,2.06,7.62,3.05,0.00,0.23,6.18,7.40,0.46,2.75,7.62,1.30,0.00,0.00,4.04,7.62,0.61,2.90,7.62,3.36,0.23,0.61,7.09,5.64,0.00,0.31,5.95,7.62,7.01,6.79,7.40,1.37,0.00,0.00,0.08,2.44,4.50,4.57,2.52,0.00,0.00]
])

new_labels = model.predict(new_samples)

for i in range(0, len(new_labels)):
    if new_labels[i] == 0:
      print(4, end='')
    elif new_labels[i] == 1:
      print(8, end='')
    elif new_labels[i] == 2:
      print(3, end='')
    elif new_labels[i] == 3:
      print(9, end='')
    elif new_labels[i] == 4:
      print(5, end='')
    elif new_labels[i] == 5:
      print(6, end='')
    elif new_labels[i] == 6:
      print(7, end='')
    elif new_labels[i] == 7:
      print(1, end='')
    elif new_labels[i] == 8:
      print(0, end='')
    elif new_labels[i] == 9:
      print(2, end='')

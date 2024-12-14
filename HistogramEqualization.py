import imageio
import numpy as np
import matplotlib.pyplot as plt

image_path = 'lowContrastImages.jpg'
image = imageio.imread(image_path, as_gray=True)

if image.max() > 1.0:
    image = image / 255.0


def histogram_equalization(image):
    # Hitung histogram dari citra
    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 1))
    cdf = histogram.cumsum()  # Hitung cumulative distribution function
    cdf_normalized = cdf / cdf.max()  # Normalisasi CDF

    # Gunakan CDF untuk equalize citra
    image_equalized = np.interp(image.flat, bin_edges[:-1], cdf_normalized)
    image_equalized = image_equalized.reshape(image.shape)

    return image_equalized


image_equalized = histogram_equalization(image)

# Periksa nilai intensitas sebelum dan sesudah equalization
print("Intensitas citra asli: min =", image.min(), ", max =", image.max())
print("Intensitas citra setelah histogram equalization: min =",
      image_equalized.min(), ", max =", image_equalized.max())

plt.figure(figsize=(12, 6))

# Tampilkan citra asli
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Citra Asli')
plt.axis('off')

# Tampilkan citra yang telah diperbaiki
plt.subplot(1, 2, 2)
plt.imshow(image_equalized, cmap='gray')
plt.title('Citra Setelah Histogram Equalization')
plt.axis('off')

plt.show()

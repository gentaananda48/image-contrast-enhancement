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

# Fungsi Peningkatan Kontras dengan Level 1.5


def enhance_contrast(image, level):
    mean_intensity = np.mean(image)
    image_enhanced = mean_intensity + level * (image - mean_intensity)
    return np.clip(image_enhanced, 0, 1)


# Menerapkan Histogram Equalization
image_equalized = histogram_equalization(image)

# Menerapkan Peningkatan Kontras dengan Level 1.5
contrast_level = 1.5
image_contrast_enhanced = enhance_contrast(image, contrast_level)

# Menampilkan Hasil
plt.figure(figsize=(18, 6))

# Tampilkan Citra Asli
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Citra Asli')
plt.axis('off')

# Tampilkan Citra dengan Histogram Equalization
plt.subplot(1, 3, 2)
plt.imshow(image_equalized, cmap='gray')
plt.title('Histogram Equalization')
plt.axis('off')

# Tampilkan Citra dengan Peningkatan Kontras Level 1.5
plt.subplot(1, 3, 3)
plt.imshow(image_contrast_enhanced, cmap='gray')
plt.title('Peningkatan Kontras Level 1.5')
plt.axis('off')

plt.show()

from scipy.ndimage import zoom

def rescaled_data(data):
    D = data.shape[2]

    start = (D - 32) // 2
    end = start + 32

    middle_parts = data[:, :, start:end]

    target_size = (128, 128, 32)
    zoom_factors = (target_size[0]/data.shape[0], target_size[1]/data.shape[1], 1)

    resized_data = zoom(middle_parts, zoom_factors, order=1)

    return resized_data
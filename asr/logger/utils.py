import io
import matplotlib.pyplot as plt


def plot_spectrogram(spectrogram_tensor, name=None):
    plt.figure(figsize=(20, 5))
    plt.imshow(spectrogram_tensor.permute(1, 0))
    plt.title(name)
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    return buffer

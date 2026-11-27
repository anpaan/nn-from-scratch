"""Download MNIST (Keras copy, ~11 MB) into data/mnist.npz.

Run from the repo root:  python -m scripts.download_mnist
"""

import os
import urllib.request

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
PATH = os.path.join("data", "mnist.npz")


def main():
    if os.path.exists(PATH) and os.path.getsize(PATH) > 10_000_000:
        print(f"{PATH} already exists, skipping")
        return
    os.makedirs("data", exist_ok=True)
    print(f"downloading {URL} ...")
    urllib.request.urlretrieve(URL, PATH)
    size = os.path.getsize(PATH)
    print(f"saved {PATH} ({size / 1e6:.1f} MB)")
    if size < 10_000_000:
        raise RuntimeError("file is too small, the download probably failed")


if __name__ == "__main__":
    main()

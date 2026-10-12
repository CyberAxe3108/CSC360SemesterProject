from PIL import Image              # Pillow: opens image files
import numpy as np                 # NumPy: lets us treat the image as an array of numbers

# ---------------------------------------------------------------
# DECODE-ONLY SCRIPT: the receiver needs just two things - the stego PNG and the secret key
# ---------------------------------------------------------------
stego_name = input("Stego image file name (example: random_stego.png): ")   # Which image to read
key = int(input("Secret key (a number): "))                                  # The shared secret key

pixels = np.array(Image.open(stego_name).convert('L')).flatten()   # Grayscale, 1D list of pixels

# Rebuild the same shuffled order of pixel positions. Same key = same order.
positions = np.random.default_rng(key).permutation(len(pixels))

lsbs = (pixels[positions] & 1).astype(np.uint8)    # Read the last bit of each pixel, in shuffled order
n_bytes = len(lsbs) // 8                           # How many full bytes we can build
raw = np.packbits(lsbs[:n_bytes * 8]).tobytes()    # Group the bits back into bytes
end = raw.find(b"##")                              # Look for the "##" end marker

if end == -1:
    print("Message not found (wrong key, or not the right image)")
else:
    print("Decoded message:", raw[:end].decode('latin-1'))   # Print the text before the marker
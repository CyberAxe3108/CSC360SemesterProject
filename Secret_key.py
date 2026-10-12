from PIL import Image              # Pillow: opens and saves image files
import numpy as np                 # NumPy: lets us treat the image as an array of numbers
from scipy.stats import chi2       # SciPy: gives us the chi-square probability (p-value)
import matplotlib.pyplot as plt    # Matplotlib: draws the chart

# ---------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------
cover_name = "SunsetTree.jpg"      # The original image
stego_name = "random_stego.png"    # Output image (keep .png!)
secret_key = 31110208               # The secret key: sender and receiver must use the SAME number

# ---------------------------------------------------------------
# PART 1: Hide a message at RANDOM pixel positions chosen by a secret key
# ---------------------------------------------------------------
def pixel_order(n_pixels, key):
    # The key seeds a random shuffle of all pixel positions.
    # Same key = same shuffle, so the receiver can rebuild the exact same order.
    return np.random.default_rng(key).permutation(n_pixels)

def encode_random(pixels, message, key):
    data = (message + "##").encode('ascii')               # Add the "##" end marker, convert to bytes
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))   # Every byte -> 8 bits (0s and 1s)
    assert len(bits) <= len(pixels), "Message too large for this image"
    positions = pixel_order(len(pixels), key)[:len(bits)]  # The first len(bits) positions of the shuffle
    stego = pixels.copy()                                  # Never change the original
    stego[positions] = (stego[positions] & 0xFE) | bits    # Put one message bit in each chosen pixel's last bit
    return stego

def decode_random(pixels, key):
    positions = pixel_order(len(pixels), key)              # Rebuild the same shuffle from the key
    lsbs = (pixels[positions] & 1).astype(np.uint8)        # Read the last bit of each pixel, in shuffled order
    n_bytes = len(lsbs) // 8                               # How many full bytes we can build
    raw = np.packbits(lsbs[:n_bytes * 8]).tobytes()        # Group the bits back into bytes
    end = raw.find(b"##")                                  # Look for the end marker
    if end == -1:
        return None                                        # Marker not found: no message (wrong key?)
    return raw[:end].decode('latin-1')                     # Return the text before the marker

cover = np.array(Image.open(cover_name).convert('L'))      # Grayscale image (2D)
H, W = cover.shape                                         # Height and width in pixels
original = cover.flatten()                                 # 1D list of pixels
n_pixels = len(original)

message = input("Enter secret message: ")                  # Ask for the secret text
stego = encode_random(original, message, secret_key)       # Hide it using the secret key
Image.fromarray(stego.reshape(H, W)).save(stego_name)      # Back to 2D and save as PNG
print("Saved", stego_name)

# Reload from the saved file and decode with the RIGHT key
saved = np.array(Image.open(stego_name).convert('L')).flatten()
print("Decoded with the right key:", decode_random(saved, secret_key))

# Try the WRONG key: the shuffle is different, so the message cannot be found
wrong = decode_random(saved, secret_key + 1)
if wrong is None:
    print("Decoded with the wrong key: Message not found")
else:
    print("Decoded with the wrong key (garbage):", repr(wrong[:40]))

# ---------------------------------------------------------------
# PART 2: Is random placement harder for the chi-square detector to catch?
# ---------------------------------------------------------------
def chi_square_pvalue(pixels):
    counts = np.bincount(pixels, minlength=256).astype(float)   # How many pixels have each value 0-255
    even = counts[0::2]                            # Counts for values 0, 2, 4, ...
    odd = counts[1::2]                             # Counts for values 1, 3, 5, ...
    expected = (even + odd) / 2                    # If a pair were perfectly equal, each would be this
    keep = expected > 5                            # Skip pairs with too few pixels to test fairly
    stat = np.sum((even[keep] - expected[keep]) ** 2 / expected[keep])   # Chi-square statistic
    return chi2.sf(stat, keep.sum() - 1)           # Near 1 = pairs look equal = likely hidden data

fractions = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]   # Share of pixels that carry a hidden bit
rng = np.random.default_rng(0)                     # Fixed seed so results repeat every run
shuffle = pixel_order(n_pixels, secret_key)        # The key-based random order
p_seq = []                                         # Results for sequential (top-left first) embedding
p_rand = []                                        # Results for random (key-based) embedding

print("\nPayload    Sequential (scan)    Random (whole image)")
print("-" * 50)
for f in fractions:
    m = int(n_pixels * f)                          # How many pixels carry hidden bits
    bits = rng.integers(0, 2, size=m).astype(np.uint8)   # Random bits (like an encrypted message)

    seq = original.copy()                          # Sequential: bits go in the first m pixels
    seq[:m] = (seq[:m] & 0xFE) | bits

    rnd = original.copy()                          # Random: bits go to m key-chosen pixels
    rnd[shuffle[:m]] = (rnd[shuffle[:m]] & 0xFE) | bits

    # Detector for sequential: scan growing chunks from the top-left and keep the best score
    best = max(chi_square_pvalue(seq[:int(n_pixels * g)]) for g in np.linspace(0.05, 1.0, 20))
    # Detector for random: the hidden bits are everywhere, so test the whole image
    whole = chi_square_pvalue(rnd)

    p_seq.append(best)
    p_rand.append(whole)
    print(f"{int(f * 100):>5}%    {best:>14.4f}    {whole:>16.4f}")

# ---------------------------------------------------------------
# PART 3: Draw the chart
# ---------------------------------------------------------------
x = [f * 100 for f in fractions]                   # Convert to percent for the x-axis
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, p_seq, color="#d9622b", linewidth=2, marker="o", markersize=8,
        markeredgecolor="white", markeredgewidth=2, label="Sequential embedding")
ax.plot(x, p_rand, color="#2a6fdb", linewidth=2, marker="o", markersize=8,
        markeredgecolor="white", markeredgewidth=2, label="Random embedding (secret key)")

# Dashed line: above this the detector flags the image as containing hidden data
ax.axhline(0.95, color="#888888", linestyle="--", linewidth=1)
ax.text(5, 0.90, "0.95: detector flags hidden data", color="#555555", fontsize=9)

ax.set_xlabel("Payload (% of pixels carrying hidden bits)")
ax.set_ylabel("Chi-square p-value  (near 1 = detected)")
ax.set_title("Random placement is harder for chi-square to detect")
ax.set_ylim(-0.05, 1.08)
ax.grid(True, color="#e5e5e5", linewidth=0.8)      # Light grid, stays in the background
ax.spines["top"].set_visible(False)                # Remove top and right border lines
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, loc="center right")

plt.tight_layout()
plt.savefig("random_vs_sequential.png", dpi=150)   # Save the chart for your slides
print("\nSaved chart as random_vs_sequential.png")
plt.show()                                         # Show the chart on screen
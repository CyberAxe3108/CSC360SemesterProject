from PIL import Image              # Pillow: opens images
import numpy as np                 # NumPy: lets us treat the image as an array of numbers
from scipy.stats import chi2       # SciPy: gives us the chi-square probability (p-value)
import matplotlib.pyplot as plt    # Matplotlib: draws the chart

# ---------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------
cover_name = "SunsetTree.jpg"      # The original image
payload_fraction = 0.5             # Hide bits in the first 50% of the pixels (try 0.1, 0.25, 1.0)

# ---------------------------------------------------------------
# STEP 1: Load the cover image and build a stego copy
# ---------------------------------------------------------------
original = np.array(Image.open(cover_name).convert('L')).flatten()   # Grayscale, 1D list of pixels
n_pixels = len(original)

n_embed = int(n_pixels * payload_fraction)         # How many pixels will carry a hidden bit
rng = np.random.default_rng(0)                     # Fixed seed so results repeat every run
bits = rng.integers(0, 2, size=n_embed).astype(np.uint8)   # Random bits (like an encrypted message)

stego = original.copy()                            # Never change the original
stego[:n_embed] = (stego[:n_embed] & 0xFE) | bits  # LSB embedding, same idea as your encoder

# ---------------------------------------------------------------
# STEP 2: The chi-square detector
# ---------------------------------------------------------------
# Idea: pixel values come in pairs (0,1) (2,3) (4,5) ... that differ only in the last bit.
# In a normal photo, the two values in a pair usually have DIFFERENT counts.
# Hiding random bits in the last bit makes the two counts in each pair almost EQUAL.
# The test measures how close to equal the pairs are.
def chi_square_pvalue(pixels):
    counts = np.bincount(pixels, minlength=256).astype(float)   # How many pixels have each value 0-255
    even = counts[0::2]                            # Counts for values 0, 2, 4, ...
    odd = counts[1::2]                             # Counts for values 1, 3, 5, ...
    expected = (even + odd) / 2                    # If a pair were perfectly equal, each would be this
    keep = expected > 5                            # Skip pairs with too few pixels to test fairly
    stat = np.sum((even[keep] - expected[keep]) ** 2 / expected[keep])   # Chi-square statistic
    df = keep.sum() - 1                            # Degrees of freedom
    return chi2.sf(stat, df)                       # p-value: close to 1 = pairs look equal = likely hidden data

# ---------------------------------------------------------------
# STEP 3: Test the original and the stego image
# ---------------------------------------------------------------
print("Pixels in image:", n_pixels, "| Pixels carrying hidden bits:", n_embed)
print("\nTesting the first", n_embed, "pixels (where the message is):")
print("  Original image p-value: %.4f" % chi_square_pvalue(original[:n_embed]))
print("  Stego image p-value:    %.4f" % chi_square_pvalue(stego[:n_embed]))
print("\nA p-value near 1 means 'pairs look equal' = likely hidden data.")
print("A p-value near 0 means 'normal photo' = probably no hidden data.")

# ---------------------------------------------------------------
# STEP 4: Test growing chunks of the image, starting from the top-left
# ---------------------------------------------------------------
fractions = np.linspace(0.05, 1.0, 20)             # 5%, 10%, ... 100% of the image
p_original = []                                    # p-values for the original image
p_stego = []                                       # p-values for the stego image
for f in fractions:
    n = int(n_pixels * f)                          # Test only the first n pixels
    p_original.append(chi_square_pvalue(original[:n]))
    p_stego.append(chi_square_pvalue(stego[:n]))

# ---------------------------------------------------------------
# STEP 5: Draw the chart
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
x = fractions * 100                                # Convert to percent for the x-axis

ax.plot(x, p_original, color="#2a6fdb", linewidth=2, marker="o", markersize=8,
        markeredgecolor="white", markeredgewidth=2, label="Original image")
ax.plot(x, p_stego, color="#d9622b", linewidth=2, marker="o", markersize=8,
        markeredgecolor="white", markeredgewidth=2, label="Stego image")

# Dotted vertical line where the hidden data ends
ax.axvline(payload_fraction * 100, color="#888888", linestyle="--", linewidth=1)
ax.text(payload_fraction * 100 + 1, 0.5, "hidden data ends here", color="#555555",
        fontsize=9, rotation=90, va="center")

ax.set_xlabel("Amount of image tested (% from the top-left)")
ax.set_ylabel("Chi-square p-value  (near 1 = likely hidden data)")
ax.set_title("Chi-square detector: original vs. stego image")
ax.set_ylim(-0.05, 1.05)
ax.grid(True, color="#e5e5e5", linewidth=0.8)      # Light grid, stays in the background
ax.spines["top"].set_visible(False)                # Remove top and right border lines
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, loc="center right")

plt.tight_layout()
plt.savefig("chi_square_result.png", dpi=150)      # Save the chart for your slides
print("\nSaved chart as chi_square_result.png")
plt.show()                                         # Show the chart on screen
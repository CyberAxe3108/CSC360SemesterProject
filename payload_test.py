from PIL import Image              # Pillow: opens images
import numpy as np                 # NumPy: lets us treat the image as an array of numbers
import matplotlib.pyplot as plt    # Matplotlib: draws the chart

# ---------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------
cover_name = "SunsetTree.jpg"                 # The original image
message_sizes = [10, 100, 1000, 10000, 50000, 100000]   # Message lengths (characters) to test

# ---------------------------------------------------------------
# STEP 1: Load the cover image
# ---------------------------------------------------------------
original = np.array(Image.open(cover_name).convert('L'))   # Grayscale: one number (0-255) per pixel
H, W = original.shape                                       # Height and width in pixels
capacity_chars = (H * W) // 8                               # 1 bit per pixel, 8 bits per character
print("Image size:", H, "x", W, "| Max message length:", capacity_chars, "characters")

# Keep only the message sizes that actually fit in this image
message_sizes = [n for n in message_sizes if n <= capacity_chars]

# ---------------------------------------------------------------
# STEP 2: A function that hides a message and measures the damage
# ---------------------------------------------------------------
rng = np.random.default_rng(0)     # Fixed seed so you get the same results every run

def test_payload(n_chars):
    # Make a random message of n_chars letters (a-z), then turn it into bytes
    text = rng.integers(97, 123, size=n_chars).astype(np.uint8)
    # Turn every byte into 8 bits (a long array of 0s and 1s)
    bits = np.unpackbits(text)

    flat = original.flatten()                       # 1D list of pixels
    stego = flat.copy()                             # Work on a copy, never the original
    # Same idea as before: clear the last bit of each pixel, then put our bit there
    stego[:len(bits)] = (stego[:len(bits)] & 0xFE) | bits

    # How many pixels actually changed value
    changed = int(np.sum(stego != flat))

    # MSE = average squared difference between original and stego pixels
    mse = np.mean((flat.astype(float) - stego.astype(float)) ** 2)
    # PSNR in dB: higher = more similar to the original
    psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float('inf')
    return len(bits), changed, psnr

# ---------------------------------------------------------------
# STEP 3: Run the test for every message size and print a table
# ---------------------------------------------------------------
results = []                                        # Will hold one row per message size
print("\nChars     Bits hidden   Pixels changed   PSNR (dB)")
print("-" * 52)
for n in message_sizes:
    bits_hidden, changed, psnr = test_payload(n)
    results.append((n, bits_hidden, changed, psnr))
    print(f"{n:<9} {bits_hidden:<13} {changed:<16} {psnr:.2f}")

# ---------------------------------------------------------------
# STEP 4: Draw the chart (PSNR vs message length)
# ---------------------------------------------------------------
sizes = [r[0] for r in results]                     # x values: message length
psnrs = [r[3] for r in results]                     # y values: PSNR

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sizes, psnrs, color="#2a6fdb", linewidth=2, marker="o", markersize=8,
        markeredgecolor="white", markeredgewidth=2)  # Line with round markers
ax.set_xscale("log")                                # Log scale: sizes grow 10x each step

# Dashed reference line: above ~40 dB the change is generally invisible
ax.axhline(40, color="#888888", linestyle="--", linewidth=1)
ax.text(sizes[0], 40.8, "40 dB: change generally invisible", color="#555555", fontsize=9)

# Write the PSNR value next to each point
for x, y in zip(sizes, psnrs):
    ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=9, color="#222222")

ax.set_xlabel("Message length (characters, log scale)")
ax.set_ylabel("PSNR (dB)  - higher = less visible change")
ax.set_title("Hiding longer messages lowers image quality (LSB)")
ax.grid(True, color="#e5e5e5", linewidth=0.8)      # Light grid, stays in the background
ax.spines["top"].set_visible(False)                 # Remove top and right border lines
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("payload_vs_psnr.png", dpi=150)         # Save the chart for your slides
print("\nSaved chart as payload_vs_psnr.png")
plt.show()                                          # Show the chart on screen
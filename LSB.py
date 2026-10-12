from PIL import Image              # Pillow: opens and saves image files
import numpy as np                 # NumPy: lets us treat the image as an array of numbers
import matplotlib.pyplot as plt    # Matplotlib: shows images and charts on screen

# ---------------------------------------------------------------
# SETTINGS: change these names in ONE place
# ---------------------------------------------------------------
cover_name = "SunsetTree.jpg"      # The original image you start with
stego_name = "My_hidden_message.png"           # Name of the new image with the hidden message (keep .png!)

# ---------------------------------------------------------------
# STEP 1: Load the cover image
# ---------------------------------------------------------------
# Open the image and convert it to grayscale ('L'), so each pixel is one number (0-255)
img = np.array(Image.open(cover_name).convert('L'))
H, W = img.shape                   # H = height in pixels, W = width in pixels
print("H:", H, "W:", W)

original = img.copy()              # Keep an untouched copy so we can compare later

# ---------------------------------------------------------------
# STEP 2: Get the secret message and turn it into bits
# ---------------------------------------------------------------
message = input("Enter secret message: ")   # Ask the user for the secret text
message += "##"                              # Add "##" at the end so the decoder knows where to stop
message = message.encode('ascii')            # Convert the text into bytes (numbers)
print("Message:", message)

# Turn every byte into 8 binary digits and join them into one long string of 0s and 1s
message_bits = ''.join([format(i, '08b') for i in message])

# Make sure the message fits: we can hide 1 bit in each pixel
assert len(message_bits) <= H * W, "Message too large for this image"

# ---------------------------------------------------------------
# STEP 3: ENCODE - hide the bits in the least significant bit (LSB) of each pixel
# ---------------------------------------------------------------
img = img.flatten()                # Turn the 2D image into one long 1D list of pixels

for idx, bit in enumerate(message_bits):     # Go through each message bit, one at a time
    val = int(img[idx])                      # Get the pixel value (0-255)
    val = (val & 0xFE) | int(bit)            # 0xFE clears the last bit, then "| bit" puts our bit there
    img[idx] = val                           # Store the changed pixel back in the image

img = img.reshape(H, W)            # Turn the 1D list back into a 2D image

# Save as PNG (lossless). Never use JPG here: JPEG compression would destroy the hidden bits.
Image.fromarray(img).save(stego_name)
print("Saved stego image as", stego_name)

# ---------------------------------------------------------------
# STEP 4: MEASURE - how much did the image change?
# ---------------------------------------------------------------
changed = original != img          # True wherever a pixel value is different from the original
print("Pixels changed:", int(np.sum(changed)), "out of", H * W)
print("Bits hidden:", len(message_bits))

# MSE = average squared difference between original and stego pixels
mse = np.mean((original.astype(float) - img.astype(float)) ** 2)
if mse == 0:
    print("PSNR: infinite (images are identical)")
else:
    # PSNR (in decibels(dB)): higher = more similar. Above ~40 dB is generally invisible to the eye.
    psnr = 10 * np.log10(255 ** 2 / mse)
    print("PSNR: %.2f dB" % psnr)

# ---------------------------------------------------------------
# STEP 5: SHOW three images side by side
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))    # Make a row of 3 plots

axes[0].imshow(original, cmap='gray')              # Left: the original image
axes[0].set_title("Original")

axes[1].imshow(img, cmap='gray')                   # Middle: the stego image (looks the same!)
axes[1].set_title("Stego (hidden message)")

axes[2].imshow(changed, cmap='gray')               # Right: white dots = pixels we modified
axes[2].set_title("Changed pixels (white)")

for ax in axes:                                    # Hide the axis numbers to keep it clean
    ax.axis('off')

plt.tight_layout()
plt.savefig("comparison.png", dpi=150)             # Save this picture for your slides
plt.show()                                         # Show it (close the window to continue)

# ---------------------------------------------------------------
# STEP 6: DECODE - read the message back out of the saved file
# ---------------------------------------------------------------
# Reload from the PNG file so we prove the message survives saving
img = np.array(Image.open(stego_name))
img = img.flatten()                # Flatten to a 1D list of pixels again

msg = ""                           # This will hold the decoded text
idx = 0                            # Position of the current pixel we are reading

while msg[-2:] != "##":            # Keep reading until the last 2 characters are "##"
    if idx + 8 > img.shape[0]:     # If we run out of pixels, the message was never found
        print("Message not found")
        break
    # Take the LSB (last binary digit) of the next 8 pixels
    bits = [bin(i)[-1] for i in img[idx:idx + 8]]
    bits = "".join(bits)           # Join the 8 digits into one string like "01001000"
    msg += chr(int(bits, 2))       # Convert the 8 bits to a number, then to a character
    idx += 8                       # Move on to the next 8 pixels
else:
    # Runs only if the loop ended because "##" was found (not because of break)
    print("Decoded message:", msg[:-2])   # Print the message without the "##" marker
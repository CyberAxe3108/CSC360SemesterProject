from PIL import Image              # Pillow: opens and saves image files
import numpy as np                 # NumPy: lets us treat the image as an array of numbers

# ---------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------
stego_png = "My_hidden_message.png"     # The stego image your encoder already created
stego_jpg = "stego_copy.jpg"            # The JPG copy this script will create

# ---------------------------------------------------------------
# A decoder function (same logic as your decoder, wrapped so we can reuse it)
# ---------------------------------------------------------------
def decode(file_name):
    img = np.array(Image.open(file_name).convert('L')).flatten()   # Load as grayscale, 1D list of pixels
    msg = ""                                    # Will hold the decoded text
    idx = 0                                     # Position of the pixel we are reading
    while msg[-2:] != "##":                     # Keep going until the "##" end marker appears
        if idx + 8 > img.shape[0]:              # Ran out of pixels: no message found
            return None
        bits = "".join(str(int(p) & 1) for p in img[idx:idx + 8])   # Last bit of the next 8 pixels
        msg += chr(int(bits, 2))                # Turn the 8 bits into one character
        idx += 8                                # Move on to the next 8 pixels
    return msg[:-2]                             # Return the message without the "##" marker

# ---------------------------------------------------------------
# STEP 1: Decode the original stego PNG (should work)
# ---------------------------------------------------------------
print("PNG result:", decode(stego_png))

# ---------------------------------------------------------------
# STEP 2: Save a JPG copy of the stego image, then try to decode it (should fail)
# ---------------------------------------------------------------
Image.open(stego_png).save(stego_jpg)           # Convert the stego PNG to JPG (compression happens here)
jpg_msg = decode(stego_jpg)                     # Try to decode the JPG copy

if jpg_msg is None:
    print("JPG result: Message not found (JPEG compression destroyed it)")
else:
    # If a "##" happened to show up, the text will still be garbled; show the start of it
    print("JPG result (garbled):", repr(jpg_msg[:60]))
    
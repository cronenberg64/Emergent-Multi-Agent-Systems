import glob
import contextlib
from PIL import Image
import os

def make_gif(input_folder="outputs", output_file="outputs/simulation.gif", duration=200):
    # Find all PNG files
    fp_in = os.path.join(input_folder, "vis_tick_*.png")
    img_paths = sorted(glob.glob(fp_in))
    
    if not img_paths:
        print(f"No images found in {input_folder}")
        return

    print(f"Found {len(img_paths)} frames. Creating GIF...")

    # Load images
    images = [Image.open(f) for f in img_paths]
    
    # Save as GIF
    images[0].save(
        output_file,
        format='GIF',
        append_images=images[1:],
        save_all=True,
        duration=duration,
        loop=0
    )
    
    print(f"GIF saved to {output_file}")

if __name__ == "__main__":
    make_gif()

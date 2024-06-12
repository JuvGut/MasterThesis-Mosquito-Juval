import glob
from PIL import Image
import os


def make_gif(frame_folder):
    folder_name = os.path.basename(frame_folder)
    frames = [Image.open(image).resize((1000,650)) for image in sorted(glob.glob(f"{frame_folder}/*.JPG"))] # if no resizing is wished, remove the .resize() part
    frame_one = frames[0]
    gif_filename = f"{folder_name}.gif"
    frame_one.save(gif_filename, format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=0)
    print(f"Saved gif to {gif_filename}")
    

if __name__ == "__main__":
    make_gif("/home/juval.gutknecht/Mosquito_Detection/data/Fails/Median_ws7_to_gif")


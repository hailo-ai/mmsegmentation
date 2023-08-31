import os
from PIL import Image

labels_list_dict = {0:0, 1:1, 2:2, 3:3, 8:4, 10:5, 13:6, 11:7, 6:8, 14:9}  # -> [road, sidewalk, building, wall, vegetation, sky, car, person, traffic_light, truck]

def process_images(root_dir):
    for root, _, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith('labelTrainIds.png'):
                image_path = os.path.join(root, filename)
                try:
                    image = Image.open(image_path)
                    pixels = image.load()

                    width, height = image.size
                    for x in range(width):
                        for y in range(height):
                            pixel_value = pixels[x, y]
                            if pixel_value not in labels_list_dict:
                                pixels[x, y] = 255
                            else:
                                pixels[x,y] = labels_list_dict[pixels[x,y]]

                    image.save(image_path)
                    print(f"Processed: {image_path}")

                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    input_path = "/data/data/cityscapes10classes/gtFine/"
    process_images(input_path)

import numpy as np
import os
import cv2
import time
from PIL import Image

import matplotlib.pyplot as plt


def pixelate_image(image_path, pixelation_factor=1, output_path=None):
    """
    Pixelates an image based on the given pixelation factor and supports PNG with transparency.
    
    Args:
        image_path (str): Path to the input image.
        pixelation_factor (int): Determines the level of pixelation (higher value = more pixelated).
        output_path (str, optional): Path to save the output pixelated image. If None, the image is not saved.
        
    Returns:
        pixelated_image (numpy.ndarray): The pixelated image as a NumPy array.
    """
    # Read the image, including alpha channel if it exists (e.g., PNGs with transparency)
    if isinstance(image_path, str):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("Invalid image path provided!")
    else:
        image = pillow_to_cv(image_path)
    
    # Check if the image has an alpha channel
    has_alpha = (image.shape[2] == 4) if len(image.shape) == 3 else False
    
    # Get the dimensions of the image
    height, width = image.shape[:2]
    
    # Resize the image to a smaller size
    small_width = max(1, width // pixelation_factor)
    small_height = max(1, height // pixelation_factor)
    if has_alpha:
        # Separate the color and alpha channels
        color_channels = image[:, :, :3]
        alpha_channel = image[:, :, 3]
        
        # Process the color channels
        small_color = cv2.resize(color_channels, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
        pixelated_color = cv2.resize(small_color, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # Process the alpha channel
        small_alpha = cv2.resize(alpha_channel, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
        pixelated_alpha = cv2.resize(small_alpha, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # Combine the color and alpha channels
        pixelated_image = np.dstack((pixelated_color, pixelated_alpha))
    else:
        # Process the image as a regular RGB/BGR image
        small_image = cv2.resize(image, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
        pixelated_image = cv2.resize(small_image, (width, height), interpolation=cv2.INTER_NEAREST)
    
    # Save the pixelated image if output path is provided
    if output_path:
        cv2.imwrite(output_path, pixelated_image)

    pixelated_image = cv_to_pillow(pixelated_image)
    return pixelated_image


def plot_resized_image(image_path, new_width, new_height):
    """
    Resizes an image to the specified dimensions and plots it.
    
    Args:
        image_path (str): Path to the input image.
        new_width (int): The desired width for the resized image.
        new_height (int): The desired height for the resized image.
    """
    # Load the image
    if isinstance(image_path, str):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("Invalid image path provided!")
    else:
        image = pillow_to_cv(image_path)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Convert the image to RGB or RGBA for Matplotlib plotting
    if len(resized_image.shape) == 3:  # For RGB or RGBA images
        if resized_image.shape[-1] == 4:  # If image has alpha channel
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGRA2RGBA)
        else:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # Plot the resized image
    plt.figure(figsize=(4, 4))
    plt.imshow(resized_image)
    plt.axis('off')  # Turn off axes for better visualization
    plt.title(f"Resized Image ({new_width}x{new_height})")
    plt.show()


def blur_image(image_path, kernel_size=7, output_path=None):
    """
    Applies a blur effect to an image.
    
    Args:
        image_path (str): Path to the input image.
        kernel_size (int): Size of the kernel used for blurring (must be odd).
        output_path (str, optional): Path to save the blurred image. If None, the image is not saved.
        
    Returns:
        blurred_image (numpy.ndarray): The blurred image as a NumPy array.
    """
    # Load the image
    if isinstance(image_path, str):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("Invalid image path provided!")
    else:
        image = pillow_to_cv(image_path)
    
    # Apply the blur using GaussianBlur
    kernel_size = (kernel_size, kernel_size)
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    
    # Save the blurred image if output path is provided
    if output_path:
        cv2.imwrite(output_path, blurred_image)
    blurred_image = cv_to_pillow(blurred_image)
    return blurred_image


def pillow_to_cv(pillow_image):
    """
    Converts a Pillow image to an OpenCV image.
    
    Args:
        pillow_image (PIL.Image.Image): The Pillow image.
    
    Returns:
        cv_image (numpy.ndarray): The OpenCV image.
    """
    # Convert Pillow image to NumPy array
    image_array = np.array(pillow_image)
    
    # Convert RGB to BGR (OpenCV uses BGR format)
    if pillow_image.mode == "RGB":
        cv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    elif pillow_image.mode == "RGBA":  # Handle images with alpha channel
        cv_image = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGRA)
    else:
        cv_image = image_array  # Grayscale or other modes remain unchanged
    
    return cv_image


def cv_to_pillow(cv_image):
    """
    Converts an OpenCV image to a Pillow image.
    
    Args:
        cv_image (numpy.ndarray): The OpenCV image.
    
    Returns:
        pillow_image (PIL.Image.Image): The Pillow image.
    """
    # Convert BGR to RGB (Pillow uses RGB format)
    if len(cv_image.shape) == 3:  # Check if the image is colored
        if cv_image.shape[2] == 4:  # Handle images with alpha channel
            pillow_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGBA))
        else:
            pillow_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    else:  # Grayscale
        pillow_image = Image.fromarray(cv_image)
    
    return pillow_image


def add_background_to_png(png_img, background_color='white'):
    """
    Add a background color to a transparent PNG image.

    Args:
        png_img (PIL.Image.Image): A PIL image object with transparency.
        background_color (tuple): A tuple representing the background color (R, G, B).

    Returns:
        PIL.Image.Image: A new image with the specified background color.
    """

    background_color_mapping = {
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'sky': (135, 206, 235)

    }

    background_color = background_color_mapping.get(background_color, (255, 255, 255))

    # Ensure the input image has an alpha channel
    png_img = png_img.convert("RGBA")

    # Create a new image with the same size and the specified background color
    background = Image.new("RGBA", png_img.size, background_color + (255,))  # Add full opacity to the background color


    # Composite the PNG image over the background
    composed_image = Image.alpha_composite(background, png_img)

    # Convert to RGB (remove alpha) if you want a non-transparent final image
    return composed_image.convert("RGB")

unique_img_count = 0

def generate_square_dynamic_images_with_lighting(
    image_path, png_img, output_dir, num_images=5, square_size=512,
    padding=50, movement_range=50, brightness_range=(0.8, 1.2), contrast_range=(0.9, 1.1),
    temperature_range=(-50, 50), simulate_night_light=False, night_light_params=None, pixelate_img=False, blur_img=False
):
    """
    Generate square images with a moving foreground, lighting variations, and optional low lighting simulation.

    Args:
        image_path (str): Path to the image file (e.g., PNG, JPG) to use as the background.
        png_img (str): Path to the PNG foreground file.
        output_dir (str): Directory to save the generated images.
        num_images (int): Number of images to generate.
        square_size (int): Size of the output square image.
        padding (int): Padding size around the foreground to show the background.
        movement_range (int): Maximum pixel range to move the PNG foreground.
        brightness_range (tuple): Range of brightness multipliers (min, max).
        contrast_range (tuple): Range of contrast multipliers (min, max).
        temperature_range (tuple): Range of temperature shifts (min, max).
        simulate_night_light (bool): Whether to simulate nighttime lighting.
        night_light_params (dict): Parameters for nighttime lighting simulation.
    """
    # Default nighttime lighting parameters
    if night_light_params is None:
        night_light_params = {"brightness": 0.5, "contrast": 0.7, "temperature_shift": -50, "noise_level": 1}

    # Load the background image
    if isinstance(image_path, str):
        background = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if background is None:
            raise ValueError("Invalid background image path provided!")
    else:
        background = pillow_to_cv(image_path)
        
    if background is None:
        raise ValueError("Failed to load the background image.")

    # Resize the background if needed
    bg_height, bg_width = background.shape[:2]
    
    # Load the PNG foreground with transparency
    foreground = png_img
    fg_width, fg_height = foreground.size

    # Calculate the target square size with padding
    if padding > 0:
        padded_fg_width = fg_width + 2 * padding
        padded_fg_height = fg_height + 2 * padding
    else:
        padded_fg_width = fg_width
        padded_fg_height = fg_height

    if square_size < max(padded_fg_width, padded_fg_height):
        raise ValueError("Square size must be larger than the foreground size including padding.")

    for i in range(num_images):
        # Randomly move the foreground within the allowed movement range
        x_center = np.random.randint(padding + movement_range, bg_width - padded_fg_width - movement_range)
        y_center = np.random.randint(padding + movement_range, bg_height - padded_fg_height - movement_range)

        # Crop a square region around the foreground
        crop_x1 = max(0, x_center - (square_size // 2))
        crop_y1 = max(0, y_center - (square_size // 2))
        crop_x2 = min(bg_width, crop_x1 + square_size)
        crop_y2 = min(bg_height, crop_y1 + square_size)
        cropped_bg = background[crop_y1:crop_y2, crop_x1:crop_x2]

        # Ensure the cropped background matches the square size
        cropped_bg = cv2.resize(cropped_bg, (square_size, square_size), interpolation=cv2.INTER_AREA)

        # Convert the cropped background to RGBA for compositing
        cropped_bg = cv2.cvtColor(cropped_bg, cv2.COLOR_BGR2RGBA)

        # Composite the foreground onto the cropped background
        cropped_bg_pil = Image.fromarray(cropped_bg, mode="RGBA")
        fg_x = (square_size - fg_width) // 2
        fg_y = (square_size - fg_height) // 2
        composite = cropped_bg_pil.copy()
        composite.paste(foreground, (fg_x, fg_y), foreground)

        # Save the base image
        base_output_path = os.path.join(output_dir, f"output_{i+1}_{unique_img_count}.png")
        if pixelate_img:
            composite = pixelate_image(composite)
        if blur_img:
            composite = blur_image(composite)

        composite = resize_image(composite, resolution=100)
        composite.save(base_output_path)
        # print(f"Saved base image: {base_output_path}")

        # Apply lighting variations
        for j in range(1):  # Generate 3 lighting variations per image
            brightness = np.random.uniform(*brightness_range)
            contrast = np.random.uniform(*contrast_range)
            temperature_shift = np.random.randint(*temperature_range)

            # Adjust lighting on the cropped background
            lighting_adjusted_bg = adjust_lighting(np.array(composite), brightness, contrast, temperature_shift)

            # Save the lighting-variation image
            lighting_output_path = os.path.join(output_dir, f"output_{i+1}_lighting_{j+1}_{unique_img_count}.png")
            img_updated = Image.fromarray(lighting_adjusted_bg)
            img_updated = resize_image(img_updated, resolution=100)
            img_updated.save(lighting_output_path)
            # print(f"Saved lighting variation: {lighting_output_path}")

            # Optionally simulate nighttime lighting
            if simulate_night_light:
                composite_np = np.array(composite.convert("RGB"))  # Ensure RGB channels for simulation
                night_light_image = simulate_night_lighting(
                    composite_np,
                    brightness=night_light_params["brightness"],
                    contrast=night_light_params["contrast"],
                    temperature_shift=night_light_params["temperature_shift"],
                    noise_level=night_light_params["noise_level"]
                )
                night_light_output_path = os.path.join(output_dir, f"output_{i+1}_night_light_{unique_img_count}.png")
                img_updated = Image.fromarray(night_light_image)
                img_updated.save(night_light_output_path)
                # print(f"Saved night-light variation: {night_light_output_path}")


def adjust_lighting(image, brightness=1.0, contrast=1.0, temperature_shift=0):
    """
    Adjusts lighting conditions of an image by modifying brightness, contrast, and color temperature.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        brightness (float): Brightness multiplier (1.0 = no change).
        contrast (float): Contrast multiplier (1.0 = no change).
        temperature_shift (int): Color temperature shift (-100 to 100; negative for cooler, positive for warmer).

    Returns:
        numpy.ndarray: Image with adjusted lighting.
    """
    # Adjust brightness and contrast
    image = cv2.convertScaleAbs(image, alpha=contrast, beta=int(255 * (brightness - 1)))

    # Apply color temperature shift
    if temperature_shift != 0:
        temp_matrix = np.zeros_like(image, dtype=np.float32)
        if temperature_shift > 0:
            temp_matrix[:, :, 2] = temperature_shift  # Increase red channel
            temp_matrix[:, :, 0] = -temperature_shift  # Decrease blue channel
        else:
            temp_matrix[:, :, 2] = temperature_shift  # Decrease red channel
            temp_matrix[:, :, 0] = -temperature_shift  # Increase blue channel

        image = np.clip(image.astype(np.float32) + temp_matrix, 0, 255).astype(np.uint8)

    return image


def simulate_night_lighting(image, brightness=0.2, contrast=0.7, temperature_shift=-50, noise_level=15):
    """
    Simulates nighttime lighting conditions on an image.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        brightness (float): Brightness multiplier (<1.0 for dim light).
        contrast (float): Contrast multiplier (<1.0 for reduced dynamic range).
        temperature_shift (int): Color temperature shift (negative for cooler tones).
        noise_level (int): Level of noise to add (0 = no noise).

    Returns:
        numpy.ndarray: Image with simulated nighttime lighting.
    """
    # Reduce brightness and contrast
    image = cv2.convertScaleAbs(image, alpha=contrast, beta=int(255 * (brightness - 1)))

    # Adjust color temperature for cooler tones
    if temperature_shift != 0:
        temp_matrix = np.zeros_like(image, dtype=np.float32)
        if temperature_shift < 0:
            # Cooler tones: increase blue, reduce red
            temp_matrix[:, :, 2] = temperature_shift  # Decrease red
            temp_matrix[:, :, 0] = -temperature_shift  # Increase blue
        image = np.clip(image.astype(np.float32) + temp_matrix, 0, 255).astype(np.uint8)

    # Add noise to simulate sensor noise in low light
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)
    return image


def resize_image(img, resolution=100):
    img = pillow_to_cv(img)
    updated_img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
    updated_img = cv_to_pillow(updated_img)
    return updated_img


def apply_image_edits_and_save(img_file_name: str, sign_input_folder: str, output_dir: str, settings: dict):
    global unique_img_count

    img_path = os.path.join(sign_input_folder, img_file_name)
    img = Image.open(img_path).convert("RGBA")
    output_folder_name = settings["folder_name"]
    output_dir = os.path.join(output_dir, output_folder_name)
    os.makedirs(output_dir, exist_ok=True)
    resolution = settings["resolution"]
    background = settings.get("background")
    background_type = settings.get("background_type")
    blur = settings.get("blur", False)
    pixelate = settings.get("pixelate", False)
    night_light = settings.get("night_light", False)
    night_light_params={"brightness": 0.8, "contrast": 0.7, "temperature_shift": -50, "noise_level": 0.2}

    if background_type == 'image':
        # background = Image.open(background)
        generate_square_dynamic_images_with_lighting(
            background,
            img,
            output_dir,
            num_images=1,
            square_size=resolution,
            padding=0,
            movement_range=20,
            brightness_range=(0.8, 1.2),
            contrast_range=(0.9, 1.1),
            temperature_range=(-40, 40),
            simulate_night_light=night_light,
            night_light_params=night_light_params,
            pixelate_img=pixelate,
            blur_img=blur
        )
    else:
        # Add a color background
        updated_img = add_background_to_png(img, background)

        updated_img = pillow_to_cv(updated_img)
        updated_img = cv2.resize(updated_img, (resolution, resolution), interpolation=cv2.INTER_AREA)
        updated_img = cv_to_pillow(updated_img)

        if night_light:
            composite_np = np.array(updated_img.convert("RGB"))  # Ensure RGB channels for simulation
            updated_img = simulate_night_lighting(
                composite_np,
                brightness=night_light_params["brightness"],
                contrast=night_light_params["contrast"],
                temperature_shift=night_light_params["temperature_shift"],
                noise_level=night_light_params["noise_level"]
            )

        if pixelate:
            updated_img = pixelate_image(updated_img)
        if blur:
            updated_img = blur_image(updated_img)

        img_output_path = os.path.join(output_dir, img_file_name)
        updated_img.save(img_output_path)



def generate_dataset(signs_png_list: list, sign_input_folder: str, output_dir: str, settings_list: list):
    global unique_img_count
    unique_img_count = 0
    for img_file_name in signs_png_list:
        for settings in settings_list:
            apply_image_edits_and_save(img_file_name, sign_input_folder, output_dir, settings)
        unique_img_count += 1


if __name__ == "__main__":
    background_dir = r'C:\Users\ali\workspace\jamk_thesis\thesis_work\creating_new_dataset\backgrounds'
    base_dir = r'C:\Users\ali\workspace\jamk_thesis\thesis_work\creating_new_dataset\transparent_dataset'
    output_dir = r'D:\dataset\pure_python_genrated_dataset'
    # os.makedirs(output_dir, exist_ok=True)

    sign_classes_list = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]
    print("We will be genrating data for these svgs only -> ", sign_classes_list)


    background_image_files = class_transparent_pngs = [file for file in os.listdir(background_dir) if file.lower().endswith(('.png', 'jpg', 'jpeg'))]
    resolution = 100
    # Create the dictionary
    background_configs = [
        {
            'folder_name': f"{os.path.splitext(img)[0].split('.')[0]}_{resolution}px",
            'resolution': resolution,
            'background_type': 'image',
            'background': os.path.join(background_dir, img),
            'night_light': True
        }
        for img in background_image_files
    ]

    # Define settings for each folder
    settings_list = [
        {'folder_name': 'white_0_100px', 'resolution': 100, 'backgrounf_type':'color', 'background': 'white', 'pixelate': True, 'blur': True},
        {'folder_name': 'white_1_100px', 'resolution': 100, 'backgrounf_type':'color', 'background': 'white', 'pixelate': False, 'blur': True},
        {'folder_name': 'white_2_100px', 'resolution': 100, 'backgrounf_type':'color', 'background': 'white', 'pixelate': True, 'blur': False},
        {'folder_name': 'black_0_100px', 'resolution': 100, 'backgrounf_type':'color', 'background': 'black', 'pixelate': True, 'blur': True},
        {'folder_name': 'black_1_100px', 'resolution': 100, 'backgrounf_type':'color', 'background': 'black', 'pixelate': False, 'blur': True},
        {'folder_name': 'black_2_100px', 'resolution': 100, 'backgrounf_type':'color', 'background': 'black', 'pixelate': True, 'blur': False},
    ] + background_configs
    
    for sign_class in sign_classes_list:
        start_time = time.time()
        print(f'Generating data for class: {sign_class}')
        sign_folder = os.path.join(base_dir, sign_class, "transparent")
        output_directory = os.path.join(output_dir, sign_class)
        os.makedirs(output_directory, exist_ok=True)
        class_transparent_pngs = [file for file in os.listdir(sign_folder) if file.lower().endswith('.png')]
        print("Raw Transparent Images Count: ", len(class_transparent_pngs))
        # print(class_transparent_pngs)

        generate_dataset(class_transparent_pngs, sign_folder, output_directory, settings_list)
        end_time = time.time()
        execution_time_minutes = (end_time - start_time) / 60
        print(f"Function executed in {execution_time_minutes:.2f} minutes")

        # process only a single class atm
        # break
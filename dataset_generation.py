"""Script to generate data for the font classification task.

Sample run:
```
python -m venv my-venv
source my-venv/bin/activate
pip install -r requirements.txt

python dataset_generation.py 100
```
"""

import colorsys
import cv2
import numpy as np
import os
import sys
import random
import traceback
import wikipedia
import easyocr
from PIL import Image, ImageDraw, ImageFont, ImageColor
from argparse import ArgumentParser
from loguru import logger
from pathlib import Path
from sklearn.cluster import KMeans
from tqdm import tqdm
from typing import Tuple, Optional
from collections import Counter


Image.MAX_IMAGE_PIXELS = None
logger.remove()
logger.add(sys.stdout, level="INFO")


def get_common_colors(
    img, colors=32, max_points=-1, N=3, colorspace="rgb", select_color="mean"
):
    max_points = int(max_points)
    img = np.array(img, dtype=np.uint8)
    h, w = img.shape[0], img.shape[1]

    img_orig_flat = img.reshape(h * w, 3)

    if colorspace == "bgr":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif colorspace == "hls":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif colorspace == "hsv":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif colorspace == "lab":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    elif colorspace == "rgb":
        img = img.copy()
    else:
        raise Exception("Unknown colorspace")

    img_flat = img.copy().reshape(h * w, 3)

    if max_points > 0 and max_points < img_flat.shape[0]:
        idx = np.random.choice(np.arange(img_flat.shape[0]), max_points, replace=False)
        kmeans = KMeans(n_clusters=colors, n_init="auto", random_state=0).fit(
            img_flat[idx]
        )
        labels = kmeans.predict(img_flat)
    else:
        kmeans = KMeans(n_clusters=colors, n_init="auto", random_state=0).fit(img_flat)
        labels = kmeans.labels_

    unique_labels, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    most_common_labels = unique_labels[sorted_indices[:N]]

    # loops for cluster center
    colors = []
    for ci in np.unique(most_common_labels):
        if select_color == "mean":
            colors.append(img_orig_flat[labels == ci, :].mean(axis=0))
        elif select_color == "median":
            colors.append(np.median(img_orig_flat[labels == ci, :], axis=0))
        else:
            raise Exception("Unknown select_color")
    return [c.astype(np.uint8) for c in colors]


def get_text_thresholded_images(image: Image):
    """
    Processes an image to extract thresholded text regions using provided bounding boxes.

    Args:
        image: A PIL Image instance.
        bounding_boxes: A list of bounding boxes, where each box is a tuple of four points (tl, tr, br, bl).

    Returns:
        A binary mask of input image in PIL format.
    """
    if image is None:
        return None
    cropped_img_ = image
    # Convert PIL Image to numpy array
    cropped_img = np.array(cropped_img_)
    # Convert to grayscale
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    # Otsu's thresholding
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_inv = cv2.bitwise_not(thresh)
    stacked = np.dstack((thresh_inv, thresh_inv, thresh_inv))
    img_th = cv2.bitwise_and(cropped_img, stacked)
    img_th[img_th > 0] = 255
    img_th = Image.fromarray(img_th).convert("L")
    # img_th.save('data/modified_text_mask.png')
    return cropped_img_, img_th


def get_hex_codes(crop, top_k=3):
    # Create a mask for the crop
    mask = Image.new("L", crop.size, 255)  # Start with all white (include all pixels)
    draw = ImageDraw.Draw(mask)

    hex_counter = Counter()
    total_pixels = 0
    for y in range(crop.height):
        for x in range(crop.width):
            if mask.getpixel((x, y)) == 255:  # Check if pixel is not in any bbox
                r, g, b = crop.getpixel((x, y))
                hex_code = f"#{r:02x}{g:02x}{b:02x}"
                hex_counter[hex_code] += 1
                total_pixels += 1

    # Get the top-k most common hex codes
    top_k_hex_codes = hex_counter.most_common(top_k)
    # Separate lists for hex codes and their percentages
    hex_codes = [hex_code for hex_code, _ in top_k_hex_codes]
    percentages = [(count / total_pixels) * 100 for _, count in top_k_hex_codes]
    return hex_codes, percentages


def create_color_mask(image, hex_color, tolerance=10):
    """
    Creates a binary mask of an image based on a hex color.

    Args:
        image (PIL.Image.Image): The input PIL image.
        hex_color (str): The hex color code (e.g., '#FF0000').
        tolerance (int): The allowed deviation in RGB values (default: 10).

    Returns:
        PIL.Image.Image: A binary (grayscale) mask image.
    """
    # 1. Convert Hex to RGB
    target_rgb = ImageColor.getrgb(hex_color)
    # 2. Convert PIL Image to Numpy Array for efficiency
    image_array = np.array(image)
    # 3. Create a mask for each color channel
    r_mask = np.abs(image_array[:, :, 0] - target_rgb[0]) <= tolerance
    g_mask = np.abs(image_array[:, :, 1] - target_rgb[1]) <= tolerance
    b_mask = np.abs(image_array[:, :, 2] - target_rgb[2]) <= tolerance
    # 4. Combine the color channels to create the final mask
    mask_array = r_mask & g_mask & b_mask
    # 5. Convert the boolean mask to an integer mask
    mask_array = mask_array.astype(np.uint8) * 255
    # 6. Create and return PIL image from mask
    mask_image = Image.fromarray(mask_array, mode="L")

    return mask_image


def convert_points_to_bounding_box(points):
    """
    Convert bounding box points to [x, y, width, height] format.

    :param points: List of tuples representing the four points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
    :return: Tuple representing the bounding box in [x, y, width, height] format.
    """
    # Unpack points
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    x4, y4 = points[3]

    # Calculate min and max for x and y
    x_min = min(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    x_max = max(x1, x2, x3, x4)
    y_max = max(y1, y2, y3, y4)

    # Calculate width and height
    width = x_max - x_min
    height = y_max - y_min

    return (x_min, y_min, width, height)


def convert_bbox_to_edges(bbox):
    """
    Convert a bounding box from [x, y, width, height] format to [left, upper, right, lower] format.

    :param bbox: Tuple representing the bounding box in [x, y, width, height] format.
    :return: Tuple representing the bounding box in [left, upper, right, lower] format.
    """
    # Unpack bounding box
    x, y, width, height = bbox

    # Calculate edges
    left = x
    upper = y
    right = x + width
    lower = y + height

    return (left, upper, right, lower)


def load_image(image_path):
    return Image.open(image_path).convert("RGB")


def rgb_to_hls(rgb):
    return colorsys.rgb_to_hls(*[x / 255.0 for x in rgb])


def hls_to_rgb(hls):
    return tuple([int(x * 255) for x in colorsys.hls_to_rgb(*hls)])


def triadic_color_hls(rgb):
    h, l, s = rgb_to_hls(rgb)
    # s = max(0.7, s)
    # FIXME: dirty hack for inverse black to white and back
    # TODO: make some threshold that will define "dark" and "white" colors
    # and inverse brightness for them
    # v, s = s, v
    l = 1.0 - l
    h_triadic1 = (h + 1 / 3) % 1
    h_triadic2 = (h + 2 / 3) % 1
    return hls_to_rgb((h_triadic1, l, s)), hls_to_rgb((h_triadic2, l, s))


def opposite_color_hls(rgb):
    h, l, s = rgb_to_hls(rgb)
    l = 1.0 - l
    h_opposite = (h + 1 / 2) % 1
    return hls_to_rgb((h_opposite, l, s))


def get_random_page_content() -> str:
    page_title = wikipedia.random(1)
    try:
        page_content = wikipedia.page(page_title).summary
    except (wikipedia.DisambiguationError, wikipedia.PageError):
        return get_random_page_content()
    return page_content


def split_string(string, min_length, max_length):
    substrings = []
    start = 0
    length = len(string)

    for i in range(length // max_length):
        substr = string[start : start + max_length]
        start += max_length
        substrings.append(substr)

    if length - start > min_length:
        substrings.append(string[start:])

    return substrings


def create_strings_from_wikipedia(minimum_length, count, lang, max_length=-1):
    """
    Create all string by randomly picking Wikipedia articles and taking sentences from them.
    """
    wikipedia.set_lang(lang)
    sentences = []

    while len(sentences) < count:
        page_content = get_random_page_content()
        processed_content = page_content.replace("\n", " ").split(". ")
        sentence_candidates = [
            s.strip() for s in processed_content if len(s.split()) > minimum_length
        ]

        for candidate in sentence_candidates:
            strings = split_string(candidate, minimum_length, max_length)
            if len(strings) > 0:
                sentences.extend(strings)
        # sentences.extend(sentence_candidates)

    return sentences[0:count]


def create_strings_from_textfile(textfile_path, min_length, max_length, count=-1):
    with open(textfile_path, "r") as f:
        lines = f.readlines()

    sentences = []
    for line in lines:
        if len(line) > min_length:
            strings = split_string(line, min_length, max_length)
            sentences.extend(strings)

        if count > 0 and len(sentences) >= count:
            break

    return sentences[0:count]


class ResizeWithPad:

    def __init__(
        self, new_shape: Tuple[int, int], padding_color: Tuple[int] = (255, 255, 255)
    ) -> None:
        self.new_shape = new_shape
        self.padding_color = padding_color

    def __call__(self, image: np.array, padding_color=None, **kwargs) -> np.array:
        """Maintains aspect ratio and resizes with padding.
        Params:
            image: Image to be resized.
            new_shape: Expected (width, height) of new image.
            padding_color: Tuple in BGR of padding color
        Returns:
            image: Resized image with padding
        """
        if padding_color is None:
            padding_color = self.padding_color
        original_shape = (image.shape[1], image.shape[0])
        ratio = float(max(self.new_shape)) / max(original_shape)
        new_size = tuple([int(x * ratio) for x in original_shape])
        image = cv2.resize(image, new_size)
        delta_w = self.new_shape[0] - new_size[0]
        delta_h = self.new_shape[1] - new_size[1]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color
        )
        return image


class CutMax:
    """Cuts the image to the maximum size"""

    def __init__(self, max_size: int = 1024) -> None:
        self.max_size = max_size

    def __call__(self, image: np.array, **kwargs) -> np.array:
        """Cuts the image to the maximum size"""
        if image.shape[0] > self.max_size:
            image = image[: self.max_size, :, :]
        if image.shape[1] > self.max_size:
            image = image[:, : self.max_size, :]
        return image


class FontGenerator:
    """
    Generate images with text and background
    1. Init background images cache
    2. Load fonts
    3. Load backgrounds images list
    4. Generate sample image
        1. Generate text from wikipedia
        2. Generate background image
            1. Get random background image from cache or load new one
            2. Random crop with random color padding
            3. Convert to grayscale if needed
        3. Or generate only color background
        4. Select random font and font size
        5. Adjust font color to contrast with background
        6. Draw text on background
    """

    def __init__(
        self,
        size=(256, 256),
        min_length=5,
        max_length=30,
        backgrounds_path="backgrounds/",
        fonts_path="fonts/",
        background_ratio=0.8,
        gray_color=False,
        background_type=1,
        background_cache_size=1000,
        source="wikipedia",
        textfile="text.txt",
        debug=False,
    ):
        """
        Generate images with text and background.

        Parameters:
        - size: Tuple[int, int] - The size of the generated images.
        - min_length: int - The minimum length of the generated text.
        - max_length: int - The maximum length of the generated text.
        - backgrounds_path: str - The path to the directory containing background images.
        - fonts_path: str - The path to the directory containing font files.
        - background_ratio: float - The ratio of background images to be used.
        - gray_color: bool - Whether to convert the background images to grayscale.
        - background_type: int - The type of background to generate.
        - background_cache_size: int - The size of the background images cache.
        - source: str - The source of the text to generate.
        - textfile_path: str - The path to the text file containing the text to generate.

        Attributes:
        - backgrounds: List[str] - The list of background image file paths.
        - fonts: Dict[str, str] - The dictionary of font names and their corresponding file paths.
        - fonts_cache: Dict[str, ImageFont] - The cache of loaded font objects.
        - backgrounds_cache: Dict[str, Image] - The cache of loaded background images.
        - text_cache: List[str] - The cache of generated text strings.
        - resizer: ResizeWithPad - The image resizer object.

        Methods:
        - load_backgrounds(): Loads the background images from the specified directory.
        - load_fonts(): Loads the font files from the specified directory.
        - get_random_font(): Returns a random font object from the loaded fonts.
        - generate_image(): Generates an image with text and background.
        - get_font_color(): Calculates the font color to contrast with the background.
        - generate_text(): Generates random text from the specified source.
        - random_crop_with_padding(): Performs a random crop of the image with padding.
        - get_random_background(): Returns a random background image from the cache or loads a new one.

        Example usage:
        generator = FontGenerator(size=(256, 256), min_length=5, max_length=30, backgrounds_path='backgrounds/', fonts_path='fonts/', background_ratio=0.8, gray_color=False, background_type=1, background_cache_size=1000, source='wikipedia', textfile_path='text.txt')
        image = generator.generate_image(text='Hello World', font_size=32, font_color=(0, 0, 0), position='center', padding=10, background_image=True)
        image.show()
        """
        self.size = size
        self.min_length = min_length
        self.max_length = max_length
        self.backgrounds_path = backgrounds_path
        self.fonts_path = fonts_path
        self.background_ratio = background_ratio
        self.background_type = background_type
        self.background_cache_size = background_cache_size
        self.gray_color = gray_color
        self.source = source
        self.textfile_path = textfile

        self.backgrounds = []
        self.fonts = {}
        self.fonts_cache = {}
        self.blacklisted_fonts = []

        self.debug = debug

        # Init background images cache
        self.load_backgrounds()
        if not self.backgrounds:
            raise FileNotFoundError(
                f"No background images found under {self.backgrounds_path}"
            )

        self.load_blacklisted_fonts("blacklisted_fonts.txt")

        self.load_fonts(self.fonts_path)
        if not self.fonts:
            raise FileNotFoundError(f"No fonts found under {self.fonts_path}")

        self.resizer = ResizeWithPad(self.size, (255, 255, 255))

    def load_backgrounds(self):
        self.backgrounds = []
        for file in os.listdir(self.backgrounds_path):
            self.backgrounds.append(os.path.join(self.backgrounds_path, file))

        # Create a cache for background images
        self.backgrounds_cache = {}
        self.text_cache = []

    def load_blacklisted_fonts(self, path: str):
        # load blacklisted fonts
        with open(path, "r") as f:
            for line in f:
                self.blacklisted_fonts.append(line.strip())

    def load_fonts(self, path: str):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".ttf") or file.endswith(".otf"):
                    if file in self.blacklisted_fonts:
                        continue
                    fontname = os.path.splitext(file)[0]
                    print(fontname, os.path.join(root, file))
                    self.fonts[fontname] = os.path.join(root, file)

    def get_random_font(self):
        font_name = random.choice(list(self.fonts.keys()))
        return self.get_font(font_name), font_name
    
    def get_font(self, font_name):
        font_path = self.fonts[font_name]
        if font_name in self.fonts_cache:
            font = self.fonts_cache[font_name]
        else:
            font = ImageFont.truetype(font_path, size=32)
            self.fonts_cache[font_name] = font
        return font

    def generate_image(
        self,
        text,
        font_name,
        font_size: int = 32,
        font_color: Optional[Tuple[int, int, int]] = (0, 0, 0),
        position: str = "center",  # center, random
        padding=10,
        background_image: bool = False,
        background_color: Optional[Tuple[int, int, int]] = None,
    ) -> Image:
        logger.debug(f"Generating image with text: {text}")
        # Generate image
        if background_image:
            image = self.get_random_background()#.resize(self.size)
            logger.debug(f"Background image with size: {image.size}")
            colors = get_common_colors(np.array(image), colors=12, max_points=1e5, N=1)
            logger.debug(f"Common colors: {colors}")
            main_color = colors[0]
            if font_color is None:
                candidates = [
                    opposite_color_hls(main_color),
                    *triadic_color_hls(main_color),
                ]
                font_color = random.choice(candidates)
            logger.debug(f"Font color: {font_color}")
        elif background_color is not None:
            image = Image.new("RGB", self.size, background_color)
            logger.debug(f"Background color: {background_color}")
        else:
            rand_color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            # Generate random color background
            image = Image.new("RGB", self.size, rand_color)
            logger.debug(f"Random color background: {rand_color}")

        draw = ImageDraw.Draw(image)

        # Select random font and font size
        font = self.get_font(font_name)
        font = font.font_variant(size=font_size)

        if font_color is None:
            # Adjust font color to contrast with background
            font_color = self.get_font_color(image)

        # Calculate position
        bbox = font.getbbox(text)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if position == "center":
            x = (self.size[0] - text_w) / 2
            y = (self.size[1] - text_h) / 2
        elif position == "random":
            # apply padding
            x = random.randint(padding, max(padding, self.size[0] - text_w - padding))
            y = random.randint(padding, max(padding, self.size[1] - text_h - padding))
        else:
            raise ValueError(f"Unknown position: {position}")

        # Draw text
        draw.text((x, y), text, fill=font_color, font=font)

        return image, font_name, font_color

    def get_font_color(self, image):
        """
        Calculate font color to contrast with background
        """
        pass

    def generate_text(self):
        """
        Generate random text from wikipedia
        """
        if len(self.text_cache) == 0:
            if self.source == "wikipedia":
                # Load text from wikipedia
                self.text_cache.extend(
                    create_strings_from_wikipedia(
                        self.min_length, 1000, "en", self.max_length
                    )
                )
            elif self.source == "textfile":
                # Load text from text file
                with open(self.textfile_path, "r") as f:
                    self.text_cache.extend(f.readlines())
                    if not self.text_cache:
                        raise ValueError(f"Text file {self.textfile_path} is empty.")

        return self.text_cache.pop()

    def random_crop_with_padding(self, image, pad_color=(255, 255, 255)):
        """
        Random crop with padding
        """
        assert image.size[0] >= self.size[0] and image.size[1] >= self.size[1]
        x = random.randint(0, image.size[0] - self.size[0])
        y = random.randint(0, image.size[1] - self.size[1])

        image = image.crop((x, y, x + self.size[0], y + self.size[1]))

        image = self.resizer(np.array(image), padding_color=pad_color)
        image = Image.fromarray(image)

        return image

    def get_random_background(self, pad_color=(255, 255, 255)):
        """
        Load background image from background cache
        """
        # Get random background image
        random_background = random.choice(self.backgrounds)

        # Load image from cache
        if random_background in self.backgrounds_cache:
            background = self.backgrounds_cache[random_background]
        else:
            background = Image.open(random_background)
            background = background.convert("RGB")
            self.backgrounds_cache[random_background] = background
        # background = background.resize(self.size)
        # Random crop with padding
        # background = self.random_crop_with_padding(background, pad_color)

        # Apply color
        if self.gray_color:
            background = background.convert("L")

        return background


def get_n_max_logits(arr: np.array, n: int):
    """
    Get n max logits from array, return indices and values
    """
    indices = np.argpartition(arr, -n)[-n:]
    indices = indices[np.argsort(-arr[indices])]
    values = arr[indices]
    return indices, values


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--N", default=4000, type=int, help="Number of generated examples")
    parser.add_argument(
        "--min_length", type=int, default=7, help="Minimum length of generated text"
    )
    parser.add_argument(
        "--max_length", type=int, default=30, help="Maximum length of generated text"
    )
    parser.add_argument("--batch_size", type=int, default=200, help="Batch size")
    parser.add_argument(
        "--max_fonts", type=int, default=3000, help="Maximum number of fonts to use"
    )
    parser.add_argument(
        "--output", type=str, default="output9", help="Output folder"
    )
    parser.add_argument(
        "--backgrounds",
        type=str,
        default="val2017/",
        help="Path for background images, supports JPG, PNG",
    )
    parser.add_argument(
        "--fonts",
        type=str,
        default="training_fonts",
        help="Path to folder with fonts in TTF or OTF format",
    )
    parser.add_argument(
        "--font_size_min", type=int, default=36, help="Minimum font size"
    )
    parser.add_argument(
        "--font_size_max", type=int, default=102, help="Maximum font size"
    )
    parser.add_argument(
        "--background_ratio",
        type=float,
        default=0.0,
        help="Ratio between results with background image and white color",
    )
    parser.add_argument(
        "--contrast_color_ratio",
        type=float,
        default=0.5,
        help="Ratio between results with contrast color and black color",
    )
    parser.add_argument(
        "--text_source",
        type=str,
        default="wikipedia",
        help="Text source: wikipedia, textfile",
    )
    parser.add_argument(
        "--textfile",
        type=str,
        default="sample_data/textfile.txt",
        help="Path to text file with sentences dataset",
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    args = parser.parse_args()
    return args


def main(args):
    # Create output folder
    os.makedirs(args.output, exist_ok=True)

    # Enable debug logger level if debug mode is on
    if args.debug:
        logger.add(sys.stdout, level="DEBUG")

    # Init font generator
    font_generator = FontGenerator(
        size=(2048, 2048),
        min_length=args.min_length,
        max_length=args.max_length,
        backgrounds_path=args.backgrounds,
        fonts_path=args.fonts,
        background_ratio=args.background_ratio,
        source=args.text_source,
        textfile=args.textfile,
    )

    ocr_reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory


    font_names = list(font_generator.fonts.keys())
    for font_name in tqdm(font_names):
        for i in tqdm(range(args.N//len(font_names)), desc=f"Generating images for {font_name}"):
            try:
                text = font_generator.generate_text()

                if np.random.rand() < args.contrast_color_ratio:
                    font_color = None
                else:
                    font_color = (0, 0, 0)

                font_size = random.randint(args.font_size_min, args.font_size_max)

                if random.random() < args.background_ratio:
                    background_image = True
                    background_color = None
                else:
                    background_image = False
                    background_color = tuple(np.random.randint(0, 256, size=3))
                    luminance = 0.2126 * background_color[0] + 0.7152 * background_color[1] + 0.0722 * background_color[2]
                    threshold = 128
                    font_color = (255, 255, 255) if luminance < threshold else (0, 0, 0)
                    print(background_color, font_color)

                # Generate image
                image, font_name, font_color = font_generator.generate_image(
                    text,
                    font_name,
                    position="random",
                    background_image=background_image,
                    font_size=font_size,
                    padding=10,
                    font_color=font_color,
                    background_color=background_color,
                )

                (Path(args.output) / font_name).mkdir(exist_ok=True)
                # use EasyOCR to crop text region
                text_blocks = ocr_reader.readtext(np.array(image), paragraph=True)
                for j, results in enumerate(text_blocks):
                    bbox, text = results[0], results[1]
                    if len(text) < 1:
                        continue
                    x_min = min(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0])
                    y_min = min(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
                    x_max = max(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0])
                    y_max = max(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
                    cropped_image = image.crop((x_min, y_min, x_max, y_max))
                    hex_codes, percentages = get_hex_codes(cropped_image, top_k=1)
                    cropped_text, text_mask = get_text_thresholded_images(
                        cropped_image)
                    msked_img = create_color_mask(cropped_image, hex_codes[0])
                    if np.random.rand() > 0.8:
                        msked_img = np.array(msked_img, dtype=np.uint8)
                        msked_img = 255 - msked_img
                        msked_img = Image.fromarray(msked_img)
                    msked_img.save(os.path.join(args.output, font_name, f"{i}_{j + 1}.jpg"))

                # Save image
                # image.save(os.path.join(args.output, font_name, f"{i}.jpg"))
            except Exception as e:
                print(f"Error while generating image {i}: {e}")
                traceback.print_exc()
                continue


if __name__ == "__main__":
    args = parse_args()
    main(args)

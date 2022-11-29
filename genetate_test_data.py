"""
Make the actual jastrow training data and split them into train and test data.
"""
import numpy as np
from PIL import Image
from PIL.Image import new
from pathlib import Path
from PIL.ImageDraw import Draw
import os
import multiprocessing as mp
from tqdm import tqdm
from random import randrange
import json
from itertools import product
import shutil
from sklearn import preprocessing

# import global variables
settings = json.load(open("settings.json", "r"))
background_size = settings["target_image_width"] * settings["initial_expansion"]
canvas_size_jastrow = settings["target_image_width"] * settings["initial_expansion"] * 2  # double size seems to be ok


# print with color
def printc(text: str, color: str = "green") -> None:
    """
    Print with colors
    params:
        text: the text to print
        color: the color to print the text in
    returns:
        None
    """
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "black": "\033[98m",
    }
    print(colors[color] + text + "\033[0m")


def update_configs(configs_variable: dict) -> dict:
    """
    Initial update of the configs_variable. called once
    at the beginning of the program.
    params:
        configs_variable: the configs_variable to update
    """
    configs_variable["radius_outer"] *= canvas_size_jastrow / 2
    configs_variable["radius_outer"] += canvas_size_jastrow / 2

    configs_variable["radius_inner"] = configs_variable["radius_outer"] - (
        (configs_variable["radius_outer"] - (canvas_size_jastrow / 2)) * configs_variable["thickness"]
    )

    return configs_variable


def generate_init_shape(configs_variable: dict) -> Image:
    """
    Generate an uncropped/un-rotated jastrow shape using the
    configs_variable.
    params:
        configs_variable: the configs_variable to use
    """
    # make empty canvas
    canvas = new("RGBA", tuple([canvas_size_jastrow] * 2), color=(0, 0, 0, 0))

    # draw the large circle
    Draw(canvas).pieslice(
        (
            *[(canvas_size_jastrow - configs_variable["radius_outer"])] * 2,
            *[configs_variable["radius_outer"]] * 2,
        ),
        start=-90 - configs_variable["angle"],
        end=-90 + configs_variable["angle"],
        fill=(255, 255, 255, 255),
    )

    # small circle
    Draw(canvas).pieslice(
        (
            *[(canvas_size_jastrow - configs_variable["radius_inner"])] * 2,
            *[configs_variable["radius_inner"]] * 2,
        ),
        start=0,
        end=360,
        fill=(0, 0, 0, 0),
    )

    return canvas


def rotate_and_crop(shape: Image, degree: int = 44) -> Image:
    """Rotate and crop the canvas to the smallest possible size.
    Params:
        canvas: the canvas to rotate and crop
        degree: the degree to rotate the canvas
    Returns:
        shape_1: the cropped canvas
        shape_2: the rotated and cropped canvas
    """

    shape_1 = shape.crop(shape.getbbox())

    rotation_1 = np.random.randint(0, 360)
    rotation_2 = np.random.randint(0, 360)

    shape_1 = shape_1.rotate(rotation_1, expand=True)
    shape_2 = shape.rotate(rotation_2, expand=True)

    shape_2 = shape_2.crop(shape_2.getbbox())

    # make shape_1 blue where it is white
    shape_1 = np.array(shape_1)
    shape_1[shape_1[:, :, 3] == 255] = [0, 0, 255, 255]
    shape_1 = Image.fromarray(shape_1)

    # make shape_2 red where it is white
    shape_2 = np.array(shape_2)
    shape_2[shape_2[:, :, 3] == 255] = [255, 0, 0, 255]
    shape_2 = Image.fromarray(shape_2)

    return shape_1, shape_2, rotation_1, rotation_2


def get_area_size(canvas: Image, color: str, tolerance: int = 10) -> int:
    """
    Get the area size of the given color.
        params:
            canvas: the canvas to get the area size from
            color: the color to get the area size from
        returns:
            area_size: the area size of the given color
    """
    canvas_array = np.array(canvas)

    match color:
        # match color with tolerance of 10
        case "red":
            return np.sum(np.all(np.abs(canvas_array - [255, 0, 0, 255]) < tolerance, axis=-1))
        case "blue":
            return np.sum(np.all(np.abs(canvas_array - [0, 0, 255, 255]) < tolerance, axis=-1))
        case _:
            raise ValueError("Color not supported.")


# make a background canvas
def make_jastrow(shape_1: Image, shape_2: Image) -> Image:
    """
    Make a jastrow canvas with the given shapes.
    return:
        background: the jastrow canvas
        paste_coordinates_shape_1: the paste coordinates of shape_1
        paste_coordinates_shape_2: the paste coordinates of shape_2
    """
    # make a background canvas
    background = new("RGBA", tuple([background_size] * 2), color=(0, 0, 0, 255))

    # place image 1 which is blue
    # at a random position on the background
    paste_coordinates_shape_1 = tuple(np.random.randint(0, background_size - i) for i in shape_1.size)
    background.paste(shape_1, paste_coordinates_shape_1, shape_1)

    # place image 2 which is red
    # at a random position on the background
    paste_coordinates_shape_2 = tuple(np.random.randint(0, background_size - i) for i in shape_2.size)
    background.paste(shape_2, paste_coordinates_shape_2, shape_2)

    return background, paste_coordinates_shape_1, paste_coordinates_shape_2




def map_between(value, range):
    """
    Map a value between a range if the value is not outside
    the range. Return the max or min if the value is outside
    the range.
    params:
        value: the value to map
        range: the range to map the value between
    returns:
        value: the mapped value
    """
    if min(range) < value < max(range):
        return (value - range[0]) / (range[1] - range[0])
    else:
        return 1 if value > max(range) else 0


"""The loop below is for generating the training data"""

# settings ----------------------------------------------
# get 100 values with equal interval between 0.1 and 0.2
variance = 0.5  # float between 0 and 1, larger the more variance
radius_outer = [1]
thicknesses = np.linspace(0.1, 0.2, 100)
angles = np.linspace(5, 45, 100)
image_per_config = 100
threshold_jastrow_coefficient = 25  # shapes under this threshold will be moved to another folder
train_test_ratio = 0.8  # float between 0 and 1, larger the more training data
variation_name = "standard"

# make combs
combs = list(product(radius_outer, thicknesses, angles))
combs = [{"radius_outer": i[0], "thickness": i[1], "angle": i[2]} for i in combs]
combs = [update_configs(i) for i in combs]

# set up progress bar
progress_bar = tqdm(total=len(combs) * image_per_config, colour="green")



def main(config: dict) -> None:
    """The main function for generating the training data
    a while loop is used to ensure that the shapes are not
    too close to each other under the condition that the
    number of shapes in the data folder is less than the
    number of images to be generated.

    params:
        num: int = the number of images to be generated
        for each config
    """

    shape = generate_init_shape(config)
    while len(os.listdir(Path("data", variation_name, "images"))) < len(combs) * image_per_config:
        try:
            # rotate and crop
            shape_1, shape_2, rotation_1, rotation_2 = rotate_and_crop(shape=shape)

            # resize the shapes
            sizes = np.random.uniform(1 - variance, 1, 2)
            shape_1_resized = shape_1.resize(tuple(round(i * sizes[0]) for i in shape_1.size), Image.Resampling.LANCZOS)
            shape_2_resized = shape_2.resize(tuple(round(i * sizes[1]) for i in shape_2.size), Image.Resampling.LANCZOS)

            # make jastrow shape
            background, paste_coordinates_shape_1, paste_coordinates_shape_2 = make_jastrow(shape_1_resized, shape_2_resized)

            # knowing the size of shape, respective rotation degrees, and paste coordinates
            # we can calculate the distance between the two shapes's contours
            distance = np.sqrt(
                (paste_coordinates_shape_1[0] - paste_coordinates_shape_2[0]) ** 2
                + (paste_coordinates_shape_1[1] - paste_coordinates_shape_2[1]) ** 2
            )

            # count the number of pixels in the shape / on the canvas
            size_shape_1 = get_area_size(canvas=shape_1_resized, color="blue")
            size_shape_2 = get_area_size(canvas=shape_2_resized, color="red")
            size_on_canvas_1 = get_area_size(canvas=background, color="blue")
            size_on_canvas_2 = get_area_size(canvas=background, color="red")

            if (size_shape_1, size_shape_2) == (size_on_canvas_1, size_on_canvas_2):

                # compute jastrow coefficient
                similarity = compute_similarity((rotation_1, rotation_2))
                distance = map_between(distance, (0, np.sqrt(2 * background_size**2)))
                jastrow_coefficient = (distance + similarity) * 100

                # set the image name
                name = "{}-{}-{}-{}-{}-{}.png".format(
                    jastrow_coefficient,
                    size_shape_1,
                    size_shape_2,
                    paste_coordinates_shape_1,
                    paste_coordinates_shape_2,
                    str(randrange(10000)).zfill(4),  # unique id
                )

                # save the image
                background = background.resize((background_size // settings["initial_expansion"],) * 2)
                background.save(Path("data", variation_name, "images", name))

                # update progress bar to the current amount of images in data folder
                current_num = len(os.listdir(Path("data", variation_name, "images")))
                progress_bar.update(current_num - progress_bar.n)

        except ValueError:
            pass


def set_label(folder: Path) -> None:
    """
    This function is used to set the label for the generated data
    params:
        folder: Path = the folder that contains the generated data
    note: the names of the images are in the format of
        name = "{}-{}-{}-{}-{}-{}.png".format(
            jastrow_coefficient,  # 0
            size_shape_1,  # 1
            size_shape_2,  # 2
            paste_coordinates_shape_1,  # 3
            paste_coordinates_shape_2,  # 4
            str(randrange(10000)).zfill(4),  # 5; unique id
        )
    """
    images = [i for i in os.listdir(folder) if i.endswith(".png")]
    labels = [i.split("-") for i in images]

    """
    Two ways to do this:
    1. calculate the pixel differences and map it to a value between 0 and 1
    where 0 is the theoretical minimum difference and 1 is the maximum difference
    2. calculate the ratio of bigger shape : smaller shape - 1 (how much the bigger
    shape is bigger than the smaller shape)

    The first method will cause the bias towards the bigger shapes
    """

    sizes_shape_1 = [float(i[1]) for i in labels]
    sizes_shape_2 = [float(i[2]) for i in labels]
    differences = [(max([i, j]) / min([i, j])) - 1 for i, j in zip(sizes_shape_1, sizes_shape_2)]

    # normalize the differences
    scaler = preprocessing.StandardScaler()
    scaler.fit(np.array(differences).reshape(-1, 1))
    differences = scaler.transform(np.array(differences).reshape(-1, 1)).reshape(-1)

    # make differences into a list of strings
    differences = [str(i) for i in differences]
    # check each element in differences is unique for naming purposes
    try:
        assert len(differences) == len(set(differences))
    except AssertionError:
        # add "0"s to the differences that are not unique
        for i in range(len(differences)):
            if differences.count(differences[i]) > 1:
                differences[i] = differences[i] + "0" * (differences.count(differences[i]) - 1)

    # rename the images
    for i, j in tqdm(zip(images, differences)):
        os.rename(Path(folder, i), Path(folder, f"{j}.png"))


def mp_generate():
    """
    Generate the data using multiprocessing
    """
    try:
        # use multiprocessing to run in parallel
        with mp.Pool(mp.cpu_count()) as pool:
            pool.map(main, combs)
    except KeyboardInterrupt:
        # kill all children
        pool.terminate()
        printc(f"Terminated at n = {progress_bar.n}", color="red")


def train_test_split():
    images = os.listdir(Path("data", variation_name, "images"))
    images = [i for i in images if i.endswith(".png")]
    train = np.random.choice(images, int(len(images) * 0.8), replace=False)
    test = [i for i in images if i not in train]
    [shutil.move(Path("data", variation_name, "images", i), Path("data", variation_name, "train", i)) for i in train]
    [shutil.move(Path("data", variation_name, "images", i), Path("data", variation_name, "test", i)) for i in test]


if __name__ == "__main__":
    # remove existing files
    shutil.rmtree("data", variation_name)

    # make data directory
    os.makedirs(Path("data", variation_name), exist_ok=True)
    os.makedirs(Path("data", variation_name, "images"), exist_ok=True)
    os.makedirs(Path("data", variation_name, "train"), exist_ok=True)
    os.makedirs(Path("data", variation_name, "test"), exist_ok=True)
    os.makedirs(Path("data", variation_name, "under_threshold"), exist_ok=True)

    mp_generate()
    filter_data()

    set_label(Path("data", variation_name, "images"))

    train_test_split()

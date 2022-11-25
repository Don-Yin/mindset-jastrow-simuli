"""
A script that generates a jastrow shape and saves it to a file.
"""

import itertools
import json
import multiprocessing
import operator
import os
from math import cos, pi, sin
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.Image import new
from PIL.ImageDraw import Draw
from tqdm import tqdm

settings = json.load(open("settings.json", "r"))
canvas_size_jastrow = settings["target_image_width"] * settings["initial_expansion"]
background_size = settings["target_image_width"] * settings["initial_expansion"]

# local configs for samples generation
configs_constant = {
    "overlap_threshold": 0.001,
    "init_angle": 20,  # from 0 - 44
    "rotate_step": 1,
}

configs_variable = {
    "radius_outer": 1,  # from 0 - 1 / this basically defines resolution
    "thickness": 0.5,  # from 0 - 1
    "angle": 45,  # from 0 - 45; note this is different from the init_angle which is for rotations
}


def update_configs(configs_variable: dict) -> dict:
    """
    Initial update of the configs_variable. called once
    at the beginning of the program.
    """
    configs_variable["radius_outer"] *= canvas_size_jastrow / 2
    configs_variable["radius_outer"] += canvas_size_jastrow / 2

    configs_variable["radius_inner"] = configs_variable["radius_outer"] - (
        (configs_variable["radius_outer"] - (canvas_size_jastrow / 2)) * configs_variable["thickness"]
    )

    return configs_variable


def get_vector(angle: float, length: float):
    """Compute the vector of the given angle and length
    params:
        angle (float): angle in radians
        length (float): length of the vector
    """
    angle = -angle
    angle -= 90
    angle = angle * pi / 180
    x, y = length * cos(angle), length * sin(angle)
    return (x, y)


def generate_shape(configs_variable: dict = configs_variable) -> Image:
    """
    Generate a uncropped/un-rotated jastrow shape using the
    configs_variable.
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


def compute_coordinates(canvas: Image) -> np.ndarray:
    """
    Compute the coordinates of the upper left and lower left
    corners of the canvas' white pixel.
    """
    # translate into arrays
    canvas_array = np.array(canvas)
    # coordinates of the white pixels at the lowest four y values
    white_pixels = np.where(canvas_array[:, :, 3] == 255)
    white_pixels = np.array([white_pixels[0], white_pixels[1]])
    white_pixels = white_pixels[:, white_pixels[1].argsort()]
    coordinates_upper_left = white_pixels[:, :1]

    # coordinates of the white pixels at the highest four x values
    white_pixels = np.where(canvas_array[:, :, 3] == 255)
    white_pixels = np.array([white_pixels[0], white_pixels[1]])
    white_pixels = white_pixels[:, white_pixels[0].argsort()]
    white_pixels = white_pixels[:, ::-1]
    white_pixels = white_pixels[:, :4]
    # select the one with the lowest y value
    white_pixels = white_pixels[:, white_pixels[1].argsort()]
    coordinates_lower_left = white_pixels[:, :1]

    return coordinates_upper_left, coordinates_lower_left


def rotate_and_crop(shape: Image, degree: int = 44, by_center=False) -> Image:
    """Rotate and crop the canvas to the smallest possible size.
    Params:
        canvas: the canvas to rotate and crop
        degree: the degree to rotate the canvas
    Returns:
        shape_1: the cropped canvas
        shape_2: the rotated and cropped canvas
    """
    _, coordinates_lower_left = compute_coordinates(shape)

    shape_1 = shape.crop(shape.getbbox())
    if by_center:
        rotation_1 = np.random.randint(0, 360)
        rotation_2 = np.random.randint(0, 360)
        shape_1 = shape_1.rotate(rotation_1, expand=True)
        shape_2 = shape.rotate(rotation_2, expand=True)
    else:
        shape_2 = shape.rotate(degree, center=(coordinates_lower_left[1], coordinates_lower_left[0]), expand=True)

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


def make_jastrow(shape_1: Image, shape_2: Image, configs_variable: dict, distance: float = None) -> Image:
    # place the first shape at the center of the background
    half_shape_size = tuple(i // 2 for i in shape_1.size)
    center_coordinates = tuple(i // 2 for i in (background_size,) * 2)
    paste_coordinates = tuple(operator.sub(*i) for i in zip(center_coordinates, half_shape_size))

    background = new("RGBA", tuple([background_size] * 2), color=(0, 0, 0, 0))
    background.paste(shape_1, paste_coordinates, shape_1)

    # compute the motion vector for moving the second shape over the first shape
    O_coordinates_upper_left, O_coordinates_lower_left = compute_coordinates(shape_1)
    R_coordinates_upper_left, R_coordinates_lower_left = compute_coordinates(shape_2)
    motion_vector = O_coordinates_upper_left - R_coordinates_lower_left

    # find a new paste coordinate so that the previous lower left coordinate is now the upper left coordinate
    paste_coordinates = (int(paste_coordinates[0] + motion_vector[1]), int(paste_coordinates[1] + motion_vector[0]))

    if distance:
        """
        Testify the overlap of the two shapes here by pasting the rotated shape on the background.
        Then apply the second motion vector.
        """
        # compute the motion vector for moving the second shape away from the first shape
        motion_vector = get_vector(
            angle=configs_variable["angle"] / 2,
            length=distance,
        )
        paste_coordinates = (int(paste_coordinates[0] + motion_vector[0]), int(paste_coordinates[1] + motion_vector[1]))

    # # paste the second image
    background.paste(shape_2, paste_coordinates, shape_2)

    return background


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


class RotateRecorder:
    def __init__(self):
        self.init_angle = configs_constant["init_angle"]
        self.overlapped = 0

    def update_overlap(self, overlap: int):
        self.overlapped = overlap

    def update_angle(self):
        self.init_angle -= configs_constant["rotate_step"]


def chunk_list(target_list: list, n: int):
    return [target_list[i : i + n] for i in range(0, len(target_list), n)]


def get_optimal_angle(input_config: dict):
    # init the original canvas + rotate recorder
    canvas_uncropped = generate_shape(configs_variable=input_config)
    rotate_recorder = RotateRecorder()

    # iteratively reduce the angle of the second shape until it touches (overlap) with the first shape
    # or stop when the angle is 0 (totally horizontal)
    while rotate_recorder.overlapped < configs_constant["overlap_threshold"] and rotate_recorder.init_angle > 0:
        shape_1, shape_2 = rotate_and_crop(canvas_uncropped, degree=rotate_recorder.init_angle)

        # compute sum of the pixel size of both shapes:
        size_cropped = get_area_size(canvas=shape_1, color="white")  # original
        size_rotated = get_area_size(canvas=shape_2, color="white")  # rotated
        size_sum = size_cropped + size_rotated

        # make a jastrow by concatenating the shapes
        # compute pixel size of the jastrow
        jastrow = make_jastrow(shape_1, shape_2, configs_variable=input_config)
        size_jastrow = get_area_size(canvas=jastrow, color="white")

        # save the settings
        previous_angle = rotate_recorder.init_angle

        # update overlap area / angle
        rotate_recorder.update_overlap(overlap=1 - (size_jastrow / size_sum))
        rotate_recorder.update_angle()

    return previous_angle


if __name__ == "__main__":

    # make grid
    grid_radius_outer = [0.8]  # from 0 - 1; basically resolution
    grid_thickness = np.linspace(0.1, 0.9, 20)  # from 0 - 1
    grid_angle = np.linspace(5, 45, 20)  # from 1 - 45

    # distances
    num_distances = 40
    distances = np.linspace(0, 0.1, num_distances) * canvas_size_jastrow  # from 0 - 1

    # make combinations of the settings
    combinations = list(itertools.product(grid_radius_outer, grid_thickness, grid_angle))
    configurations = [{"radius_outer": i[0], "thickness": i[1], "angle": i[2]} for i in combinations]
    configurations = [update_configs(config) for config in configurations]

    with multiprocessing.Pool(os.cpu_count()) as pool:
        results = list(tqdm(pool.imap(get_optimal_angle, configurations), total=len(configurations)))
        # results = pool.map(get_optimal_angle, configurations)
        configs_to_save = [{"configuration": config, "optimal_angle": result} for config, result in zip(configurations, results)]

    # save the configs
    json.dump(configs_to_save, open(Path("saved_configs.json"), "w"))

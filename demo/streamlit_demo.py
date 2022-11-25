import json

import streamlit as st

from demo.make_demo_data import generate_shape, make_jastrow, rotate_and_crop

st.set_page_config(layout="wide")


class Demo:
    def __init__(self):
        # read the json file
        self.data = json.load(open("data_demo.json"))
        # self.outer_radius = sorted(list(set([float(i[0]) for i in attributes]))) # 3800.0
        self.thickness = sorted([i["configuration"]["thickness"] for i in self.data])
        self.radian = sorted([i["configuration"]["angle"] for i in self.data])

        self.remove_logo()
        self.sidebar()
        self.main()

    def main(self):
        canvas_uncropped = generate_shape(configs_variable=self.selected_config)
        canvas_cropped, canvas_rotated = rotate_and_crop(canvas_uncropped, degree=self.get_optimal_angle_by_config(self.selected_config))
        jastrow = make_jastrow(canvas_cropped, canvas_rotated, configs_variable=self.selected_config, distance=self.selected_distance)
        jastrow = jastrow.crop(jastrow.getbbox())
        st.image(jastrow, width=jastrow.size[0] // 3)

    def get_optimal_angle_by_config(self, config):
        return [
            i for i in self.data if i["configuration"]["thickness"] == config["thickness"] and i["configuration"]["angle"] == config["angle"]
        ][0]["optimal_angle"]

    def sidebar(self):
        with st.sidebar:
            st.title("Jastrow Demo")

            selected_thickness = st.select_slider(label="Thickness", options=self.thickness, value=self.thickness[0])
            selected_radian = st.select_slider(label="Length", options=self.radian, value=self.radian[0])

            self.selected_distance = st.slider(label="Distance", min_value=1, max_value=1200, value=1, step=1)
            self.selected_config = {"thickness": selected_thickness, "angle": selected_radian, "radius_outer": 3800.0}
            self.selected_config["radius_inner"] = self.selected_config["radius_outer"] - (
                (self.selected_config["radius_outer"] - (4000 / 2)) * self.selected_config["thickness"]
            )

    def remove_logo(self):
        hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

        # MainMenu {visibility: hidden;}


if __name__ == "__main__":
    Demo()

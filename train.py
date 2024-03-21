import argparse
import itertools
import math
import os
from contextlib import nullcontext
import random

import numpy as np
import mlx
import mlx.nn.layers as F
import mlx.utils

from stable_diffusion.clip import CLIPTextModel
from stable_diffusion.vae import CVAE
from stable_diffusion.unet import UNetModel
from stable_diffusion.tokenizer import Tokenizer
from datasets import Dataset

ROOT_MODEL_PATH = "./rvxl4/snapshots/a302f56626174497a679d51745089bae3cc9fee7"

if __name__ == '__main__':
    # https://huggingface.co/SG161222/RealVisXL_V4.0/blob/main/**/config.json
    text_encoder = CLIPTextModel.from_pretrained(f"${ROOT_MODEL_PATH}/text_encoder")
    text_encoder_2 = CLIPTextModel.from_pretrained(f"${ROOT_MODEL_PATH}/text_encoder_2")
    vae = CVAE.from_pretrained(f"{ROOT_MODEL_PATH}/vae")
    unet = UNetModel.from_pretrained(f"${ROOT_MODEL_PATH}/unet")
    tokenizer = Tokenizer.from_pretrained(f"${ROOT_MODEL_PATH}/tokenizer")
    tokenizer_2 = Tokenizer.from_pretrained(f"${ROOT_MODEL_PATH}/tokenizer_2")


    dset = Dataset(
        instance_data_root="",
        instance_prompt="photo of a ewpp woman"
    )

    # TODO: compose the above models into a general LoRA pipeline.. then train

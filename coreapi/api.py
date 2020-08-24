import base64
import io
from typing import List

from fastapi import FastAPI
import numpy as np
from PIL import Image
from pydantic import BaseModel
from skimage import measure

from corebreakout import CoreSegmenter
from corebreakout import utils
from coreapi.config import CONFIG

# By default, models from corebreakout's assets.zip
def load_model():
    return CoreSegmenter(**CONFIG)

app = FastAPI()


# Define classes for what Label-tool accepts

class InputBytes(BaseModel):
    b64: str


class Instance(BaseModel):
    input_bytes: InputBytes


class Instances(BaseModel):
    instances: List[Instance]


@app.post("/labels")
def core_labels(images: Instances):
    labels = []
    for instance in images:
        labels.append(segment_image(instance))
    return {"masks": labels}


def segment_image(instance: Instance, model=load_model()):
    image_bytes = base64.decodebytes(instance[1][0].input_bytes.b64.encode())
    image_arr = np.array(Image.open(io.BytesIO(image_bytes)))

    # Use just the mask-generating parts of the segmenter instead of:
    # return model.segment(image_arr, depth_range=[1.0, 4.0], layout_params={'col_height': 100})
    preds = model.model.detect([image_arr], verbose=0)[0]
    col_labels = utils.masks_to_labels(preds['masks'])
    # TODO measure.regionprops(col_labels)
    return col_labels

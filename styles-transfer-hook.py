import logging
import PIL.Image
import io
import numpy as np


LOG = logging.getLogger(__name__)

def init_hook(**params):
    LOG.info("Init hooks {}".format(params))


def preprocess(inputs,ctx):
    images = inputs['image']
    batch = []
    sizes = []
    for image in images:
        image = PIL.Image.open(io.BytesIO(image))
        sizes.append(image.size)
        image = image.convert('RGB').resize((512,512),PIL.Image.BILINEAR)
        image = np.asarray(image)
        image = np.transpose(image, (2,0,1))
        batch.append(image)
    batch = np.stack(batch)
    ctx.sizes = sizes

    return {'Reshape/placeholder_port_0': batch}

def postprocess(outputs,ctx):
    original_size = ctx.sizes[0]
    for k,v in outputs.items():
        outputs = v[0]

        break
    outputs = np.transpose(outputs, (1,2,0))
    outputs = np.clip(outputs, 0, 255.0)
    im = PIL.Image.fromarray(np.uint8(outputs))
    im = im.resize(original_size,PIL.Image.BILINEAR)
    with io.BytesIO() as output:
        im.save(output,format='PNG')
        contents = output.getvalue()
    return {'output':contents}

import logging
import PIL.Image
import io
import numpy as np
import pickle
import os

LOG = logging.getLogger(__name__)


def log(func):
    def decorator(*args, **kwargs):
        LOG.info('Running %s...' % func.__name__)
        return func(*args, **kwargs)

    return decorator

@log
def init_hook(**params):
    LOG.info("Init hooks {}".format(params))
    


@log
def preprocess(inputs,**kwargs):
    LOG.info('Preprocess: {}, args: {}'.format(inputs,kwargs))
    images = inputs['image']
    batch = []
    for image in images:
        image = PIL.Image.open(io.BytesIO(image))
        image = image.resize((512,512))
        image = np.asarray(image)
        image = np.transpose(image, (2,0,1))
        batch.append(image)
    batch = np.stack(batch)
    LOG.info('Batch shape: {}'.format(batch.shape))
    return {'Reshape/placeholder_port_0': batch}

@log
def postprocess(outputs,**kwargs):
    for k,v in outputs.items():
        outputs = v[0]
        LOG.info('Use {} as output,{}'.format(k,v.shape))
        break
    outputs = np.transpose(outputs, (1,2,0))
    outputs = np.clip(outputs, 0, 255.0)
    im = PIL.Image.fromarray(np.uint8(outputs))
    with io.BytesIO() as output:
        im.save(output,format='PNG')
        contents = output.getvalue()
    return {'output':contents}

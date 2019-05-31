import base64
import io
import json
import logging

import cv2
import numpy as np


LOG = logging.getLogger(__name__)


def init_hook(**params):
    LOG.info('Init with params:')
    LOG.info(json.dumps(params, indent=2))


def prepare_image(image):
    x = image.transpose([2, 0, 1])
    x = np.expand_dims(x, axis=0)
    return x


def result_image(out):
    out = out.clip(min=0, max=1)
    out = out.squeeze().transpose([1, 2, 0]) * 255
    out = out.astype(np.uint8)
    return out


def process(inputs, ctx, **kwargs):
    bgr_image = _load_image(inputs, 'input', bgr=True)

    model_inputs = kwargs['model_inputs']
    # Get [W, H] from [B, C, H, W]
    low_input_size = tuple(model_inputs['0'][:-3:-1])
    high_input_size = tuple(model_inputs['1'][:-3:-1])
    factor = int(high_input_size[1] / low_input_size[1])

    low_image = cv2.resize(bgr_image, low_input_size, interpolation=cv2.INTER_AREA)
    high_image = cv2.resize(bgr_image, high_input_size, interpolation=cv2.INTER_CUBIC)

    outputs = ctx.driver.predict({
        '0': prepare_image(low_image),
        '1': prepare_image(high_image),
    })
    output = list(outputs.values())[0]
    output = result_image(output)
    # output = output.squeeze().transpose([1, 2, 0])
    # output = np.clip((output * 255).astype(np.uint8), a_min=0, a_max=255)

    output = cv2.resize(
        output,
        (bgr_image.shape[1] * factor, bgr_image.shape[0] * factor),
        interpolation=cv2.INTER_AREA,
    )
    original_resized = image_resize(bgr_image, width=bgr_image.shape[1] * factor)

    im_bytes = cv2.imencode('.jpg', output)[1].tostring()
    im_bytes_resized = cv2.imencode('.jpg', original_resized)[1].tostring()

    return {'output': im_bytes, 'resized': im_bytes_resized}


def _load_image(inputs, image_key, bgr=False):
    image = inputs.get(image_key)
    if image is None:
        raise RuntimeError('Missing "{0}" key in inputs. Provide an image in "{0}" key'.format(image_key))
    if len(image.shape) == 0:
        image = np.stack([image.tolist()])

    if len(image.shape) < 3:
        image = cv2.imdecode(np.frombuffer(image[0], np.uint8), cv2.IMREAD_COLOR)
        if not bgr:
            image = image[:, :, ::-1]

    return image


def _boolean_string(s):
    if isinstance(s, bool):
        return s

    s = s.lower()
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[0], image.shape[1]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def result_table_string(result_dict, ctx):
    table = []

    def crop_from_box(box, normalized_coordinates=True):
        left, right = box[0], box[2]
        top, bottom = box[1], box[3]
        if normalized_coordinates:
            left, right = left * ctx.image.width, right * ctx.image.width
            top, bottom = top * ctx.image.height, bottom * ctx.image.height

        cropped = ctx.image.crop((left, top, right, bottom))
        cropped = image_resize(cropped, width=256)
        image_bytes = io.BytesIO()
        cropped.convert('RGB').save(image_bytes, format='JPEG', quality=80)

        return image_bytes.getvalue()

    def append(type_, name, prob, image):
        encoded = image
        if image is not None:
            encoded = base64.encodebytes(image).decode()

        table.append(
            {
                'type': type_,
                'name': name,
                'prob': float(prob),
                'image': encoded
            }
        )

    if len(result_dict.get('face_boxes', [])) > 0:
        for prob, box in zip(result_dict['face_scores'], result_dict['face_boxes']):
            append('face', 'face', prob, crop_from_box(box))

    return json.dumps(table)

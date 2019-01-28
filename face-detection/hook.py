import base64
import io
import json
import logging

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import six


LOG = logging.getLogger(__name__)
factor = 0.7


def init_hook(**params):
    fac = params.get('factor')
    if fac:
        fac = float(fac)
        global factor
        factor = fac
        LOG.info('Factor is "%s" now.' % factor)


def preprocess(inputs, ctx, **kwargs):
    image = inputs.get('input')
    if image is None:
        raise RuntimeError('Missing "input" key in inputs. Provide an image in "input" key')

    if isinstance(image[0], (six.string_types, bytes)):
        image = Image.open(io.BytesIO(image[0]))

        image = image.convert('RGB')

    if isinstance(image, np.ndarray) and image.shape[2] == 4:
        # Convert RGBA -> RGB
        rgba_image = Image.fromarray(image)
        image = rgba_image.convert('RGB')

    data = image.resize((300, 300), Image.ANTIALIAS)
    data = np.array(data).transpose([2, 0, 1]).reshape(1, 3, 300, 300)
    # convert to BGR
    data = data[:, ::-1, :, :]

    ctx.image = image

    return {'data': data}


def draw_rectangle(draw, coordinates, color, width=1):
    for i in range(width):
        rect_start = (coordinates[0][0] - i, coordinates[0][1] - i)
        rect_end = (coordinates[1][0] + i, coordinates[1][1] + i)
        draw.rectangle((rect_start, rect_end), outline=color)


def add_overlays(frame, boxes, labels=None):
    draw = ImageDraw.Draw(frame)
    if boxes is not None:
        for i, face in enumerate(boxes):
            face_bb = face.astype(int)
            draw_rectangle(
                draw,
                [(face_bb[0], face_bb[1]), (face_bb[2], face_bb[3])],
                (0, 255, 0), width=2
            )

            if labels:
                draw.text(
                    (face_bb[0] + 4, face_bb[1] + 5),
                    labels[i], font=ImageFont.load_default(),
                )


def image_resize(image, width=None, height=None, inter=Image.ANTIALIAS):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.height, image.width

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
    resized = image.resize(dim, inter)

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


def postprocess(outputs, ctx):
    outputs = list(outputs.values())[0].reshape([-1, 7])
    # 7 values:
    # class_id, label, confidence, x_min, y_min, x_max, y_max
    # Select boxes where confidence > factor
    bboxes_raw = outputs[outputs[:, 2] > factor]

    w, h = ctx.image.size

    if bboxes_raw is not None:
        bboxes = np.zeros([len(bboxes_raw), 4])
        labels = []
        for i, box in enumerate(bboxes_raw):
            bboxes[i][0] = box[3] * w
            bboxes[i][1] = box[4] * h
            bboxes[i][2] = box[5] * w
            bboxes[i][3] = box[6] * h
            labels.append('%.3f. %s' % (box[2], box[1]))

        result = {'face_boxes': bboxes_raw[:, 3:7], 'face_scores': bboxes_raw[:, 2]}
        table = result_table_string(result, ctx)
        add_overlays(ctx.image, bboxes, labels)
    else:
        table = []

    image_bytes = io.BytesIO()

    im = ctx.image
    im.save(image_bytes, format='JPEG', quality=80)

    return {
        'output': image_bytes.getvalue(),
        'table_output': table
        # 'boxes': ctx.bounding_boxes,
        # 'labels': np.array(labels)
    }

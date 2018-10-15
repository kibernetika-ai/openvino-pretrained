import io
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

        add_overlays(ctx.image, bboxes, labels)

    image_bytes = io.BytesIO()

    im = ctx.image
    im.save(image_bytes, format='PNG')

    return {
        'output': image_bytes.getvalue(),
        # 'boxes': ctx.bounding_boxes,
        # 'labels': np.array(labels)
    }

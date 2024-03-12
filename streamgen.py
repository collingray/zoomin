import torch
from diffusers import StableDiffusionUpscalePipeline
from diffusers.utils import load_image
from diffusers import StableDiffusionInpaintPipeline
import PIL
from flask import Flask, Response, stream_with_context, request
import tempfile
from threading import Thread
from datetime import datetime, timedelta
from time import sleep
import os
import cv2
from queue import Queue
from io import BytesIO


frames_per_second = 30
keyframes_per_second = 0.5
frames_per_keyframe = int(frames_per_second / keyframes_per_second)
crop_per_keyframe = 0.8
image_size = (512, 512)
inpaint_pipe = None


def rescale_and_mask(image, downscale_percent):
    rescaled_size = (int(image.width * downscale_percent), int(image.height * downscale_percent))
    rescaled_image = image.resize(rescaled_size, resample=3)
    # the output image is the original downscaled by downscale_percent and padded to the original size
    output_image = PIL.Image.new(image.mode, (image.width, image.height), (0, 0, 0))
    output_image.paste(rescaled_image, (int((image.width - rescaled_image.width) / 2), int((image.height - rescaled_image.height) / 2)))
    # the mask is a white image with the same size as the original, with a black rectangle in the center
    mask = PIL.Image.new("L", (image.width, image.height), 255)
    mask.paste(PIL.Image.new("L", rescaled_size, 0), (int((image.width - rescaled_image.width) / 2), int((image.height - rescaled_image.height) / 2)))

    return output_image, mask


def iterate_image(image, prompt="a boundless harbor scene fractal beautiful escher", downscale_percent=0.9):
    zoomed_out, mask = rescale_and_mask(image, downscale_percent=downscale_percent)
    output_image = inpaint_pipe(prompt=prompt, image=zoomed_out, mask_image=mask, num_inference_steps=35).images[0]
    return output_image


def center_crop(image, percent):
    h, w = image.size
    new_h = int(h * percent)
    new_w = int(w * percent)
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    return image.crop((left, top, left + new_w, top + new_h))


def center_crop_rescale(image, percent):
    h, w = image.size
    cropped = center_crop(image, percent)
    return cropped.resize((w, h), resample=3)


def encode_frame(frame):
    img_buffer = BytesIO()
    frame.save(img_buffer, format="JPEG")

    # Return the output frame in the byte format
    return b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + img_buffer.getvalue() + b'\r\n'


def generate_keyframes(init_img, out_buffer):
    global prompt
    curr_img = init_img
    out_buffer.put(curr_img)

    while True:
        curr_img = iterate_image(curr_img, downscale_percent=crop_per_keyframe, prompt=prompt)
        out_buffer.put(curr_img)


# The amount to crop each frame, 1 for frame_idx = total_frames-1, total_crop for frame_idx = 0
def crop_for_frame(total_crop, frame_idx, total_frames):
    return total_crop + ((1 - total_crop) * (frame_idx / (total_frames - 1)))


def generate_frames(in_buffer, out_buffer):
    while True:
        keyframe = in_buffer.get()
        for i in range(frames_per_keyframe):
            encoded = encode_frame(center_crop_rescale(keyframe, crop_for_frame(crop_per_keyframe, i, frames_per_keyframe)))
            if encoded:
                out_buffer.put(encoded)


def generate_video_stream():
    starting_image = PIL.Image.new("RGB", image_size, (255, 0, 0))
    keyframe_buffer = Queue(maxsize=10)
    frame_buffer = Queue(maxsize=frames_per_second*10)

    keyframe_generator_thread = Thread(target=generate_keyframes, args=(starting_image, keyframe_buffer))
    keyframe_generator_thread.daemon = True  # This ensures the thread exits when the main program does
    keyframe_generator_thread.start()

    frame_generator_thread = Thread(target=generate_frames, args=(keyframe_buffer, frame_buffer))
    frame_generator_thread.daemon = True  # This ensures the thread exits when the main program does
    frame_generator_thread.start()

    while True:
        yield frame_buffer.get()
        sleep(1 / frames_per_second)


app = Flask(__name__)
stream_generator = None
prompt = "a boundless harbor scene fractal beautiful escher"


@app.route('/video')
def video():
    return Response(stream_with_context(stream_generator), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/settings', methods=['POST'])
def settings():
    global prompt
    json = request.get_json()
    prompt = json['prompt']
    return "Settings updated"


if __name__ == '__main__':
    # Load the model
    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16)
    inpaint_pipe = inpaint_pipe.to("cuda")
    # Disable the overzealous safety checker
    inpaint_pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

    stream_generator = generate_video_stream()

    app.run(host='0.0.0.0', port=6969, debug=True, threaded=True, use_reloader=False)
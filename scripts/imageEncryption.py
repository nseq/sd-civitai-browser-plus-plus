from PIL import Image as PILImage, PngImagePlugin, _util, ImagePalette
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, Response
from urllib.parse import unquote
from pathlib import Path
from PIL import Image
import gradio as gr
import numpy as np
import asyncio
import hashlib
import base64
import sys
import io
import os

from modules.paths_internal import models_path
from modules import shared, images
from modules.api import api

Embed  = shared.cmd_opts.embeddings_dir
Models = Path(models_path)

password = getattr(shared.cmd_opts, 'enc_pw', None)
image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.avif']
image_keys = ['Encrypt', 'EncryptPwdSha']

def set_shared_options():
    section = ("encrypt_image_is_enable", "Encrypt image")
    option = shared.OptionInfo(default="Yes", label="Whether the encryption plug-in is enabled", section=section)
    option.do_not_save = True
    shared.opts.add_option("encrypt_image_is_enable", option)
    shared.opts.data['encrypt_image_is_enable'] = "Yes"

def get_range(input: str, offset: int, range_len=4):
    offset = offset % len(input)
    return (input * 2)[offset:offset + range_len]

def get_sha256(input: str):
    return hashlib.sha256(input.encode('utf-8')).hexdigest()

def shuffle_arr_v2(arr, key):
    sha_key = get_sha256(key)
    arr_len = len(arr)

    for i in range(arr_len):
        s_idx = arr_len - i - 1
        to_index = int(get_range(sha_key, i, range_len=8), 16) % (arr_len - i)
        arr[s_idx], arr[to_index] = arr[to_index], arr[s_idx]

    return arr

def encrypt_image_v3(image: Image.Image, psw):
    try:
        width = image.width
        height = image.height
        x_arr = np.arange(width)
        shuffle_arr_v2(x_arr,psw) 
        y_arr = np.arange(height)
        shuffle_arr_v2(y_arr,get_sha256(psw))
        pixel_array = np.array(image)
        
        _pixel_array = pixel_array.copy()
        for x in range(height): 
            pixel_array[x] = _pixel_array[y_arr[x]]
        pixel_array = np.transpose(pixel_array, axes=(1, 0, 2))
        
        _pixel_array = pixel_array.copy()
        for x in range(width): 
            pixel_array[x] = _pixel_array[x_arr[x]]
        pixel_array = np.transpose(pixel_array, axes=(1, 0, 2))

        return pixel_array
    except Exception as e:
        if "axes don't match array" in str(e):
            return np.array(image)

def decrypt_image_v3(image: Image.Image, psw):
    try:
        width = image.width
        height = image.height
        x_arr = np.arange(width)
        shuffle_arr_v2(x_arr, psw)
        y_arr = np.arange(height)
        shuffle_arr_v2(y_arr, get_sha256(psw))
        pixel_array = np.array(image)

        _pixel_array = pixel_array.copy()
        for x in range(height): 
            pixel_array[y_arr[x]] = _pixel_array[x]
        pixel_array = np.transpose(pixel_array, axes=(1, 0, 2))

        _pixel_array = pixel_array.copy()
        for x in range(width): 
            pixel_array[x_arr[x]] = _pixel_array[x]
        pixel_array = np.transpose(pixel_array, axes=(1, 0, 2))

        return pixel_array

    except Exception as e:
        if "axes don't match array" in str(e):
            return np.array(image)

class EncryptedImage(PILImage.Image):
    __name__ = "EncryptedImage"

    @staticmethod
    def from_image(image: PILImage.Image):
        image = image.copy()
        img = EncryptedImage()
        img.im = image.im
        img._mode = image.mode
        if image.im.mode:
            try:
                img.mode = image.im.mode
            except Exception:
                pass

        img._size = image.size
        img.format = image.format
        if image.mode in ("P", "PA"):
            img.palette = image.palette.copy() if image.palette else ImagePalette.ImagePalette()

        img.info = image.info.copy()
        return img

    def save(self, fp, format=None, **params):
        filename = ""
        encryption_type = self.info.get('Encrypt')

        if isinstance(fp, Path):
            filename = str(fp)
        elif _util.is_path(fp):
            filename = fp
        elif fp == sys.stdout:
            try:
                fp = sys.stdout.buffer
            except AttributeError:
                pass

        if not filename and hasattr(fp, "name") and _util.is_path(fp.name):
            filename = fp.name

        if not filename or not password:
            super().save(fp, format=format, **params)
            return

        if encryption_type == 'pixel_shuffle_3':
            super().save(fp, format=format, **params)
            return

        pnginfo = params.get('pnginfo', PngImagePlugin.PngInfo())
        if not pnginfo:
            pnginfo = PngImagePlugin.PngInfo()

        for key, value in (self.info or {}).items():
            if value:
                pnginfo.add_text(key, str(value))

        back_img = PILImage.new('RGBA', self.size)
        back_img.paste(self)

        try:
            encrypted_img = PILImage.fromarray(encrypt_image_v3(self, get_sha256(password)))
            self.paste(encrypted_img)
            encrypted_img.close()
        except Exception as e:
            if "axes don't match array" in str(e):
                fn = Path(filename)
                os.system(f'rm -f {fn}')
                return

        pnginfo.add_text('Encrypt', 'pixel_shuffle_3')
        pnginfo.add_text('EncryptPwdSha', get_sha256(f'{get_sha256(password)}Encrypt'))

        params.update(pnginfo=pnginfo)
        super().save(fp, format=self.format, **params)
        self.paste(back_img)
        back_img.close()

def open(fp, *args, **kwargs):
    try:
        if not _util.is_path(fp) or not Path(fp).suffix:
            return super_open(fp, *args, **kwargs)

        if isinstance(fp, bytes):
            return encode_pil_to_base64(fp)

        img = super_open(fp, *args, **kwargs)
        try:
            pnginfo = img.info or {}

            if password and img.format.lower() == PngImagePlugin.PngImageFile.format.lower():
                if pnginfo.get("Encrypt") == 'pixel_shuffle_3':
                    decrypted_img = PILImage.fromarray(decrypt_image_v3(img, get_sha256(password)))
                    img.paste(decrypted_img)
                    decrypted_img.close()
                    pnginfo["Encrypt"] = None

            img.info = pnginfo

            return EncryptedImage.from_image(img)

        except Exception as e:
            print(f"Error in 203 : {fp} : {e}")
            return None

        finally:
            img.close()

    except Exception as e:
        print(f"Error in 210 : {fp} : {e}")
        return None

def encode_pil_to_base64(img: PILImage.Image):
    pnginfo = img.info or {}

    with io.BytesIO() as output_bytes:
        if pnginfo.get("Encrypt") == 'pixel_shuffle_3':
            img.paste(PILImage.fromarray(decrypt_image_v3(img, get_sha256(password))))

        pnginfo["Encrypt"] = None
        img.save(output_bytes, format=PngImagePlugin.PngImageFile.format, quality=shared.opts.jpeg_quality)
        bytes_data = output_bytes.getvalue()

    return base64.b64encode(bytes_data)

_executor = ThreadPoolExecutor(max_workers=100)
_semaphore_factory = lambda: asyncio.Semaphore(min(os.cpu_count() * 2, 10))
_semaphores = {}
p_cache = {}

def imgResize(image, target_height=500):
    width, height = image.size
    if height > target_height:
        aspect_ratio = width / height
        new_width = int(target_height * aspect_ratio)
        return image.resize((new_width, target_height), PILImage.Resampling.LANCZOS)
    return image

async def imgAsync(fp, image_keys, should_resize=False):
    loop = asyncio.get_running_loop()
    if loop not in _semaphores:
        _semaphores[loop] = _semaphore_factory()
    semaphore = _semaphores[loop]

    try:
        async with semaphore:
            if fp in p_cache:
                return p_cache[fp]

            try:
                content = await loop.run_in_executor(
                    _executor,
                    lambda: imgProcess(fp, image_keys, should_resize)
                )
            except Exception as e:
                print(f"Error in 256 : {fp}, Error: {e}")
                return None

            p_cache[fp] = content
            return content
    except Exception as e:
        print(f"Error in 262 : {fp}: {e}")
        try:
            with open(fp, 'rb') as f:
                return f.read()
        except Exception as inner_e:
            print(f"Error in 267 : {inner_e}")
            return None
    finally:
        if fp in p_cache:
            del p_cache[fp]

def imgProcess(fp, image_keys, should_resize):
    try:
        with PILImage.open(fp) as image:
            try:
                image.verify()
            except Exception as e:
                print(f"Invalid image file: {fp}: {e}")
                return None

            if should_resize:
                image = imgResize(image)
                image.save(fp)

            pnginfo = image.info or {}

            if not all(k in pnginfo for k in image_keys):
                try:
                    EncryptedImage.from_image(image).save(fp)
                    image = PILImage.open(fp)
                    pnginfo = image.info or {}
                except Exception as e:
                    print(f"Error in 294 : {fp}: {e}")
                    return None

            buffered = io.BytesIO()
            info = PngImagePlugin.PngInfo()

            for key, value in pnginfo.items():
                if value is None or key == 'icc_profile':
                    continue
                if isinstance(value, bytes):
                    try:
                        info.add_text(key, value.decode('utf-8'))
                    except UnicodeDecodeError:
                        try:
                            info.add_text(key, value.decode('utf-16'))
                        except UnicodeDecodeError:
                            info.add_text(key, str(value))
                            print(f"Error decoding '{key}' in hook http. {fp}")
                else:
                    info.add_text(key, str(value))

            image.save(buffered, format=PngImagePlugin.PngImageFile.format, pnginfo=info)
            image.close()
            return buffered.getvalue()
    except Exception as e:
        print(f"Error in 319 : {fp}: {e}")
        return None

def hook_http_request(app: FastAPI):
    @app.middleware("http")
    async def image_decrypting(req: Request, call_next):
        endpoint = '/' + req.scope.get('path', 'err').strip('/')

        def process_query(endpoint, prefixes, param):
            if endpoint.startswith(prefixes):
                query_string = unquote(req.scope.get('query_string', b'').decode('utf-8'))
                return next((sub.split('=')[1] for sub in query_string.split('&') if sub.startswith(param)), '')
            return None

        path = process_query(endpoint, ('/infinite_image_browsing/image-thumbnail', '/infinite_image_browsing/file'), 'path=')
        if path:
            endpoint = f'/file={path}'

        fn = process_query(endpoint, '/sd_extra_networks/thumb', 'filename=')
        if fn:
            endpoint = f'/file={fn}'

        if endpoint.startswith('/file='):
            fp = Path(endpoint[6:])
            ext = fp.suffix.lower().split('?')[0]

            if 'card-no-preview.png' in str(fp):
                return await call_next(req)

            if ext in image_extensions:
                should_resize = str(Models) in str(fp) or str(Embed) in str(fp)
                content = await imgAsync(fp, image_keys, should_resize)
                if content:
                    return Response(content=content, media_type="image/png")
                return await call_next(req)

        return await call_next(req)

def image_encryption_started(_: gr.Blocks, app: FastAPI):
    app.middleware_stack = None
    set_shared_options()
    hook_http_request(app)
    app.build_middleware_stack()

if PILImage.Image.__name__ != 'EncryptedImage':
    super_open = PILImage.open
    super_encode_pil_to_base64 = api.encode_pil_to_base64
    super_modules_images_save_image = images.save_image
    super_api_middleware = api.api_middleware

    if password is not None:
        PILImage.Image = EncryptedImage
        PILImage.open = open
        api.encode_pil_to_base64 = encode_pil_to_base64

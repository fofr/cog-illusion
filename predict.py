from typing import List

from PIL.Image import LANCZOS
from PIL import Image
import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetInpaintPipeline,
)


CACHE_DIR = "hf-cache"


def resize_for_condition_image(input_image, width, height):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(min(width, height)) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=LANCZOS)
    return img


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.controlnet_txt2img_pipe = (
            StableDiffusionControlNetPipeline.from_pretrained(
                CACHE_DIR, torch_dtype=torch.float16
            ).to("cuda")
        )

        self.controlnet_img2img_pipe = (
            StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                CACHE_DIR, torch_dtype=torch.float16
            ).to("cuda")
        )

        self.controlnet_inpainting_pipe = (
            StableDiffusionControlNetInpaintPipeline.from_pretrained(
                CACHE_DIR, torch_dtype=torch.float16
            ).to("cuda")
        )

        self.controlnet_txt2img_pipe.enable_xformers_memory_efficient_attention()
        self.controlnet_img2img_pipe.enable_xformers_memory_efficient_attention()
        self.controlnet_inpainting_pipe.enable_xformers_memory_efficient_attention()

    def resize_image(self, image):
        original_width, original_height = image.size
        print(f"Original dimensions: Width: {original_width}, Height: {original_height}")
        resized_width, resized_height = self.resize_to_allowed_dimensions(original_width, original_height)
        print(f"Resized dimensions: Width: {resized_width}, Height: {resized_height}")
        resized_image = image.resize((resized_width, resized_height))
        return resized_image, resized_width, resized_height

    def get_allowed_dimensions(self, base=512, max_dim=1024):
        """
        Function to generate allowed dimensions optimized around a base up to a max
        """
        allowed_dimensions = []
        for i in range(base, max_dim + 1, 64):
            for j in range(base, max_dim + 1, 64):
                allowed_dimensions.append((i, j))
        return allowed_dimensions

    def resize_to_allowed_dimensions(self, width, height):
        """
        Function adapted from Lucataco's implementation of SDXL-Controlnet for Replicate
        """
        allowed_dimensions = self.get_allowed_dimensions()
        # Calculate the aspect ratio
        aspect_ratio = width / height
        print(f"Aspect Ratio: {aspect_ratio:.2f}")
        # Find the closest allowed dimensions that maintain the aspect ratio
        closest_dimensions = min(
            allowed_dimensions, key=lambda dim: abs(dim[0] / dim[1] - aspect_ratio)
        )
        return closest_dimensions

    # Define the arguments and types the model takes as input
    def predict(
        self,
        prompt: str = Input(default="a painting of a 19th century town"),
        negative_prompt: str = Input(
            description="The negative prompt to guide image generation.",
            default="ugly, disfigured, low quality, blurry, nsfw",
        ),
        num_inference_steps: int = Input(
            description="Number of diffusion steps", ge=20, le=100, default=40
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance",
            default=7.5,
            ge=0.1,
            le=30.0,
        ),
        seed: int = Input(description="Seed", default=-1),
        width: int = Input(default=768),
        height: int = Input(default=768),
        num_outputs: int = Input(
            description="Number of outputs", ge=1, le=4, default=1
        ),
        control_image: Path = Input(
            description="Control image",
            default=None,
        ),
        image: Path = Input(
            description="Optional img2img",
            default=None,
        ),
        mask_image: Path = Input(
            description="Optional mask for inpainting",
            default=None,
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using img2img / inpaint. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.8,
        ),
        controlnet_conditioning_scale: float = Input(
            description="How strong the controlnet conditioning is",
            ge=0.0,
            le=4.0,
            default=2.2,
        ),
    ) -> List[Path]:
        seed = torch.randint(0, 2**32, (1,)).item() if seed == -1 else seed
        if control_image is None:
            raise ValueError("Give an image for prediction")
        else:
            control_image = Image.open(str(control_image))

        # Common arguments
        common_args = dict(
            prompt=[prompt] * num_outputs,
            negative_prompt=[negative_prompt] * num_outputs,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=torch.Generator().manual_seed(seed),
            num_inference_steps=num_inference_steps,
        )

        pipe = self.controlnet_txt2img_pipe
        if mask_image is not None:
            print("Inpainting mode")
            mask_image = Image.open(str(mask_image))
            image = Image.open(str(image))

            image, new_width, new_height = self.resize_image(image)
            mask_image = mask_image.resize((new_width, new_height))
            control_image = control_image.resize((new_width, new_height))

            mode_args = dict(
                width=new_width,
                height=new_height,
                mask_image=[mask_image] * num_outputs,
                image=[image] * num_outputs,
                control_image=[control_image] * num_outputs,
                strength=prompt_strength,
            )

            pipe = self.controlnet_inpainting_pipe
        elif image is not None:
            print("img2img mode")
            image = Image.open(str(image))
            image, new_width, new_height = self.resize_image(image)
            control_image = control_image.resize((new_width, new_height))

            mode_args = dict(
                width=new_width,
                height=new_height,
                image=[image] * num_outputs,
                control_image=[control_image] * num_outputs,
                strength=prompt_strength,
            )

            pipe = self.controlnet_img2img_pipe
        else:
            print("txt2img mode")
            control_image = control_image.resize((width, height))
            mode_args = dict(
                width=width,
                height=height,
                image=[control_image] * num_outputs,
            )

        out = pipe(**common_args, **mode_args)

        outputs = []
        for i, image in enumerate(out.images):
            fname = f"output-{i}.png"
            image.save(fname)
            outputs.append(Path(fname))

        return outputs

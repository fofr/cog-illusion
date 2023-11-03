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
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=torch.Generator().manual_seed(seed),
            num_inference_steps=num_inference_steps,
        )

        pipe = self.controlnet_txt2img_pipe
        if mask_image is not None:
            mask_image = Image.open(str(mask_image))
            image = Image.open(str(image))

            mode_args = dict(
                mask_image=[mask_image] * num_outputs,
                image=[image] * num_outputs,
                control_image=[control_image] * num_outputs,
                strength=prompt_strength,
            )

            print("Inpainting mode")
            pipe = self.controlnet_inpainting_pipe
        elif image is not None:
            image = Image.open(str(image))

            mode_args = dict(
                image=[image] * num_outputs,
                control_image=[control_image] * num_outputs,
                strength=prompt_strength,
            )

            print("img2img mode")
            pipe = self.controlnet_img2img_pipe
        else:
            mode_args = dict(
                image=[control_image] * num_outputs,
            )

            print("txt2img mode")

        out = pipe(**common_args, **mode_args)

        outputs = []
        for i, image in enumerate(out.images):
            fname = f"output-{i}.png"
            image.save(fname)
            outputs.append(Path(fname))

        return outputs

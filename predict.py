from typing import List

from PIL.Image import LANCZOS
from PIL import Image
import torch
from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionControlNetPipeline, EulerDiscreteScheduler


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
        # torch.backends.cuda.matmul.allow_tf32 = True
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            CACHE_DIR, torch_dtype=torch.float16
        ).to("cuda")
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.enable_xformers_memory_efficient_attention()

    # Define the arguments and types the model takes as input
    def predict(
        self,
        prompt: str = Input(
            default="a painting of a 19th century town"
        ),
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
        image: Path = Input(
            description="Input image",
            default=None,
        ),
        controlnet_conditioning_scale: float = Input(
            description="How strong the controlnet conditioning is",
            ge=0.0,
            le=4.0,
            default=2.2,
        ),
    ) -> List[Path]:
        seed = torch.randint(0, 2**32, (1,)).item() if seed == -1 else seed
        if image is None:
            raise ValueError("Give an image for prediction")
        else:
            image = Image.open(str(image))
        out = self.pipe(
            prompt=[prompt] * num_outputs,
            negative_prompt=[negative_prompt] * num_outputs,
            image=[image] * num_outputs,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=torch.Generator().manual_seed(seed),
            num_inference_steps=num_inference_steps,
        )

        outputs = []
        for i, image in enumerate(out.images):
            fname = f"output-{i}.png"
            image.save(fname)
            outputs.append(Path(fname))

        return outputs

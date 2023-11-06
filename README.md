# cog-illusion

[![Replicate](https://replicate.com/fofr/illusions/badge)](https://replicate.com/fofr/illusions)

This is an implementation of [Monster Labs' QR code control net](https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run python script/download-weights

Then, you can run predictions:

    cog predict -i prompt="..." -i image=@...

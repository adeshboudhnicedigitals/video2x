# Discord guide: upscaling anime video to 4K

Source: pasted by the user on 2026-10-09 from a Discord server that upscales video to 4K. The author is not named. The text below is the guide as pasted (two posts), kept verbatim apart from Markdown headings. It is a third-party opinion, not something we measured. See `spike_tensorrt_kaggle.ipynb` and `docs/hypotheses.md` for our own numbers.

## Post 1: tools and hardware

This FAQ on the OpenModelDB sums it up well: https://openmodeldb.info/docs/faq

I recommend people try chaiNNer and AnimeJanai first. They're both free. However, please experiment with all of them in the FAQ list. There could be better ones out there I haven't tried yet. 90% of the models you need are in this guide within OpenModelDB.

chaiNNer can be used for quickly testing multiple different models or for upscaling entire videos. All models must be imported into this app from the OpenModelDB or elsewhere. You can also convert PyTorch -> ONNX models here which is important for use in AnimeJanai. https://chainner.app/download

AnimeJanai is a free and fairly simple GUI for upscaling video. It's one of the fastest upscalers I've tested besides Topaz. It comes with preinstalled models you can swap between. You can also import new models (ONNX only). If I'm testing new models, I'll use chaiNNer to test several combinations, convert them to onnx if needed, then copy paste them over into Janai. Its TensorRT implementation is generally faster than chaiNNer from what I've seen. https://github.com/the-database/AnimeJaNaiConverterGui (try version 0.0.9 if you run into issues)

Real Video Enhancer is a GUI that works exclusively with PyTorch models. The preinstalled models are quite good. Also use this if you aren't able to convert PyTorch models to ONNX for AnimeJanai. https://github.com/TNTwise/REAL-Video-Enhancer

### Recommended Models

This is a list of 300+ models to try lol https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/tag/models

See the post below this for my full Model description. I ran out of space here lol.

### Hardware requirements

Something with Tensor cores. Anything Nvidia 20 series or newer. Vram is pretty important too for the initial inference. I have 12Gb of vram and maybe 20% of models I test don't work. I haven't tested anything with AMD or Intel GPUs. Assume render speeds will be slower than Nvidia cards.

I use these apps below. They're not required but they're free.

- Davinci Resolve: https://www.blackmagicdesign.com/products/davinciresolve
- Handbrake: https://handbrake.fr/
- FFmpeg: https://www.ffmpeg.org/download.html
- Subtitle Edit: https://www.nikse.dk/subtitleedit

I'll continue to update this guide as I see fit.

## Post 2: recommended models and multi-pass workflow

### Recommended Models

This is a list of 300+ models to try lol https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/tag/models

In general, I believe the AnimeJanai models are the best if you don't mind running multiple passes. There could be better models out there that can accomplish more with less upscale passes.

I usually use one of these four models for the first upscale pass. The first pass is the most important since it builds the foundation. Think of it like a skill tree. Three of them are Janai models.

Google Drive folder with the models: https://drive.google.com/drive/folders/1kJx9HL-Td56L_q7RbtDcQ-UYICuxQm2v?usp=sharing

Note: When I refer to "details", I'm referring to the texture of the backgrounds like trees, clouds, grass, rocks, buildings, sidewalks, etc.

- **V1beta34 (beta):** Beta works well with most resolutions and art styles. Its detail boost in addition to its sharpening is very impressive.
- **V2_Compact (v2):** Stronger sharpener than beta but doesn't boost details as much. v2 can usually stand on it's own without needing additional passes. It can still benefit from running a second pass with a model of your choosing. Watch out for over-sharpening.
- **HD_V3Sharp1 (hdsharp):** It's a very precise, consistent, and generally lowkey sharpener. It can thin lines better but has less detail boosting than beta.
- **animefilm-light (anifilm 1.0):** Works with basically all anime old or new. 1980s-2020s. Good detail preservation. I used this for all Naruto and Avatar upscales.
- **animefilm 1.5:** Works more specifically with older anime. It's faster and can sometimes thin lines better than version 1.0. It's a bit better working with lower quality footage. Beware of tiling issues, they are lines that run vertical during playback. There's an anifilm 2.0 I haven't tried yet.

### What's with all this "first pass" talk?

I tend to use 2-4 different models back to back for most of my upscales. As I upscale, I resize them down to 1080p, 1440p, or 2160p before upscaling again. Lots of factors go into determining how far to downscale the image.

Everything mentioned above assumes you have decent quality source footage. If you have poor footage and want to clean it up, that's a whole separate set of models. To start, just try using "realesr-animevideov3". I've included it in the Gdrive link above. If you have trouble upscaling with animejanai, please try downloading an older version like version 0.0.9. There's also an animejanai discord server.

## Notes for this project (added when saved)

- Tools named here that match our TensorRT idea: AnimeJaNaiConverterGui (ONNX, TensorRT), chaiNNer (PyTorch to ONNX conversion), VSGAN-tensorrt-docker (300+ models in one release).
- The guide's multi-pass idea (upscale, resize down to 1080p/1440p/2160p, upscale again with another model) is a different workflow from our single x2 pass followed by a resize.
- Hardware claim: tensor cores needed (NVIDIA 20 series or newer). A Kaggle T4 qualifies; the RTX PRO 6000 Blackwell does too.
- Model files are on a Google Drive folder and a GitHub release, not in this repo.

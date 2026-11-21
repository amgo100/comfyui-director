# ComfyUI Director

A timeline for building long videos in ComfyUI from **Qwen Image 2.1** keyframes and **Wan** clips, inspired by LTX Director.
It adds a full-screen **Director** panel (sidebar icon, floating button, or `Alt+D`) and a small set of `Director*` nodes.
The panel never asks you to wire nodes: it builds the right graph for each step, queues it on your normal ComfyUI queue and files the result back into the project.

## Install

1. Copy the `comfyui-director` folder into `ComfyUI/custom_nodes/` and restart ComfyUI. It has no extra pip dependencies (it uses PyAV, Pillow and torch, which ComfyUI already ships with).
2. Copy `config.example.json` to `config.json` and change the model file names if yours differ. These are the defaults for every project; individual projects can override them in **Settings → Models**. The defaults match the Comfy-Org repackaged files:

| Purpose | Folder | Default file |
|---|---|---|
| Keyframes: text→image, edit, inpaint (one model) | `diffusion_models` | `qwen_image_2.1_int8_convrot.safetensors` (or `qwen_image_2.1_bf16`) |
| Qwen Image 2.1 text encoder, VAE | `text_encoders`, `vae` | `qwen3vl_8b_int8_convrot.safetensors`, `qwen_image_2.1_vae_bf16.safetensors` |
| Fast keyframes (optional) | `loras` | `qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors` (official 8-step Turbo LoRA) |
| Image→video, first/last frame | `diffusion_models` | `wan2.2_i2v_{high,low}_noise_14B_fp8_scaled.safetensors` |
| Text→video | `diffusion_models` | `wan2.2_t2v_{high,low}_noise_14B_fp8_scaled.safetensors` |
| Context bridging and video inpaint | `diffusion_models` | `wan2.1_vace_14B_fp16.safetensors` |
| Wan text encoder, VAE | `text_encoders`, `vae` | `umt5_xxl_fp8_e4m3fn_scaled.safetensors`, `wan_2.1_vae.safetensors` |
| Fast video (optional) | `loras` | Wan 2.2 lightx2v 4-step (high/low), lightx2v I2V distill for VACE |
| Interpolate (optional) | `frame_interpolation` | `film_net_fp16.safetensors` (or a RIFE model) |
| Final video upscaling | `upscale_models` | `4x-UltraSharp.pth` (any ESRGAN-family model ComfyUI can load) |

Keyframes use 25 steps at cfg 1 (euler/simple), or 8 steps with the Turbo LoRA. At cfg 1 ComfyUI skips the negative prompt; raise `qwen21.quality.cfg` (2 at most) if you want it to count. To use the older Qwen-Image-2512 + Qwen-Image-Edit-2511 pair instead, set `"image_model": "qwen"`; that path is kept unchanged in the `qwen` section.

For a two-stage VACE model (for example Wan 2.2 Fun VACE), set both `wan.vace.high` and `wan.vace.low`. All other graphs pick up the change automatically.

## Wan 2.2 Fun models

Four Wan 2.2 Fun A14B models (two-stage, high/low noise) extend what a clip can do. They are set up in **Settings → Models**, and file names default to the Comfy-Org repackaged fp8 files.

| Model | Where it appears | What it does |
|---|---|---|
| **Fun InP** | Models tab → *First/last-frame clips use: Wan 2.2 Fun InP* | A dedicated start/end-frame model (`WanFunInpaintToVideo`). It replaces Wan 2.2 I2V for every clip with keyframes at both ends (or only an end), and the method shows as *Fun InP*. |
| **Fun Camera** | Clip inspector → *Camera & motion control → Camera move* | Starts from the clip's start frame (a keyframe or the previous clip's last frame) and makes a camera move: static, pan up/down/left/right, zoom in/out, or rotate clockwise/anticlockwise, at an adjustable speed. Uses `WanCameraEmbedding` and `WanCameraImageToVideo`. An end frame is ignored. |
| **Fun Control** | Clip inspector → *Camera & motion control → Follow a control video* | Motion and layout follow a control video. The source can be another clip's selected take or an uploaded video (converted to the project's fps), with an optional start offset. Preprocessing is *None* (already depth, pose or edges) or *Canny* (ComfyUI's built-in edge detector). The start frame is passed as the appearance reference (`Wan22FunControlToVideo` `ref_image`), so it shapes the look but is not pinned as frame 1. Using an existing clip with Canny re-renders it in a new style. |
| **Fun VACE** | Models tab → Wan presets → *VACE: Wan 2.2 Fun VACE A14B* | Swaps the context bridging and video inpainting from Wan 2.1 VACE 14B to the two-stage Wan 2.2 Fun VACE. The other preset switches back. |

Camera and control modes take precedence over the clip's method. A clip waits until its start frame and its control clip (if any) have selected takes, and auto-fill follows those dependencies. Fun Camera, Control and InP use the Wan 2.2 I2V lightx2v 4-step LoRAs in fast mode; Fun VACE uses the T2V ones.

## Choosing models per project

**Settings → Models** lets each project use different files from the `config.json` defaults. Every model the pipeline uses is a slot:
- **Keyframes (Qwen Image 2.1):** diffusion model, text encoder, VAE, weight dtype, fast-mode LoRA, steps and CFG. The model family itself can also be switched to the Qwen-Image 2512 / Edit 2511 pair.
- **Video (Wan):** text encoder, VAE, dtypes, and the high- and low-noise models and fast LoRAs for image→video, text→video and VACE, plus steps, CFG and shift.

The dropdowns list what ComfyUI can see, and a default whose file is missing is marked "not found". The loader node is picked from where a file lives:

| File | Loader |
|---|---|
| `models/diffusion_models/…` | `UNETLoader` (with the chosen weight dtype) |
| `models/checkpoints/…` (all-in-one fine-tunes) | `CheckpointLoaderSimple`. Set the text encoder and VAE to *From checkpoint* to use the ones inside it; the checkpoint is loaded once per graph. |
| `*.gguf` in `models/unet` or `models/text_encoders` | `UnetLoaderGGUF` / `CLIPLoaderGGUF`. Needs the ComfyUI-GGUF custom nodes; without them you get a clear error. |

Setting a Wan *low noise* slot to **None** switches that method to single-stage sampling, which suits single-model fine-tunes. If you use a distilled or Turbo model, set its *fast-mode LoRA* to **None** and adjust the steps. Overrides are stored in the project, so other projects keep their own setup.

## Concepts

**Project.** A folder under `ComfyUI/output/director/<project>/` that holds `project.json` plus `keyframes/`, `clips/`, `masks/`, `imports/` and `renders/`. Every generated file stays there, so nothing is lost when you pick a different take.

**Keyframe.** A still image with a *recipe* and a list of *takes*. All keyframe work runs on Qwen Image 2.1, which does text→image and editing in one model and accepts up to 16 reference images.
- *Text → image*: optionally with reference keyframes (character, location, style). The output is made at the keyframe size, which matches the video's aspect ratio.
- *Edit*: `image1` is the keyframe being edited and sets the output size. Further keyframes are references. The prompt refers to them as `<image1>`, `<image2>` and so on, e.g. "the keeper from `<image2>` walks up the path in `<image1>`". This is how keyframes are built from other keyframes.
- *Import*: an uploaded image.
- *Grab*: a frame taken from any clip.

**Project references.** In Settings you can mark keyframes, such as a character sheet or a location plate, as references for every keyframe run (text→image, edit and inpaint). They are appended after the keyframe's own images, and the inspector shows which `<imageN>` each one becomes. This keeps a character consistent across shots without training a LoRA. LoRAs still work and can be combined with references.

**Region edits.** A derived (edit) keyframe can be limited to a painted region of `image1`: *What to change → Only a painted region…* in the inspector. The region is stored with the keyframe, so every generation repaints only that area, and *Edit region…* changes it later. *↳ Derive by painting a region…* (inspector → Improve) does the same in one step from the keyframe you are looking at.

Every run adds takes. One take is ★ selected, and every clip that uses the keyframe uses that take. To improve a keyframe, use *Inpaint region* (under the viewer, or *Improve → 🖌 Inpaint selected take…* in the inspector). Qwen Image 2.1 has no mask input of its own, so the pack crops around the painted area, edits the crop (as `image1`) with a latent noise mask, and stitches it back with a feathered edge. Everything outside the mask stays pixel-exact, and the result is added as another take of the same keyframe.

**Clip (segment).** One shot on the timeline. A clip can be one of three kinds:
- *Generated*: has a start frame, an end frame, a prompt and LoRAs.
- *Imported video*: converted on import to the project's fps and resolution.
- *Still*: a keyframe held for N frames.

The start of a generated clip can be: none, a keyframe, or **continue from the previous clip**. The end can be: none, a keyframe, or **lead into the next clip**.

**Method (auto).** The method is chosen from what is attached to the clip:

| Start | End | Method |
|---|---|---|
| keyframe | keyframe | Wan 2.2 first/last frame (`WanFirstLastFrameToVideo`) |
| keyframe | – | Wan 2.2 image→video |
| – | keyframe | first/last frame with only an end frame |
| – | – | Wan 2.2 text→video |
| previous clip / next clip (with *motion context* on) | any | **VACE bridge** |

With the VACE bridge, the last *N* frames of the previous clip (and/or the first *N* of the next one) go in as fixed context. The model continues the motion instead of restarting from one still frame, which removes most of the speed and direction jumps at clip boundaries. Single keyframes can be added as context frames on either side. You can override the method for any clip.

**Takes and selection.** Generate several takes per run, compare them in the take strip (click to view, double-click or ★ to select) and keep the best one. Downstream clips that continue from this one are generated from the take you selected.

**Progress.** The header shows the running job's stage (loading a model, encoding, sampling step n/m, decoding, saving), an overall percentage, elapsed time, an estimate of the time left, and how many jobs are queued. The percentage weights each node of the graph by its usual share of run time and follows ComfyUI's per-node events, so model loading and decoding count too. Placeholders in the take strip show the same progress for each take being generated.

**Auto-fill.** Queues everything whose inputs are ready, waits, picks up whatever became ready because of those results, and keeps going down the timeline until nothing more can be generated. The first take of each item is auto-selected, so a fully planned timeline can be filled in one pass. Afterwards you can regenerate any clip you don't like.

Auto-fill works through **one model at a time**. Switching between Qwen Image and the different Wan models means reloading many gigabytes of weights. Consecutive jobs that use the same weights reuse the model ComfyUI already has loaded. So each step queues one group of ready jobs that share their diffusion models, chosen in this order:
1. the group whose weights are already loaded;
2. otherwise keyframes, because they unlock clips;
3. otherwise the largest group.

Work that becomes ready in the meantime waits for a later step with the same model. Each item is tried at most once per run: a job that fails is reported and not retried, and the run ends with a summary.

**Stale takes.** Every generated take records which take of each keyframe and clip it was made from: its parent and reference keyframes, its start and end keyframes, the clip it continues from or leads into, and its control clip. If one of those later gets a different selected take, the take is marked **out of date**. That happens when an input is regenerated, inpainted, re-selected or swapped for another keyframe. Clips built from an out-of-date item are marked as *downstream*, since they will be out of date once it is regenerated.
- The header shows **↻ N stale**. Its dialog lists each item with the reason, for example "keyframe 'Lighthouse' has a different take selected".
- Each item can be regenerated (one new take, which replaces the selection) or kept as it is.
- **↻ Update all** runs auto-fill on the stale items. Each regenerated take makes whatever was built from it stale in turn, so the update cascades down the timeline. It is model-grouped like auto-fill, and each item is regenerated at most once per run.
- Old takes are never deleted.
- Projects created before this feature treat their current selection as up to date.
- Prompt edits don't count as stale. Regenerate those clips yourself.

**LoRAs.** Each keyframe has its own LoRA list (Qwen), and each clip has its own LoRA list (Wan). Project-wide image and video LoRAs, for example a character LoRA, are merged into every run. For Wan 2.2, each video LoRA can target the high-noise stage, the low-noise stage or both.

**Video inpaint.** On a clip take, paint a region on the current frame and choose a frame range. Wan VACE regenerates that region over those frames. Everything outside the mask is composited back pixel-exact, the frames are spliced into the original clip, and the result is saved as a new take.

**Render.** *▶ Render draft* streams all selected takes into one mp4 at the project size. Duplicate boundary frames are dropped where one clip starts on the frame the previous one ended on, and trims are applied. *✨ Final* makes the delivery version (see [Final video](#final-video-upscaling-to-1080p)). *Interpolate ×2* on a draft render runs FILM on the whole render; for the final video, set the frame rate in ✨ Final instead.

## Typical session

1. **New project** → **Settings**: set the resolution (832×480 or 1280×720), the style text that is added to every prompt, and any project-wide LoRAs.
2. Make the anchor keyframes. Import or generate a character sheet and mark it as a project reference in Settings. Use *Text→Image* for the first shot, then *Derive* for later shots so the location and character stay consistent. Generate 2–4 takes each and select the best.
3. Build the timeline. You can drag a keyframe onto the left or right half of a clip, use *+ Clip starting here* from a keyframe, use *+ Extend* (continues from the previous clip), or *Import video*.
4. Write what happens in each clip, then press **⚡ Auto-fill gaps**, or **Generate takes** on individual clips.
5. Review the clips. Select better takes, inpaint any problem areas, use *Frame → keyframe* to start a new branch from any frame, and *Split* clips that are too long.
6. **▶ Render draft** to check the cut, then **✨ Final** to upscale and render the delivery version.

## Final video: upscaling to 1080p

Clips are generated at the project size (832×480 or 1280×720) because that is where Wan is fast. **✨ Final** in the header upscales the timeline to the delivery size, by default **1920×1080**. The output keeps the project's aspect ratio at the chosen short side (720p, 1080p, 1440p or 2160p); within 3% of 16:9 it snaps to the standard size, so 832×480 becomes 1920×1080 with a thin strip cropped.

Finishing works **per take**, not on the render. Each clip's selected take, and each keyframe held as a still, gets a *finished* version stored with the take together with the settings it was made with. As a result:
- memory stays bounded (one clip at a time, never the whole film);
- re-rendering costs nothing, and picking a different take or changing one clip only re-finishes that clip;
- changing the finishing settings marks existing finished versions as *outdated* (amber **HD** badge), never mixing them silently.

The dialog lists every item with its state. *✨ Finish N* queues everything missing or outdated, and *▶ Render final* writes the mp4. A clip that isn't finished is resized in the render, and the result reports it. In the viewer, *✨ Finish* does a single take, and *Show draft / Show finished* compares the two.

| Method | What it does | Cost |
|---|---|---|
| **Resize only** | Antialiased resize. | None; nothing is queued. |
| **Upscale model** (default) | An ESRGAN-family model (`models/upscale_models`, e.g. 4x-UltraSharp) on every frame, then resized to the output size. Frames go through in small batches, so the 4× intermediate never exists for a whole clip. | Seconds per clip. Fine texture can shimmer slightly, since each frame is upscaled on its own. |
| **Wan refine + upscale model** | Upscales to the refine size (720p by default) and runs a short low-denoise pass (0.3) of the **Wan 2.2 text→video low-noise** model with the clip's prompt and LoRAs. This adds real detail that stays consistent over time. The result is colour-matched back to the draft (`DirectorColorMatch`) and upscaled to the output size. Clips longer than one generation (imports) are refined in overlapping windows that are crossfaded together. | About one fast generation per clip at the refine size (4 steps with the lightx2v LoRA when *Fast video* is on, otherwise 10 steps at CFG 3.5). Refining at 1080p takes roughly 2–3× as long. |

**Frame rate.** The final video can be interpolated ×2–×4 (16 → 32 fps and so on). Interpolation runs per clip, after upscaling. Each clip starts and ends on its original frames, so joins stay exact: a clip that continues the previous one keeps the in-between frames leading into it, and a cut holds the last frame. Every clip is then exactly *m* times its draft length, which keeps timing (and any later soundtrack) in sync with the draft.

**Memory.** Frames are held in RAM as 32-bit floats: about 25 MB per 1080p frame. An 81-frame clip at 1080p needs about 2 GB, or about 6 GB with ×2 interpolation. Interpolating at 4K needs several times that.

Defaults are in `config.json` → `finish` (output short side, method, upscale model, refine size, denoise, model and steps). Each project can override them in **Settings → Finishing** or directly in the ✨ Final dialog.

## Nodes

All graphs use core ComfyUI nodes plus these nodes. Open any queued job from the queue history to inspect or change it by hand.

| Node | Purpose |
|---|---|
| `DirectorLoadImage` / `DirectorLoadMask` / `DirectorLoadVideoFrames` | Load project-relative files. Paths are restricted to the project folder. |
| `DirectorVaceBridge` | Builds `[context before] + [gray frames to generate] + [context after]` with masks, padded to Wan's 4k+1 length. |
| `DirectorVaceInpaint`, `DirectorCompositeFrames` | Masked VACE control video, and an exact composite back onto the original frames. |
| `DirectorColorMatch` | Per-frame gain/offset colour correction measured on reference pixels (context frames, a ring around a mask, or with no mask the whole frame), interpolated over frames without reference. |
| `DirectorMaskCrop`, `DirectorMaskStitch` | Crop-and-stitch inpainting at full model resolution (crops rounded to multiples of 32 for Qwen Image 2.1). |
| `DirectorEmptyLatent` | Empty latent of any channel count and scale; used for Qwen Image 2.1 text→image (64 channels, 1/16) at the video's aspect ratio. |
| `DirectorUpscaleFrames` | Exact-size upscale: an optional upscale model, then an antialiased resize with a centre crop. Runs a few frames at a time. |
| `DirectorPadFrames`, `DirectorFrameRange`, `DirectorJoinFrames` | Pad to Wan's 4k+1 length and trim back; join overlapping refine windows with a crossfade. |
| `DirectorSaveImage`, `DirectorSaveVideo` | Write results and register them as takes. `DirectorSaveVideo` can also trim context frames and splice frames into a source clip. With a `finish:<segment\|keyframe>:<id>:<take>` target, both store the take's finished version instead. |

## HTTP API (used by the panel)

`GET/POST /director/projects`, `GET /director/project/{id}` (with a `finish` status block), `POST …/op` (fine-grained edits, so running jobs never lose results), `POST …/build` (returns API-format prompts; finishing actions are `finish.segment`, `finish.keyframe` and `finish.all`), `GET …/plan` (the next auto-fill step: one model group; `?stale=1` includes stale takes, `?loaded=` is the loaded weights, `?skip=` the ids already tried), `POST …/upload`, `POST …/grab`, `POST …/render` (`{"final": true}` for the finished render), `GET /director/file/{id}/{path}`.

## Limits and notes

- **Colour drift with VACE.** VACE output, especially with a speed LoRA and in warm scenes, can carry a colour cast. `DirectorColorMatch` measures the cast on pixels that must not change and removes it: the ring around the mask in video inpaint, and the context frames when bridging clips. It works well for inpainting (every frame has reference pixels). For bridges it can only measure at the ends, so a cast that appears only mid-clip is reduced, not removed. If a cast remains, try in **Settings → Models**: lower *VACE fast LoRA strength* (e.g. 0.5), turn off *Fast video* for that clip, or set *VACE fill* to 1.0 (the white fill ComfyUI's own VACE templates use). *VACE colour match* sets the correction strength (0 = off).

- Qwen Image 2.1 sampling uses ModelSamplingFlux to reproduce the model's resolution-dependent shift (it is fitted at 1024×1024), so keyframes larger than 1 MP keep the right schedule. Turn this off with `qwen21.resolution_aware_shift`. For edits with many references on a tight VRAM budget, set `qwen21.cache` (`device: cpu`, `dtype: int8`) to add the experimental Qwen Image 2.1 Cache node.
- The 2.1 prompt-enhancer models are not wired in.

- Each generated clip is at most `max_segment_frames` long (81 frames, about 5 s at 16 fps). Longer stretches are built by chaining clips with *Extend* or *Split*. VACE context keeps the motion continuous across the joins.
- The video-inpaint mask is static across the frame range. Paint generously around moving objects. A tracked mask (SAM) would be a natural next step.
- Long chains of image→video clips can drift in colour. Using VACE context instead of single-frame continuation, and re-anchoring on a keyframe every few clips, keeps this under control.

`examples/` contains API-format graphs produced by the panel for each operation, for reference. The `qi21_*` files are the Qwen Image 2.1 keyframe graphs: text→image with and without references, edit with references, and inpaint. The `finish_*` files are the finishing graphs: an upscale model, Wan refine with ×2 interpolation, and a still. The others cover the video side and the legacy Qwen path.

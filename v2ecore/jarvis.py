"""A QA assistant for filling in v2e's input requirements."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from getpass import getuser

import cv2
import questionary

from v2ecore.constants import IMPORTANT_MESSAGE_STYLE


def get_arguments() -> dict[str, Any]:
    """Get arguments for v2e."""

    output_width, output_height = set_output_dimension(
        args.output_width,
        args.output_height,
        args.dvs128,
        args.dvs240,
        args.dvs346,
        args.dvs640,
        args.dvs1024,
        logger,
    )

    arguments = {
        **input_video_arguments(),
        **output_video_arguments(),
        **dvs_config_arguments(),
        **dvs_output_video_arguments(),
        **dvs_events_saving_arguments(),
        **slomo_arguments(),
    }

    if arguments["skip_video_output"]:
        arguments["output_folder"] = None
        arguments["vid_orig"] = None
        arguments["vid_slomo"] = None

    return arguments


def input_video_arguments() -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    v2e_user = getuser()
    questionary.print(f"Welcome to v2e, {v2e_user}!", style=IMPORTANT_MESSAGE_STYLE)

    arguments["user"] = v2e_user

    # Asking about the input video.
    while True:
        input_video = questionary.path(
            "Please select the input video video file or a folder that contains "
            "video frames. (Use Tab to autocomplete)"
        ).ask()

        input_video = Path(input_video).absolute()

        if not input_video.exists():
            questionary.print(
                f"The input video {input_video} does not exist. Please provide "
                "a valid video file path.",
                style=IMPORTANT_MESSAGE_STYLE,
            )
        else:
            break

    arguments["input_video"] = input_video

    # On frame rate.
    confirm = False
    frame_rate = 0.0
    if input_video.is_file():
        try:
            frame_rate = float(cv2.VideoCapture(str(input_video)).get(cv2.CAP_PROP_FPS))

            confirm = questionary.confirm(
                f"The video's frame rate {frame_rate}fps. Would you like to override?"
            ).ask()
        except ValueError:
            pass

    prepend_message = (
        "The video is passed as a folder. " if input_video.is_dir() else ""
    )
    if confirm or input_video.is_dir():
        while True:
            frame_rate = questionary.text(
                f"{prepend_message}Please specify the frame rate."
            ).ask()

            try:
                frame_rate = float(frame_rate)
                if frame_rate <= 0:
                    questionary.print(
                        f"The frame rate {frame_rate} is less than or equal to 0. "
                        "Please provide a valid frame rate.",
                        style=IMPORTANT_MESSAGE_STYLE,
                    )
                else:
                    break
            except ValueError:
                questionary.print(
                    f"The frame rate {frame_rate} is not a valid number. Please "
                    "provide a valid frame rate.",
                    style=IMPORTANT_MESSAGE_STYLE,
                )

    arguments["input_frame_rate"] = frame_rate

    # Finer control on the input video.
    input_slowdown_factor = 1.0
    start_time = None
    stop_time = None
    crop = None
    hdr = False
    confirm = questionary.confirm(
        f"""
v2e provides additional control on the input video. These options are:

- `input_slowdown_factor` ({input_slowdown_factor}): Sets the known slow-motion factor
   of the input video, i.e. how much the video is slowed down, i.e., the ratio of
   shooting frame rate to playback frame rate. `input_slowmotion_factor < 1` for
   sped-up video and `input_slowmotion_factor > 1` for slowmotion video.
   If an input video is shot at 120fps yet is presented as a 30fps video (has
   specified playback frame rate of 30Hz, according to file's FPS setting), then set
   `input_slowdown_factor=4`. It means that each input frame represents
   (1/30)/4 s=(1/120)s. If input is video with intended frame intervals of 1ms that is 
   in AVI file with default 30 FPS playback spec, then use ((1/30)s)*(1000Hz)=33.33333.

- `start_time` ({start_time}): Start at this time in seconds in video. Use None to 
   start at beginning of source video.

- `stop_time` ({stop_time}): Stop at this time in seconds in video. Use None to end 
   at end of source video.

- `crop` ({crop}): Crop input video by (left, right, top, bottom) pixels.
  E.g. CROP=(100, 100, 0, 0) crops 100 pixels from left and right of input frames.

- `hdr` ({hdr}): Treat input video as high dynamic range (HDR) logarithmic, 
  i.e. skip the linlog conversion step.

Would you like to configure these options?"""
    ).ask()

    if confirm:
        # Input slow down factor.
        # Start time and stop time.
        # Crop.
        # HDR.
        pass

    arguments["input_slowdown_factor"] = input_slowdown_factor
    arguments["start_time"] = start_time
    arguments["stop_time"] = stop_time
    arguments["hdr"] = hdr

    if (
        not input_start_time is None
        and not input_stop_time is None
        and is_float(input_start_time)
        and is_float(input_stop_time)
        and input_stop_time <= input_start_time
    ):
        logger.error(
            f"stop time {input_stop_time} must be later than start time {input_start_time}"
        )
        v2e_quit(1)

    return arguments


def output_video_arguments() -> dict[str, Any]:
    arguments: dict[str, Any] = {}

    # Asking about the output video.
    output_folder = questionary.path(
        "Please specify the output folder. (Use Tab to autocomplete)"
    ).ask()

    output_folder = Path(output_folder).absolute()
    output_folder.mkdir(parents=True, exist_ok=True)

    arguments["output_folder"] = output_folder

    output_folder = set_output_folder(
        args.output_folder,
        input_file,
        args.unique_output_folder if not args.overwrite else False,
        args.overwrite,
        args.output_in_place if (not synthetic_input) else False,
        logger,
    )

    # Asking about the output

    # Finer control on the output video.
    avi_frame_rate = 30
    output_in_place = False
    overwrite = False
    unique_output_folder = True
    auto_timestamp_resolution = False
    timestamp_resolution = 10

    arguments["avi_frame_rate"] = avi_frame_rate
    arguments["output_in_place"] = output_in_place
    arguments["overwrite"] = overwrite
    arguments["unique_output_folder"] = unique_output_folder
    arguments["auto_timestamp_resolution"] = auto_timestamp_resolution
    arguments["timestamp_resolution"] = timestamp_resolution

    return arguments


def dvs_config_arguments() -> dict[str, Any]:
    arguments: dict[str, Any] = {}

    # Finer control on the DVS output.
    positive_threshold = 0.2
    negative_threshold = 0.2
    sigma_threshold = 0.03
    cutoff_hz = 300
    leak_rate_hz = 0.01
    shot_noise_rate_hz = 0.001
    photoreceptor_noise = False
    leak_jitter_fraction = 0.1
    noise_rate_cov_decades = 0.1
    refactory_period = 0.0005
    dvs_emulator_seed = 0
    show_dvs_model_state = None
    save_dvs_model_state = None

    if leak_rate_hz > 0 and sigma_thres == 0:
        logger.warning(
            "leak_rate_hz>0 but sigma_thres==0, "
            "so all leak events will be synchronous"
        )

    arguments["positive_threshold"] = positive_threshold
    arguments["negative_threshold"] = negative_threshold
    arguments["sigma_threshold"] = sigma_threshold
    arguments["cutoff_hz"] = cutoff_hz
    arguments["leak_rate_hz"] = leak_rate_hz
    arguments["shot_noise_rate_hz"] = shot_noise_rate_hz
    arguments["photoreceptor_noise"] = photoreceptor_noise
    arguments["leak_jitter_fraction"] = leak_jitter_fraction
    arguments["noise_rate_cov_decades"] = noise_rate_cov_decades
    arguments["refactory_period"] = refactory_period
    arguments["dvs_emulator_seed"] = dvs_emulator_seed
    arguments["show_dvs_model_state"] = show_dvs_model_state
    arguments["save_dvs_model_state"] = save_dvs_model_state

    return arguments


def dvs_output_video_arguments() -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    # DVS output video options.
    dvs_exposure = ""
    dvs_video_file_name = "dvs-video.avi"
    dvs_video_full_scale = 2
    skip_dvs_video_output = False
    no_preview = False

    arguments["dvs_exposure"] = dvs_exposure
    arguments["dvs_video_file_name"] = dvs_video_file_name
    arguments["dvs_video_full_scale"] = dvs_video_full_scale
    arguments["skip_dvs_video_output"] = skip_dvs_video_output
    arguments["no_preview"] = no_preview

    return arguments


def dvs_events_saving_arguments() -> dict[str, Any]:
    arguments: dict[str, Any] = {}

    ddd_output = False
    dvs_h5 = False
    dvs_aedat2 = False
    dvs_text = False

    ddd_output = questionary.confirm("Save DVS events in DDD17 and DDD20 format?").ask()
    dvs_h5 = questionary.confirm("Save DVS events in HDF5 format?").ask()
    dvs_aedat2 = questionary.confirm("Save DVS events in AEDAT2 format?").ask()
    dvs_text = questionary.confirm("Save DVS events in text format?").ask()

    arguments["ddd_output"] = ddd_output
    arguments["dvs_h5"] = dvs_h5
    arguments["dvs_aedat2"] = dvs_aedat2
    arguments["dvs_text"] = dvs_text

    return arguments


def slomo_arguments() -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    # SloMo Options
    disable_slomo = False
    slomo_model = "SuperSloMo39.ckpt"
    batch_size = 8
    save_videos = False
    slomo_stats_plot = False

    arguments["disable_slomo"] = disable_slomo
    arguments["slomo_model"] = slomo_model
    arguments["batch_size"] = batch_size
    arguments["save_videos"] = save_videos
    arguments["slomo_stats_plot"] = slomo_stats_plot

    return arguments

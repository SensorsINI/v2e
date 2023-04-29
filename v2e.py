#!/usr/bin/env python
"""Python code for extracting frames from video file and synthesizing fake DVS events
from this video after SuperSloMo has generated interpolated frames from the original
video frames.

@author: Tobi Delbruck, Yuhuang Hu, Zhe He
@contact: tobi@ini.uzh.ch, yuhuang.hu@ini.uzh.ch, zhehe@student.ethz.ch
"""
# todo refractory period for pixel
from __future__ import annotations

import glob
import importlib
import os
import sys
import time
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch
from engineering_notation import EngNumber as eng  # only from pip
from loguru import logger
from tqdm import tqdm

import v2ecore.desktop as desktop
from v2ecore.base_synthetic_input import base_synthetic_input
from v2ecore.emulator import EventEmulator
from v2ecore.jarvis import get_arguments
from v2ecore.renderer import EventRenderer
from v2ecore.renderer import ExposureMode
from v2ecore.slomo import SuperSloMo
from v2ecore.v2e_args import NO_SLOWDOWN
from v2ecore.v2e_args import v2e_check_dvs_exposure_args
from v2ecore.v2e_utils import all_images
from v2ecore.v2e_utils import check_lowpass
from v2ecore.v2e_utils import ImageFolderReader
from v2ecore.v2e_utils import read_image
from v2ecore.v2e_utils import setup_input_video
from v2ecore.v2e_utils import v2e_quit


# torch device
torch_device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu").type
logger.info(f"torch device is {torch_device}")
if torch_device == "cpu":
    logger.warning(
        "CUDA GPU acceleration of pytorch operations is not available; "
        "see https://pytorch.org/get-started/locally/ "
        "to generate the correct conda install command to enable GPU-accelerated CUDA."
    )


def main() -> None:
    # Get arguments.
    args = get_arguments()

    # Write arguments to a log file.
    #  infofile = write_args_info(args, output_folder, other_args, command_line)

    # Set input file.
    input_video = args["input_video"]
    synthetic_input = args["synthetic_input"]
    input_slowmotion_factor: float = args["input_slowmotion_factor"]

    # Set output video
    timestamp_resolution: float = args["timestamp_resolution"]
    auto_timestamp_resolution: bool = args["auto_timestamp_resolution"]

    if input_video is not None and synthetic_input is not None:
        raise ValueError(
            f"Both `input_video` '{input_video}' and `synthetic_input` "
            f"'{synthetic_input}' are specified - you can only specify one of them."
        )

    if input_video is None and synthetic_input is None:
        raise ValueError(
            f"Both `input_video` '{input_video}' and `synthetic_input` "
            f"'{synthetic_input}' are not provided - you must provide one of them."
        )

    # Set output folder.
    output_folder = args["output_folder"]

    # Set output width and height based on the arguments
    output_height, output_width = args["output_height"], args["output_width"]

    # setup synthetic input classes and method
    synthetic_input_module = None
    synthetic_input_class = None
    synthetic_input_instance: base_synthetic_input | None = None
    synthetic_input_next_frame_method = None
    if synthetic_input is not None:
        try:
            synthetic_input_module = importlib.import_module(synthetic_input)
            if "." in synthetic_input:
                classname = synthetic_input[synthetic_input.rindex(".") + 1 :]
            else:
                classname = synthetic_input
            synthetic_input_class: synthetic_input = getattr(
                synthetic_input_module, classname
            )
            synthetic_input_instance: base_synthetic_input = synthetic_input_class(
                width=output_width,
                height=output_height,
                preview=not args["no_preview"],
                arg_list=other_args,
                avi_path=os.path.join(output_folder, args.vid_orig),
                parent_args=args,
            )  # TODO output folder might not be unique, could write to first output folder

            if not isinstance(synthetic_input_instance, base_synthetic_input):
                logger.error(
                    f"synthetic input instance of {synthetic_input} is of type {type(synthetic_input_instance)}, but it should be a sublass of synthetic_input;"
                    f"there is no guarentee that it implements the necessary methods"
                )
            synthetic_input_next_frame_method = getattr(
                synthetic_input_class, "next_frame"
            )
            synthetic_input_total_frames_method = getattr(
                synthetic_input_class, "total_frames"
            )

            srcNumFramesToBeProccessed = synthetic_input_instance.total_frames()

            logger.info(
                f"successfully instanced {synthetic_input_instance} with method {synthetic_input_next_frame_method}:"
                "{synthetic_input_module.__doc__}"
            )

        except ModuleNotFoundError as e:
            logger.error(f"Could not import {synthetic_input}: {e}")
            v2e_quit(1)
        except AttributeError as e:
            logger.error(f"{synthetic_input} method incorrect?: {e}")
            v2e_quit(1)

    # check to make sure there are no other arguments that might be bogus misspelled arguments in case
    # we don't have synthetic input class to pass these to
    if synthetic_input_instance is None and len(other_args) > 0:
        logger.error(
            f"There is no synthetic input class specified but there are extra arguments {other_args} that are probably incorrect"
        )
        v2e_quit(1)

    num_frames = 0
    srcNumFramesToBeProccessed = 0
    srcDurationToBeProcessed = float("NaN")

    # input file checking
    #  if (not input_file or not os.path.isfile(input_file)
    #      or not os.path.isdir(input_file)) \
    #          and not base_synthetic_input:
    if (
        not args["disable_slomo"]
        and args["auto_timestamp_resolution"] is False
        and timestamp_resolution is None
    ):
        raise ValueError(
            "`auto_timestamp_resolution` is set to False, "
            "then `timestamp_resolution must be set to "
            "some desired DVS event timestamp resolution in seconds, "
            "e.g. 0.01"
        )

    if auto_timestamp_resolution is True and timestamp_resolution is not None:
        logger.info(
            f"`auto_timestamp_resolution=True` and "
            f"`timestamp_resolution={timestamp_resolution}`: "
            f"Limiting automatic upsampling to maximum timestamp interval."
        )

    # Visualization
    avi_frame_rate = args.avi_frame_rate
    dvs_vid = args.dvs_vid
    dvs_vid_full_scale = args.dvs_vid_full_scale

    # Debug feature: if show slomo stats
    slomo_stats_plot = args.slomo_stats_plot

    # DVS exposure
    exposure_mode, exposure_val, area_dimension = v2e_check_dvs_exposure_args(args)
    if exposure_mode == ExposureMode.DURATION:
        dvsFps = 1.0 / exposure_val

    time_run_started = time.time()

    slomo_timestampe_resoltion_s = None

    if synthetic_input is None:
        logger.info("opening video input file " + input_video)

        (
            cap,
            src_fps,
            output_height,
            output_width,
            src_num_frames,
            src_total_duration,
            start_frame,
            stop_frame,
            src_num_frames_to_be_proccessed,
            src_duration_to_be_processed,
            start_time,
            stop_time,
            src_frame_interval_s,
            slowdown_factor,
            slomo_timestampe_resoltion_s,
        ) = setup_input_video(args)

        # the SloMo model, set no SloMo model if no slowdown
        slomo = None
        if not args["disable_slomo"] and (
            args["auto_timestamp_resolution"] or slowdown_factor != NO_SLOWDOWN
        ):
            slomo = SuperSloMo(
                model=args["slomo_model"],
                auto_upsample=args["auto_timestamp_resolution"],
                upsampling_factor=slowdown_factor,
                video_path=args["output_folder"],
                vid_orig=args["vid_orig"],
                vid_slomo=args["vid_slomo"],
                preview=not args["no_preview"],
                batch_size=args["batch_size"],
            )

    if synthetic_input is None and not auto_timestamp_resolution:
        logger.info(
            f"\n events will have timestamp resolution "
            f"{eng(slomo_timestampe_resoltion_s)}s,"
        )
        if exposure_mode == ExposureMode.DURATION and dvsFps > (
            1 / slomo_timestampe_resoltion_s
        ):
            logger.warning(
                f"DVS video frame rate={dvsFps}Hz is larger than "
                f"the effective DVS frame rate of {1/slomo_timestampe_resoltion_s}Hz; "
                "DVS video will have blank frames."
            )

    if synthetic_input is None:
        logger.info(
            f"Source video {str(input_video)} has total {src_num_frames} frames "
            f"with total duration {eng(src_total_duration)}s.\n "
            f"Source video is {eng(src_fps)}fps with slowmotion_factor "
            f"{eng(input_slowmotion_factor)} (frame interval "
            f"{src_frame_interval_s}s),\n"
            f"Will convert {stop_frame-start_frame+1} frames {start_frame} to "
            f"{stop_frame}\n"
            f"(From {start_time}s to {stop_time}s, duration {stop_time-start_time}s)"
        )

        if exposure_mode == ExposureMode.DURATION:
            dvsNumFrames = np.math.floor(
                dvsFps * srcDurationToBeProcessed / input_slowmotion_factor
            )
            dvsDuration = dvsNumFrames / dvsFps
            dvsPlaybackDuration = dvsNumFrames / avi_frame_rate
            start_time = start_frame / srcFps
            stop_time = (
                stop_frame / srcFps
            )  # todo something replicated here, already have start and stop times

            logger.info(
                "v2e DVS video will have constant-duration frames"
                f"at {eng(dvsFps)}fps (accumulation time {eng(1/dvsFps)}s), "
                f"DVS video will have {dvsNumFrames} frames with duration "
                f"{eng(dvsDuration)}s and playback duration {eng(dvsPlaybackDuration)}s"
            )
        elif exposure_mode == ExposureMode.SOURCE:
            logger.info(
                f"v2e DVS video will have constant-duration frames "
                f"at the source video {eng(src_fps)} fps (accumulation time "
                f"{eng(src_frame_interval_s)}s)"
            )
        else:
            logger.info(
                "v2e DVS video will have constant-count frames with {exposure_val} "
                "events."
            )

    num_pixels = output_width * output_height

    hdr: bool = args.hdr
    if hdr:
        logger.info("Treating input as HDR logarithmic video")

    scidvs: bool = args.scidvs
    if scidvs:
        logger.info("Simulating SCIDVS pixel")

    emulator = EventEmulator(
        pos_thres=args["positive_threshold"],
        neg_thres=args["negative_threshold"],
        sigma_thres=args["sigma_threshold"],
        cutoff_hz=args["cutoff_hz"],
        leak_rate_hz=args["leak_rate_hz"],
        shot_noise_rate_hz=args["shot_noise_rate_hz"],
        photoreceptor_noise=args["photoreceptor_noise"],
        leak_jitter_fraction=args["leak_jitter_fraction"],
        noise_rate_cov_decades=args["noise_rate_cov_decades"],
        refractory_period_s=args["refractory_period"],
        seed=args.dvs_emulator_seed,
        output_folder=args["output_folder"],
        dvs_h5=args["dvs_h5"],
        dvs_aedat2=args["dvs_aedat2"],
        dvs_text=args["dvs_text"],
        show_dvs_model_state=args.show_dvs_model_state,
        save_dvs_model_state=args.save_dvs_model_state,
        output_width=output_width,
        output_height=output_height,
        device=torch_device,
        cs_lambda_pixels=args.cs_lambda_pixels,
        cs_tau_p_ms=args.cs_tau_p_ms,
        hdr=hdr,
        scidvs=scidvs,
    )

    if args.dvs_params is not None:
        logger.warning(
            f"--dvs_param={args.dvs_params} option overrides your "
            f"selected options for threshold, threshold-mismatch, "
            f"leak and shot noise rates"
        )
        emulator.set_dvs_params(args.dvs_params)

    eventRenderer = EventRenderer(
        output_path=output_folder,
        dvs_vid=dvs_vid,
        preview=not args["no_preview"],
        full_scale_count=dvs_vid_full_scale,
        exposure_mode=exposure_mode,
        exposure_value=exposure_val,
        area_dimension=area_dimension,
        avi_frame_rate=args.avi_frame_rate,
    )

    if synthetic_input_next_frame_method is not None:
        # array to batch events for rendering to DVS frames
        events = np.zeros((0, 4), dtype=np.float32)
        (fr, fr_time) = synthetic_input_instance.next_frame()
        num_frames += 1
        i = 0
        with tqdm(
            total=synthetic_input_instance.total_frames(), desc="dvs", unit="fr"
        ) as pbar:
            with torch.no_grad():
                while fr is not None:
                    newEvents = emulator.generate_events(fr, fr_time)
                    pbar.update(1)
                    i += 1
                    if (
                        newEvents is not None
                        and newEvents.shape[0] > 0
                        and not args.skip_video_output
                    ):
                        events = np.append(events, newEvents, axis=0)
                        events = np.array(events)
                        if i % args["batch_size"] == 0:
                            eventRenderer.render_events_to_frames(
                                events, height=output_height, width=output_width
                            )
                            events = np.zeros((0, 4), dtype=np.float32)
                    (fr, fr_time) = synthetic_input_instance.next_frame()
                    num_frames += 1
            # process leftover events
            if len(events) > 0 and not args.skip_video_output:
                eventRenderer.render_events_to_frames(
                    events, height=output_height, width=output_width
                )
    else:  # video file folder or (avi/mp4) file input
        # timestamps of DVS start at zero and end with
        # span of video we processed
        srcVideoRealProcessedDuration = (
            stop_time - start_time
        ) / input_slowmotion_factor
        num_frames = srcNumFramesToBeProccessed
        inputHeight = None
        inputWidth = None
        inputChannels = None
        if start_frame > 0:
            logger.info(f"skipping to frame {start_frame}")
            for i in tqdm(range(start_frame), unit="fr", desc="src"):
                if isinstance(cap, ImageFolderReader):
                    if i < start_frame - 1:
                        ret, _ = cap.read(skip=True)
                    else:
                        ret, _ = cap.read()
                else:
                    ret, _ = cap.read()
                if not ret:
                    raise ValueError(
                        "something wrong, got to end of file before "
                        "reaching start_frame"
                    )

        logger.info(f"processing frames {start_frame} to {stop_frame} from video input")

        c_l = 0
        c_r = None
        c_t = 0
        c_b = None
        if args.crop is not None:
            c = args.crop
            if len(c) != 4:
                logger.error(
                    f"--crop must have 4 elements (you specified --crop={args.crop}"
                )
                v2e_quit(1)

            c_l = c[0] if c[0] > 0 else 0
            c_r = -c[1] if c[1] > 0 else None
            c_t = c[2] if c[2] > 0 else 0
            c_b = -c[3] if c[3] > 0 else None
            logger.info(
                f"cropping video by (left,right,top,bottom)=({c_l},{c_r},{c_t},{c_b})"
            )

        with TemporaryDirectory() as source_frames_dir:
            if os.path.isdir(input_video):  # folder input
                inputWidth = cap.frame_width
                inputHeight = cap.frame_height
                inputChannels = cap.frame_channels
            else:
                inputWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                inputHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                inputChannels = 1 if int(cap.get(cv2.CAP_PROP_MONOCHROME)) else 3
            logger.info(
                f"Input video {input_video} has W={inputWidth} x H={inputHeight} "
                f"frames each with {inputChannels} channels"
            )

            if (output_width is None) and (output_height is None):
                output_width = inputWidth
                output_height = inputHeight
                logger.warning(
                    f"output size ({output_width}x{output_height}) was set "
                    "automatically to input video size\n"
                    "Are you sure you want this? "
                    "It might be slow.\n Consider using\n "
                    "    --output_width=346 --output_height=260\n "
                    "to match Davis346."
                )

                # set emulator output width and height for the last time
                emulator.output_width = output_width
                emulator.output_height = output_height

            logger.info(
                f"*** Stage 1/3: "
                f"Resizing {src_num_frames_to_be_proccessed} input frames "
                f"to output size "
                f"(with possible RGB to luma conversion)"
            )
            for inputFrameIndex in tqdm(
                range(src_num_frames_to_be_proccessed), desc="rgb2luma", unit="fr"
            ):
                # read frame
                ret, inputVideoFrame = cap.read()
                num_frames += 1
                if ret == False:
                    logger.warning(f"could not read frame {inputFrameIndex} from {cap}")
                    continue
                if inputVideoFrame is None or np.shape(inputVideoFrame) == ():
                    logger.warning(
                        f"empty video frame number {inputFrameIndex} in {cap}"
                    )
                    continue
                if not ret or inputFrameIndex + start_frame > stop_frame:
                    break

                if args.crop is not None:
                    # crop the frame, indices are y,x, UL is 0,0
                    if c_l + (c_r if c_r is not None else 0) >= inputWidth:
                        logger.error(
                            f"left {c_l}+ right crop {c_r} is larger than image width {inputWidth}"
                        )
                        v2e_quit(1)
                    if c_t + (c_b if c_b is not None else 0) >= inputHeight:
                        logger.error(
                            f"top {c_t}+ bottom crop {c_b} is larger than image height {inputHeight}"
                        )
                        v2e_quit(1)

                    inputVideoFrame = inputVideoFrame[
                        c_t:c_b, c_l:c_r
                    ]  # https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python

                if (
                    output_height
                    and output_width
                    and (inputHeight != output_height or inputWidth != output_width)
                ):
                    dim = (output_width, output_height)
                    (fx, fy) = (
                        float(output_width) / inputWidth,
                        float(output_height) / inputHeight,
                    )
                    inputVideoFrame = cv2.resize(
                        src=inputVideoFrame,
                        dsize=dim,
                        fx=fx,
                        fy=fy,
                        interpolation=cv2.INTER_AREA,
                    )
                if inputChannels == 3:  # color
                    if inputFrameIndex == 0:  # print info once
                        logger.info("\nConverting input frames from RGB color to luma")
                    # TODO would break resize if input is gray frames
                    # convert RGB frame into luminance.
                    inputVideoFrame = cv2.cvtColor(
                        inputVideoFrame, cv2.COLOR_BGR2GRAY
                    )  # much faster

                    # TODO add vid_orig output if not using slomo

                # save frame into numpy records
                save_path = os.path.join(
                    source_frames_dir, str(inputFrameIndex).zfill(8) + ".npy"
                )
                np.save(save_path, inputVideoFrame)
                # print("Writing source frame {}".format(save_path), end="\r")
            cap.release()

            with TemporaryDirectory() as interpFramesFolder:
                interpTimes = None
                # make input to slomo
                if slomo is not None and (
                    auto_timestamp_resolution or slowdown_factor != NO_SLOWDOWN
                ):
                    # interpolated frames are stored to tmpfolder as
                    # 1.png, 2.png, etc

                    logger.info(
                        f"*** Stage 2/3: SloMo upsampling from " f"{source_frames_dir}"
                    )
                    interpTimes, avgUpsamplingFactor = slomo.interpolate(
                        source_frames_dir,
                        interpFramesFolder,
                        (output_width, output_height),
                    )
                    avgTs = src_frame_interval_s / avgUpsamplingFactor
                    logger.info(
                        "SloMo average upsampling factor={:5.2f}; "
                        "average DVS timestamp resolution={}s".format(
                            avgUpsamplingFactor, eng(avgTs)
                        )
                    )
                    # check for undersampling wrt the
                    # photoreceptor lowpass filtering

                    if args["cutoff_hz"] > 0:
                        logger.info(
                            "Using auto_timestamp_resolution. "
                            "checking if cutoff hz is ok given "
                            f"samplee rate {1/avgTs}"
                        )
                        check_lowpass(args["cutoff_hz"], 1 / avgTs, logger)

                    # read back to memory
                    interpFramesFilenames = all_images(interpFramesFolder)
                    # number of frames
                    n = len(interpFramesFilenames)
                else:
                    logger.info(
                        f"*** Stage 2/3:turning npy frame files to png "
                        f"from {source_frames_dir}"
                    )
                    interpFramesFilenames = []
                    n = 0
                    src_files = sorted(
                        glob.glob("{}".format(source_frames_dir) + "/*.npy")
                    )
                    for frame_idx, src_file_path in tqdm(
                        enumerate(src_files), desc="npy2png", unit="fr"
                    ):
                        src_frame = np.load(src_file_path)
                        tgt_file_path = os.path.join(
                            interpFramesFolder, str(frame_idx) + ".png"
                        )
                        interpFramesFilenames.append(tgt_file_path)
                        n += 1
                        cv2.imwrite(tgt_file_path, src_frame)
                    interpTimes = np.array(range(n))

                # compute times of output integrated frames
                nFrames = len(interpFramesFilenames)
                # interpTimes is in units of 1 per input frame,
                # normalize it to src video time range
                f = srcVideoRealProcessedDuration / (
                    np.max(interpTimes) - np.min(interpTimes)
                )
                # compute actual times from video times
                interpTimes = f * interpTimes
                # debug
                if slomo_stats_plot:
                    from matplotlib import pyplot as plt  # TODO debug

                    dt = np.diff(interpTimes)
                    fig = plt.figure()
                    ax1 = fig.add_subplot(111)
                    ax1.set_title("Slo-Mo frame interval stats (close to continue)")
                    ax1.plot(interpTimes)
                    ax1.plot(interpTimes, "x")
                    ax1.set_xlabel("frame")
                    ax1.set_ylabel("frame time (s)")
                    ax2 = ax1.twinx()
                    ax2.plot(dt * 1e3)
                    ax2.set_ylabel("frame interval (ms)")
                    logger.info("close plot to continue")
                    fig.show()

                # array to batch events for rendering to DVS frames
                events = np.zeros((0, 4), dtype=np.float32)

                logger.info(
                    f"*** Stage 3/3: emulating DVS events from " f"{nFrames} frames"
                )

                # parepare extra steps for data storage
                # right before event emulation
                if args.ddd_output:
                    emulator.prepare_storage(nFrames, interpTimes)

                # generate events from frames and accumulate events to DVS frames for output DVS video
                with tqdm(total=nFrames, desc="dvs", unit="fr") as pbar:
                    with torch.no_grad():
                        for i in range(nFrames):
                            fr = read_image(interpFramesFilenames[i])
                            newEvents = emulator.generate_events(fr, interpTimes[i])

                            pbar.update(1)
                            if (
                                newEvents is not None
                                and newEvents.shape[0] > 0
                                and not args.skip_video_output
                            ):
                                events = np.append(events, newEvents, axis=0)
                                events = np.array(events)
                                if i % args["batch_size"] == 0:
                                    eventRenderer.render_events_to_frames(
                                        events, height=output_height, width=output_width
                                    )
                                    events = np.zeros((0, 4), dtype=np.float32)
                    # process leftover events
                    if len(events) > 0 and not args.skip_video_output:
                        eventRenderer.render_events_to_frames(
                            events, height=output_height, width=output_width
                        )

    # Clean up
    eventRenderer.cleanup()
    emulator.cleanup()
    if slomo is not None:
        slomo.cleanup()
    if synthetic_input_instance is not None:
        synthetic_input_instance.cleanup()

    if num_frames == 0:
        logger.error("no frames read from file")

    totalTime = time.time() - time_run_started
    framePerS = num_frames / totalTime
    sPerFrame = totalTime / num_frames if num_frames > 0 else None
    throughputStr = (
        (str(eng(framePerS)) + "fr/s")
        if framePerS > 1
        else (str(eng(sPerFrame)) + "s/fr")
    )
    timestr = (
        f"done processing {num_frames} frames in {eng(totalTime)}s "
        f"({throughputStr})\n **************** see output folder "
        f"{output_folder}"
    )
    logger.info(
        "generated total {eng(emulator.num_events_total)} events "
        "({eng(emulator.num_events_on)} on, "
        "{eng(emulator.num_events_off)} off)"
    )
    total_time = emulator.t_previous
    rate_total = emulator.num_events_total / total_time
    rate_on_total = emulator.num_events_on / total_time
    rate_off_total = emulator.num_events_off / total_time
    rate_per_pixel = rate_total / num_pixels
    rate_on_per_pixel = rate_on_total / num_pixels
    rate_off_per_pixel = rate_off_total / num_pixels
    logger.info(
        f"Avg event rate for N={num_pixels} px and total time ={total_time:.3f} s"
        f"\n\tTotal: {eng(rate_total)}Hz ({eng(rate_on_total)}Hz on, {eng(rate_off_total)}Hz off)"
        f"\n\tPer pixel:  {eng(rate_per_pixel)}Hz ({eng(rate_on_per_pixel)}Hz on, {eng(rate_off_per_pixel)}Hz off)"
    )
    if totalTime > 60:
        try:
            from plyer import notification

            logger.info("generating desktop notification")
            notification.notify(title="v2e done", message=timestr, timeout=3)
        except Exception as e:
            logger.info(f"could not show notification: {e}")

    # try to show desktop
    # suppress folder opening if it's not necessary
    if not args.skip_video_output and not args.no_preview:
        try:
            logger.info(f"showing {output_folder} in desktop")
            desktop.open(os.path.abspath(output_folder))
        except Exception as e:
            logger.warning(f"{e}: could not open '{output_folder}' in desktop")
    logger.info(timestr)
    sys.exit(0)


if __name__ == "__main__":
    main()
    v2e_quit()

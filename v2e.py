#!/usr/bin/env python
"""
Python code for extracting frames from video file and synthesizing fake DVS
events from this video after SuperSloMo has generated interpolated
frames from the original video frames.

@author: Tobi Delbruck, Yuhuang Hu, Zhe He
@contact: tobi@ini.uzh.ch, yuhuang.hu@ini.uzh.ch, zhehe@student.ethz.ch
@latest update: Apr 2020
"""
# todo refractory period for pixel

import glob
import argparse
from pathlib import Path
from shutil import rmtree

import argcomplete
import cv2
import numpy as np
import os
from tempfile import TemporaryDirectory, mkdtemp
from engineering_notation import EngNumber  # only from pip
from scripts.regsetup import description
from tqdm import tqdm
from tqdm import trange
# may only apply to windows
try:
    from gooey import Gooey # pip install Gooey
except Exception:
    pass

import v2e.desktop as desktop
from v2e.v2e_utils import all_images, read_image,  \
    check_lowpass, v2e_quit
from v2e.v2e_args import v2e_args, write_args_info, v2e_check_dvs_exposure_args
from v2e.v2e_args import NO_SLOWDOWN
from v2e.renderer import EventRenderer, ExposureMode
from v2e.slomo import SuperSloMo
from v2e.emulator import EventEmulator
from v2e.v2e_utils import inputVideoFileDialog
import logging

logging.basicConfig()
root = logging.getLogger()
root.setLevel(logging.INFO)
# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output/7995762#7995762
logging.addLevelName(
    logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(
        logging.WARNING))
logging.addLevelName(
    logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(
        logging.ERROR))
logger = logging.getLogger(__name__)


@Gooey(program_name="v2e", default_size=(575, 600))  # uncomment if you are lucky enough to be able to install Gooey, which requires wxPython
def get_args():
    parser = argparse.ArgumentParser(
        description='v2e: generate simulated DVS events from video.',
        epilog='Run with no --input to open file dialog', allow_abbrev=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = v2e_args(parser)
    parser.add_argument(
        "--rotate180", type=bool, default=False,
        help="rotate all output 180 deg.")
    # https://kislyuk.github.io/argcomplete/#global-completion
    # Shellcode (only necessary if global completion is not activated -
    # see Global completion below), to be put in e.g. .bashrc:
    # eval "$(register-python-argcomplete v2e.py)"
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    return args

def makeOutputFolder(output_folder,suffix_counter, overwrite, unique_output_folder):
    if overwrite and unique_output_folder:
        logger.error("specify either --overwrite or --unique_output_folder")
        v2e_quit()
    if suffix_counter>0:
        output_folder=output_folder+'-'+str(suffix_counter)
    nonEmptyFolderExists = not overwrite and os.path.exists(output_folder) \
        and os.listdir(output_folder)
    if nonEmptyFolderExists and not overwrite and not unique_output_folder:
        logger.error(
            'non-empty output folder {} already exists \n '
            '- use --overwrite or --unique_output_folder'.format(os.path.abspath(output_folder), nonEmptyFolderExists))
        v2e_quit()

    if nonEmptyFolderExists and unique_output_folder:
        makeOutputFolder(output_folder, suffix_counter+1, overwrite, unique_output_folder)
    logger.info('making output folder {}'.format(output_folder))
    if not os.path.exists(output_folder): os.mkdir(output_folder)


def main():

    args = get_args()
    overwrite = args.overwrite
    output_folder = args.output_folder
    unique_output_folder = args.unique_output_folder

    makeOutputFolder(output_folder, 0, overwrite, unique_output_folder)

    if (args.output_width is not None) ^ (args.output_width is not None):
        logger.error(
            'provide both or neither of output_width and output_height')
        v2e_quit()
    input_file = args.input
    if not input_file:
        input_file = inputVideoFileDialog()
        if not input_file:
            logger.info('no file selected, quitting')
            v2e_quit()

    output_width: int = args.output_width
    output_height: int = args.output_height
    if (output_width is None) ^ (output_height is None):
        logger.error('set neither or both of output_width and output_height')
        v2e_quit()

    # input file checking
    if not input_file or not Path(input_file).exists():
        logger.error('input file {} does not exist'.format(input_file))
        v2e_quit()

    start_time = args.start_time
    stop_time = args.stop_time

    input_slowmotion_factor = args.input_slowmotion_factor
    timestamp_resolution = args.timestamp_resolution

    pos_thres = args.pos_thres
    neg_thres = args.neg_thres
    sigma_thres = args.sigma_thres
    cutoff_hz = args.cutoff_hz
    leak_rate_hz = args.leak_rate_hz
    if leak_rate_hz > 0 and sigma_thres == 0:
        logger.warning(
            'leak_rate_hz>0 but sigma_thres==0, '
            'so all leak events will be synchronous')
    shot_noise_rate_hz = args.shot_noise_rate_hz
    avi_frame_rate = args.avi_frame_rate
    dvs_vid = args.dvs_vid
    dvs_vid_full_scale = args.dvs_vid_full_scale
    dvs_h5 = args.dvs_h5
    # dvs_np = args.dvs_np
    dvs_aedat2 = args.dvs_aedat2
    dvs_text = args.dvs_text
    vid_orig = args.vid_orig
    vid_slomo = args.vid_slomo

    preview = not args.no_preview
    rotate180 = args.rotate180
    batch_size = args.batch_size

    exposure_mode, exposure_val, area_dimension = \
        v2e_check_dvs_exposure_args(args)

    infofile = write_args_info(args, output_folder)

    fh = logging.FileHandler(infofile)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    import time
    time_run_started = time.time()

    logger.info("opening video input " + input_file)

    cap = cv2.VideoCapture(input_file)
    srcFps = cap.get(cv2.CAP_PROP_FPS)
    if srcFps == 0:
        logger.error('source {} fps is 0'.format(input_file))
        v2e_quit()

    srcFrameIntervalS = (1. / srcFps)/input_slowmotion_factor

    slowdown_factor = int(np.ceil(srcFrameIntervalS/timestamp_resolution))
    if slowdown_factor < 1:
        slowdown_factor = 1
        logger.warning(
            'timestamp resolution={}s is greater than source '
            'frame interval={}s, will not use upsampling'
            .format(timestamp_resolution, srcFrameIntervalS))

    logger.info(
        'src video frame rate={:.2f} Hz with slowmotion_factor={:.2f}, '
        'timestamp resolution={:.3f} ms, computed slomo upsampling factor={}'
        .format(
            srcFps, input_slowmotion_factor, timestamp_resolution*1000,
            slowdown_factor))

    slomoTimestampResolutionS = srcFrameIntervalS / slowdown_factor
    # https://stackoverflow.com/questions/25359288/how-to-know-total-number-of-frame-in-a-file-with-cv2-in-python
    srcNumFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if srcNumFrames < 2:
        logger.warning(
            'num frames is less than 2, probably cannot be determined '
            'from cv2.CAP_PROP_FRAME_COUNT')

    if slomoTimestampResolutionS > timestamp_resolution:
        logger.warning(
            'upsampled src frame intervals of {}s is larger than '
            'the desired DVS timestamp resolution of {}s'
            .format(slomoTimestampResolutionS, timestamp_resolution))

    check_lowpass(cutoff_hz, 1/slomoTimestampResolutionS, logger)

    # the SloMo model, set no SloMo model if no slowdown
    if slowdown_factor != NO_SLOWDOWN:
        slomo = SuperSloMo(
            model=args.slomo_model, slowdown_factor=slowdown_factor,
            video_path=output_folder, vid_orig=vid_orig, vid_slomo=vid_slomo,
            preview=preview, batch_size=batch_size)
    else:
        slomo = None

    srcTotalDuration = (srcNumFrames - 1) / srcFps
    start_frame = int(srcNumFrames * (start_time / srcTotalDuration)) \
        if start_time else 0
    stop_frame = int(srcNumFrames * (stop_time / srcTotalDuration)) \
        if stop_time else srcNumFrames
    srcNumFramesToBeProccessed = stop_frame-start_frame+1
    srcDurationToBeProcessed = srcNumFramesToBeProccessed/srcFps
    start_time=start_frame/srcFps
    stop_time=stop_frame/srcFps

    if exposure_mode == ExposureMode.DURATION:
        dvsFps = 1./exposure_val
        dvsNumFrames = np.math.floor(dvsFps * srcDurationToBeProcessed/input_slowmotion_factor)
        dvsDuration = dvsNumFrames / dvsFps
        dvsPlaybackDuration = dvsNumFrames / avi_frame_rate
        logger.info(
            '\n\n{} has {} frames with duration {}s, '
            '\nsource video is {}fps with slowmotion_factor {} (frame interval {}s),'
            '\n slomo will have {}fps,'
            '\n events will have timestamp resolution {}s,'
            '\n v2e DVS video will have {}fps (accumulation time {}s), '
            '\n DVS video will have {} frames with duration {}s '
            'and playback duration {}s\n'
            .format(input_file, srcNumFrames, EngNumber(srcTotalDuration),
                    EngNumber(srcFps), EngNumber(input_slowmotion_factor), EngNumber(srcFrameIntervalS),
                    EngNumber(srcFps * slowdown_factor),
                    EngNumber(slomoTimestampResolutionS),
                    EngNumber(dvsFps), EngNumber(1 / dvsFps),
                    dvsNumFrames, EngNumber(dvsDuration),
                    EngNumber(dvsPlaybackDuration)))
        if dvsFps > (1/slomoTimestampResolutionS):
            logger.warning(
                'DVS video frame rate={}Hz is larger than '
                'the effective DVS frame rate of {}Hz; '
                'DVS video will have blank frames'.format(
                    dvsFps, (1/slomoTimestampResolutionS)))
    else:
        logger.info(
            '\n\n{} has {} frames with duration {}s, '
            '\nsource video is {}fps (frame interval {}s),'
            '\n slomo will have {}fps,'
            '\n events will have timestamp resolution {}s,'
            '\n v2e DVS video will have constant count '
            'frames with {} events), '
            .format(input_file, srcNumFrames, EngNumber(srcTotalDuration),
                    EngNumber(srcFps), EngNumber(srcFrameIntervalS),
                    EngNumber(srcFps * slowdown_factor),
                    EngNumber(slomoTimestampResolutionS),
                    exposure_val))

    emulator = EventEmulator(
        pos_thres=pos_thres, neg_thres=neg_thres,
        sigma_thres=sigma_thres, cutoff_hz=cutoff_hz,
        leak_rate_hz=leak_rate_hz, shot_noise_rate_hz=shot_noise_rate_hz,
        output_folder=output_folder, dvs_h5=dvs_h5, dvs_aedat2=dvs_aedat2,
        dvs_text=dvs_text)

    if args.dvs_params:
        emulator.set_dvs_params(args.dvs_params)

    eventRenderer = EventRenderer(
        output_path=output_folder,
        dvs_vid=dvs_vid, preview=preview, full_scale_count=dvs_vid_full_scale,
        exposure_mode=exposure_mode,
        exposure_value=exposure_val,
        area_dimension=area_dimension)

    # timestamps of DVS start at zero and end with span of video we processed
    ts0 = 0
    ts1 = (stop_time-start_time)/input_slowmotion_factor
    num_frames = srcNumFramesToBeProccessed
    inputHeight = None
    inputWidth = None
    inputChannels = None
    if start_frame > 0:
        logger.info('skipping to frame {}'.format(start_frame))
        for i in tqdm(range(start_frame), unit='fr', desc='src'):
            ret, _ = cap.read()
            if not ret:
                raise ValueError(
                    'something wrong, got to end of file before '
                    'reaching start_frame')

    logger.info(
        'processing frames {} to {} from video input'.format(
            start_frame, stop_frame))

    with TemporaryDirectory() as source_frames_dir:
        inputWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        inputHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        inputChannels = 3

        if (output_width is None) and (output_height is None):
            output_width = inputWidth
            output_height = inputHeight
            logger.warning(
                'output size ({}x{}) was set automatically to '
                'input video size\n    Are you sure you want this? '
                'It might be slow.\n    Consider using '
                '--output_width and --output_height'
                .format(output_width, output_height))

        logger.info('Resizing input frames to output size (with possible RGG to luma conversion)')
        for inputFrameIndex in tqdm(range(srcNumFramesToBeProccessed),desc='rgb2luma',unit='fr'):
                # read frame
                ret, inputVideoFrame = cap.read()

                if not ret or inputFrameIndex+start_frame > stop_frame:
                    break

                if output_height and output_width and \
                        (inputHeight != output_height or
                         inputWidth != output_width):
                    dim = (output_width, output_height)
                    (fx, fy) = (float(output_width)/inputWidth,
                                float(output_height)/inputHeight)
                    inputVideoFrame = cv2.resize(
                        src=inputVideoFrame, dsize=dim, fx=fx, fy=fy,
                        interpolation=cv2.INTER_AREA)
                if inputChannels == 3:  # color
                    if inputFrameIndex == 0:  # print info once
                        logger.info(
                            '\nConverting input frames from RGB color to luma')
                    # TODO would break resize if input is gray frames
                    # convert RGB frame into luminance.
                    inputVideoFrame = cv2.cvtColor(
                        inputVideoFrame, cv2.COLOR_BGR2GRAY)  # much faster

                # save frame into numpy records
                save_path = os.path.join(
                    source_frames_dir, str(inputFrameIndex).zfill(8)+".npy")
                np.save(save_path, inputVideoFrame)
                # print("Writing source frame {}".format(save_path), end="\r")
        cap.release()

        with TemporaryDirectory() as interpFramesFolder:
            # make input to slomo
            if slowdown_factor != NO_SLOWDOWN:
                # interpolated frames are stored to tmpfolder as
                # 1.png, 2.png, etc
                slomo.interpolate(
                    source_frames_dir, interpFramesFolder,
                    (output_width, output_height))
                # read back to memory
                interpFramesFilenames = all_images(interpFramesFolder)
            else:
                logger.info('turning npy frame files to png from {}'.format(source_frames_dir))
                interpFramesFilenames = []
                src_files = sorted(
                    glob.glob("{}".format(source_frames_dir)+"/*.npy"))
                for frame_idx, src_file_path in tqdm(enumerate(src_files),desc='npy2png',unit='fr'):
                    src_frame = np.load(src_file_path)
                    tgt_file_path = os.path.join(
                        interpFramesFolder, str(frame_idx)+".png")
                    interpFramesFilenames.append(tgt_file_path)
                    cv2.imwrite(tgt_file_path, src_frame)

            # number of frames
            n = len(interpFramesFilenames)

            # compute times of output integrated frames
            interpTimes = np.linspace(
                start=ts0, stop=ts1, num=n, endpoint=False)

            # interpolate events
            # get some progress bar
            events = np.zeros((0, 4), dtype=np.float32)
            num_batches = (n // (slowdown_factor*batch_size))+1

            with tqdm(total=num_batches * slowdown_factor*batch_size, desc='dvs',unit='fr') as pbar:
                for batch_idx in (range(num_batches)):
                    # events = np.zeros((0, 4), dtype=np.float32)
                    for sub_img_idx in range(slowdown_factor*batch_size):
                        image_idx = batch_idx*(slowdown_factor*batch_size)+sub_img_idx
                        # at the end of the file
                        if image_idx > n-1:
                            break
                        fr = read_image(interpFramesFilenames[image_idx])
                        newEvents = emulator.generate_events(
                            fr, interpTimes[image_idx])
                        # events = np.array(events)  # remove first None element
                        eventRenderer.render_events_to_frames(
                            newEvents, height=output_height, width=output_width)
                        # if newEvents is not None and newEvents.shape[0] > 0:
                        #     events = np.append(events, newEvents, axis=0)
                        pbar.update(1)

    if num_frames == 0:
        logger.error('no frames read from file')
        v2e_quit()
    totalTime = (time.time() - time_run_started)
    framePerS = num_frames/totalTime
    sPerFrame = 1/framePerS
    throughputStr = (str(EngNumber(framePerS))+'fr/s') \
        if framePerS > 1 else (str(EngNumber(sPerFrame))+'s/fr')
    logger.info(
        'done processing {} frames in {}s ({})\n see output folder {}'
        .format(num_frames,
                EngNumber(totalTime),
                throughputStr,
                output_folder))
    logger.info('generated total {} events ({} on, {} off)'
                .format(EngNumber(emulator.num_events_total),
                        EngNumber(emulator.num_events_on),
                        EngNumber(emulator.num_events_off)))
    logger.info(
        'avg event rate {}Hz ({}Hz on, {}Hz off)'
        .format(
            EngNumber(emulator.num_events_total/srcDurationToBeProcessed),
            EngNumber(emulator.num_events_on/srcDurationToBeProcessed),
            EngNumber(emulator.num_events_off/srcDurationToBeProcessed)))
    try:
        desktop.open(os.path.abspath(output_folder))
    except Exception as e:
        logger.warning(
            '{}: could not open {} in desktop'.format(e, output_folder))
    eventRenderer.cleanup()
    if slomo is not None:
        slomo.cleanup()

main()
v2e_quit()


import argparse
import os
import logging

from v2ecore.renderer import ExposureMode

logger = logging.getLogger(__name__)

# there is no slow down when slowdown_factor = 1
NO_SLOWDOWN = 1


def expandpath(path):
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def output_file_check(arg):
    if arg.lower() == "none":
        return None
    return arg


# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got {v}')


def v2e_args(parser):
    # check and add prefix if running script in subfolder
    dir_path = os.getcwd()
    if dir_path.endswith('ddd'):
        prepend = '../../'
    else:
        prepend = ''

    # general arguments for output folder, overwriting, etc
    outGroupGeneral = parser.add_argument_group('Output: General')
    outGroupGeneral.add_argument(
        "-o", "--output_folder", type=expandpath, default='v2e-output',
        help="folder to store outputs.")
    outGroupGeneral.add_argument(
        "--avi_frame_rate", type=int, default=30,
        help="frame rate of output AVI video files; "
             "only affects playback rate. ")

    outGroupGeneral.add_argument(
        "--output_in_place", default=True, type=str2bool,
        const=True, nargs='?',
        help="store output files in same folder as source video.")
    outGroupGeneral.add_argument(
        "--overwrite", action="store_true",
        help="overwrites files in existing folder "
             "(checks existence of non-empty output_folder).")
    outGroupGeneral.add_argument(
        "--unique_output_folder", default=True, type=str2bool,
        const=True, nargs='?',
        help="If specifying --output_folder, makes unique output "
             "folder based on output_folder, e.g. output1 "
             "(if non-empty output_folder already exists)")

    # timestamp resolution
    timestampResolutionGroup = parser.add_argument_group(
        'DVS timestamp resolution')
    timestampResolutionGroup.add_argument(
        "--auto_timestamp_resolution", default=True, type=str2bool,
        const=True, nargs='?',
        help="(Ignored by --disable_slomo.) "
             "\nIf True (default), upsampling_factor is automatically "
             "determined to limit maximum movement between frames to 1 pixel."
             "\nIf False, --timestamp_resolution sets the upsampling factor "
             "for input video."
             "\nCan be combined with --timestamp_resolution to "
             "ensure DVS events have at most some resolution.")
    timestampResolutionGroup.add_argument(
        "--timestamp_resolution", type=float,
        help="(Ignored by --disable_slomo.) "
             "Desired DVS timestamp resolution in seconds; "
             "determines slow motion upsampling factor;  "
             "the video will be upsampled from source fps to "
             "achieve the at least this timestamp resolution."
             "I.e. slowdown_factor = (1/fps)/timestamp_resolution; "
             "using a high resolution e.g. of 1ms will result in slow "
             "rendering since it will force high upsampling ratio."
             "\nCan be combind with --auto_timestamp_resolution to "
             "limit upsampling to a maximum limit value.")

    # DVS model parameters
    modelGroup = parser.add_argument_group('DVS model')
    modelGroup.add_argument(
        "--output_height", type=int, default=None,
        help="Height of output DVS data in pixels. "
             "If None, same as input video. "
             "Use --output_height=346 for Davis346.")
    modelGroup.add_argument(
        "--output_width", type=int, default=None,
        help="Width of output DVS data in pixels. "
             "If None, same as input video. "
             "Use --output_width=260 for Davis346.")

    modelGroup.add_argument(
        "--dvs_params", type=str, default=None,
        help="Easy optional setting of parameters for DVS model:"
             "None, 'clean', 'noisy'; 'clean' turns off noise and "
             "makes threshold variation zero. 'noisy' sets "
             "limited bandwidth and adds leak events and shot noise."
             "This option by default will disable user set "
             "DVS parameters. To use custom DVS paramters, "
             "use None here.")
    modelGroup.add_argument(
        "--pos_thres", type=float, default=0.2,
        help="threshold in log_e intensity change to "
             "trigger a positive event.")
    modelGroup.add_argument(
        "--neg_thres", type=float, default=0.2,
        help="threshold in log_e intensity change to "
             "trigger a negative event.")
    modelGroup.add_argument(
        "--sigma_thres", type=float, default=0.03,
        help="1-std deviation threshold variation in log_e intensity change.")

    modelGroup.add_argument(
        "--cutoff_hz", type=float, default=300,
        help="photoreceptor IIR lowpass filter "
             "cutoff-off 3dB frequency in Hz - "
             "see https://ieeexplore.ieee.org/document/4444573."
             "CAUTION: See interaction with timestamp_resolution and auto_timestamp_resolution; "
             "check output logger warnings."
    )
    modelGroup.add_argument(
        "--leak_rate_hz", type=float, default=0.01,
        # typical for normal illumination levels with Davis cameras
        help="leak event rate per pixel in Hz - "
             "see https://ieeexplore.ieee.org/abstract/document/7962235")
    modelGroup.add_argument(
        "--shot_noise_rate_hz", type=float, default=0.001,
        # default for good lighting, very low rate
        help="Temporal noise rate of ON+OFF events in "
             "darkest parts of scene; reduced in brightest parts. ")
    modelGroup.add_argument(
        "--dvs_emulator_seed", type=int, default=0,
        help="Set to a integer >0 to use a fixed random seed."
             "default is 0 which means the random seed is not fixed.")

    modelGroup.add_argument(
        "--show_dvs_model_state", type=str, default=None,
        # default for good lighting, very low rate
        help="one of new_frame baseLogFrame lpLogFrame0 lpLogFrame1 "
             "diff_frame (without quotes)")

    # common camera types
    camGroup = modelGroup.add_argument_group(
        'DVS camera sizes (overrides --output_width and --output_height')
    camAction = camGroup.add_mutually_exclusive_group()
    camAction.add_argument(
        '--dvs128', action='store_true',
        help='Set size for 128x128 DVS (DVS128)')
    camAction.add_argument(
        '--dvs240', action='store_true',
        help='Set size for 240x180 DVS (Davis240)')
    camAction.add_argument(
        '--dvs346', action='store_true',
        help='Set size for 346x260 DVS (Davis346)')
    camAction.add_argument(
        '--dvs640', action='store_true',
        help='Set size for 640x480 DVS')
    camAction.add_argument(
        '--dvs1024', action='store_true',
        help='Set size for 1024x768 DVS')

    # slow motion frame synthesis
    sloMoGroup = parser.add_argument_group(
        'SloMo upsampling (see also "DVS timestamp resolution" group)')
    sloMoGroup.add_argument(
        "--disable_slomo", action='store_true',
        help="Disables slomo interpolation; the output DVS events "
             "will have exactly the timestamp resolution of "
             "the source video "
             "(which is perhaps modified by --input_slowmotion_factor).")
    sloMoGroup.add_argument(
        "--slomo_model", type=expandpath,
        default=prepend+"input/SuperSloMo39.ckpt",
        help="path of slomo_model checkpoint.")
    sloMoGroup.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size in frames for SuperSloMo. "
             "Batch size 8-16 is recommended if your GPU "
             "has sufficient memory.")
    sloMoGroup.add_argument(
        "--vid_orig", type=str, default="video_orig.avi",
        help="Output src video at same rate as slomo video "
             "(with duplicated frames).")
    sloMoGroup.add_argument(
        "--vid_slomo", type=str, default="video_slomo.avi",
        help="Output slomo of src video slowed down by slowdown_factor.")

    # TODO in general, allow reuse of slomo output
    #  sloMoGroup.add_argument(
    #     "--slomo_use_saved", action="store_true",
    #     help="Use previously-generated vid_slomo as input video "
    #          "instead of generating new one. "
    #          "Caution: saved video must have correct slowdown_factor.")

    sloMoGroup.add_argument(
        '--slomo_stats_plot', action='store_true',
        help="show a plot of slomo statistics")

    # input file handling
    inGroup = parser.add_argument_group('Input file handling')
    inGroup.add_argument(
        "-i", "--input", type=expandpath,
        help="Input video file or a image folder; "
             "leave empty for file chooser dialog."
             "If the input is a folder, the folder should contain "
             "a ordered list of image files."
             "In addition, the user has to set the frame rate manually.")
    inGroup.add_argument(
        "--input_frame_rate", type=float,
        help="Manually define the video frame rate when the video is "
             "presented as a list of image files."
             "When the input video is a video file, this "
             "option will be ignored.")
    inGroup.add_argument(
        "--input_slowmotion_factor", type=float, default=1.0,
        help="Sets the known slow-motion factor of the input video, "
             "i.e. how much the video is slowed down, i.e., "
             "the ratio of shooting frame rate to playback frame rate. "
             "input_slowmotion_factor<1 for sped-up video and "
             "input_slowmotion_factor>1 for slowmotion video."
             "If an input video is shot at 120fps yet is presented as a "
             "30fps video "
             "(has specified playback frame rate of 30Hz, "
             "according to file's FPS setting), "
             "then set --input_slowdown_factor=4."
             "It means that each input frame represents (1/30)/4 s=(1/120)s."
             "If input is video with intended frame intervals of "
             "1ms that is in AVI file "
             "with default 30 FPS playback spec, "
             "then use ((1/30)s)*(1000Hz)=33.33333.")
    inGroup.add_argument(
        "--start_time", type=float, default=None,
        help="Start at this time in seconds in video. "
             "Use None to start at beginning of source video.")
    inGroup.add_argument(
        "--stop_time", type=float, default=None,
        help="Stop at this time in seconds in video. "
             "Use None to end at end of source video.")

    # synthetic input handling
    syntheticInputGroup = parser.add_argument_group('Synthetic input')
    syntheticInputGroup.add_argument(
        "--synthetic_input", type=str,
        help="Input from class SYNTHETIC_INPUT that "
             "has methods next_frame() and total_frames()."
             "Disables file input and SuperSloMo frame interpolation. "
             "SYNTHETIC_INPUT.next_frame() should return a frame of "
             "the correct resolution (see DVS model arguments) "
             "which is array[y][x] with "
             "pixel [0][0] at upper left corner and pixel values 0-255. "
             "SYNTHETIC_INPUT must be resolvable from the classpath. "
             "SYNTHETIC_INPUT is the module name without .py suffix."
             "See example moving_dot.py."
    )

    # DVS output video
    outGroupDvsVideo = parser.add_argument_group('Output: DVS video')
    outGroupDvsVideo.add_argument(
        "--dvs_exposure", nargs='+', type=str, default='duration 0.01',
        help="Mode to finish DVS frame event integration: "
             "duration time: Use fixed accumulation time in seconds, "
             "e.g. --dvs_exposure duration .005; "
             "count n: Count n events per frame, -dvs_exposure count 5000; "
             "area_event N M: frame ends when any area of M x M pixels "
             "fills with N events, -dvs_exposure area_count 500 64")
    outGroupDvsVideo.add_argument(
        "--dvs_vid", type=str, default="dvs-video.avi",
        help="Output DVS events as AVI video at frame_rate.")
    outGroupDvsVideo.add_argument(
        "--dvs_vid_full_scale", type=int, default=2,
        help="Set full scale event count histogram count for DVS videos "
             "to be this many ON or OFF events for full white or black.")
    outGroupDvsVideo.add_argument(
        "--skip_video_output", action="store_true",
        help="Skip producing video outputs, including the original video, "
             "SloMo video, and DVS video. "
             "This mode also prevents showing preview of output (cf --no_preview).")
    outGroupDvsVideo.add_argument(
        "--no_preview", action="store_true",
        help="disable preview in cv2 windows for faster processing.")
# outGroupDvsVideo.add_argument(
    #     "--frame_rate", type=float,
    #     help="implies --dvs_exposure duration 1/framerate.  "
    #          "Equivalent frame rate of --dvs_vid output video; "
    #          "the events will be accummulated as this sample rate; "
    #          "DVS frames will be accumulated for duration 1/frame_rate")

    # DVS output as events
    dvsEventOutputGroup = parser.add_argument_group('Output: DVS events')
    dvsEventOutputGroup.add_argument(
        "--dvs_h5", type=output_file_check, default="None",
        help="Output DVS events as hdf5 event database.")
    dvsEventOutputGroup.add_argument(
        "--dvs_aedat2", type=output_file_check, default='v2e-dvs-events.aedat',
        help="Output DVS events as DAVIS346 camera AEDAT-2.0 event file "
             "for jAER; one file for real and one file for v2e events.")
    dvsEventOutputGroup.add_argument(
        "--dvs_text", type=output_file_check, default='v2e-dvs-events.txt',
        help="Output DVS events as text file with one event per "
             "line [timestamp (float s), x, y, polarity (0,1)].")
    #  dvsEventOutputGroup.add_argument(
    #      "--dvs_numpy", type=output_file_check, default="None",
    #      help="Accumulates DVS events to memory and writes final numpy data "
    #           "file with this name holding vector of events. "
    #           "WARNING: memory use is unbounded.")

    # # perform basic checks, however this fails if script adds
    # # more arguments later
    # args = parser.parse_args()
    # if args.input and not os.path.isfile(args.input):
    #     logger.error('input file {} not found'.format(args.input))
    #     quit(1)
    # if args.slomo_model and not os.path.isfile(args.slomo_model):
    #     logger.error(
    #         'slomo model checkpoint {} not found'.format(args.slomo_model))
    #     quit(1)

    return parser


def write_args_info(args, path) -> str:
    '''
    Writes arguments to logger and file named from startup __main__
    Parameters
    ----------
    args: parser.parse_args()

    Returns
    -------
    full path to file
    '''
    import __main__
    arguments_list = 'arguments:\n'
    for arg, value in args._get_kwargs():
        arguments_list += "{}:\t{}\n".format(arg, value)
    logger.info(arguments_list)
    basename = os.path.basename(__main__.__file__)
    argsFilename = basename.strip('.py') + '-args.txt'
    filepath = os.path.join(path, argsFilename)
    with open(filepath, "w") as f:
        f.write(arguments_list)
    return filepath


def v2e_check_dvs_exposure_args(args):
    if not args.dvs_exposure:
        raise ValueError(
            "define --dvs_exposure method. "
            "See extended usage.")

    dvs_exposure = args.dvs_exposure
    exposure_mode = None
    exposure_val = None
    area_dimension = None
    try:
        exposure_mode = ExposureMode[dvs_exposure[0].upper()]
    except Exception:
        raise ValueError(
            "dvs_exposure first parameter '{}' must be 'duration','count', "
            "or 'area_event'".format(dvs_exposure[0]))

    if exposure_mode == ExposureMode.AREA_COUNT and not len(dvs_exposure) == 3:
        raise ValueError("area_event argument needs three parameters, "
                         "e.g. 'area_count 500 16'")
    elif not exposure_mode == ExposureMode.AREA_COUNT and \
            not len(dvs_exposure) == 2:
        raise ValueError("duration or count argument needs two parameters, "
                         "e.g. 'duration 0.01' or 'count 3000'")

    if not exposure_mode == ExposureMode.AREA_COUNT:
        try:
            exposure_val = float(dvs_exposure[1])
        except Exception:
            raise ValueError(
                "dvs_exposure second parameter must be a number, "
                "either duration or event count")
    else:
        try:
            exposure_val = int(dvs_exposure[1])
            area_dimension = int(dvs_exposure[2])
        except Exception:
            raise ValueError(
                "area_count must be N M, where N is event count and M "
                "is area dimension in pixels")
    s = 'DVS frame expsosure mode {}'.format(exposure_mode)
    if exposure_mode == ExposureMode.DURATION:
        s = s+': frame rate {}'.format(1./exposure_val)
    elif exposure_mode == ExposureMode.COUNT:
        s = s+': {} events/frame'.format(exposure_val)
    elif exposure_mode == ExposureMode.AREA_COUNT:
        s = s+': {} events per {}x{} pixel area'.format(
            exposure_val, area_dimension, area_dimension)

    logger.info(s)
    return exposure_mode, exposure_val, area_dimension

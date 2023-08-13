import argparse
import os
import logging
import time
from pathlib import Path
from v2ecore.emulator import EventEmulator

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

def none_or_str(value): # https://stackoverflow.com/questions/48295246/how-to-pass-none-keyword-as-command-line-argument
    """ Use in argparse add_argument as type none_or_str
    """
    if value == 'None' or value=='':
        return None
    return value

# https://stackoverflow.com/questions/3853722/how-to-insert-newlines-on-argparse-help-text
class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
            # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


def tuple_type(strings):
    """ From https://stackoverflow.com/questions/33564246/passing-a-tuple-as-command-line-argument
    Allows passing tuple as argument and returning a tuple in the args.xxxx
    """
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)


def v2e_args(parser):
    """

    Parameters
    ----------
    parser
        the argparse object to be populated with arguments

    Returns
    -------
     the parser with all the standard v2e arguments

    """
    v2ecore_path=os.path.dirname(__file__)



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
        "--output_in_place", default=False, type=str2bool,
        const=True, nargs='?',
        help="store output files in same folder as source video (in same folder as frames if using folder of frames).")
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
    outGroupGeneral.add_argument(
        "--skip_video_output", action="store_true",
        help="Skip producing video outputs, including the original video, "
             "SloMo video, and DVS video. "
             "This mode also prevents showing preview of output "
             "(cf --no_preview).")

    # timestamp resolution
    timestampResolutionGroup = parser.add_argument_group(
        'DVS timestamp resolution')
    timestampResolutionGroup.add_argument(
        "--auto_timestamp_resolution", default=True, type=str2bool,
        const=True, nargs='?',
        help="(Ignored by --disable_slomo or --synthetic_input.) "
             "\nIf True (default), upsampling_factor is automatically "
             "determined to limit maximum movement between frames to 1 pixel."
             "\nIf False, --timestamp_resolution sets the upsampling factor "
             "for input video."
             "\nCan be combined with --timestamp_resolution to "
             "ensure DVS events have at most some resolution.")
    timestampResolutionGroup.add_argument(
        "--timestamp_resolution", type=float,
        help="(Ignored by --disable_slomo or --synthetic_input.) "
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
        "--dvs_params", type=str, default=None,
        help="Easy optional setting of parameters for DVS model:"
             "None, 'clean', 'noisy'; 'clean' turns off noise, sets unlimited bandwidth and "
             "makes threshold variation small. 'noisy' sets "
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
             "CAUTION: See interaction with timestamp_resolution "
             "and auto_timestamp_resolution; "
             "check output logger warnings. The input sample rate (frame rate) must be fast enough to for accurate IIR lowpass filtering."
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
    modelGroup.add_argument('--photoreceptor_noise',action='store_true',
            help='Create temporal noise by injecting Gaussian noise to the log photoreceptor before lowpass filtering.'
                 'This way, more accurate statistics of temporal noise will tend to result (in particular, alternating ON and OFF noise events)'
                 'but the noise rate will only approximate the desired noise rate;'
                 ' the photoreceptor noise will be computed to result in the --shot_noise_rate noise value. '
                 'Overrides the default shot noise mechanism reported in 2020 v2e paper.')
    modelGroup.add_argument(
        "--leak_jitter_fraction", type=float, default=0.1,
        help="Jitter of leak noise events relative to the (FPN) "
             "interval, drawn from normal distribution")
    modelGroup.add_argument(
        "--noise_rate_cov_decades", type=float, default=0.1,
        help="Coefficient of Variation of noise rates (shot and leak) "
             "in log normal distribution decades across pixel array"
             "WARNING: currently only in leak events")
    modelGroup.add_argument(
        "--refractory_period", type=float, default=0.0005,
        help="Refractory period in seconds, default is 0.5ms."
             "The new event will be ignore if the previous event is "
             "triggered less than refractory_period ago."
             "Set to 0 to disable this feature.")
    modelGroup.add_argument(
        "--dvs_emulator_seed", type=int, default=0,
        help="Set to a integer >0 to use a fixed random seed."
             "default is 0 which means the random seed is not fixed.")

    modelGroup.add_argument(
        "--show_dvs_model_state", nargs='+', default=None,
        help="One or more space separated list model states. Do not use '='. E.g. '--show_dvs_model_state all'. "
             f"Possible models states are (without quotes) either 'all' or chosen from {EventEmulator.MODEL_STATES.keys()}")

    modelGroup.add_argument(
        "--save_dvs_model_state", action="store_true",
        help="save the model states that are shown (cf --show_dvs_model_state) to avi files")

    modelGroup.add_argument("--record_single_pixel_states",type=tuple_type, default=None,
        help=f"Record internal states of a single pixel specified by (x,y) tuple to '{EventEmulator.SINGLE_PIXEL_STATES_FILENAME}'."
             f"The file is a pickled binary dict that has the state arrays over time imcluding a time array."
             "Pixel location can also be specified as x,y without ()")

    # common camera types
    camGroup = parser.add_argument_group(
        'DVS camera sizes (selecting --dvs346, --dvs640, etc. overrides --output_width and --output_height')
    camGroup.add_argument(
        "--output_height", type=int, default=None,
        help="Height of output DVS data in pixels. "
             "If None, same as input video. "
             "Use --output_height=260 for Davis346.")
    camGroup.add_argument(
        "--output_width", type=int, default=None,
        help="Width of output DVS data in pixels. "
             "If None, same as input video. "
             "Use --output_width=346 for Davis346.")

    camAction = camGroup.add_mutually_exclusive_group()
    camAction.add_argument(
        '--dvs128', action='store_true',
        help='Set size for 128x128 DVS (DVS128)')
    camAction.add_argument(
        '--dvs240', action='store_true',
        help='Set size for 240x180 DVS (DAVIS240)')
    camAction.add_argument(
        '--dvs346', action='store_true',
        help='Set size for 346x260 DVS (DAVIS346)')
    camAction.add_argument(
        '--dvs640', action='store_true',
        help='Set size for 640x480 DVS (DAVIS640)')
    camAction.add_argument(
        '--dvs1024', action='store_true',
        help='Set size for 1024x768 DVS (not supported for AEDAT-2.0 output since there is no jAER DVS1024 camera')

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
        default=os.path.join(v2ecore_path,"../input/SuperSloMo39.ckpt"),
        help="path of slomo_model checkpoint.")
    sloMoGroup.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size in frames for SuperSloMo. "
             "Batch size 8-16 is recommended if your GPU "
             "has sufficient memory.")
    sloMoGroup.add_argument(
        "--vid_orig", type=none_or_str, default="video_orig.avi",
        help="Output src video at same rate as slomo video "
             "(with duplicated frames). "
             "Specify emtpy string or 'None' to skip output.")

    sloMoGroup.add_argument(
        "--vid_slomo", type=none_or_str, default="video_slomo.avi",
        help="Output slomo of src video slowed down by slowdown_factor."
             "Specify emtpy string or 'None' to skip output.")

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
        help="Either override the video file metadata frame rate or manually define the video frame rate when the video is "
             "presented as a list of image files. Overrides the stored (metadata) frame rate of input video. "
             "This option overrides the --input_slowmotion_factor argument in case "
             "the input is from a video file."
             )
    # note R| for SmartFormatter
    inGroup.add_argument(
        "--input_slowmotion_factor", type=float, default=1.0,
        help="R|(See --input_frame_rate argument too.) Sets the known slow-motion factor of the input video, "
             "\ni.e. how much the video is slowed down, i.e., "
             "\nthe ratio of shooting frame rate to playback frame rate. "
             "\ninput_slowmotion_factor<1 for sped-up video and "
             "\ninput_slowmotion_factor>1 for slowmotion video."
             "\nIf an input video is shot at 120fps yet is presented as a "
             "30fps video "
             "\n(has specified playback frame rate of 30Hz, "
             "\naccording to file's FPS setting), "
             "\nthen set --input_slowdown_factor=4."
             "\nIt means that each input frame represents (1/30)/4 s=(1/120)s."
             "\nIf input is video with intended frame intervals of "
             "\n1ms that is in AVI file "
             "\nwith default 30 FPS playback spec, "
             "\nthen use ((1/30)s)*(1000Hz)=33.33333.")

    inGroup.add_argument(
        "--start_time", type=float, default=None,
        help="Start at this time in seconds in video. "
             "Use None to start at beginning of source video.")
    inGroup.add_argument(
        "--stop_time", type=float, default=None,
        help="Stop at this time in seconds in video. "
             "Use None to end at end of source video.")
    inGroup.add_argument(
        "--crop", type=tuple_type, default=None,
        help="Crop input video by (left, right, top, bottom) pixels. "
             "E.g. CROP=(100,100,0,0) crops 100 pixels "
             "from left and right of input frames."
             " CROP can also be specified as L,R,T,B without ()")
    inGroup.add_argument('--hdr',action='store_true',
                         help='Treat input video as high dynamic range (HDR) logarithmic, '
                              'i.e. skip the linlog conversion step. '
                              'Use --hdr for HDR input with floating '
                              'point gray scale input videos. Units of log input are based '
                              'on white 255 pixels have values ln(255)=5.5441')

    # synthetic input handling
    syntheticInputGroup = parser.add_argument_group('Synthetic input')
    syntheticInputGroup.add_argument(
        "--synthetic_input", type=str,
        help="Input from class SYNTHETIC_INPUT that "
             "\nhas methods next_frame() and total_frames()."
             "\nDisables file input and SuperSloMo frame interpolation and the DVS timestamp resolution is set by the times returned by next_frame() method. "
             "\nSYNTHETIC_INPUT.next_frame() should return a tuple (frame, time) with frame having"
             "\nthe correct resolution (see DVS model arguments) "
             "\nwhich is array[y][x] with "
             "\npixel [0][0] at upper left corner and pixel values 0-255. The time is a float in seconds."
             "\nSYNTHETIC_INPUT must be resolvable from the classpath. "
             "\nSYNTHETIC_INPUT is the module name without .py suffix."
             "\nSee example moving_dot.py."
    )

    # DVS output video
    outGroupDvsVideo = parser.add_argument_group('Output: DVS video')
    outGroupDvsVideo.add_argument(
        "--dvs_exposure", nargs='+', type=str, default=['duration', '0.01'],
        help="R|Mode to finish DVS frame event integration:"
             "\n\tduration time: Use fixed accumulation time in seconds, e.g. "
             "\n\t\t--dvs_exposure duration .005; "
             "\n\tcount n: Count n events per frame,e.g."
             "\n\t\t--dvs_exposure count 5000;"
             "\n\tarea_count M N: frame ends when any area of N x N "
             "pixels fills with M events, e.g."
             "\n\t\t-dvs_exposure area_count 500 64"
             "\n\tsource: each DVS frame is from one source frame (slomo or original, depending on if slomo is used)")
    outGroupDvsVideo.add_argument(
        "--dvs_vid", type=none_or_str, default="dvs-video.avi",
        help="Output DVS events as AVI video at frame_rate. To suppress, supply empty argument or 'None'.")
    outGroupDvsVideo.add_argument(
        "--dvs_vid_full_scale", type=int, default=2,
        help="Set full scale event count histogram count for DVS videos "
             "to be this many ON or OFF events for full white or black.")
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
        "--ddd_output", action="store_true",
        help="Save frames, frame timestamp and corresponding event index"
             " in HDF5 format used for DDD17 and DDD20 datasets. Default is False.")
    dvsEventOutputGroup.add_argument(
        "--dvs_h5", type=output_file_check, default=None,
        help="Output DVS events as hdf5 event database.")
    dvsEventOutputGroup.add_argument(
        "--dvs_aedat2", type=output_file_check, default=None,
        help="Output DVS events as DAVIS346 camera AEDAT-2.0 event file "
             "for jAER; one file for real and one file for v2e events. To suppress, supply argument None. ")
    dvsEventOutputGroup.add_argument(
        "--dvs_aedat4", type=output_file_check, default=None,
        help="Output DV AEDAT-4.0 event file "
             "To suppress, supply argument None. ")
    dvsEventOutputGroup.add_argument(
        "--dvs_text", type=output_file_check, default=None,
        help="Output DVS events as text file with one event per "
             "line [timestamp (float s), x, y, polarity (0,1)]. ")
    dvsEventOutputGroup.add_argument('--label_signal_noise', action='store_true',
                                 help='append to the --dvs_text file a column,'
                                      'containing list of signal and shot noise events. '
                                      'Each row of the CSV appends a 1 for signal and 0 for noise.'
                                      '** Notes: 1: requires activating --dvs_text option; '
                                      '2: requires disabling --photoreceptor_noise option '
                                      '(because when noise arises from photoreceptor it is '
                                      'impossible to tell if event was caused by signal or noise.'
                                      '3: Only labels shot noise events (because leak noise events arise from leak and cannot be distinguished from photoreceptor input).')

    #  dvsEventOutputGroup.add_argument(
    #      "--dvs_numpy", type=output_file_check, default="None",
    #      help="Accumulates DVS events to memory and writes final numpy data "
    #           "file with this name holding vector of events. "
    #           "WARNING: memory use is unbounded.")


    # center surround DVS emulation
    csdvs=parser.add_argument_group('Center-Surround DVS')
    csdvs.add_argument('--cs_lambda_pixels',type=float,default=None,help='space constant of surround in pixels, None to disable.  '
                                                                         'This space constant lambda is sqrt(1/gR) '
                                                                         'where g is the transverse conductance and R is the lateral resistance.')
    csdvs.add_argument('--cs_tau_p_ms',type=float,default=None,help='time constant of photoreceptor center of diffuser in ms, or 0 to disable for instantaneous surround. '
                                                                    'Defined as C/g where C is capacitance and '
                                                                    'g is the transverse conductance from photoreceptor to horizontal cell network. '
                                                                    'This time is'
                                                                    'the time constant for h cell diffuser network response '
                                                                    'time to global input to photoreceptors. If set to zero,'
                                                                    ' then the simulation of diffuser runs until it converges, '
                                                                    'i.e. until the maximum change between timesteps '
                                                                    'is smaller than a threshold value')

    # SCIDVS pixel study
    scidvs_group = parser.add_argument_group('SCIDVS pixel')
    scidvs_group.add_argument('--scidvs', action='store_true',help='Simulate proposed SCIDVS pixel with nonlinear adapatation and high gain')


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


def write_args_info(args, path, other_args=None, command_line=None) -> str:
    '''
    Writes arguments to logger and file named from startup __main__
    Parameters
    ----------
    args: parser.parse_args()
    path: file to write to
    other_args: possible extra arguments\
    command_line: the whole command line

    Returns
    -------
    full path to file
    '''
    import __main__
    arguments_list = '\n*** arguments:\n'
    for arg, value in sorted(args._get_kwargs()):
        arguments_list += "{}:\t{}\n".format(arg, value)
    logger.info(arguments_list)
    other_arguments_list=None
    if other_args is not None and len(other_args)>0:
        other_arguments_list = '\n**** extra other arguments (please check if you are misspelling intended arguments):\n'
        for arg in sorted(other_args):
            other_arguments_list += "{}\n".format(arg)
        logger.warning(other_arguments_list)
        time.sleep(2)
    basename = os.path.basename(__main__.__file__)
    argsFilename = basename.strip('.py') + '-args.txt'
    filepath = os.path.join(path, argsFilename)
    with open(filepath, "w") as f:
        f.write(arguments_list)
        if other_arguments_list is not None:
            f.write(other_arguments_list)
        f.write(f'\n*** command line:\n'+command_line)
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
            " 'area_count' or 'source'".format(dvs_exposure[0]))

    if exposure_mode == ExposureMode.SOURCE:
        logger.info('DVS video exposure mode is SOURCE')
        return exposure_mode,None,None
    if exposure_mode == ExposureMode.AREA_COUNT and not len(dvs_exposure) == 3:
        raise ValueError("area_event argument needs three parameters:  "
                         "'area_count M N'; frame ends when any area of M x M pixels "
                         "fills with N events")
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

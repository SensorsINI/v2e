
import os
import logging

from v2e.renderer import ExposureMode

logger=logging.getLogger(__name__)


def v2e_args(parser):
    dir_path = os.getcwd()  # check and add prefix if running script in subfolder
    if dir_path.endswith('ddd'):
        prepend='../../'
    else:
        prepend=''


    modelGroup=parser.add_argument_group('DVS model')
    modelGroup.add_argument("--dvs_params", type=str, default='clean',
                        help="Easy optional setting of parameters for DVS model: 'clean', 'noisy'")
    modelGroup.add_argument("--pos_thres", type=float, default=0.2,
                        help="threshold in log_e intensity change to trigger a positive event.")
    modelGroup.add_argument("--neg_thres", type=float, default=0.2,
                        help="threshold in log_e intensity change to trigger a negative event.")
    modelGroup.add_argument("--sigma_thres", type=float, default=0.03,
                        help="1-std deviation threshold variation in log_e intensity change.")
    modelGroup.add_argument("--cutoff_hz", type=float, default=0,
                        help="photoreceptor second-order IIR lowpass filter cutoff-off 3dB frequency in Hz - see https://ieeexplore.ieee.org/document/4444573")
    modelGroup.add_argument("--leak_rate_hz", type=float, default=0.01,
                        help="leak event rate per pixel in Hz - see https://ieeexplore.ieee.org/abstract/document/7962235")
    modelGroup.add_argument("--shot_noise_rate_hz", type=float, default=0,
                        help="Temporal noise rate of ON+OFF events in darkest parts of scene; reduced in brightest parts. ")


    sloMoGroup=parser.add_argument_group('SloMo upsampling')
    sloMoGroup.add_argument("--slomo_model", type=str, default=prepend+"input/SuperSloMo39.ckpt", help="path of slomo_model checkpoint.")
    sloMoGroup.add_argument("--segment_size", type=int, default=1, help="segment size for SuperSloMo. Video is split to chunks of this many frames, and within each segment, batch mode CNN inference of optic flow takes place. Video will be processed segment by segment.")
    sloMoGroup.add_argument("--batch_size", type=int, default=1, help="batch size in frames for SuperSloMo. Must be less than or equal to seqment_size.")
    sloMoGroup.add_argument("--no_preview", action="store_true", help="disable preview in cv2 windows for faster processing.")
    sloMoGroup.add_argument("--slowdown_factor", type=int, default=10,
                        help="slow motion factor; if the input video has frame rate fps, then the DVS events will have time resolution of 1/(fps*slowdown_factor).")
    sloMoGroup.add_argument("--vid_orig", type=str, default="video_orig.avi", help="output src video at same rate as slomo video (with duplicated frames).")
    sloMoGroup.add_argument("--vid_slomo", type=str, default="video_slomo.avi", help="output slomo of src video slowed down by slowdown_factor.")

    inGroup=parser.add_argument_group('Input')
    inGroup.add_argument("-i", "--input", type=str, help="input video file; leave empty for file chooser dialog.")
    inGroup.add_argument("--start_time", type=float, default=None, help="start at this time in seconds in video.")
    inGroup.add_argument("--stop_time", type=float, default=None, help="stop at this time in seconds in video.")

    outGroupGeneral=parser.add_argument_group('Output: General')
    outGroupGeneral.add_argument("-o", "--output_folder", type=str, required=True, help="folder to store outputs.")
    outGroupGeneral.add_argument("--overwrite", action="store_true", help="overwrites files in existing folder (checks existence of non-empty output_folder).")

    outGroupDvsVideo = parser.add_argument_group('Output: DVS video')
    outGroupDvsVideo.add_argument("--dvs_vid", type=str, default="dvs-video.avi", help="output DVS events as AVI video at frame_rate.")
    outGroupDvsVideo.add_argument("--dvs_vid_full_scale", type=int, default=2, help="set full scale event count histogram count for DVS videos to be this many ON or OFF events for full white or black.")
    outGroupDvsVideo.add_argument("--output_height",
                        help="height of output DVS data in pixels. If None, same as input video.")
    outGroupDvsVideo.add_argument("--output_width", 
                        help="width of output DVS data in pixels. If None, same as input video.")
    outGroupDvsVideo.add_argument("--frame_rate", type=int,
                                  help="implies --dvs_exposure duration 1/framerate.  Equivalent frame rate of --dvs_vid output video; the events will be accummulated as this sample rate; DVS frames will be accumulated for duration 1/frame_rate")
    outGroupDvsVideo.add_argument("--dvs_exposure", nargs='+', type=str, help="mode to finish DVS event integration: duration time: accumulation time in seconds; count n: count n events per frame; area_event N M: frame ends when any area of M x M pixels fills with N events")

    dvsEventOutputGroup = parser.add_argument_group('Output: DVS events')
    dvsEventOutputGroup.add_argument("--dvs_h5", type=str, default=None, help="output DVS events as hdf5 event database.")
    dvsEventOutputGroup.add_argument("--dvs_aedat2", type=str, default=None, help="output DVS events as DAVIS346 camera AEDAT-2.0 event file for jAER; one file for real and one file for v2e events.")
    dvsEventOutputGroup.add_argument("--dvs_text", type=str, default=None, help="output DVS events as text file with one event per line [timestamp (float s), x, y, polarity (0,1)].")
    dvsEventOutputGroup.add_argument("--dvs_numpy", type=str, default=None, help="accumulates DVS events to memory and writes final numpy data file with this name holding vector of events. WARNING: memory use is unbounded.")

    # # perform basic checks, however this fails if script adds more arguments later
    # args = parser.parse_args()
    # if args.input and not os.path.isfile(args.input):
    #     logger.error('input file {} not found'.format(args.input))
    #     quit(1)
    # if args.slomo_model and not os.path.isfile(args.slomo_model):
    #     logger.error('slomo model checkpoint {} not found'.format(args.slomo_model))
    #     quit(1)

    return parser


def write_args_info(args, path)-> str:
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
    basename=os.path.basename(__main__.__file__)
    argsFilename = basename.strip('.py') + '-args.txt'
    filepath=os.path.join(path, argsFilename)
    with open(filepath, "w") as f:
        f.write(arguments_list)
    return filepath


def v2e_check_dvs_exposure_args(args):
    if not args.frame_rate and not args.dvs_exposure:
        raise ValueError('either define --frame_rate or --dvs_exposure. See extended usage.')

    if args.frame_rate and args.dvs_exposure:
        raise ValueError('either define --frame_rate or --dvs_exposure. See extended usage.')

    if args.frame_rate:
        dvs_exposure=('duration',1./float(args.frame_rate))
        logger.info('--frame_rate option implies constant duration DVS frames')
    else:
        dvs_exposure=args.dvs_exposure
    exposure_mode=None
    exposure_val=None
    area_dimension=None
    try:
        exposure_mode=ExposureMode[dvs_exposure[0].upper()]
    except:
        raise ValueError("dvs_exposure first parameter '{}' must be 'duration','count', or 'area_event'".format(dvs_exposure[0]))

    if exposure_mode==ExposureMode.AREA_COUNT and not len(dvs_exposure)==3:
        raise ValueError("area_event argument needs three parameters, e.g. 'area_count 500 16'")
    elif not exposure_mode==ExposureMode.AREA_COUNT and not len(dvs_exposure)==2:
        raise ValueError("duration or count argument needs two parameters, e.g. 'duration 0.01' or 'count 3000'")

    if not exposure_mode==ExposureMode.AREA_COUNT:
        try:
             exposure_val=float(dvs_exposure[1])
        except:
            raise ValueError('dvs_exposure second parameter must be a number, either duration or event count')
    else:
        try:
             exposure_val=int(dvs_exposure[1])
             area_dimension=int(dvs_exposure[2])
        except:
            raise ValueError('area_count must be N M, where N is event count and M is area dimension in pixels')
    s='DVS frame expsosure mode {}'.format(exposure_mode)
    if exposure_mode==ExposureMode.DURATION:
        s=s+': frame rate {}'.format(1./exposure_val)
    elif exposure_mode==ExposureMode.COUNT:
        s=s+': {} events/frame'.format(exposure_val)
    elif exposure_mode==ExposureMode.AREA_COUNT:
        s=s+': {} events per {}x{} pixel area'.format(exposure_val,area_dimension,area_dimension)

    logger.info(s)
    return exposure_mode, exposure_val, area_dimension

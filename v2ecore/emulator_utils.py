"""Collections of emulator utilities.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""
import logging
import math
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def lin_log(x, threshold=20):
    """
    linear mapping + logarithmic mapping.

    :param x: float or ndarray
        the input linear value in range 0-255 TODO assumes 8 bit
    :param threshold: float threshold 0-255
        the threshold for transition from linear to log mapping

    Returns: the log value
    """
    # converting x into np.float64.
    if x.dtype is not torch.float64:  # note float64 to get rounding to work
        x = x.double()

    f = (1./threshold) * math.log(threshold)

    y = torch.where(x <= threshold, x*f, torch.log(x))

    # important, we do a floating point round to some digits of precision
    # to avoid that adding threshold and subtracting it again results
    # in different number because first addition shoots some bits off
    # to never-never land, thus preventing the OFF events
    # that ideally follow ON events when object moves by
    rounding = 1e8
    y = torch.round(y*rounding)/rounding

    return y.float()


def rescale_intensity_frame(new_frame):
    """Rescale intensity frames.

    make sure we get no zero time constants
    limit max time constant to ~1/10 of white intensity level
    """
    return (new_frame+20)/275.


def low_pass_filter(
        log_new_frame,
        lp_log_frame0,
        lp_log_frame1,
        inten01,
        delta_time,
        cutoff_hz=0):
    """Compute intensity-dependent low-pass filter.

    # Arguments
        log_new_frame: new frame in lin-log representation.
        lp_log_frame0:
        lp_log_frame1:
        inten01:
        delta_time:
        cutoff_hz:

    # Returns
        new_lp_log_frame0
        new_lp_log_frame1

    """
    if cutoff_hz <= 0:
        # unchange
        return log_new_frame, log_new_frame

    # else low pass
    tau = 1/(math.pi*2*cutoff_hz)

    # make the update proportional to the local intensity
    # the more intensity, the shorter the time constant
    eps = inten01*(delta_time/tau)
    if eps>0.3:
        logger.warning(f'IIR lowpass filter update has large update eps={eps:.2f} from delta_time/tau={delta_time:.3g}/{tau:.3g}')
    eps = torch.clamp(eps, max=1)  # keep filter stable

    # first internal state is updated
    new_lp_log_frame0 = (1-eps)*lp_log_frame0+eps*log_new_frame

    # then 2nd internal state (output) is updated from first
    # Note that observations show that one pole is nearly always dominant,
    # so the 2nd stage is just copy of first stage
    new_lp_log_frame1 = lp_log_frame0
    # (1-eps)*self.lpLogFrame1+eps*self.lpLogFrame0 # was 2nd-order,
    # now 1st order.

    return new_lp_log_frame0, new_lp_log_frame1


def subtract_leak_current(base_log_frame,
                          leak_rate_hz,
                          delta_time,
                          pos_thres,
                          leak_jitter_fraction,
                          noise_rate_array):
    """Subtract leak current from base log frame."""

    rand = torch.randn(
        noise_rate_array.shape, dtype=torch.float32,
        device=noise_rate_array.device)

    curr_leak_rate = \
        leak_rate_hz*noise_rate_array*(1-leak_jitter_fraction*rand)

    delta_leak = delta_time*curr_leak_rate*pos_thres  # this is a matrix

    # ideal model
    #  delta_leak = delta_time*leak_rate_hz*pos_thres  # this is a matrix

    return base_log_frame-delta_leak


def compute_event_map(diff_frame, pos_thres, neg_thres):
    """Compute event map.

    Prepare positive and negative event frames that later will be used
    for generating events.
    """
    # extract positive and negative differences
    pos_frame = F.relu(diff_frame)
    neg_frame = F.relu(-diff_frame)

    # compute quantized number of ON and OFF events for each pixel
    pos_evts_frame = torch.div(
        pos_frame, pos_thres, rounding_mode="floor").type(torch.int32)
    neg_evts_frame = torch.div(
        neg_frame, neg_thres, rounding_mode="floor").type(torch.int32)

    #  max_events = max(pos_evts_frame.max(), neg_evts_frame.max())

    #  # boolean array (max_events, height, width)
    #  # positive events and negative
    #  pos_evts_cord = torch.arange(
    #      1, max_events+1, dtype=torch.int32,
    #      device=diff_frame.device).unsqueeze(-1).unsqueeze(-1).repeat(
    #          1, diff_frame.shape[0], diff_frame.shape[1])
    #  neg_evts_cord = pos_evts_cord.clone().detach()
    #
    #  # generate event cords
    #  pos_evts_cord_post = (pos_evts_cord >= pos_evts_frame.unsqueeze(0))
    #  neg_evts_cord_post = (neg_evts_cord >= neg_evts_frame.unsqueeze(0))

    return pos_evts_frame, neg_evts_frame
    #  return pos_evts_cord_post, neg_evts_cord_post, max_events


def compute_photoreceptor_noise_voltage(rate_hz, f3db, pos_thr, neg_thr) -> float:
    """
     Computes the necessary photoreceptor noise voltage to result in obseved shot noise rate at low light intensity.
     This computation relies on the known f3dB photoreceptor lowpass filter cutoff frequency and the known (nominal) event threshold.
     emulator.py injects Gaussian distributed noise to the photoreceptor that should in principle generate the desired shot noise events.

     See the file media/noise_event_rate_simulation.xlsx for the simulation data and curve fit.

    Parameters
    -----------
     rate_hz: float
        the desired pixel shot noise rate in hz
     f3db: float
        the cutoff frequency in Hz
     pos_thr:float
        on threshold in ln units
     neg_thr:float
        off threshold in ln units. The on and off thresholds are averaged to obtain a single threshold.

    Returns
    -----------
    float
         Noise voltage Gaussian RMS value
    """

    # Excel 6th order polynomial fit to Rui's simulation results of noise rate Rn in Hz per f3db bandwidth as a function of noise voltage Vn compared to threshold voltage thr
    # x = log10(thr/Vn)
    # y = log10(Rn/f3db)
    # y = -1.4883*x^^6 - 6.1187*x^^5 - 7.5137*x^^4 - 2.2628*x^^3 + 0.6095*x^^2 - 1.9891*x + 3.7154

    # y = log10(thr/Vn)
    # x = log10(Rn/f3db)
    # y = -0.0026x3 - 0.0127x2 - 0.0486x + 0.6512
    # see the plot Fig. 3 from Graca, Rui, and Tobi Delbruck. 2021. “Unraveling the Paradox of Intensity-Dependent DVS Pixel Noise.” arXiv [eess.SY]. arXiv. http://arxiv.org/abs/2109.08640.
    thr= (pos_thr + neg_thr) / 2
    rate_per_bw=2*rate_hz/f3db # simulation data are on ON event rates, so we double the desired rate here
    if rate_per_bw>0.5:
        logger.warning(f'shot noise rate per hz of bandwidth is larger than 0.1 (rate_hz={rate_hz} Hz, 3dB bandwidth={f3db} Hz)')
    x=math.log10(rate_per_bw)
    if x<-4.0:
        logger.warning(f'desired noise rate of {rate_hz}Hz is too low to accurately compute a threshold value')
    elif x>3.0:
        logger.warning(f'desired noise rate of {rate_hz}Hz is too large to accurately compute a threshold value')
    y= -0.0026*x**3 - 0.0127*x**2 - 0.0486*x + 0.6512

    vn_per_thr=10**y
    vn=vn_per_thr*thr
    logger.info(
        f'Computed photoreceptor_noise_vrms={vn:.3f} for shot_noise_rate_hz={rate_hz}, cutoff_hz={f3db} Hz and average on/off threshold={thr}')

    return vn


def generate_shot_noise(
        shot_noise_rate_hz,
        delta_time,
        num_iters,
        shot_noise_inten_factor,
        inten01,
        pos_thres_pre_prob,
        neg_thres_pre_prob):
    """Generate shot noise.

    """
    # new shot noise generator, generate for the entire batch
    shot_noise_factor = (
        (shot_noise_rate_hz/2)*delta_time/num_iters) * \
        ((shot_noise_inten_factor-1)*inten01+1)
    # =1 for inten=0 and SHOT_NOISE_INTEN_FACTOR for inten=1

    # probability for each pixel is
    # dt*rate*nom_thres/actual_thres.
    # That way, the smaller the threshold,
    # the larger the rate
    one_minus_shot_ON_prob_this_sample = \
        1 - shot_noise_factor*pos_thres_pre_prob
    shot_OFF_prob_this_sample = \
        shot_noise_factor*neg_thres_pre_prob

    # for shot noise
    rand01 = torch.rand(
        size=[num_iters]+list(inten01.shape),
        dtype=torch.float32,
        device=inten01.device)  # draw_frame samples

    # pre compute all the shot noise cords
    shot_on_cord = torch.gt(
        rand01, one_minus_shot_ON_prob_this_sample.unsqueeze(0))
    shot_off_cord = torch.lt(
        rand01, shot_OFF_prob_this_sample.unsqueeze(0))

    return shot_on_cord, shot_off_cord

    # old shot noise, generate at every iteration.
    # the right device
    #  device = base_log_frame.device

    # array with True where ON noise event
    #  shot_ON_cord = rand01 > (1-shot_ON_prob_this_sample)
    #
    #  shot_OFF_cord = rand01 < shot_OFF_prob_this_sample

    # get shot noise event ON and OFF cordinates
    #  shot_ON_xy = shot_ON_cord.nonzero(as_tuple=True)
    #  shot_ON_count = shot_ON_xy[0].shape[0]
    #
    #  shot_OFF_xy = shot_OFF_cord.nonzero(as_tuple=True)
    #  shot_OFF_count = shot_OFF_xy[0].shape[0]

    #  self.num_events_on += shotOnCount
    #  self.num_events_off += shotOffCount
    #  self.num_events_total += shotOnCount+shotOffCount

    # update log_frame
    #  base_log_frame += shot_ON_cord*pos_thres
    #  base_log_frame -= shot_OFF_cord*neg_thres

    #  if shot_ON_count > 0:
    #      shot_ON_events = torch.ones(
    #          (shot_ON_count, 4), dtype=torch.float32, device=device)
    #      shot_ON_events[:, 0] *= ts
    #      shot_ON_events[:, 1] = shot_ON_xy[1]
    #      shot_ON_events[:, 2] = shot_ON_xy[0]
    #
    #      base_log_frame += shot_ON_cord*pos_thres
    #  else:
    #      shot_ON_events = torch.zeros(
    #          (0, 4), dtype=torch.float32, device=device)
    #
    #  if shot_OFF_count > 0:
    #      shot_OFF_events = torch.ones(
    #          (shot_OFF_count, 4), dtype=torch.float32, device=device)
    #      shot_OFF_events[:, 0] *= ts
    #      shot_OFF_events[:, 1] = shot_OFF_xy[1]
    #      shot_OFF_events[:, 2] = shot_OFF_xy[0]
    #      shot_OFF_events[:, 3] *= -1
    #
    #      base_log_frame -= shot_OFF_cord*neg_thres
    #  else:
    #      shot_OFF_events = torch.zeros(
    #          (0, 4), dtype=torch.float32, device=device)
    # end temporal noise

    #  return shot_ON_events, shot_OFF_events, base_log_frame
    #  return shot_ON_cord, shot_OFF_cord, base_log_frame
    #  return shot_ON_cord, shot_OFF_cord


if __name__ == "__main__":

    temp_input = torch.randint(0, 256, (1280, 720), dtype=torch.float32).cuda()

    for i in range(1000):
        temp_out = lin_log(temp_input, threshold=20)

    pass

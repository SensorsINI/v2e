"""Collections of emulator utilities.

Author: Yuhuang Hu, Tobi Delbruck
Email : yuhuang.hu@ini.uzh.ch, tobi@ini.uzh.ch
"""
import logging
import math
import sys

import numpy as np
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
        lp_log_frame,
        inten01,
        delta_time,
        cutoff_hz=0):
    """Compute intensity-dependent low-pass filter.

    # Arguments
        log_new_frame: new frame in lin-log representation.
        lp_log_frame:
        inten01: the scaling of filter time constant array, or None to not scale
        delta_time:
        cutoff_hz:

    # Returns
        new_lp_log_frame
    """
    if cutoff_hz <= 0:
        # unchanged
        return log_new_frame

    # else low pass
    tau = 1/(math.pi*2*cutoff_hz)

    # make the update proportional to the local intensity
    # the more intensity, the shorter the time constant
    if inten01 is not None:
        eps = inten01*(delta_time/tau)
        max_eps = torch.max(eps)
        if max_eps >0.3:
            IIR_MAX_WARNINGS = 10
            if low_pass_filter.iir_warning_count<IIR_MAX_WARNINGS:
                logger.warning(f'IIR lowpass filter update has large maximum update eps={max_eps:.2f} from delta_time/tau={delta_time:.3g}/{tau:.3g}')
                low_pass_filter.iir_warning_count+=1
                if low_pass_filter.iir_warning_count==IIR_MAX_WARNINGS:
                    logger.warning(f'Supressing further warnings about inaccurate IIR lowpass filtering; check timestamp resolution and DVS photoreceptor cutoff frequency')

        eps = torch.clamp(eps, max=1)  # keep filter stable
    else:
        eps=delta_time/tau

    # first internal state is updated
    new_lp_log_frame = (1-eps)*lp_log_frame+eps*log_new_frame

    # then 2nd internal state (output) is updated from first
    # Note that observations show that one pole is nearly always dominant,
    # so the 2nd stage is just copy of first stage

    # (1-eps)*self.lpLogFrame1+eps*self.lpLogFrame0 # was 2nd-order,
    # now 1st order.

    return new_lp_log_frame

low_pass_filter.iir_warning_count=0


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
    """
        Compute event maps, i.e. 2d arrays of [width,height] containing quantized number of ON and OFF events.

    Args:
        diff_frame:  the input difference frame between stored log intensity and current frame log intensity [width, height]
        pos_thres:  ON threshold values [width, height]
        neg_thres:  OFF threshold values [width, height]

    Returns:
        pos_evts_frame, neg_evts_frame;  2d Tensors of integer ON and OFF event counts
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


def compute_photoreceptor_noise_voltage(shot_noise_rate_hz, f3db, sample_rate_hz, pos_thr, neg_thr, sigma_thr) -> float:
    """
     Computes the necessary photoreceptor noise voltage to result in observed shot noise rate at low light intensity.
     This computation relies on the known f3dB photoreceptor lowpass filter cutoff frequency and the known (nominal) event threshold.
     emulator.py injects Gaussian distributed noise to the photoreceptor that should in principle generate the desired shot noise events.

     See the file media/noise_event_rate_simulation.xlsx for the simulation data and curve fit.

    Parameters
    -----------
     shot_noise_rate_hz: float
        the desired pixel shot noise rate in hz
     f3db: float
        the 1st-order IIR RC lowpass filter cutoff frequency in Hz
     sample_rate_hz: float
        the sample rate (up-sampled frame rate) before IIR lowpassing the noise
     pos_thr:float
        on threshold in ln units
     neg_thr:float
        off threshold in ln units. The on and off thresholds are averaged to obtain a single threshold.
     sigma_thr: float
        the std deviations of the thresholds

    Returns
    -----------
    float
         Noise signal Gaussian RMS value in log_e units, to be added as Gaussian source directly to log photoreceptor output signal
    """

    def compute_vn_from_log_rate_per_hz(thr, x):
        # y = log10(thr/Vn)
        # x = log10(Rn/f3db)
        # see the plot Fig. 3 from Graca, Rui, and Tobi Delbruck. 2021. “Unraveling the Paradox of Intensity-Dependent DVS Pixel Noise.” arXiv [eess.SY]. arXiv. http://arxiv.org/abs/2109.08640.
        # the fit is computed in media/noise_event_rate_simulation.xlsx spreadsheet
        y = -0.0026 * x ** 3 - 0.036 * x ** 2 - 0.1949 * x + 0.321
        thr_per_vn = 10 ** y  # to get thr/vn
        vn = thr / thr_per_vn  # compute necessary vn to give us this noise rate per pixel at this pixel bandwidth
        return vn

    # check if we already estimated the required noise for this sample rate
    if not compute_photoreceptor_noise_voltage.last_sample_rate is None:
        diff=np.abs(sample_rate_hz/compute_photoreceptor_noise_voltage.last_sample_rate-1)
        if diff<0.1:
            return compute_photoreceptor_noise_voltage.last_vn # return cached value

    rate_per_bw= (shot_noise_rate_hz / f3db) / 2 # simulation data are on ON event rates, divide by 2 here to end up with correct total rate
    if rate_per_bw>0.5:
        logger.warning(f'shot noise rate per hz of bandwidth is larger than 0.1 (rate_hz={shot_noise_rate_hz} Hz, 3dB bandwidth={f3db} Hz)')
    x=math.log10(rate_per_bw)
    if x<-5.0:
        logger.warning(f'desired noise rate of {shot_noise_rate_hz}Hz is too low to accurately compute a threshold value')
    elif x>0.0:
        logger.warning(f'desired noise rate of {shot_noise_rate_hz}Hz is too large to accurately compute a threshold value')

    # now we need to numerically estimate the required Vnrms given the thresholds and the sigma thresholds,
    # since the noise rate varies dramatically with threshold
    N=300 # num samples
    pos_samps=pos_thr+sigma_thr*np.random.default_rng().standard_normal(N)
    neg_samps=neg_thr+sigma_thr*np.random.default_rng().standard_normal(N)
    thrs=np.vstack((pos_samps,neg_samps))
    mins=np.min(thrs,axis=0)
    vns=np.zeros_like(mins)
    for i in range(N):
        thr=mins[i]

        vn = compute_vn_from_log_rate_per_hz(thr, x)
        vns[i]=vn

    vn=np.mean(vns)
    # now we need to find the scaling factor from white noise to get the correct noise vn after RC lowpass.
    # # to get this NEB factor, we generate white samples here, lowpass filter them the same exact way
    # as we do in the emulator (i.e. with same IIR time constant and sample rate)
    # compute the variance, and scale the amplitude to give us vn
    compute_photoreceptor_noise_voltage.last_sample_rate=sample_rate_hz
    tau=1/(f3db*2*math.pi)
    dt=1/sample_rate_hz
    t=np.arange(0,1000*tau,dt)
    rin = vn*np.random.default_rng().standard_normal(t.shape) # generated Gaussian random sequence with amplitude vn RMS
    rms_in=np.std(rin) # check the RMS, should be vn
    rout=np.zeros_like(rin)
    # RC lowpass the noise
    eps=dt/tau
    eps_limit=.1
    if eps>eps_limit:
        logger.warning(f'\neps={eps:.3f} for IIR lowpass is >{eps_limit}, either reduce timestep (currently {dt:.3f}s) (using higher frame rate) or decrease cutuff_hz (currently {f3db:.3f} Hz)'
                       f'\n\tExpect the generated shot noise rate to be significantly lower than the desired rate.'
                       f'\n\tConsider not using --photoreceptor_noise option if you only want simple Poisson shot noise without temporal correlation of lowpass filtering and ON/OFF events.')
    rout[0]=0 # init value is mean 0
    # lp filter the sequence with same tau and dt as v2e
    for i in range(1,len(rin)):
        rout[i]=rout[i-1]*(1-eps)+rin[i]*eps
    rms_out=np.std(rout) # compute the amplitude of this noise
    scale=rms_in/rms_out #
    vnscaled=scale*vn # divide the computed vn to get the necessary vn to add before RC lowpass filtering
    new_rms_out=np.std(scale*rin) # check RMS of scaled noise

    compute_photoreceptor_noise_voltage.last_vn=vnscaled
    # rout*=vnscaled
    # stdout=np.std(rout)
    # import matplotlib.pyplot as plt
    # plt.plot(t,rin,t,rout)
    # plt.xlabel('time (s)')
    # plt.ylabel('filtered noise')
    # plt.show()
    if not compute_photoreceptor_noise_voltage.vrms_computation_printed:
        logger.info(
        f'For desired shot_noise_rate_hz={shot_noise_rate_hz} Hz, computed photoreceptor_noise_rms={vn:.3f} in ln units,'
        f' scaled by factor {scale:.3f} to {vnscaled:.3f} before 1st-order lowpass with sample rate {sample_rate_hz:.3} Hz, '
        f'sample interval dt={dt*1000:.3f} ms,'
        f', cutoff_hz={f3db} Hz, tau={tau*1000:.3f} ms,  Rn/f3dB={rate_per_bw:.3g} Hz, '
        f' and nominal on/off threshold={pos_thr}/{neg_thr} +/- {sigma_thr:.3f} ln units.'
        # f' The sample lowpass filtered has RMS amplitude {stdout:.3f}.'
        )
        compute_photoreceptor_noise_voltage.vrms_computation_printed=True
    return vnscaled

compute_photoreceptor_noise_voltage.vrms_computation_printed=False
compute_photoreceptor_noise_voltage.last_sample_rate=None
compute_photoreceptor_noise_voltage.last_vn=None

def generate_shot_noise(
        shot_noise_rate_hz,
        delta_time,
        shot_noise_inten_factor,
        inten01,
        pos_thres_pre_prob,
        neg_thres_pre_prob):
    """Generate shot noise.
    :param shot_noise_rate_hz: the rate per pixel in hz
    :param delta_time: the delta time for this frame in seconds
    :param shot_noise_inten_factor: factor to model the slight increase
        of shot noise with intensity when shot noise dominates at low intensity
    :param inten01: the pixel light intensities in this frame; shape is used to generate output
    :param pos_thres_pre_prob: per pixel factor to generate more
        noise from pixels with lower ON threshold: self.pos_thres_nominal/self.pos_thres
    :param neg_thres_pre_prob: same for OFF

    :returns: shot_on_coord, shot_off_coord, each are (h,w) arrays of on and off boolean True for noise events per pixel
    """
    # new shot noise generator, generate for the entire batch of iterations over this frame

    if shot_noise_rate_hz*delta_time>1:
        logger.warning(f'shot_noise_rate_hz*delta_time={shot_noise_rate_hz:.2f}*{delta_time:.2g}={shot_noise_rate_hz*delta_time:.2f} is too large, decrease timestamp resolution or sample rate')

    # shot noise factor is the probability of generating an OFF event in this frame (which is tiny typically)
    # we compute it by taking half the total shot noise rate (OFF only),
    # multiplying by the delta time of this frame,
    # and multiplying by the intensity factor
    # division by num_iter is correct if generate_shot_noise is called outside the iteration loop, unless num_iter=1 for calling outside loop
    shot_noise_factor = (
        (shot_noise_rate_hz/2)*delta_time) * \
        ((shot_noise_inten_factor-1)*inten01+1) # =1 for inten=0 and SHOT_NOISE_INTEN_FACTOR for inten=1 # TODO check this logic again, the shot noise rate should increase with intensity but factor is negative here

    # probability for each pixel is
    # dt*rate*nom_thres/actual_thres.
    # That way, the smaller the threshold,
    # the larger the rate
    one_minus_shot_ON_prob_this_sample = \
        1 - shot_noise_factor*pos_thres_pre_prob # ON shot events are generated when uniform sampled random number from range 0-1 is larger than this; the larger shot_noise_factor, the larger the noise rate
    shot_OFF_prob_this_sample = \
        shot_noise_factor*neg_thres_pre_prob # OFF shot events when 0-1 sample less than this

    # for shot noise generate rands from 0-1 for each pixel
    rand01 = torch.rand(
        size=inten01.shape,
        dtype=torch.float32,
        device=inten01.device)  # draw_frame samples

    # precompute all the shot noise cords, gets binary array size of chip
    shot_on_cord = torch.gt(
        rand01, one_minus_shot_ON_prob_this_sample)
    shot_off_cord = torch.lt(
        rand01, shot_OFF_prob_this_sample)

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

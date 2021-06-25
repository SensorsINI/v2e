"""Collections of emulator utilities.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

import math
import torch
import torch.nn.functional as F


def lin_log(x, threshold=20):
    """
    linear mapping + logarithmic mapping.

    :param x: float or ndarray
        the input linear value in range 0-255
    :param threshold: float threshold 0-255
        the threshold for transisition from linear to log mapping
    """
    # converting x into np.float32.
    if x.dtype is not torch.float64:  # note float64 to get rounding to work
        x = x.double()

    f = (1./threshold) * math.log(threshold)

    y = torch.where(x <= threshold, x*f, torch.log(x))

    # important, we do a floating point round to some digits of precision
    # to avoid that adding threshold and subtracting it again results
    # in different number because first addition shoots some bits off
    # to never-never land, thus preventing the OFF events
    # that ideally follow ON events when object moves by
    rounding = float(10**8)
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
    eps = torch.clamp(eps, max=1)  # keep filter stable
    #  eps = torch.where(
    #      eps > 1,
    #      torch.tensor(1, dtype=eps.dtype, device=eps.device),
    #      eps)

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
                          pos_thres_nominal):
    """Subtract leak current from base log frame."""

    delta_leak = delta_time*leak_rate_hz*pos_thres_nominal  # scalars

    return base_log_frame - delta_leak


def compute_event_map(diff_frame, pos_thres, neg_thres):
    """Compute event map.

    Prepare positive and negative event frames that later will be used
    for generating events.
    """

    # extract positive and negative differences
    pos_frame = F.relu(diff_frame)
    neg_frame = F.relu(-diff_frame)
    #  pos_frame = torch.where(
    #      diff_frame > 0, diff_frame,
    #      torch.tensor(0, dtype=diff_frame.dtype, device=diff_frame.device))
    #  neg_frame = torch.where(
    #      diff_frame < 0, -1*diff_frame,
    #      torch.tensor(0, dtype=diff_frame.dtype, device=diff_frame.device))

    # compute quantized number of ON and OFF events for each pixel
    pos_evts_frame = (pos_frame // pos_thres).type(torch.int32)
    neg_evts_frame = (neg_frame // neg_thres).type(torch.int32)

    return pos_evts_frame, neg_evts_frame


def generate_shot_noise(
        inten01,
        base_log_frame,
        shot_noise_rate_hz,
        delta_time,
        num_iters,
        pos_thres_pre_prob,
        pos_thres,
        neg_thres_pre_prob,
        neg_thres,
        ts):
    """Generate shot noise.

    """
    # the right device
    device = inten01.device

    # NOISE: add temporal noise here by
    # simple Poisson process that has a base noise rate
    # self.shot_noise_rate_hz.
    # If there is such noise event,
    # then we output event from each such pixel

    # the shot noise rate varies with intensity:
    # for lowest intensity the rate rises to parameter.
    # the noise is reduced by factor
    # SHOT_NOISE_INTEN_FACTOR for brightest intensities
    SHOT_NOISE_INTEN_FACTOR = 0.25
    shot_noise_factor = (
        (shot_noise_rate_hz/2)*delta_time/num_iters) * \
        ((SHOT_NOISE_INTEN_FACTOR-1)*inten01+1)
    # =1 for inten=0 and SHOT_NOISE_INTEN_FACTOR for inten=1

    rand01 = torch.rand(
        size=inten01.shape,
        dtype=torch.float32).to(device)  # draw samples

    # probability for each pixel is
    # dt*rate*nom_thres/actual_thres.
    # That way, the smaller the threshold,
    # the larger the rate
    shot_ON_prob_this_sample = shot_noise_factor*pos_thres_pre_prob
    # array with True where ON noise event
    shot_ON_cord = rand01 > (1-shot_ON_prob_this_sample)

    shot_OFF_prob_this_sample = shot_noise_factor*neg_thres_pre_prob
    # array with True where OFF noise event
    shot_OFF_cord = rand01 < shot_OFF_prob_this_sample

    # get shot noise event ON and OFF cordinates
    shot_ON_xy = shot_ON_cord.nonzero(as_tuple=True)
    shot_ON_count = shot_ON_xy[0].shape[0]

    shot_OFF_xy = shot_OFF_cord.nonzero(as_tuple=True)
    shot_OFF_count = shot_OFF_xy[0].shape[0]

    #  self.num_events_on += shotOnCount
    #  self.num_events_off += shotOffCount
    #  self.num_events_total += shotOnCount+shotOffCount

    if shot_ON_count > 0:
        shot_ON_events = torch.hstack(
            (torch.ones((shot_ON_count, 1),
                        dtype=torch.float32).to(device)*ts,
             shot_ON_xy[1].unsqueeze(dim=1),
             shot_ON_xy[0].unsqueeze(dim=1),
             torch.ones((shot_ON_count, 1),
                        dtype=torch.float32).to(device)*1))
        base_log_frame += shot_ON_cord*pos_thres
    else:
        shot_ON_events = torch.zeros((0, 4), dtype=torch.float32).to(device)

    if shot_OFF_count > 0:
        shot_OFF_events = torch.hstack(
            (torch.ones((shot_OFF_count, 1),
                        dtype=torch.float32).to(device)*ts,
             shot_OFF_xy[1].unsqueeze(dim=1),
             shot_OFF_xy[0].unsqueeze(dim=1),
             torch.ones((shot_OFF_count, 1),
                        dtype=torch.float32).to(device)*-1))
        base_log_frame -= shot_OFF_cord*neg_thres
    else:
        shot_OFF_events = torch.zeros((0, 4), dtype=torch.float32).to(device)
    # end temporal noise

    return shot_ON_events, shot_OFF_events


if __name__ == "__main__":

    temp_input = torch.randint(0, 256, (1280, 720), dtype=torch.float32).cuda()

    for i in range(1000):
        temp_out = lin_log(temp_input, threshold=20)

    pass

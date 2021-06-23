"""Collections of emulator utilities.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

import numpy as np
import jax.numpy as jnp
from jax import random
from jax import jit
from jax.config import config

# configure jax
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')


def jax_lin_log(x, threshold=20):
    """
    linear mapping + logarithmic mapping.

    :param x: float or ndarray
        the input linear value in range 0-255
    :param threshold: float threshold 0-255
        the threshold for transisition from linear to log mapping
    """
    # converting x into np.float32.
    if x.dtype is not jnp.float64:  # note float64 to get rounding to work
        x = x.astype(jnp.float64)

    f = (1/threshold) * jnp.log(threshold)

    y = jnp.where(x <= threshold, x*f, jnp.log(x))

    # important, we do a floating point round to some digits of precision
    # to avoid that adding threshold and subtracting it again results
    # in different number because first addition shoots some bits off
    # to never-never land, thus preventing the OFF events
    # that ideally follow ON events when object moves by
    y = jnp.around(y, 8)

    return y


def jax_rescale_intensity_frame(new_frame):
    """Rescale intensity frames.

    make sure we get no zero time constants
    limit max time constant to ~1/10 of white intensity level
    """
    return new_frame+20/275.


def jax_low_pass_filter(
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
    tau = 1/(jnp.pi*2*cutoff_hz)

    # make the update proportional to the local intensity
    # the more intensity, the shorter the time constant
    eps = inten01*(delta_time/tau)
    eps = jnp.where(eps > 1, 1, eps)

    # first internal state is updated
    new_lp_log_frame0 = (1-eps)*lp_log_frame0+eps*log_new_frame

    # then 2nd internal state (output) is updated from first
    # Note that observations show that one pole is nearly always dominant,
    # so the 2nd stage is just copy of first stage
    new_lp_log_frame1 = lp_log_frame0
    # (1-eps)*self.lpLogFrame1+eps*self.lpLogFrame0 # was 2nd-order,
    # now 1st order.

    return new_lp_log_frame0, new_lp_log_frame1


def jax_subtract_leak_current(base_log_frame,
                              leak_rate_hz,
                              delta_time,
                              pos_thres_nominal):
    """Subtract leak current from base log frame."""

    delta_leak = delta_time*leak_rate_hz*pos_thres_nominal  # scalars

    return base_log_frame - delta_leak


def jax_compute_event_map(diff_frame, pos_thres, neg_thres):
    """Compute event map.

    Prepare positive and negative event frames that later will be used
    for generating events.
    """

    # extract positive and negative differences
    pos_frame = jnp.where(diff_frame > 0, diff_frame, 0)
    neg_frame = jnp.where(diff_frame < 0, -1*diff_frame, 0)

    # compute quantized number of ON and OFF events for each pixel
    pos_evts_frame = (pos_frame // pos_thres).astype(jnp.int32)
    neg_evts_frame = (neg_frame // neg_thres).astype(jnp.int32)

    # max number of iterations
    num_iters = max(pos_evts_frame.max(), neg_evts_frame.max())

    return pos_evts_frame, neg_evts_frame, num_iters


def jax_generate_shot_noise(
        inten01,
        base_log_frame,
        shot_noise_rate_hz,
        delta_time,
        num_iters,
        pos_thres_pre_prob,
        pos_thres,
        neg_thres_pre_prob,
        neg_thres,
        ts,
        jax_key):
    """Generate shot noise.

    """
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
    shotNoiseFactor = (
        (shot_noise_rate_hz/2)*delta_time/num_iters) * \
        ((SHOT_NOISE_INTEN_FACTOR-1)*inten01+1)
    # =1 for inten=0 and SHOT_NOISE_INTEN_FACTOR for inten=1

    rand01 = random.uniform(
        key=jax_key,
        shape=inten01.shape,
        dtype=jnp.float32)  # draw samples

    # probability for each pixel is
    # dt*rate*nom_thres/actual_thres.
    # That way, the smaller the threshold,
    # the larger the rate
    shotOnProbThisSample = shotNoiseFactor*pos_thres_pre_prob
    # array with True where ON noise event
    shot_on_cord = rand01 > (1-shotOnProbThisSample)

    shotOffProbThisSample = shotNoiseFactor*neg_thres_pre_prob
    # array with True where OFF noise event
    shot_off_cord = rand01 < shotOffProbThisSample

    shotOnXy = shot_on_cord.nonzero()
    shotOnCount = shotOnXy[0].shape[0]

    shotOffXy = shot_off_cord.non_zero()
    shotOffCount = shotOffXy[0].shape[0]

    #  self.num_events_on += shotOnCount
    #  self.num_events_off += shotOffCount
    #  self.num_events_total += shotOnCount+shotOffCount

    pos_thr = shot_on_cord*pos_thres
    neg_thr = shot_off_cord*neg_thres

    if shotOnCount > 0:
        shotONEvents = np.hstack(
            (np.ones((shotOnCount, 1), dtype=np.float32)*ts,
             shotOnXy[1][..., np.newaxis],
             shotOnXy[0][..., np.newaxis],
             np.ones((shotOnCount, 1), dtype=np.float32)*1))

        base_log_frame += pos_thr
    if shotOffCount > 0:
        shotOFFEvents = np.hstack(
            (np.ones((shotOffCount, 1), dtype=np.float32)*ts,
             shotOffXy[1][..., np.newaxis],
             shotOffXy[0][..., np.newaxis],
             np.ones((shotOffCount, 1), dtype=np.float32)*-1))

        base_log_frame -= neg_thr
    # end temporal noise

    return shotONEvents, shotOFFEvents


# JAX JIT compiled functions

# Linear Log mapping
lin_log = jit(jax_lin_log)

# rescale intensity frame
rescale_intensity_frame = jit(jax_rescale_intensity_frame)

# intensity dependent low pas filter
low_pass_filter = jit(jax_low_pass_filter)

# subtract leak current
subtract_leak_current = jit(jax_subtract_leak_current)

# compute event map
compute_event_map = jit(jax_compute_event_map)

# generate shot noise
generate_shot_noise = jit(jax_generate_shot_noise)


if __name__ == "__main__":

    # jax
    key = random.PRNGKey(0)
    temp_input = random.randint(key, (1280, 720), 0, 256).astype(np.float32)

    for i in range(1000):
        #  temp_out = jax_lin_log(temp_input, threshold=20)
        temp_out = lin_log(temp_input, threshold=20)

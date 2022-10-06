# v1.6.2


## Bug fixes

 - Fix x,y output order in ae_text_output
 - Add warnings about extraneous extra arguments if there is no synthetic input class to pass them to
 - Document the order of fields in event generation in emulator.py
 - Fixed usage command to correctly show DVS output size options and to warn users about missing output size
 - Fixed docstrings for methods to correct x,y order of event fields

# v1.6.0

##  New features
  - **HDR (high dynamic range)**: The **_--hdr option_** treats inputs as logarithmic gray scale. See  [scripts/particles.py](https://github.com/SensorsINI/v2e/blob/master/scripts/particles.py) for script that that uses the --hdr argument to generate moving particles with very high brightness compared to background. At line 77, see
```python
       if self.parent_args.hdr:
            self.bg=np.log(self.bg)
            self.fg=np.log(self.fg)
```
  - **More accurate shot noise modeling**: The **-_-photoreceptor_noise_** option simulates shot noise events by injecting a 1st-order IIR lowpass (RC) filtered white noise source right after photoreceptor lowpass filtering (the noise is summed here to avoid intensity-dependent lowpass filtering of the noise). The result is more realistic statistics of shot noise events.
##    Bug fixes
  - fixed DDD conversion script [ddd/ddd_extract_data.py](https://github.com/SensorsINI/v2e/blob/master/dataset_scripts/ddd/ddd_extract_data.py#L12-L12)  to work properly.
  - improved argument usage descriptions.
  - fixed order of x,y events in AEDAT text output in [v2ecore/output/ae_text_output.py](https://github.com/SensorsINI/v2e/blob/master/dataset_scripts/ddd/ddd_extract_data.py#L12-L12).
  - fixed several synthetic input scripts to work and include option arguments
  - fixed logging to log the script extra options

# v1.5.0
##    New Features
  - v2e can display and save as AVI videos all the internal DVS model states via the **_--show_dvs_model_state_** and **_--save_dvs_model_state_** options.
  - v2e can model the **CSDVS** (center-surround DVS) proposed in Delbruck, Tobi, Chenghan Li, Rui Graca, and Brian Mcreynolds. 2022. “_Utility and Feasibility of a Center Surround Event Camera_.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/2202.13076.
##    Bug fixes
- More improvements to user experience with improved CUDA supported pytorch installation instructions and warnings printed if CUDA is not available.
- More bug fixes for synthetic input script handling and more example scripts, e.g. _spots.py_.
- Displayed DVS model state cv2 windows can now be dragged and resized by included cv2.pollkey() calls periodically.


# v1.4.3
##    New Features
  - Try [opening v2e in google colab](https://colab.research.google.com/drive/1czx-GJnx-UkhFVBbfoACLVZs8cYlcr_M?usp=sharing).
  - There are improved noise models of leak and shot noise.
    Both now model frozen log normal rate distributions; see the **_--noise_rate_cov_decades_** option.
    The leak noise events have timing jitter that matches observed jitter; see **_--leak_jitter_fraction_** option.
  - v2e models finite refractory period; see the **_--refractory_period_** option.
  - _setup.py_ includes dependencies and now installs _v2e_ as a script to your conda environment to run from command line.
  -  HDF output will also output the frames, to allow modeling of DAVIS cameras with APS+DVS output.
  - Added ability to generate input frames from python class.
    See [base_synthetic_input.py](https://github.com/SensorsINI/v2e/blob/master/v2ecore/base_synthetic_input.py) and command line argument **--synthetic_input**=<module_name>.
  - Synthetic input can recieve command line arguments; see [particle.py](https://github.com/SensorsINI/v2e/blob/master/scripts/particles.py).
  - Desktop notification is  generated for long running conversion completion.
  - Ability to crop input frames added; see --crop_input option.
##    Bug fixes
- Better argument checking and error reporting.
- For input folder, --start option skips frames without needing to load them from disk for huge speedup.


# v1.4.2
##    New Features
- The `EventEmulator` is now accelerated by PyTorch. If GPU is not available, the emulation will use CPU. Early results show that the emulation on GPU and CPU has similar performance.
- When saving with HDF5, the emulator will also optionally save three additional dataset: 1. `frames` luma frames, 2. `frame_ts` frame timestamps and 3. "frame_idxs" corresponding event index for dual modality saving and faster indexing. At default, this is disabled for faster and cheaper emulation. Can be enabled by adding `--davis_output` flag. This feature is currently only available for video, not synthetic input.
- More realistic leak events model. Add two options: `leak_jitter_fraction` and `noise_rate_cov_decades` to control leak events variation. Disable by setting them to 0.
- Add `refractory_period`: events that arrives at time smaller than the `refractory_period` will be filtered. Disable by setting it to 0.

##    Bug fixes
- A shot noise event and a regular event can occur at the same timestamp. This is fixed. Now if these two types of events occurred together, only one event will be output.

# v1.4.1
##    New Features
- a new option --dvs_emulator_seed is introduced. This option allows the user to use a fixed random seed. The default is 0 which means the emulator uses a system defined random seed.

##    Bug fixes
- Shot noise events polarity was wrongly set to 1 always. This makes the shot noise events are always positive events. Fixed this in emulator.
- If not using fixed chipset, v2e was wrongly quitting. This is partially fixed by removing a quitting behavior and resetting both output_width an output_height to None. However, since the AEDAT2 saving requires standard chipset, we recommend user to use HDF5 saving method. For synthetic inputs, please set both output_width and output_width (or using a standard chipset), otherwise there will be errors.
- Event Renderer always at 30 fps. Fixed this so that user can preset the frame rate.
- The argument skip_video_output still outputs original video and the slow motion video. This is fixed so the behavior matches with the description.

# v1.4.0
##    New features
- Input frames can be generated by python module that is specified on command line, see moving-dot.py for sample.
- Video files output can be suppressed with --skip_video_files.
- Write in place option --output_in_place added to save v2e output in same folder as source video.
- The input can be either a single video file or a folder that contains a list of ordered image files. The user has to set --input_frame_rate manually when the video is presented as a folder.
- Change package name from v2e to v2ecore so that it can be run systemwide.
- Several new warnings and checks added for bad combination of --timestamp_resolution and --cutoff_hz in relation to IIR filtering.
- Added --disable_slomo switch to completely disable frame interpolation when user generates the src frames at sufficient sample rate.

##    Bug fixes
- Fixed option to specify both --auto_timestamp_resolution and --timestamp_resolution at same time, to limit maximum timestep for DVS events.
- Fixed warning messages to warn users about bad combination of upsampling frame rate and lowpass filtering.
- Fixed subtle timing problems with events that cause some skew for some videos that generate many events per frame.
- Fixed the problem when initializing the first frame that causes a burst of leak events. Set the leak initial state accounting for random pixel ON thresholds.
- Changed so that DVS parameter selection is by default None, so that user selections are not overridden. Added warning that such options will be overridden by using the --dvs_params option.
- Set default 'noisy' shot noise rate to more realistic 5Hz/pixel value.
- Fixed (we hope) very subtle bug in generating ideal ON and OFF events where the ON events were not followed by OFF events for a moving white dot. This was caused by floating point roundoff that caused bits to disappear when adding and subtracting the threshold from the memorized log intensity frame, resulting in nonsymetrical output.
- Changed lin_log function to include floating point rounding to 5 digits precision, to prevent subtle and hard to understand effects in synthetic input.

# v1.3.1
- initial release to accompany the v2e arxiv paper

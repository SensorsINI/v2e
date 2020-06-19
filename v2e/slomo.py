"""Super SloMo class for dvs simulator project.
    @author: Zhe He
    @contact: hezhehz@live.cn
    @latest update: 2019-May-27th

    lightly modified based on this implementation: \
        https://github.com/avinashpaliwal/Super-SloMo
"""

import torch
import os
import numpy as np
import cv2
import glob

from tqdm import tqdm

import torchvision.transforms as transforms

from v2e.v2e_utils import video_writer
import v2e.dataloader as dataloader
import v2e.model as model

from PIL import Image
import logging
import atexit
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning,
    module="torch.nn.functional")
# https://github.com/fastai/fastai/issues/2370

logger = logging.getLogger(__name__)


class SuperSloMo(object):
    """Super SloMo class
        @author: Zhe He
        @contact: hezhehz@live.cn
        @latest update: 2019-May-27th
    """

    def __init__(
        self,
        model,
        slowdown_factor,
        batch_size=1,
        video_path=None,
        vid_orig='original.avi',
        vid_slomo='slomo.avi',
        preview=False,
        avi_frame_rate=30
    ):
        """
        init

        Parameters
        ----------
        model: str,
            path of the stored Pytorch checkpoint.
        slowdown_factor: int,
            slow motion factor.
        batch_size: int,
            batch size.
        video_path: str or None,
            str path to folder where you want videos of original and
            slomo video to be stored, else None
        vid_orig: str or None,
            name of output original (input) video at slo motion rate,
            needs video_path to be set too
        vid_slomo: str or None,
            name of slomo video file, needs video_path to be set too
        """

        if torch.cuda.is_available():
            self.device = "cuda:0"
            logger.info('CUDA available, running on GPU :-)')
        else:
            self.device = "cpu"
            logger.warning('CUDA not available, will be slow :-(')
        self.checkpoint = model
        self.batch_size = batch_size
        if not isinstance(slowdown_factor, int) or slowdown_factor < 2:
            raise ValueError(
                'slowdown_factor={} but must be an int value>1'
                .format(slowdown_factor))
        self.sf = slowdown_factor
        self.video_path = video_path
        self.preview = preview
        self.preview_resized = False
        self.vid_orig = vid_orig
        self.vid_slomo = vid_slomo
        self.avi_frame_rate = avi_frame_rate

        # initialize the Transform instances.
        self.to_tensor, self.to_image = self.__transform()
        self.ori_writer = None
        self.slomo_writer = None  # will be constructed on first need
        self.numOrigVideoFramesWritten = 0
        self.numSlomoVideoFramesWritten = 0

        atexit.register(self.cleanup)
        self.model_loaded = False

    def cleanup(self):
        logger.info("Closing video writers for original and slomo videos...")
        if self.ori_writer:
            logger.info(
                'closing original video AVI after '
                'writing {} frames'.format(self.numOrigVideoFramesWritten))
            self.ori_writer.release()
        if self.slomo_writer:
            logger.info(
                'closing slomo video AVI after '
                'writing {} frames'.format(self.numSlomoVideoFramesWritten))
            self.slomo_writer.release()
        cv2.destroyAllWindows()

    def __transform(self):
        """create the Transform instances.

        Returns
        -------
        to_tensor: Pytorch Transform instance.
        to_image: Pytorch Transform instance.
        """
        mean = [0.428]
        std = [1]
        normalize = transforms.Normalize(mean=mean, std=std)
        negmean = [x * -1 for x in mean]
        revNormalize = transforms.Normalize(mean=negmean, std=std)

        if (self.device == "cpu"):
            to_tensor = transforms.Compose([transforms.ToTensor()])
            to_image = transforms.Compose([transforms.ToPILImage()])
        else:
            to_tensor = transforms.Compose([transforms.ToTensor(),
                                            normalize])
            to_image = transforms.Compose([revNormalize,
                                           transforms.ToPILImage()])
        return to_tensor, to_image

    def __load_data(self, source_frame_path, frame_size):
        """Return a Dataloader instance, which is constructed with \
            APS frames.

        Parameters
        ---------
        images: np.ndarray, [N, W, H]
            input APS frames.

        Returns
        -------
        videoFramesloader: Pytorch Dataloader instance.
        frames.dim: new size.
        frames.origDim: original size.
        """
        #  frames = dataloader.Frames(images, transform=self.to_tensor)
        frames = dataloader.FramesDirectory(
            source_frame_path, frame_size, transform=self.to_tensor)
        videoFramesloader = torch.utils.data.DataLoader(
                frames,
                batch_size=self.batch_size,
                shuffle=False)
        return videoFramesloader, frames.dim, frames.origDim

    def __model(self, dim):
        """Initialize the pytorch model

        Parameters
        ---------
        dim: tuple
            size of resized images.

        Returns
        -------
        flow_estimator: nn.Module
        warpper: nn.Module
        interpolator: nn.Module
        """
        if not os.path.isfile(self.checkpoint):
            raise FileNotFoundError(
                'SuperSloMo model checkpoint '+str(self.checkpoint) +
                ' does not exist or is not readable')
        logger.info('loading SuperSloMo model from ' + str(self.checkpoint))

        flow_estimator = model.UNet(2, 4)
        flow_estimator.to(self.device)
        for param in flow_estimator.parameters():
            param.requires_grad = False
        interpolator = model.UNet(12, 5)
        interpolator.to(self.device)
        for param in interpolator.parameters():
            param.requires_grad = False

        warper = model.backWarp(dim[0],
                                dim[1],
                                self.device)
        warper = warper.to(self.device)

        # dict1 = torch.load(self.checkpoint, map_location='cpu')
        # fails intermittently on windows

        dict1 = torch.load(self.checkpoint, map_location=self.device)
        interpolator.load_state_dict(dict1['state_dictAT'])
        flow_estimator.load_state_dict(dict1['state_dictFC'])

        return flow_estimator, warper, interpolator

    def interpolate(self, source_frame_path, output_folder, frame_size):
        """Run interpolation. \
            Interpolated frames will be saved in folder self.output_path.

        Parameters
        ----------
        source_frame_path: path that contains source file
        output_folder:str, folder that stores the interpolated images,
            numbered 1:N*slowdown_factor.
        frame_size: tuple (width, height)


        Frames will include the input frames, i.e.
        if there are 2 input frames and slowdown_factor=10,
        there will be 10 frames written,
        starting with the first input frame, and ending before
        the 2nd input frame.

        If  slowdown factor=2, then the first output frame will be
        the first input frame, and the 2nd output frame will be
        a new synthetic frame halfway to the 2nd frame.

        If the slowdown_factor is 3, then there will
        the first input frame followed by 2 more interframes.

        The output will never include the 2nd input frame.

        Frames will include the input frames,
        i.e. if there are 2 input frames and slowdown_factor=10,
        there will be ?? frames written
        """
        if not output_folder:
            raise Exception(
                'output_folder is None; it must be supplied to store '
                'the interpolated frames')

        video_frame_loader, dim, ori_dim = self.__load_data(
            source_frame_path, frame_size)
        if not self.model_loaded:
            (self.flow_estimator, self.warper,
             self.interpolator) = self.__model(dim)
            self.model_loaded = True

        # construct AVI video output writer now that we know the frame size
        if self.video_path is not None and self.vid_orig is not None and \
                not self.ori_writer:
            self.ori_writer = video_writer(
                os.path.join(self.video_path, self.vid_orig),
                ori_dim[1],
                ori_dim[0], frame_rate=self.avi_frame_rate
            )

        if self.video_path is not None and self.vid_slomo is not None and \
                not self.slomo_writer:
            self.slomo_writer = video_writer(
                os.path.join(self.video_path, self.vid_slomo),
                ori_dim[1],
                ori_dim[0], frame_rate=self.avi_frame_rate
            )

        # prepare preview
        if self.preview:
            self.name = str(__file__)
            cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)

        frameCounter = 1
        # torch.cuda.empty_cache()
        with torch.no_grad():
            #  logger.debug(
            #      "using " + str(output_folder) +
            #      " to store interpolated frames")
            nImages = len(video_frame_loader)
            #  nImages = images.shape[0]
            disableTqdm = nImages <= max(self.batch_size, 4)
            unit='fr' if self.batch_size==1 else 'batch'
            for _, (frame0, frame1) in enumerate(
                    tqdm(video_frame_loader, desc='slomo-interp',
                         unit=unit, disable=disableTqdm), 0):

                I0 = frame0.to(self.device)
                I1 = frame1.to(self.device)
                # actual number of frames, account for <batch_size
                num_batch_frames = I0.shape[0]

                flowOut = self.flow_estimator(torch.cat((I0, I1), dim=1))
                F_0_1 = flowOut[:, :2, :, :]
                F_1_0 = flowOut[:, 2:, :, :]

                # for preview
                if self.preview:
                    start_frame_count = frameCounter
                # Generate intermediate frames
                for intermediateIndex in range(0, self.sf):
                    t = (intermediateIndex + 0.5) / self.sf
                    temp = -t * (1 - t)
                    fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

                    F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                    F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

                    g_I0_F_t_0 = self.warper(I0, F_t_0)
                    g_I1_F_t_1 = self.warper(I1, F_t_1)

                    intrpOut = self.interpolator(
                        torch.cat(
                            (I0, I1, F_0_1, F_1_0,
                             F_t_1, F_t_0, g_I1_F_t_1,
                             g_I0_F_t_0), dim=1))

                    F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                    F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                    V_t_0 = torch.sigmoid(intrpOut[:, 4:5, :, :])
                    V_t_1 = 1 - V_t_0

                    g_I0_F_t_0_f = self.warper(I0, F_t_0_f)
                    g_I1_F_t_1_f = self.warper(I1, F_t_1_f)

                    wCoeff = [1 - t, t]

                    Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f +
                            wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / \
                           (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

                    # Save intermediate frame
                    for batchIndex in range(num_batch_frames):

                        img = self.to_image(Ft_p[batchIndex].cpu().detach())
                        img_resize = img.resize(ori_dim, Image.BILINEAR)

                        save_path = os.path.join(
                            output_folder,
                            str(frameCounter + self.sf * batchIndex) + ".png")
                        img_resize.save(save_path)

                    frameCounter += 1

                # for preview
                if self.preview:
                    stop_frame_count = frameCounter

                    for frame_idx in range(
                            start_frame_count,
                            stop_frame_count+self.sf*(num_batch_frames-1)):
                        frame_path = os.path.join(
                            output_folder, str(frame_idx) + ".png")
                        frame = cv2.imread(frame_path)
                        cv2.imshow(self.name, frame)
                        if not self.preview_resized:
                            cv2.resizeWindow(self.name, 800, 600)
                            self.preview_resized = True
                        # wait minimally since interp takes time anyhow
                        cv2.waitKey(1)

                # Set counter accounting for batching of frames
                frameCounter += self.sf * (self.batch_size - 1)

            # write input frames into video
            # don't duplicate each frame if called using rotating buffer
            # of two frames in a row
            if self.ori_writer:
                src_files = sorted(
                    glob.glob("{}".format(source_frame_path)+"/*.npy"))

                for frame_idx, src_file_path in enumerate(
                        tqdm(src_files, desc='slomo-interp',
                             unit='fr', disable=disableTqdm), 0):
                    src_frame = np.load(src_file_path)
                    self.ori_writer.write(
                        cv2.cvtColor(src_frame, cv2.COLOR_GRAY2BGR))
                    self.numOrigVideoFramesWritten += 1

            frame_paths = self.__all_images(output_folder)
            # write slomo frames into video
            # will not duplicate frames if called in 2-frame loop (tobi thinks)
            if self.slomo_writer:
                # tqdm(frame_paths,desc='slomo-write-slomo-vid',unit='fr'):
                for path in frame_paths:
                    frame = self.__read_image(path)
                    self.slomo_writer.write(
                        cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
                    self.numSlomoVideoFramesWritten += 1
                    # if cv2.waitKey(int(1000/30)) & 0xFF == ord('q'):
                    #     break

    def __all_images(self, data_path):
        """Return path of all input images. Assume that the ascending order of
        file names is the same as the order of time sequence.

        Parameters
        ----------
        data_path: str
            path of the folder which contains input images.

        Returns
        -------
        List[str]
            sorted in numerical order.
        """
        images = glob.glob(os.path.join(data_path, '*.png'))
        if len(images) == 0:
            raise ValueError(("Input folder is empty or images are not in"
                              " 'png' format."))
        images_sorted = sorted(
                images,
                key=lambda line: int(line.split(os.sep)[-1].split('.')[0]))
        # only works for linux separators with /,
        # use os.sep according to
        # https://stackoverflow.com/questions/16010992
        # /how-to-use-directory-separator-in-both-linux-and-windows-in-python
        return images_sorted

    @staticmethod
    def __read_image(path):
        """Read image.

        Parameters
        ----------
        path: str
            path of image.

        Return
        ------
            np.ndarray
        """
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return img

    def get_interpolated_timestamps(self, ts):
        """ Interpolate the timestamps.

        Parameters
        ----------
        ts: np.array, np.float64,
            timestamps of input frames.

        Returns
        -------
        np.array, np.float64,
            interpolated timestamps.
        """
        new_ts = []
        for i in range(ts.shape[0] - 1):
            start, end = ts[i], ts[i + 1]
            interpolated_ts = np.linspace(
                start,
                end,
                self.sf,
                endpoint=False) + 0.5 * (end - start) / self.sf
            new_ts.append(interpolated_ts)
        new_ts = np.hstack(new_ts)

        return new_ts

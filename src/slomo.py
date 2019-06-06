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

import torchvision.transforms as transforms
import torch.nn.functional as F

import dataloader
import model

from PIL import Image
from tqdm import tqdm


def video_writer(output_path, height, width):
    """
    Return a video writer.
    @params:
        output_path: str,
            path to store output video.
        height: int,
            height of a frame.
        width: int,
            width of a frame.
    @return:
        an instance of cv2.VideoWriter.
    """

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
                output_path,
                fourcc,
                30.0,
                (width, height))
    return out


class SuperSloMo(object):
    """Super SloMo class
        @author: Zhe He
        @contact: hezhehz@live.cn
        @latest update: 2019-May-27th

        ------

        @methods:
    """

    def __init__(
        self,
        checkpoint,
        slow_factor,
        output_path,
        batch_size=1,
        video_path=None,
        rotate=False
    ):
        """
        init
        @params:
            checkpoint: str,
                path of the stored Pytorch checkpoint.
            slow_factor: int,
                slow motion factor.
            output_path: str,
                a temporary path to store interpolated frames.
            batch_size: int,
                batch size.
            video_path: str or None,
                str if videos need to be stored else None
            rotate: bool,
                True if frames need to be rotated else False
        """

        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.sf = slow_factor
        self.output_path = output_path
        self.video_path = video_path
        self.rotate = rotate

        # initialize the Transform instances.
        self.to_tensor, self.to_image = self.__transform()

    def __transform(self):
        """create the Transform instances.

            @Return:
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

    def __load_data(self, images):
        """Return a Dataloader instance, which is constructed with \
            APS frames.

            @Params:
                images: np.ndarray, [N, W, H]
                    input APS frames.

            @Return:
                videoFramesloader: Pytorch Dataloader instance.
                frames.dim: new size.
                frames.origDim: original size.
        """
        frames = dataloader.Frames(images, transform=self.to_tensor)
        videoFramesloader = torch.utils.data.DataLoader(
                frames,
                batch_size=self.batch_size,
                shuffle=False)
        return videoFramesloader, frames.dim, frames.origDim

    def __model(self, dim):
        """Initialize the pytorch model
            @Params:
                dim: tuple
                    size of resized images.
            @Return:
                flow_estimator: nn.Module
                warpper: nn.Module
                interpolator: nn.Module
        """

        flow_estimator = model.UNet(2, 4)
        flow_estimator.to(self.device)
        for param in flow_estimator.parameters():
            param.requires_grad = False
        interpolator = model.UNet(12, 5)
        interpolator.to(self.device)
        for param in interpolator.parameters():
            param.requires_grad = False

        warpper = model.backWarp(dim[0],
                                 dim[1],
                                 self.device)
        warpper = warpper.to(self.device)

        dict1 = torch.load(self.checkpoint, map_location='cpu')
        interpolator.load_state_dict(dict1['state_dictAT'])
        flow_estimator.load_state_dict(dict1['state_dictFC'])

        return flow_estimator, warpper, interpolator

    def interpolate(self, images):
        """Run interpolation.
            Interpolated frames will be saved in folder self.output_path.
            @Params:
                images: np.ndarray, [N, W, H]
        """

        video_frame_loader, dim, ori_dim = self.__load_data(images)
        flow_estimator, warpper, interpolator = self.__model(dim)

        frameCounter = 1

        with torch.no_grad():
            for _, (frame0, frame1) in enumerate(tqdm(video_frame_loader), 0):

                I0 = frame0.to(self.device)
                I1 = frame1.to(self.device)

                flowOut = flow_estimator(torch.cat((I0, I1), dim=1))
                F_0_1 = flowOut[:, :2, :, :]
                F_1_0 = flowOut[:, 2:, :, :]

                # Generate intermediate frames
                for intermediateIndex in range(0, self.sf):
                    t = (intermediateIndex + 0.5) / self.sf
                    temp = -t * (1 - t)
                    fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

                    F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                    F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

                    g_I0_F_t_0 = warpper(I0, F_t_0)
                    g_I1_F_t_1 = warpper(I1, F_t_1)

                    intrpOut = interpolator(
                        torch.cat(
                            (I0, I1, F_0_1, F_1_0,
                             F_t_1, F_t_0, g_I1_F_t_1,
                             g_I0_F_t_0), dim=1))

                    F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                    F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                    V_t_0 = torch.sigmoid(intrpOut[:, 4:5, :, :])
                    V_t_1 = 1 - V_t_0

                    g_I0_F_t_0_f = warpper(I0, F_t_0_f)
                    g_I1_F_t_1_f = warpper(I1, F_t_1_f)

                    wCoeff = [1 - t, t]

                    Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f +
                            wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / \
                        (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

                    # Save intermediate frame
                    for batchIndex in range(self.batch_size):

                        img = self.to_image(Ft_p[batchIndex].cpu().detach())
                        img_resize = img.resize(ori_dim, Image.BILINEAR)

                        save_path = os.path.join(
                            self.output_path,
                            str(frameCounter + self.sf * batchIndex) + ".png")
                        img_resize.save(save_path)
                    frameCounter += 1

                # Set counter accounting for batching of frames
                frameCounter += self.sf * (self.batch_size - 1)

        if self.video_path is not None:
            ori_writer = video_writer(
                os.path.join(self.video_path, "original.avi"),
                ori_dim[0],
                ori_dim[1]
            )

            slomo_writer = video_writer(
                os.path.join(self.video_path, "slomo.avi"),
                ori_dim[0],
                ori_dim[1]
            )

            # write input frames into video
            for frame in images:
                if self.rotate:
                    frame = np.rot90(frame, k=2)
                for _ in range(self.sf):
                    slomo_writer.write(
                        cv2.cvtColor(
                            frame,
                            cv2.COLOR_GRAY2BGR
                        )
                    )
                if cv2.waitKey(int(1000/30)) & 0xFF == ord('q'):
                    break

            frame_paths = self.__all_images(self.output_path)
            # write slomo frames into video
            for path in frame_paths:
                frame = self.__read_image(path)
                if self.rotate:
                    frame = np.rot90(frame, k=2)
                ori_writer.write(
                    cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                )
                if cv2.waitKey(int(1000/30)) & 0xFF == ord('q'):
                    break

    def __all_images(self, data_path):
        """Return path of all input images. Assume that the ascending order of
        file names is the same as the order of time sequence.

        @Args:
            data_path: str
                path of the folder which contains input images.
        @Return:
            List[str]
                sorted in numerical order.
        """
        images = glob.glob(os.path.join(data_path, '*.png'))
        if len(images) == 0:
            raise ValueError(("Input folder is empty or images are not in"
                              " 'png' format."))
        images_sorted = sorted(
                images,
                key=lambda line: int(line.split('/')[-1].split('.')[0]))
        return images_sorted

    @staticmethod
    def __read_image(path):
        """Read image.
        @Args:
            path: str
                path of image.
        @Return:
            np.ndarray
        """
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return img

    def get_ts(self, ts):
        """
        Interpolate the timestamps.
        @params:
            ts: np.array, np.float64,
                timestamps.
        @return:
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

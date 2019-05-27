"""Super SloMo class for dvs simulator project.
    @author: Zhe He
    @contact: hezhehz@live.cn
    @latest update: 2019-May-27th

    lightly modified based on this implementation: \
        https://github.com/avinashpaliwal/Super-SloMo
"""

import torch
import tqdm
import os

import torchvision.transforms as transforms
import torch.nn.functional as F

import dataloader
import model

from PIL import Image


class SuperSloMo(object):
    """Super SloMo class
        @author: Zhe He
        @contact: hezhehz@live.cn
        @latest update: 2019-May-27th

        ------

        @methods:
            

    """

    def __init__(self,
                 checkpoint,
                 slow_factor,
                 output_path,
                 batch_size=1):
        """init"""

        if torch.cude.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.sf = slow_factor
        self.output_path = output_path

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

    def run(self, images):
        """Run interpolation.
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

                # Save reference frames in output folder
                # for batchIndex in range(args.batch_size):
                #     (TP(frame0[batchIndex].detach())).resize(videoFrames.origDim, Image.BILINEAR).save(os.path.join(outputPath, str(frameCounter + args.sf * batchIndex) + ".png"))
                # frameCounter += 1

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
                    V_t_0 = F.sigmoid(intrpOut[:, 4:5, :, :])
                    V_t_1 = 1 - V_t_0

                    g_I0_F_t_0_f = warpper(I0, F_t_0_f)
                    g_I1_F_t_1_f = warpper(I1, F_t_1_f)

                    wCoeff = [1 - t, t]

                    Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

                    # Save intermediate frame
                    for batchIndex in range(self.batch_size):

                        img = self.to_image(Ft_p[batchIndex].cpu().detach())
                        img_resize = img.resize(ori_dim, Image.BILINEAR)
                        save_path = os.path.join(
                            self.output_path,
                            str(frameCounter + self.sf * batchIndex) + ".png")
                        img_resize.save(save_path)
                        # (self.to_image(Ft_p[batchIndex].cpu().detach())).resize(ori_dim, Image.BILINEAR).save(os.path.join(self.output_path, str(frameCounter + self.sf * batchIndex) + ".png"))
                    frameCounter += 1

                # Set counter accounting for batching of frames
                frameCounter += self.sf * (self.batch_size - 1)

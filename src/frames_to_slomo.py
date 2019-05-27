#!/usr/bin/env python3

import argparse
import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np

import model
import dataloader

from PIL import Image

from tqdm import tqdm


# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",
                    type=str,
                    required=True,
                    help='path of checkpoint for pretrained model')
parser.add_argument("--fps",
                    type=float,
                    default=30,
                    help='specify fps of output video. Default: 30.')
parser.add_argument("--sf",
                    type=int,
                    required=True,
                    help=('specify the slomo factor N. This will increase the',
                          ' frames by Nx. Example sf=2 ==> 2x frames'))
parser.add_argument("--batch_size",
                    type=int,
                    default=1,
                    help=('Specify batch size for faster conversion. This',
                          ' will depend on your cpu/gpu memory. Default: 1'))
parser.add_argument("--output",
                    type=str,
                    default="output.mp4",
                    help='Specify output file name. Default: output.mp4')
args = parser.parse_args()


def check():
    """
    Checks the validity of commandline arguments.

    Parameters
    ----------
        None

    Returns
    -------
        error : string
            Error message if error occurs otherwise blank string.
    """

    error = ""
    if (args.sf < 2):
        error = "Error: --sf/slomo factor has to be atleast 2"
    if (args.batch_size < 1):
        error = "Error: --batch_size has to be atleast 1"
    if (args.fps < 1):
        error = "Error: --fps has to be atleast 1"
    return error


def main():
    # Check if arguments are okay
    error = check()
    if error:
        print(error)
        exit(1)

    outputPath = os.path.join("tmpSuperSloMo")
    os.mkdir(outputPath)

    # Initialize transforms
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mean = [0.428]
    std = [1]
    normalize = transforms.Normalize(mean=mean,
                                     std=std)

    negmean = [x * -1 for x in mean]
    revNormalize = transforms.Normalize(mean=negmean, std=std)

    # Temporary fix for issue #7
    # https://github.com/avinashpaliwal/Super-SloMo/issues/7 -
    # - Removed per channel mean subtraction for CPU.
    if (device == "cpu"):
        transform = transforms.Compose([transforms.ToTensor()])
        TP = transforms.Compose([transforms.ToPILImage()])
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        normalize])
        TP = transforms.Compose([revNormalize, transforms.ToPILImage()])

    # Load data
    array = np.load(args.array)
    frames = dataloader.Frames(array, transform=transform)

    videoFramesloader = torch.utils.data.DataLoader(
            frames,
            batch_size=args.batch_size,
            shuffle=False)

    # Initialize model
    flowComp = model.UNet(2, 4)
    flowComp.to(device)
    for param in flowComp.parameters():
        param.requires_grad = False
    ArbTimeFlowIntrp = model.UNet(12, 5)
    ArbTimeFlowIntrp.to(device)
    for param in ArbTimeFlowIntrp.parameters():
        param.requires_grad = False

    flowBackWarp = model.backWarp(frames.dim[0],
                                  frames.dim[1],
                                  device)
    flowBackWarp = flowBackWarp.to(device)

    dict1 = torch.load(args.checkpoint, map_location='cpu')
    ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
    flowComp.load_state_dict(dict1['state_dictFC'])

    # Interpolate frames
    frameCounter = 1

    with torch.no_grad():
        for _, (frame0, frame1) in enumerate(tqdm(videoFramesloader), 0):

            I0 = frame0.to(device)
            I1 = frame1.to(device)

            flowOut = flowComp(torch.cat((I0, I1), dim=1))
            F_0_1 = flowOut[:, :2, :, :]
            F_1_0 = flowOut[:, 2:, :, :]

            # Save reference frames in output folder
            # for batchIndex in range(args.batch_size):
            #     (TP(frame0[batchIndex].detach())).resize(videoFrames.origDim, Image.BILINEAR).save(os.path.join(outputPath, str(frameCounter + args.sf * batchIndex) + ".png"))
            # frameCounter += 1

            # Generate intermediate frames
            for intermediateIndex in range(0, args.sf):
                t = (intermediateIndex + 0.5) / args.sf
                temp = -t * (1 - t)
                fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

                F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

                g_I0_F_t_0 = flowBackWarp(I0, F_t_0)
                g_I1_F_t_1 = flowBackWarp(I1, F_t_1)
                
                intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))
                    
                F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                V_t_0   = F.sigmoid(intrpOut[:, 4:5, :, :])
                V_t_1   = 1 - V_t_0
                    
                g_I0_F_t_0_f = flowBackWarp(I0, F_t_0_f)
                g_I1_F_t_1_f = flowBackWarp(I1, F_t_1_f)
                
                wCoeff = [1 - t, t]

                Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

                # Save intermediate frame
                for batchIndex in range(args.batch_size):
                    (TP(Ft_p[batchIndex].cpu().detach())).resize(frames.origDim, Image.BILINEAR).save(os.path.join(outputPath, str(frameCounter + args.sf * batchIndex) + ".png"))
                frameCounter += 1
            
            # Set counter accounting for batching of frames
            frameCounter += args.sf * (args.batch_size - 1)

    # Remove temporary files
    # rmtree(extractionDir)

    exit(0)


main()

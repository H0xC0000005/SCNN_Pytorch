import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SCNN(nn.Module):
    def __init__(
            self,
            input_size,
            ms_ks=9,
            pretrained=True
    ):
        """
        Argument
            ms_ks: kernel size in message passing conv
        """
        super(SCNN, self).__init__()
        self.message_passing = None
        self.conv_after_backbone = None
        self.final_conv = None
        self.final_pool = None
        self.final_fc = None
        self.fc_input_feature = None
        self.backbone = None
        self.pretrained = pretrained
        self.net_init(input_size, ms_ks)
        if not pretrained:
            self.weight_init()

        """
        predefined hyperparams
        """
        self.scale_background = 0.4  # ?
        self.scale_seg = 1.0  # ?
        self.scale_exist = 0.1  # ?

        # predefined loss (cross entropy)
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([self.scale_background, 1, 1, 1, 1]))
        self.bce_loss = nn.BCELoss()

    def forward(self, img, seg_gt=None, exist_gt=None):
        # CNN backbone as SCNN can just be the last layer of a feature mapping
        x = self.backbone(img)
        x = self.conv_after_backbone(x)
        x = self.message_passing_forward(x)
        x = self.final_conv(x)

        seg_pred = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
        x = self.final_pool(x)
        x = x.view(-1, self.fc_input_feature)
        exist_pred = self.final_fc(x)

        if seg_gt is not None and exist_gt is not None:
            loss_seg = self.ce_loss(seg_pred, seg_gt)
            loss_exist = self.bce_loss(exist_pred, exist_gt)
            loss = loss_seg * self.scale_seg + loss_exist * self.scale_exist
        else:
            print(f"warning: in SCNN forward() seg_gt and exist_gt have at least 1 None so "
                  f"all losses are set to 0")
            loss_seg = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss_exist = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss = torch.tensor(0, dtype=img.dtype, device=img.device)

        return seg_pred, exist_pred, loss_seg, loss_exist, loss

    def message_passing_forward(self, x):
        """
        convolution through 4 dimensions
        """
        Vertical = [True, True, False, False]
        Reverse = [False, True, False, True]
        for ms_conv, v, r in zip(self.message_passing, Vertical, Reverse):
            x = self.message_passing_once(x, ms_conv, v, r)
        return x

    def message_passing_once(self, x, conv, vertical=True, reverse=False):
        """
        actually this is spacial convolution over a simple direction

        Argument:
        ----------
        x: input tensor
        vertical: vertical message passing or horizontal
        reverse: False for up-down or left-right, True for down-up or right-left

        conv: a bit magic, this is applied to each output slice and size keep unchanged, so
        it implicitly requires padding
        """
        nB, C, H, W = x.shape  # [batch channel height width], for a result of convolutions
        if vertical:
            slices = [x[:, :, i:(i + 1), :] for i in range(H)]  # horizontal slices for a single sample
            dim = 2  # slice on height ie. get horizontal slices
        else:
            slices = [x[:, :, :, i:(i + 1)] for i in range(W)]  # same, verical slices
            dim = 3
        if reverse:
            slices = slices[::-1]
        # at this point slices is in specified order

        out = [slices[0]]
        for i in range(1, len(slices)):
            # convolution @ add through dimension
            out.append(slices[i] + F.relu(conv(out[i - 1])))
        if reverse:
            # flip back to original order if reversed previously
            out = out[::-1]
        return torch.cat(out, dim=dim)  # return to original format

    def net_init(self, input_size, ms_ks):
        input_w, input_h = input_size
        self.fc_input_feature = 5 * int(input_w / 16) * int(input_h / 16)
        self.backbone = models.vgg16_bn(pretrained=self.pretrained).features

        # ----------------- process backbone -----------------
        for i in [34, 37, 40]:
            # magic numbers to get padding, not sure what these layers are for
            # but these lines are just copying conv weights, not sure about their purpose
            conv = self.backbone._modules[str(i)]
            # dilated conv need more padding
            dilated_conv = nn.Conv2d(
                conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride,
                padding=tuple(p * 2 for p in conv.padding), dilation=2, bias=(conv.bias is not None)
            )
            dilated_conv.load_state_dict(conv.state_dict())
            self.backbone._modules[str(i)] = dilated_conv
        # also not sure why pop these modules, may be dropouts accd to original paper
        self.backbone._modules.pop('33')
        self.backbone._modules.pop('43')

        # ----------------- SCNN part -----------------

        # cap 2 final conv layers to backbone features
        self.conv_after_backbone = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()  # (nB, 128, 36, 100)
        )

        # ----------------- add message passing -----------------

        # these conv are passed into self.message_passing_once() to process each slice of a sample
        self.message_passing = nn.ModuleList()
        self.message_passing.add_module('up_down', nn.Conv2d(128, 128, (1, ms_ks), padding=(0, ms_ks // 2), bias=False))
        self.message_passing.add_module('down_up', nn.Conv2d(128, 128, (1, ms_ks), padding=(0, ms_ks // 2), bias=False))
        self.message_passing.add_module('left_right',
                                        nn.Conv2d(128, 128, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))
        self.message_passing.add_module('right_left',
                                        nn.Conv2d(128, 128, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))
        # (nB, 128, 36, 100) as message passing does not change a single sample size

        # ----------------- SCNN part -----------------
        self.final_conv = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 5, 1)  # get (nB, 5, 36, 100)
        )

        self.final_pool = nn.Sequential(
            nn.Softmax(dim=1),  # (nB, 5, 36, 100)
            nn.AvgPool2d(2, 2),  # (nB, 5, 18, 50)
        )
        self.final_fc = nn.Sequential(
            nn.Linear(self.fc_input_feature, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data[:] = 1.
                m.bias.data.zero_()

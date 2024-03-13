from torchsummary import summary
from collections import OrderedDict
import torch
import torch.nn as nn


# class tHighResolutionNet(nn.Module):
#
#     def __init__(self, cl, cfg, **kwargs):
#         super(tHighResolutionNet, self).__init__()
#
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
#                                bias=False)
#         self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.stage1_cfg = cfg['MODEL']['EXTRA']['STAGE1']
#         num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
#         block = blocks_dict[self.stage1_cfg['BLOCK']]
#         num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
#         self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
#         stage1_out_channel = block.expansion * num_channels
#
#         self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
#         num_channels = self.stage2_cfg['NUM_CHANNELS']
#         block = blocks_dict[self.stage2_cfg['BLOCK']]
#         num_channels = [
#             num_channels[i] * block.expansion for i in range(len(num_channels))]
#         self.transition1 = self._make_transition_layer(
#             [stage1_out_channel], num_channels)
#         self.stage2, pre_stage_channels = self._make_stage(
#             self.stage2_cfg, num_channels)
#
#         self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
#         num_channels = self.stage3_cfg['NUM_CHANNELS']
#         block = blocks_dict[self.stage3_cfg['BLOCK']]
#         num_channels = [
#             num_channels[i] * block.expansion for i in range(len(num_channels))]
#         self.transition2 = self._make_transition_layer(
#             pre_stage_channels, num_channels)
#         self.stage3, pre_stage_channels = self._make_stage(
#             self.stage3_cfg, num_channels)
#
#         self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
#         num_channels = self.stage4_cfg['NUM_CHANNELS']
#         block = blocks_dict[self.stage4_cfg['BLOCK']]
#         num_channels = [
#             num_channels[i] * block.expansion for i in range(len(num_channels))]
#         self.transition3 = self._make_transition_layer(
#             pre_stage_channels, num_channels)
#         self.stage4, pre_stage_channels = self._make_stage(
#             self.stage4_cfg, num_channels, multi_scale_output=True)
#
#         # Classification Head
#         self.incre_modules, self.downsamp_modules, \
#         self.final_layer = self._make_head(pre_stage_channels)
#
#         self.classifier = nn.Linear(2048, 1000)
#
#         # addition
#         self.avgpool_1 = nn.AdaptiveAvgPool2d((1, 1))
#         self.maxpool_1 = nn.AdaptiveMaxPool2d((1, 1))
#         self.avgpool_2 = nn.AdaptiveAvgPool2d((2, 2))
#         self.maxpool_2 = nn.AdaptiveMaxPool2d((2, 2))
#         self.avgpool_3 = nn.AdaptiveAvgPool2d((3, 3))
#         self.maxpool_3 = nn.AdaptiveMaxPool2d((3, 3))
#         self.avgpool_4 = nn.AdaptiveAvgPool2d((4, 4))
#         self.maxpool_4 = nn.AdaptiveMaxPool2d((4, 4))
#         self.classifier = ClassBlock(6144, 751, 0.5)
#         self.cl = cl
#
#     def _make_head(self, pre_stage_channels):
#         head_block = Bottleneck
#         head_channels = [32, 64, 128, 256]
#
#         # Increasing the #channels on each resolution
#         # from C, 2C, 4C, 8C to 128, 256, 512, 1024
#         incre_modules = []
#         for i, channels in enumerate(pre_stage_channels):
#             incre_module = self._make_layer(head_block,
#                                             channels,
#                                             head_channels[i],
#                                             1,
#                                             stride=1)
#             incre_modules.append(incre_module)
#         incre_modules = nn.ModuleList(incre_modules)
#
#         # downsampling modules
#         downsamp_modules = []
#         for i in range(len(pre_stage_channels) - 1):
#             in_channels = head_channels[i] * head_block.expansion
#             out_channels = head_channels[i + 1] * head_block.expansion
#
#             downsamp_module = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channels,
#                           out_channels=out_channels,
#                           kernel_size=3,
#                           stride=2,
#                           padding=1),
#                 nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
#                 nn.ReLU(inplace=True)
#             )
#
#             downsamp_modules.append(downsamp_module)
#         downsamp_modules = nn.ModuleList(downsamp_modules)
#
#         final_layer = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=head_channels[3] * head_block.expansion,
#                 out_channels=2048,
#                 kernel_size=1,
#                 stride=1,
#                 padding=0
#             ),
#             nn.BatchNorm2d(2048, momentum=BN_MOMENTUM),
#             nn.ReLU(inplace=True)
#         )
#
#         return incre_modules, downsamp_modules, final_layer
#
#     def _make_transition_layer(
#             self, num_channels_pre_layer, num_channels_cur_layer):
#         num_branches_cur = len(num_channels_cur_layer)
#         num_branches_pre = len(num_channels_pre_layer)
#
#         transition_layers = []
#         for i in range(num_branches_cur):
#             if i < num_branches_pre:
#                 if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
#                     transition_layers.append(nn.Sequential(
#                         nn.Conv2d(num_channels_pre_layer[i],
#                                   num_channels_cur_layer[i],
#                                   3,
#                                   1,
#                                   1,
#                                   bias=False),
#                         nn.BatchNorm2d(
#                             num_channels_cur_layer[i], momentum=BN_MOMENTUM),
#                         nn.ReLU(inplace=True)))
#                 else:
#                     transition_layers.append(None)
#             else:
#                 conv3x3s = []
#                 for j in range(i + 1 - num_branches_pre):
#                     inchannels = num_channels_pre_layer[-1]
#                     outchannels = num_channels_cur_layer[i] \
#                         if j == i - num_branches_pre else inchannels
#                     conv3x3s.append(nn.Sequential(
#                         nn.Conv2d(
#                             inchannels, outchannels, 3, 2, 1, bias=False),
#                         nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
#                         nn.ReLU(inplace=True)))
#                 transition_layers.append(nn.Sequential(*conv3x3s))
#
#         return nn.ModuleList(transition_layers)
#
#     def _make_layer(self, block, inplanes, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
#             )
#
#         layers = []
#         layers.append(block(inplanes, planes, stride, downsample))
#         inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def _make_stage(self, layer_config, num_inchannels,
#                     multi_scale_output=True):
#         num_modules = layer_config['NUM_MODULES']
#         num_branches = layer_config['NUM_BRANCHES']
#         num_blocks = layer_config['NUM_BLOCKS']
#         num_channels = layer_config['NUM_CHANNELS']
#         block = blocks_dict[layer_config['BLOCK']]
#         fuse_method = layer_config['FUSE_METHOD']
#
#         modules = []
#         for i in range(num_modules):
#             # multi_scale_output is only used last module
#             if not multi_scale_output and i == num_modules - 1:
#                 reset_multi_scale_output = False
#             else:
#                 reset_multi_scale_output = True
#
#             modules.append(
#                 HighResolutionModule(num_branches,
#                                      block,
#                                      num_blocks,
#                                      num_inchannels,
#                                      num_channels,
#                                      fuse_method,
#                                      reset_multi_scale_output)
#             )
#             num_inchannels = modules[-1].get_num_inchannels()
#
#         return nn.Sequential(*modules), num_inchannels
#
#     def forward(self, x):
#         sr = x
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.layer1(x)
#
#         x_list = []
#         for i in range(self.stage2_cfg['NUM_BRANCHES']):
#             if self.transition1[i] is not None:
#                 x_list.append(self.transition1[i](x))
#             else:
#                 x_list.append(x)
#         y_list_s2 = self.stage2(x_list)
#
#         x_list = []
#         for i in range(self.stage3_cfg['NUM_BRANCHES']):
#             if self.transition2[i] is not None:
#                 x_list.append(self.transition2[i](y_list_s2[-1]))
#             else:
#                 x_list.append(y_list_s2[i])
#         y_list_s3 = self.stage3(x_list)
#
#         x_list = []
#         for i in range(self.stage4_cfg['NUM_BRANCHES']):
#             if self.transition3[i] is not None:
#                 x_list.append(self.transition3[i](y_list_s3[-1]))
#             else:
#                 x_list.append(y_list_s3[i])
#
#         y_list = self.stage4(x_list)
#
#         for i in range(4):
#             y_list[i] = self.incre_modules[i](y_list[i])
#
#         y_list[0] = self.avgpool_4(y_list[0]) + 0.5 * self.maxpool_4(y_list[0])
#         y_list[1] = self.avgpool_2(y_list[1]) + 0.2 * self.maxpool_2(y_list[1])
#         y_list[2] = self.avgpool_2(y_list[2]) + 0.2 * self.maxpool_2(y_list[2])
#         y_list[3] = self.avgpool_1(y_list[3]) + 0.2 * self.maxpool_1(y_list[3])
#
#         y_list[0] = y_list[0].view((y_list[0].shape)[0], -1)
#         y_list[1] = y_list[1].view((y_list[1].shape)[0], -1)
#         y_list[2] = y_list[2].view((y_list[2].shape)[0], -1)
#         y_list[3] = y_list[3].view((y_list[3].shape)[0], -1)
#
#         y_tlp = torch.cat((y_list[0], y_list[1], y_list[2], y_list[3]), 1)
#         y_cls = self.classifier(y_tlp)
#
#         if self.cl:
#             return y_tlp, y_list[0], y_list[1], y_list[2], y_list[3], y_cls, sr
#         self.classifier.classifier = nn.Sequential()
#         y_cls_ft = self.classifier(y_tlp)
#         return torch.cat((y_tlp, y_cls_ft), 1)
#
#     def init_weights(self, pretrained='', ):
#         logger.info('=> init weights from normal distribution')
#         for m in self.modules():
#             if m.__class__.__name__ == 'VDSR':
#                 break
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(
#                     m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#         if os.path.isfile(pretrained):
#             pretrained_dict = torch.load(pretrained)
#             logger.info('=> loading pretrained model {}'.format(pretrained))
#             model_dict = self.state_dict()
#             pretrained_dict = {k: v for k, v in pretrained_dict.items()
#                                if k in model_dict.keys()}
#             for k, _ in pretrained_dict.items():
#                 logger.info(
#                     '=> loading {} pretrained model {}'.format(k, pretrained))
#             model_dict.update(pretrained_dict)
#             self.load_state_dict(model_dict)


# def get_cls_tnet(cl, config, **kwargs):
#     model = tHighResolutionNet(cl, config, **kwargs)
#     model.init_weights('hrnetv2_w32_imagenet_pretrained.pth')
#     return model

model = torch.load('/mnt/data/code/Deep-High-Resolution-Representation-Learning-for-Cross-Resolution-Person-Re-identification-main/model/ft_net_v7/net_last.pth')
net = nn.Sequential(OrderedDict(model))
summary(model, input_size=(3, 64, 128))
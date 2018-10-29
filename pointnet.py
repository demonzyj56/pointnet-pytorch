"""PointNet model."""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PointNet(nn.Module):
    """PointNet for classification task."""

    def __init__(self, in_channels, out_channels, p=0.3):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv5 = nn.Conv1d(128, 1024, kernel_size=1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, out_channels)
        self.dropout1 = nn.Dropout(p=p)
        self.dropout2 = nn.Dropout(p=p)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.input_transform = TNet(in_channels, in_channels)
        self.feature_transform = TNet(64, 64)

        self.reset_parameters()

    def forward(self, input):
        """
        Parameters
        ----------
        input: torch tensor, [N, in_channels, H]
            Input point clouds. N is the batch size, in_channels is the number
            of input channel (3 for xyz coordinates and 6 for additional
            normal vectors), and H is the size of each point cloud model.
        """
        transform1 = self.input_transform(input)
        x = torch.bmm(transform1, input)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        transform2 = self.feature_transform(x)
        x = torch.bmm(transform2, x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.max_pool1d(x, kernel_size=x.size(-1)).squeeze(-1)
        x = F.relu(self.bn6(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x, transform1, transform2

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1.)
                m.bias.data.zero_()
        self.input_transform.reset_parameters()
        self.feature_transform.reset_parameters()


class TNet(nn.Module):
    """TNet as transformation matrix for input points.
    TNet takes a fixed set of point clouds as input, and output a
    transformation matrix of KxK."""

    def __init__(self, in_channels, K):
        """
        Parameters
        ----------
        in_channels: int
            Input dimension of point clouds.
        K: int
            Ouput transformation matrix size.
        """
        super(TNet, self).__init__()
        self.K = K
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, K**2)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.reset_parameters()

    def forward(self, input):
        x = F.relu(self.bn1(self.conv1(input)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, kernel_size=x.size(-1)).squeeze(-1)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        out = self.fc3(x).view(-1, self.K, self.K)
        return out

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1.)
                m.bias.data.zero_()
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.copy_(torch.eye(self.K).view(-1))


class PointNetPartSem(nn.Module):
    """Point net for part segmentation (with ShapeNet part dataset)."""

    def __init__(self, in_channels, num_categories, part_num):
        super(PointNetPartSem, self).__init__()
        # base network
        self.input_transform = TNet(in_channels, in_channels)
        self.feature_transform = TNet(128, 128)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, 64, kernel_size=1),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.Conv1d(128, 128, kernel_size=1),
            nn.Conv1d(128, 512, kernel_size=1),
            nn.Conv1d(512, 2048, kernel_size=1),
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(n.out_channels) for n in self.convs
        ])
        # cls network
        self.cls_fc1 = nn.Linear(2048, 256)
        self.cls_fc2 = nn.Linear(256, 256)
        self.cls_fc3 = nn.Linear(256, num_categories)
        self.cls_bn1 = nn.BatchNorm1d(256)
        self.cls_bn2 = nn.BatchNorm1d(256)
        self.cls_dp1 = nn.Dropout(p=0.3)
        # seg network
        seg_in_channels = 2048 + num_categories + \
            sum([n.out_channels for n in self.convs])
        self.seg_conv1 = nn.Conv1d(seg_in_channels, 256, kernel_size=1)
        self.seg_conv2 = nn.Conv1d(256, 256, kernel_size=1)
        self.seg_conv3 = nn.Conv1d(256, 128, kernel_size=1)
        self.seg_conv4 = nn.Conv1d(128, part_num, kernel_size=1)
        self.seg_bn1 = nn.BatchNorm1d(256)
        self.seg_bn2 = nn.BatchNorm1d(256)
        self.seg_bn3 = nn.BatchNorm1d(128)
        self.seg_dp1 = nn.Dropout(p=0.2)
        self.seg_dp2 = nn.Dropout(p=0.2)
        self.num_categories = num_categories
        self.part_num = part_num

        self.reset_parameters()

    def forward(self, point_cloud, input_label):
        num_points = point_cloud.size(-1)
        transform1 = self.input_transform(point_cloud)
        x = torch.bmm(transform1, point_cloud)
        out = []
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if i == 3:
                transform2 = self.feature_transform(x)
                x = torch.bmm(transform2, x)
            x = F.relu(bn(conv(x)))
            out.append(x)
        x = F.max_pool1d(x, kernel_size=x.size(-1)).squeeze(-1)  # batch_size x 2048
        # cls branch
        cls_out = F.relu(self.cls_bn1(self.cls_fc1(x)))
        cls_out = F.relu(self.cls_bn2(self.cls_fc2(cls_out)))
        cls_out = self.cls_dp1(cls_out)
        cls_out = self.cls_fc3(cls_out)
        # seg branch
        input_label_one_hot = torch.zeros(
            input_label.size(0), self.num_categories,
            dtype=point_cloud.dtype, device=point_cloud.device
        )
        input_label_one_hot.scatter_(1, input_label.view(-1, 1), 1.)  # batch_size x num_categories
        seg_out = torch.cat([x, input_label_one_hot], dim=1).unsqueeze(-1)
        seg_out = seg_out.expand(seg_out.size(0), seg_out.size(1), num_points)
        seg_out = torch.cat([seg_out]+out, dim=1)
        seg_out = F.relu(self.seg_bn1(self.seg_conv1(seg_out)))
        seg_out = self.seg_dp1(seg_out)
        seg_out = F.relu(self.seg_bn2(self.seg_conv2(seg_out)))
        seg_out = self.seg_dp2(seg_out)
        seg_out = F.relu(self.seg_bn3(self.seg_conv3(seg_out)))
        seg_out = self.seg_conv4(seg_out)

        return cls_out, seg_out

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1.)
                m.bias.data.zero_()
        self.input_transform.reset_parameters()
        self.feature_transform.reset_parameters()

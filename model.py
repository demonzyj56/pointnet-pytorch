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
        self.pool = nn.AdaptiveMaxPool1d(1)
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
        x = self.pool(x).squeeze(-1)  # 2-D tensor [N, 1024]
        # NOTE: paper says only one dropout layer, but code has two.
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
                m.reset_parameters()
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
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, K**2)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, input):
        x = F.relu(self.bn1(self.conv1(input)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
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
                m.reset_parameters()
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.copy_(torch.eye(self.K).view(-1))

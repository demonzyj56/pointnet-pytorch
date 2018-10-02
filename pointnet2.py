"""PointNet++ with multi-scale grouping."""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
# from functions import FarthestPointSampleFunction
# from functions import BallPointQueryFunction
from extensions.ball_point_query import BallPointQueryFunction
from extensions.farthest_point_sample import FarthestPointSampleFunction

logger = logging.getLogger(__name__)


class PointNet2MSG(nn.Module):
    """
    For each point set at multiple scales, each point is assigned to a centroid
    which is sampled from the point set using `furthest point sampling`.  The
    features within each group are then pooled and concatenated as the feature
    for that point set.
    """

    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: int
            Input channel size for input point clouds.  This should be 3 for
            xyz coordinates and 6 for (coordinates+normals), and more for other
            cases
        :param out_channels: int
            Number of output classes.
        """
        super(PointNet2MSG, self).__init__()
        msg0_channels = in_channels - 3
        self.fps1 = FarthestPointSample(512)
        self.msg1 = nn.ModuleList([
            SetAbstraction(0.1, 16, msg0_channels, [32, 32, 64]),
            SetAbstraction(0.2, 32, msg0_channels, [64, 64, 128]),
            SetAbstraction(0.4, 128, msg0_channels, [64, 96, 128])
        ])
        msg1_channels = 320
        self.fps2 = FarthestPointSample(128)
        self.msg2 = nn.ModuleList([
            SetAbstraction(0.2, 32, msg1_channels, [64, 64, 128]),
            SetAbstraction(0.4, 64, msg1_channels, [128, 128, 256]),
            SetAbstraction(0.8, 128, msg1_channels, [128, 128, 256])
        ])
        msg2_channels = 640
        self.msg3 = SetGlobalAbstraction(msg2_channels, [256, 512, 1024])
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, out_channels)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(p=0.4)
        self.dp2 = nn.Dropout(p=0.4)
        self.reset_parameters()

    def forward(self, pt_coordinates, pt_features=None):
        centroids1 = self.fps1(pt_coordinates)  # (batch_size, num_centroids1)
        pt_features = torch.cat([
            m(pt_coordinates, pt_features, centroids1) for m in self.msg1
        ], dim=1)  # (batch_zise, feat_len, num_centroids1)
        pt_coordinates = pt_coordinates.gather(
            2, centroids1.unsqueeze(1).repeat(1, pt_coordinates.size(1), 1)
        )  # (batch_size, 3, num_centroids1)
        centroids2 = self.fps2(pt_coordinates)
        pt_features = torch.cat([
            m(pt_coordinates, pt_features, centroids2) for m in self.msg2
        ], dim=1)
        pt_coordinates = pt_coordinates.gather(
            2, centroids2.unsqueeze(1).repeat(1, pt_coordinates.size(1), 1)
        )
        pt_features = self.msg3(pt_coordinates, pt_features)
        pt_features = F.relu(self.bn1(self.fc1(pt_features)))
        pt_features = self.dp1(pt_features)
        pt_features = F.relu(self.bn2(self.fc2(pt_features)))
        pt_features = self.dp2(pt_features)
        pt_features = self.fc3(pt_features)
        return pt_features

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1.)
                m.bias.data.zero_()
        for m in (self.msg1, self.msg2):
            for mm in m:
                mm.reset_parameters()
        self.msg3.reset_parameters()


class PointNet2SSG(nn.Module):
    """Single scale version of PointNet2."""
    def __init__(self, in_channels, out_channels):
        super(PointNet2SSG, self).__init__()
        ssg0_channels = in_channels - 3
        self.fps1 = FarthestPointSample(512)
        self.ssg1 = SetAbstraction(0.2, 32, ssg0_channels, [64, 64, 128])
        self.fps2 = FarthestPointSample(128)
        self.ssg2 = SetAbstraction(0.4, 64, 128, [128, 128, 256])
        self.ssg3 = SetGlobalAbstraction(256, [256, 512, 1024])
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, out_channels)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)
        self.reset_parameters()

    def forward(self, pt_coordinates, pt_features=None):
        centroids1 = self.fps1(pt_coordinates)
        pt_features = self.ssg1(pt_coordinates, pt_features, centroids1)
        pt_coordinates = pt_coordinates.gather(
            2, centroids1.unsqueeze(1).repeat(1, pt_coordinates.size(1), 1)
        )
        centroids2 = self.fps2(pt_coordinates)
        pt_features = self.ssg2(pt_coordinates, pt_features, centroids2)
        pt_coordinates = pt_coordinates.gather(
            2, centroids2.unsqueeze(1).repeat(1, pt_coordinates.size(1), 1)
        )
        pt_features = self.ssg3(pt_coordinates, pt_features)
        pt_features = F.relu(self.bn1(self.fc1(pt_features)))
        pt_features = self.dp1(pt_features)
        pt_features = F.relu(self.bn2(self.fc2(pt_features)))
        pt_features = self.dp2(pt_features)
        pt_features = self.fc3(pt_features)
        return pt_features

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1.)
                m.bias.data.zero_()
        for m in (self.ssg1, self.ssg2, self.ssg3):
            m.reset_parameters()


class FarthestPointSample(nn.Module):
    """Do (approximate) farthest point sampling over input points."""

    def __init__(self, num_centroids):
        super(FarthestPointSample, self).__init__()
        self.num_centroids = num_centroids

    def forward(self, pt_coordinates):
        if self.num_centroids > pt_coordinates.size(-1):
            centroids = torch.arange(pt_coordinates.size(-1), dtype=torch.int64,
                                     device=pt_coordinates.device,
                                     requires_grad=False)
            return centroids.unsqueeze(0).repeat(pt_coordinates.size(0), 1)
        # out = torch.zeros(pt_coordinates.size(0), self.num_centroids,
        #                   dtype=torch.int64, device=pt_coordinates.device,
        #                   requires_grad=False)
        # for i in range(out.size(0)):
        #     out[i] = torch.randperm(pt_coordinates.size(-1))[:self.num_centroids]
        out = FarthestPointSampleFunction.apply(pt_coordinates,
                                                self.num_centroids)
        return out


class BallPointQuery(nn.Module):

    def __init__(self, radius, max_samples):
        super(BallPointQuery, self).__init__()
        self.radius = radius
        self.max_samples = max_samples

    def forward(self, pt_coordinates, centroids):
        """
        :param pt_coordinates: [batch_size, 3, num_points]
            Input point cloud coordinates.
        :param centroids: [batch_size, 3, num_centroids]
            Coordinates of sampled centroids
        :return: group_idx: [batch_size, num_centroids, max_samples]
            Index for selecting number of max samples for each batch for each
            centroid.
        """
        out = BallPointQueryFunction.apply(pt_coordinates, centroids,
                                           self.radius, self.max_samples)
        return out


class SetAbstraction(nn.Module):
    """Given centroid index for each point set, this module does the following:
    1) For each centroid, sample a number of points within the given radius.
    2) The feature of each point is annotated with centerized coordinates, and
        go through mlp layers.
    """
    def __init__(self, radius, max_samples, in_channels, mlp_channels):
        """
        :param radius: float
            The radius of each group.
        :param max_samples: float
            Maximum number for each group.
        :param in_channels: int
            Feature length for input features (excluding coordinates), since we
            input point coordinates and features separately and the former
            dimension is always 3.
        :param mlp_channels: list
            A list giving channel size of each mlp layer.
        """
        super(SetAbstraction, self).__init__()
        self.max_samples = max_samples
        inc = [in_channels+3] + list(mlp_channels[:-1])
        self.point_query = BallPointQuery(radius, max_samples)
        self.mlps = nn.ModuleList([
            nn.Conv2d(i, o, kernel_size=1) for i, o in zip(inc, mlp_channels)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm2d(o) for o in mlp_channels])
        self.pool = nn.MaxPool2d([1, max_samples])
        self.reset_parameters()

    def forward(self, pt_coordinates, pt_features, centroid_idx):
        """
        :param pt_coordinates: [batch_size, 3, num_points]
            The xyz coordinates for each point.
        :param pt_features: [batch_size, in_channels, num_points] or None
            Features for each point. If it is None, then use pt_coordinates
            as features.
        :param centroid_idx: [batch_size, num_centroids]
            Indices at which the points are centroids.
        :return:
            features: (batch_size, mlp_channels[-1], num_centroids)
                The pooled features for each group.
        """
        num_centroids = centroid_idx.size(-1)
        # (batch_size, 3, num_centroids)
        centroids = pt_coordinates.gather(
            2, centroid_idx.unsqueeze(1).repeat(1, pt_coordinates.size(1), 1)
        )
        # (batch_size, num_centroids, max_samples)
        group_idx = self.point_query(pt_coordinates, centroids)
        group_idx = group_idx.view(group_idx.size(0), -1)
        # (batch_size, 3, num_centroids x max_samples)
        grouped_coordinates = pt_coordinates.gather(
            2, group_idx.unsqueeze(1).repeat(1, pt_coordinates.size(1), 1)
        )
        # (batch, 3, num_centroids, max_samples)
        grouped_coordinates = grouped_coordinates.view(
            grouped_coordinates.size(0), grouped_coordinates.size(1),
            num_centroids, self.max_samples
        )
        grouped_coordinates.sub_(
            centroids.unsqueeze(-1).expand_as(grouped_coordinates)
        )
        if pt_features is None:
            cat_features = grouped_coordinates
        else:
            grouped_features = pt_features.gather(
                2, group_idx.unsqueeze(1).repeat(1, pt_features.size(1), 1)
            )
            grouped_features = grouped_features.view(
                grouped_features.size(0), grouped_features.size(1),
                num_centroids, self.max_samples
            )
            cat_features = torch.cat([grouped_coordinates, grouped_features],
                                     dim=1)
        for mlp, bn in zip(self.mlps, self.bns):
            cat_features = F.relu(bn(mlp(cat_features)))
        cat_features = self.pool(cat_features).squeeze(-1)
        return cat_features

    def reset_parameters(self):
        """Parameter initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.zero_()


class SetGlobalAbstraction(nn.Module):
    """Similar to set abstraction, but reducing all points to a single centroid,
    i.e. (0, 0, 0).  This is equivalent to concatenating pt coordinates and
    pt features and going through the network.
    """

    def __init__(self, in_channels, mlp_channels):
        """
        :param in_channels: int
            Feature length for input features (excluding coordinates), since we
            input point coordinates and features separately and the former
            dimension is always 3.
        :param mlp_channels: list
            A list giving channel size of each mlp layer.
        """
        super(SetGlobalAbstraction, self).__init__()
        inc = [in_channels+3] + list(mlp_channels[:-1])
        self.mlps = nn.ModuleList([
            nn.Conv1d(i, o, kernel_size=1) for i, o in zip(inc, mlp_channels)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(o) for o in mlp_channels])
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.reset_parameters()

    def forward(self, pt_coordinates, pt_features):
        """
        :param pt_coordinates: [batch_size, 3, num_points]
            The xyz coordinates for each point.
        :param pt_features: [batch_size, in_channels, num_points] or None
            Features for each point. If it is None, then use pt_coordinates
            as features.
        :return:
            features: (batch_size, mlp_channels[-1])
                The pooled features for each group.
        """
        if pt_features is None:
            cat_features = pt_coordinates
        else:
            cat_features = torch.cat([pt_coordinates, pt_features], dim=1)
        for mlp, bn in zip(self.mlps, self.bns):
            cat_features = F.relu(bn(mlp(cat_features)))
        cat_features = self.pool(cat_features).squeeze(-1)
        return cat_features

    def reset_parameters(self):
        """Parameter initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1.)
                m.bias.data.zero_()


if __name__ == '__main__':
    net = PointNet2MSG(3, 40)
    pts = torch.randn(16, 3, 1024)
    net, pts = net.cuda(), pts.cuda()
    out = net(pts)
    print(out.size())

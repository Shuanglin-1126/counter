import torch
from torch.nn import Module

class Post_Prob(Module):
    def __init__(self, sigma, c_size, stride, background_ratio, use_background, device):
        super(Post_Prob, self).__init__()
        assert c_size % stride == 0

        self.sigma = sigma
        self.bg_ratio = background_ratio
        self.device = device
        # coordinate is same to image space, set to constant since crop size is same
        # 坐标与图像空间相同，设置为常量，因为裁剪大小相同（self.cood为网格点）
        # self.cood为论文公式中的Xm，keypoints为论文公式中的Zn
        self.cood = torch.arange(0, c_size, step=stride,
                                 dtype=torch.float32, device=device) + stride / 2
        # 在cood前加一个值为1维度，[c_size]-->[1,c_size]
        self.cood.unsqueeze_(0)
        self.softmax = torch.nn.Softmax(dim=0)
        self.use_bg = use_background

    def forward(self, points, st_sizes):
        num_points_per_image = [len(points_per_image) for points_per_image in points]
        all_points = torch.cat(points, dim=0) # 沿第一个维度拼接points

        if len(all_points) > 0:
            x = all_points[:, 0].unsqueeze_(1)
            y = all_points[:, 1].unsqueeze_(1)
            # 根据pytorch的广播规则，使向量具有相同的shape（填充值由已有值复制而来）
            # x_dis中（i，j）代表第i个关键点与第j个网格点的距离
            x_dis = -2 * torch.matmul(x, self.cood) + x * x + self.cood * self.cood
            y_dis = -2 * torch.matmul(y, self.cood) + y * y + self.cood * self.cood
            y_dis.unsqueeze_(2)
            x_dis.unsqueeze_(1)
            dis = y_dis + x_dis
            # 将dis重新排列为二维张量，第一维不变，-1表示第二维自动计算
            dis = dis.view((dis.size(0), -1))
            # 将dis沿第一维分割，每部分大小由num中元素指定
            dis_list = torch.split(dis, num_points_per_image)
            prob_list = []
            for dis, st_size in zip(dis_list, st_sizes):
                if len(dis) > 0:
                    # 不使用背景点即无y0
                    if self.use_bg:
                        # 沿着dis的第一个维度进行检索，返回每一维中（最小值，最小值索引），
                        # 取元组中第一维即最小值，clamp保证其大于等于0
                        min_dis = torch.clamp(torch.min(dis, dim=0, keepdim=True)[0], min=0.0)
                        # 获取背景距离（对应论文的3.3节公式13，没看懂）
                        bg_dis = (st_size * self.bg_ratio) ** 2 / (min_dis + 1e-5)
                        dis = torch.cat([dis, bg_dis], 0)  # concatenate background distance to the last
                    # 获得高斯分布
                    dis = -dis / (2.0 * self.sigma ** 2)
                    # 将概率归一化，使其和为1
                    prob = self.softmax(dis)
                else:
                    prob = None
                prob_list.append(prob)
        else:
            prob_list = []
            for _ in range(len(points)):
                prob_list.append(None)
        return prob_list



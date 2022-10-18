import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction="mean", weight=None, num_classes=2, soft_label=False):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.soft_label = soft_label

        self.weight = weight
        self.num_classes = num_classes

    def forward(self, inputs, target):
        # The shape of the inputs is [batch_size, number_classes]
        # The shape of the target is [batch_size, number_classes], a soft but not hard label
        log_softmax_predicts = torch.log_softmax(inputs, dim=-1)
        if self.soft_label:
            cross_entropy_loss = -torch.sum(log_softmax_predicts * target, dim=-1)
        else:
            hard_label = torch.argmax(target, dim=-1)

            # 已知预测分布Q, 与P相比计算交叉熵损失
            # TODO 这里的one hot可能会有问题，顺序未打乱时容易出现一致的情形
            cross_entropy_loss = -torch.sum(
                log_softmax_predicts
                * F.one_hot(hard_label, num_classes=self.num_classes),
                dim=-1,
            )

        if self.reduction == "mean":
            return torch.mean(cross_entropy_loss, dim=-1)
        elif self.reduction == "sum":
            return torch.sum(cross_entropy_loss, dim=-1)
        else:
            return cross_entropy_loss


class FocalLoss(nn.Module):
    # 参考链接：https://cloud.tencent.com/developer/article/1422612
    def __init__(self, reduction="mean", gamma=2, alpha=0.25, epislon=1e-8):
        super(FocalLoss, self).__init__()
        self.reduction = reduction
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)
        self.epsilon = epislon

    def forward(self, inputs, target):
        # The shape of the input is [batch_size, number_classes]
        # The shape of the target is [batch_size, number_classes], a soft but not hard label
        # assert preds.dim()==2 and labels.dim()==1
        inputs = inputs + self.epsilon
        soft_max_predicts = torch.softmax(inputs, dim=-1)
        log_softmax_predicts = torch.log(soft_max_predicts)
        ce_loss = -log_softmax_predicts * target
        weight = target * torch.pow((1 - soft_max_predicts), self.gamma)
        focal_loss = self.alpha * weight * ce_loss
        # TODO 这里应该不需要计算
        # focal_loss = torch.max(focal_loss, dim=-1)[0]
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        elif self.reduction == "sum":
            return torch.sum(focal_loss)
        else:
            return focal_loss


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=True,
    ):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class AsymmetricLossOptimized(nn.Module):
    """ Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations"""

    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=False,
    ):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = (
            self.anti_targets
        ) = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(
                1 - self.xs_pos - self.xs_neg,
                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets,
            )
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


class ASLSingleLabel(nn.Module):
    """
    This loss is intended for single-label classification problems
    """

    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction="mean"):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        """
        "input" dimensions: - (batch_size, number_classes)
        "target" dimensions: - (batch_size, number_classes)
        """
        target = torch.argmax(target, dim=-1)
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(
            1, target.long().unsqueeze(1), 1
        )

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(
            1 - xs_pos - xs_neg,
            self.gamma_pos * targets + self.gamma_neg * anti_targets,
        )
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(
                self.eps / num_classes
            )

        # loss calculation
        loss = -self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == "mean":
            loss = loss.mean()

        return loss.sum()


class PolyLossWithCE(nn.Module):
    # 参考文献：P OLY L OSS : A P OLYNOMIAL E XPANSION P ERSPEC TIVE OF C LASSIFICATION L OSS F UNCTIONS
    def __init__(
        self, reduction="mean", weight=None, ignore_index=-100, epsilon=1.0,
    ):
        super(PolyLossWithCE, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon

        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, inputs, target):
        # The shape of the input is [batch_size, number_classes]
        # The shape of the target is [batch_size, number_classes], a soft but not hard label
        hard_label = torch.argmax(target, dim=-1)
        ce_loss = F.cross_entropy(
            inputs,
            hard_label,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        pt = torch.sum(target * F.softmax(input=inputs, dim=-1), axis=-1)
        poly_loss = ce_loss + self.epsilon * (1 - pt)
        if self.reduction == "mean":
            return torch.mean(poly_loss, dim=-1)
        elif self.reduction == "sum":
            return torch.sum(poly_loss, dim=-1)
        else:
            return poly_loss


class PolyLossWithFocal(nn.Module):
    # 参考文献：P OLY L OSS : A P OLYNOMIAL E XPANSION P ERSPEC TIVE OF C LASSIFICATION L OSS F UNCTIONS
    def __init__(
        self, reduction="mean", weight=None, gamma=2, epsilon=1.0,
    ):
        super(PolyLossWithFocal, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        self.gamma = gamma

        self.weight = weight

    def forward(self, inputs, target):
        # The shape of the input is [batch_size, number_classes]
        # The shape of the target is [batch_size, number_classes], a soft but not hard label
        p = torch.sigmoid(inputs)
        pt = target * p + (1 - target) * (1 - p)

        inputs = inputs + 1e-8
        soft_max_predicts = torch.softmax(inputs, dim=-1)
        log_softmax_predicts = torch.log(soft_max_predicts)
        ce_loss = -log_softmax_predicts * target
        weight = target * torch.pow((1 - soft_max_predicts), self.gamma)
        focal_loss = 0.25 * weight * ce_loss
        # focal_loss = torch.max(focal_loss, dim=-1)[0]

        poly_loss = focal_loss + self.epsilon * torch.pow((1 - pt), self.gamma + 1)
        # poly_loss = torch.max(poly_loss, dim=-1)[0]

        if self.reduction == "mean":
            return torch.mean(poly_loss)
        elif self.reduction == "sum":
            return torch.sum(poly_loss)
        else:
            return poly_loss


if __name__ == "__main__":
    Loss1 = CrossEntropyLoss(reduction="none", soft_label=False, num_classes=8)
    data_input = torch.randn((32, 8))
    data_target = torch.rand(32, 8)
    loss = Loss1(data_input, data_target)
    print(loss)

    Loss2 = torch.nn.CrossEntropyLoss(reduction="none")
    loss2 = Loss2(data_input, torch.argmax(data_target, dim=-1))
    print(loss2)

    Loss3 = PolyLossWithCE(reduction="none")
    loss3 = Loss3(data_input, data_target)
    print(loss3)

    Loss4 = PolyLossWithFocal(reduction="none")
    loss4 = Loss4(data_input, data_target)
    print(loss4)

    Loss5 = FocalLoss(reduction="none")
    loss5 = Loss5(data_input, data_target)
    print(loss5)
    
    # import torch
    # torch.nn.MSELoss()    
    # torch.nn.L1Loss()
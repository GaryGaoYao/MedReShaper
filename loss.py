import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import SSIM
from net_inception import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sobel_filter(in_channels):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_x = sobel_x.repeat(in_channels, 1, 1, 1)
    sobel_y = sobel_y.repeat(in_channels, 1, 1, 1)
    return sobel_x, sobel_y

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        sobel_x, sobel_y = sobel_filter(3)  # Get the filters as separate tensors
        self.sobel_x = sobel_x.to(device)  # Move each tensor to the device individually
        self.sobel_y = sobel_y.to(device)

    def forward(self, input, target, mask=None):
        input_edges_x = F.conv2d(input, self.sobel_x, padding=1, groups=3)
        input_edges_y = F.conv2d(input, self.sobel_y, padding=1, groups=3)
        target_edges_x = F.conv2d(target, self.sobel_x, padding=1, groups=3)
        target_edges_y = F.conv2d(target, self.sobel_y, padding=1, groups=3)
        if mask is not None:
            input_edges_x *= mask
            input_edges_y *= mask
            target_edges_x *= mask
            target_edges_y *= mask
        loss = F.mse_loss(input_edges_x, target_edges_x) + F.mse_loss(input_edges_y, target_edges_y)
        return loss


# Charbonnier Loss
def charbonnier_loss(pred, target, epsilon=1e-3, mask=None):
    loss = torch.sqrt((pred - target) ** 2 + epsilon ** 2)
    return torch.mean(loss * mask) if mask is not None else torch.mean(loss)


# Total Variation Loss
def total_variation_loss(img, weight=1e-3, mask=None):
    tv_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    tv_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
    if mask is not None:
        mask_h = mask[:, :, 1:, :]
        mask_w = mask[:, :, :, 1:]
        return weight * (torch.mean(tv_h * mask_h) + torch.mean(tv_w * mask_w))
    return weight * (torch.mean(tv_h) + torch.mean(tv_w))


# Gradient Loss
class GradientLoss(nn.Module):
    def __init__(self, in_channels=3, device='cuda'):
        super(GradientLoss, self).__init__()
        sobel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view(1, 1, 3, 3)
        sobel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view(1, 1, 3, 3)
        self.sobel_x = sobel_x.repeat(in_channels, 1, 1, 1).to(device)
        self.sobel_y = sobel_y.repeat(in_channels, 1, 1, 1).to(device)

    def gradient(self, input_tensor):
        grad_x = F.conv2d(input_tensor, self.sobel_x, padding=1, groups=input_tensor.shape[1])
        grad_y = F.conv2d(input_tensor, self.sobel_y, padding=1, groups=input_tensor.shape[1])
        return grad_x, grad_y

    def forward(self, output, target, mask=None):
        output_grad_x, output_grad_y = self.gradient(output)
        target_grad_x, target_grad_y = self.gradient(target)
        if mask is not None:
            output_grad_x *= mask
            output_grad_y *= mask
            target_grad_x *= mask
            target_grad_y *= mask
        loss = (F.mse_loss(output_grad_x, target_grad_x) +
                F.mse_loss(output_grad_y, target_grad_y) +
                F.l1_loss(output_grad_x, target_grad_x) +
                F.l1_loss(output_grad_y, target_grad_y))
        return loss


# Perceptual Loss
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.vgg_layers = nn.Sequential(*list(vgg)[:23])  # Only up to the 23rd layer
        self.vgg_layers.eval()
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def forward(self, generated_image, target_image, mask=None):
        if mask is not None:
            # Resize mask to match the image dimensions and apply it before feature extraction
            mask_resized = F.interpolate(mask, size=generated_image.shape[2:], mode='nearest')
            # Ensuring mask is broadcastable over the channel dimension
            mask_resized = mask_resized.expand_as(generated_image)
            # Apply mask without modifying the input tensors in-place
            generated_image = generated_image * mask_resized
            target_image = target_image * mask_resized

        generated_features = self.vgg_layers(generated_image)
        target_features = self.vgg_layers(target_image)

        loss_mse = F.mse_loss(generated_features, target_features)
        loss_l1 = F.l1_loss(generated_features, target_features)
        return loss_mse + loss_l1


# Compute Difference Area
def compute_difference_area(input_image, target_image, threshold=0.1):
    diff = torch.abs(input_image - target_image)
    diff_mask = (diff > threshold).float()
    difference_area = torch.sum(diff_mask)
    return diff_mask, difference_area


# Loss Functions Initialization
edge_loss = EdgeLoss().to(device)
mse_loss = nn.MSELoss().to(device)
l1_loss = nn.L1Loss().to(device)
color_loss = GradientLoss().to(device)
ssim_loss = SSIM(data_range=1, size_average=True).to(device)
perceptual_loss = PerceptualLoss().to(device)


# Calculate Losses with Mask
def calculate_losses(output, target, mask=None):
    losses = {
        'loss_edge': edge_loss(output, target, mask),
        'loss_l1': l1_loss(output * mask, target * mask) if mask is not None else l1_loss(output, target),
        'loss_mse': mse_loss(output * mask, target * mask) if mask is not None else mse_loss(output, target),
        'loss_ssim': 1 - ssim_loss(output * mask, target * mask) if mask is not None else 1 - ssim_loss(output, target),
        'loss_perceptual': perceptual_loss(output, target, mask),
        'loss_charbonnier': charbonnier_loss(output, target, mask=mask),
        'loss_tv': total_variation_loss(output, mask=mask),
        'loss_color': color_loss(output, target, mask)
    }
    return losses


# Combined Loss Calculation with Mask
def combined_loss(defect, output, target):
    loss_names = [
        'loss_edge', 'loss_l1', 'loss_mse', 'loss_ssim',
        'loss_perceptual', 'loss_charbonnier', 'loss_tv', 'loss_color'
    ]

    # Compute the difference mask for defect areas
    diff_mask, _ = compute_difference_area(defect, target, threshold=0.1)

    # Calculate the losses using the mask
    losses_diff_area = calculate_losses(output, target, mask=diff_mask)
    losses_whole_pic = calculate_losses(output, target)

    # Dynamic weighting based on relative magnitudes
    max_loss_value_diff = max(loss.item() for loss in losses_diff_area.values())
    weighted_losses_diff = {name: loss.item() / max_loss_value_diff for name, loss in losses_diff_area.items()}

    max_loss_value_whole = max(loss.item() for loss in losses_whole_pic.values())
    weighted_losses_whole = {name: loss.item() / max_loss_value_whole for name, loss in losses_whole_pic.items()}

    total_loss_diff_area = sum(weighted_losses_diff[name] * losses_diff_area[name] for name in loss_names)
    total_loss_whole_pic = sum(weighted_losses_whole[name] * losses_whole_pic[name] for name in loss_names)

    total_loss = 100 * (total_loss_diff_area + total_loss_whole_pic)

    return total_loss

class CompoundPAMLoss_improved(nn.Module):
    def __init__(self, weight_charb=1.0, weight_ssim=1.0, weight_grad=1.0, device='cuda'):
        super(CompoundPAMLoss, self).__init__()
        self.charbonnier = charbonnier_loss
        self.ssim = SSIM(data_range=1.0, size_average=True).to(device)
        self.gradient = GradientLoss(in_channels=3, device=device)
        self.weight_charb = weight_charb
        self.weight_ssim = weight_ssim
        self.weight_grad = weight_grad

    def forward(self, output, target, defect_input=None):
        # Step 1: compute defect region mask (optional)
        if defect_input is not None:
            mask, _ = compute_difference_area(defect_input, target, threshold=0.1)
        else:
            mask = None

        # Step 2: compute each loss component
        loss_charb = self.charbonnier(output, target, mask=mask)
        loss_ssim = 1 - self.ssim(output * mask, target * mask) if mask is not None else 1 - self.ssim(output, target)
        loss_grad = self.gradient(output, target, mask=mask)

        # Step 3: weighted combination
        total_loss = (
            self.weight_charb * loss_charb +
            self.weight_ssim * loss_ssim +
            self.weight_grad * loss_grad
        )

        return total_loss

import torch
import torch.nn as nn
from loguru import logger

from pare.losses.keypoints import JointsMSELoss
from pare.losses.segmentation import CrossEntropy
from pare.utils.geometry import batch_rodrigues, rotmat_to_rot6d


class HMRLoss(nn.Module):
    def __init__(
            self,
            shape_loss_weight=0,
            keypoint_loss_weight=5.,
            pose_loss_weight=1.,
            smpl_part_loss_weight=1.,
            beta_loss_weight=0.001,
            openpose_train_weight=0.,
            gt_train_weight=1.,
            loss_weight=60.,
            estimate_var=False,
            uncertainty_loss='MultivariateGaussianNegativeLogLikelihood',
    ):
        super(HMRLoss, self).__init__()
        self.criterion_shape = nn.L1Loss()
        self.criterion_keypoints = nn.MSELoss(reduction='none')
        self.estimate_var = estimate_var

        if self.estimate_var:
            self.criterion_regr = eval(uncertainty_loss)()  # AleatoricLoss
        else:
            self.criterion_regr = nn.MSELoss()

        self.loss_weight = loss_weight
        self.gt_train_weight = gt_train_weight
        self.pose_loss_weight = pose_loss_weight
        self.beta_loss_weight = beta_loss_weight
        self.shape_loss_weight = shape_loss_weight
        self.keypoint_loss_weight = keypoint_loss_weight
        self.openpose_train_weight = openpose_train_weight
        self.smpl_part_loss_weight = smpl_part_loss_weight

    def forward(self, pred, gt):
        pred_cam = pred['pred_cam']
        pred_betas = pred['pred_shape_var'] if self.estimate_var else pred['pred_shape']
        pred_rotmat = pred['pred_pose_var'] if self.estimate_var else pred['pred_pose']
        pred_joints = pred['smpl_joints3d']
        pred_vertices = pred['smpl_vertices']
        pred_projected_keypoints_2d = pred['smpl_joints2d']

        gt_pose = gt['pose']
        gt_pose_conf = gt['pose_conf']
        gt_betas = gt['betas']
        gt_joints = gt['pose_3d']
        gt_vertices = gt['vertices']
        gt_keypoints_2d = gt['keypoints']
        has_smpl = gt['has_smpl'].bool()
        has_pose_3d = gt['has_pose_3d'].bool()

        smpl_loss_f = smpl_losses_uncertainty if self.estimate_var else smpl_losses

        # Compute loss on SMPL parameters
        loss_regr_pose, loss_regr_betas = smpl_loss_f(
            pred_rotmat,
            pred_betas,
            gt_pose,
            gt_betas,
            has_smpl,
            gt_pose_conf,
            criterion=self.criterion_regr,
        )

        # Compute 2D reprojection loss for the keypoints
        loss_keypoints = projected_keypoint_loss(
            pred_projected_keypoints_2d,
            gt_keypoints_2d,
            self.openpose_train_weight,
            self.gt_train_weight,
            criterion=self.criterion_keypoints,
        )

        # Compute 3D keypoint loss
        loss_keypoints_3d = keypoint_3d_loss(
            pred_joints,
            gt_joints,
            has_pose_3d,
            criterion=self.criterion_keypoints,
        )

        # Per-vertex loss for the shape
        loss_shape = shape_loss(
            pred_vertices,
            gt_vertices,
            has_smpl,
            criterion=self.criterion_shape,
        )

        loss_shape *= self.shape_loss_weight
        loss_keypoints *= self.keypoint_loss_weight
        loss_keypoints_3d *= self.keypoint_loss_weight
        loss_regr_pose *= self.pose_loss_weight
        loss_regr_betas *= self.beta_loss_weight
        loss_cam = ((torch.exp(-pred_cam[:, 0] * 10)) ** 2).mean()

        loss_dict = {
            'loss/loss_keypoints': loss_keypoints,
            'loss/loss_keypoints_3d': loss_keypoints_3d,
            'loss/loss_regr_pose': loss_regr_pose,
            'loss/loss_regr_betas': loss_regr_betas,
            'loss/loss_shape': loss_shape,
            'loss/loss_cam': loss_cam,
        }

        if 'pred_segm_rgb' in pred.keys():
            loss_part_segm = self.criterion_part(pred['pred_segm_rgb'], gt['gt_segm_rgb'])
            loss_part_segm *= self.smpl_part_loss_weight
            loss_dict['loss/loss_part_segm'] = loss_part_segm

        loss = sum(loss for loss in loss_dict.values())

        loss *= self.loss_weight

        loss_dict['loss/total_loss'] = loss

        return loss, loss_dict


class HMRCamLoss(nn.Module):
    def __init__(
            self,
            shape_loss_weight=0,
            keypoint_loss_weight=5.,
            pose_loss_weight=1.,
            smpl_part_loss_weight=1.,
            beta_loss_weight=0.001,
            openpose_train_weight=0.,
            gt_train_weight=1.,
            loss_weight=60.,
    ):
        super(HMRCamLoss, self).__init__()
        self.criterion_shape = nn.L1Loss()
        self.criterion_keypoints = nn.MSELoss(reduction='none')

        self.criterion_regr = nn.MSELoss()

        self.loss_weight = loss_weight
        self.gt_train_weight = gt_train_weight
        self.pose_loss_weight = pose_loss_weight
        self.beta_loss_weight = beta_loss_weight
        self.shape_loss_weight = shape_loss_weight
        self.keypoint_loss_weight = keypoint_loss_weight
        self.openpose_train_weight = openpose_train_weight
        self.smpl_part_loss_weight = smpl_part_loss_weight

    def forward(self, pred, gt):
        pred_cam = pred['pred_cam']
        pred_betas = pred['pred_shape']
        pred_rotmat = pred['pred_pose']
        pred_joints = pred['smpl_joints3d']
        pred_vertices = pred['smpl_vertices']
        pred_projected_keypoints_2d = pred['smpl_joints2d']

        gt_pose = gt['pose']
        gt_pose_conf = gt['pose_conf']
        gt_betas = gt['betas']
        gt_joints = gt['pose_3d']
        gt_vertices = gt['vertices']

        has_smpl = gt['has_smpl'].bool()
        has_pose_3d = gt['has_pose_3d'].bool()

        img_size = gt['orig_shape'].rot90().T.unsqueeze(1)  # image size (H,W) -> (W,H)

        # normalize predicted keypoints between -1 and 1 to compute the loss
        pred_projected_keypoints_2d[:, :, :2] = 2 * (pred_projected_keypoints_2d[:, :, :2] / img_size) - 1

        # normalize gt keypoints between -1 and 1 to compute the loss
        gt_keypoints_2d_full_img = gt['keypoints_orig'].clone()
        gt_keypoints_2d_full_img[:, :, :2] = 2 * (gt_keypoints_2d_full_img[:, :, :2] / img_size) - 1

        smpl_loss_f = smpl_losses

        # Compute loss on SMPL parameters
        loss_regr_pose, loss_regr_betas = smpl_loss_f(
            pred_rotmat,
            pred_betas,
            gt_pose,
            gt_betas,
            has_smpl,
            gt_pose_conf,
            criterion=self.criterion_regr,
        )

        # Compute 2D reprojection loss for the keypoints
        loss_keypoints = projected_keypoint_loss(
            pred_projected_keypoints_2d,
            gt_keypoints_2d_full_img,
            self.openpose_train_weight,
            self.gt_train_weight,
            criterion=self.criterion_keypoints,
            reduce='none',
        )

        # scale keypoints with img_H/bbox_h and img_W/bbox_w to have
        # loss magnitude identical to HMR
        loss_keypoints_scale = img_size.squeeze(1) / (gt['scale'] * 200.).unsqueeze(-1)
        loss_keypoints = loss_keypoints * loss_keypoints_scale.unsqueeze(1)
        loss_keypoints = loss_keypoints.mean()

        # Compute 3D keypoint loss
        loss_keypoints_3d = keypoint_3d_loss(
            pred_joints,
            gt_joints,
            has_pose_3d,
            criterion=self.criterion_keypoints,
        )

        # Per-vertex loss for the shape
        loss_shape = shape_loss(
            pred_vertices,
            gt_vertices,
            has_smpl,
            criterion=self.criterion_shape,
        )

        loss_shape *= self.shape_loss_weight
        loss_keypoints *= self.keypoint_loss_weight
        loss_keypoints_3d *= self.keypoint_loss_weight
        loss_regr_pose *= self.pose_loss_weight
        loss_regr_betas *= self.beta_loss_weight
        loss_cam = ((torch.exp(-pred_cam[:, 0] * 10)) ** 2).mean()

        loss_dict = {
            'loss/loss_keypoints': loss_keypoints,
            'loss/loss_keypoints_3d': loss_keypoints_3d,
            'loss/loss_regr_pose': loss_regr_pose,
            'loss/loss_regr_betas': loss_regr_betas,
            'loss/loss_shape': loss_shape,
            'loss/loss_cam': loss_cam,
        }

        if 'pred_segm_rgb' in pred.keys():
            loss_part_segm = self.criterion_part(pred['pred_segm_rgb'], gt['gt_segm_rgb'])
            loss_part_segm *= self.smpl_part_loss_weight
            loss_dict['loss/loss_part_segm'] = loss_part_segm

        loss = sum(loss for loss in loss_dict.values())

        loss *= self.loss_weight

        loss_dict['loss/total_loss'] = loss

        # import IPython; IPython.embed(); exit()

        return loss, loss_dict


def projected_keypoint_loss(
        pred_keypoints_2d,
        gt_keypoints_2d,
        openpose_weight,
        gt_weight,
        criterion,
        reduce='mean',

):
    """ Compute 2D reprojection loss on the keypoints.
    The loss is weighted by the confidence.
    The available keypoints are different for each dataset.
    """
    conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
    conf[:, :25] *= openpose_weight
    conf[:, 25:] *= gt_weight
    if reduce == 'mean':
        loss = (conf * criterion(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
    elif reduce == 'none':
        loss = (conf * criterion(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1]))
    else:
        raise ValueError(f'{reduce} value is not defined!')
    return loss


def keypoint_loss(
        pred_keypoints_2d,
        gt_keypoints_2d,
        criterion,
):
    """ Compute 2D reprojection loss on the keypoints.
    The loss is weighted by the confidence.
    The available keypoints are different for each dataset.
    """
    loss = criterion(pred_keypoints_2d, gt_keypoints_2d).mean()
    return loss


def heatmap_2d_loss(
        pred_heatmaps_2d,
        gt_heatmaps_2d,
        joint_vis,
        criterion,
):
    """ Compute 2D reprojection loss on the keypoints.
    The loss is weighted by the confidence.
    The available keypoints are different for each dataset.
    """
    loss = criterion(pred_heatmaps_2d, gt_heatmaps_2d, joint_vis)
    return loss


def keypoint_3d_loss(
        pred_keypoints_3d,
        gt_keypoints_3d,
        has_pose_3d,
        criterion,
):
    """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
    The loss is weighted by the confidence.
    """
    pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
    conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
    gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
    gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
    conf = conf[has_pose_3d == 1]
    pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
    if len(gt_keypoints_3d) > 0:
        gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
        pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
        return (conf * criterion(pred_keypoints_3d, gt_keypoints_3d)).mean()
    else:
        return torch.FloatTensor(1).fill_(0.).to(pred_keypoints_3d.device)


def np_keypoint_3d_loss(
        pred_joints,
        gt_joints,
        has_pose_3d,
        criterion,
):
    """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
    The loss is weighted by the confidence.
    """
    # pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
    # conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
    # gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()

    # gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
    # conf = conf[has_pose_3d == 1]
    # pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]

    # gt_pelvis = (gt_joints[:, 2, :] + gt_joints[:, 3, :]) / 2
    # gt_keypoints_3d = gt_joints - gt_pelvis[:, None, :]
    # pred_pelvis = (pred_joints[:, 2, :] + pred_joints[:, 3, :]) / 2
    # pred_keypoints_3d = pred_joints - pred_pelvis[:, None, :]
    return criterion(pred_joints, gt_joints).mean()


def shape_loss(
        pred_vertices,
        gt_vertices,
        has_smpl,
        criterion,
):
    """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
    pred_vertices_with_shape = pred_vertices[has_smpl == 1]
    gt_vertices_with_shape = gt_vertices[has_smpl == 1]
    if len(gt_vertices_with_shape) > 0:
        return criterion(pred_vertices_with_shape, gt_vertices_with_shape)
    else:
        return torch.FloatTensor(1).fill_(0.).to(pred_vertices.device)


def smpl_losses_uncertainty(
        pred_rot6d,
        pred_betas,
        gt_pose,
        gt_betas,
        has_smpl,
        criterion,
):
    pred_rot6d_valid = pred_rot6d[has_smpl == 1]
    gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1, 3)).view(-1, 24, 3, 3)[has_smpl == 1]
    gt_rot6d_valid = rotmat_to_rot6d(gt_rotmat_valid)
    pred_betas_valid = pred_betas[has_smpl == 1]
    gt_betas_valid = gt_betas[has_smpl == 1]
    if len(pred_rot6d_valid) > 0:
        loss_regr_pose = criterion(pred_rot6d_valid, gt_rot6d_valid)
        loss_regr_betas = criterion(pred_betas_valid, gt_betas_valid)
    else:
        loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(pred_rot6d.device)
        loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(pred_rot6d.device)
    return loss_regr_pose, loss_regr_betas


def smpl_losses(
        pred_rotmat,
        pred_betas,
        gt_pose,
        gt_betas,
        has_smpl,
        pose_conf,
        criterion,
):
    pred_rotmat_valid = pred_rotmat[has_smpl == 1]
    gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1, 3)).view(-1, 24, 3, 3)[has_smpl == 1]
    pred_betas_valid = pred_betas[has_smpl == 1]
    gt_betas_valid = gt_betas[has_smpl == 1]
    pose_conf = pose_conf[has_smpl == 1].unsqueeze(-1).unsqueeze(-1)
    if len(pred_rotmat_valid) > 0:
        loss_regr_pose = (pose_conf * criterion(pred_rotmat_valid, gt_rotmat_valid)).mean()
        loss_regr_betas = criterion(pred_betas_valid, gt_betas_valid).mean()
    else:
        loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(pred_rotmat.device)
        loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(pred_rotmat.device)
    return loss_regr_pose, loss_regr_betas

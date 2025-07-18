# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import OBBMetrics, batch_probiou
from ultralytics.utils.plotting import output_to_rotated_target, plot_images


class OBBValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on an Oriented Bounding Box (OBB) model.

    Example:
        ```python
        from ultralytics.models.yolo.obb import OBBValidator

        args = dict(model='yolov8n-obb.pt', data='coco8-seg.yaml')
        validator = OBBValidator(args=args)
        validator(model=args['model'])
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize OBBValidator and set task to 'obb', metrics to OBBMetrics."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = 'obb'
        self.metrics = OBBMetrics(save_dir=self.save_dir, plot=True, on_plot=self.on_plot)

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        super().init_metrics(model)
        val = self.data.get(self.args.split, '')  # validation path
        self.is_dota = isinstance(val, str) and 'DOTA' in val  # is COCO

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(preds,
                                       self.args.conf,
                                       self.args.iou,
                                       labels=self.lb,
                                       nc=self.nc,
                                       multi_label=True,
                                       agnostic=self.args.single_cls,
                                       max_det=self.args.max_det,
                                       rotated=True)

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class.
            labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
                Each label is of the format: class, x1, y1, x2, y2.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """
        iou = batch_probiou(gt_bboxes, torch.cat([detections[:, :4], detections[:, -1:]], dim=-1))
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def _prepare_batch(self, si, batch):
        idx = batch['batch_idx'] == si
        cls = batch['cls'][idx].squeeze(-1)
        bbox = batch['bboxes'][idx]
        ori_shape = batch['ori_shape'][si]
        imgsz = batch['img'].shape[2:]
        ratio_pad = batch['ratio_pad'][si]
        if len(cls):
            bbox[..., :4].mul_(torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]])  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad, xywh=True)  # native-space labels
        prepared_batch = dict(cls=cls, bbox=bbox, ori_shape=ori_shape, imgsz=imgsz, ratio_pad=ratio_pad)
        return prepared_batch

    def _prepare_pred(self, pred, pbatch):
        predn = pred.clone()
        ops.scale_boxes(pbatch['imgsz'], predn[:, :4], pbatch['ori_shape'], ratio_pad=pbatch['ratio_pad'],
                        xywh=True)  # native-space pred
        return predn

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        # plot_images(batch['img'],
        #             *output_to_rotated_target(preds, max_det=self.args.max_det),
        #             paths=batch['im_file'],
        #             fname=self.save_dir / f'val_batch{ni}_pred.jpg',
        #             names=self.names,
        #             on_plot=self.on_plot)  # pred
        plot_images(batch['img'][:, :3],
                    *output_to_rotated_target(preds, max_det=self.args.max_det),
                    paths=batch['im_file'],
                    fname=self.save_dir / f'val_batch{ni}_pred.jpg',
                    names=self.names,
                    on_plot=self.on_plot)  # pred
        if batch['img'].shape[1]==6:
            plot_images(batch['img'][:, 3:],
                        *output_to_rotated_target(preds, max_det=self.args.max_det),
                        paths=batch['im_file'],
                        fname=self.save_dir / f'val_batch{ni}_pred(extra).jpg',
                        names=self.names,
                        on_plot=self.on_plot)  # pred

    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        rbox = torch.cat([predn[:, :4], predn[:, -1:]], dim=-1)
        poly = ops.xywhr2xyxyxyxy(rbox).view(-1, 8)
        for i, (r, b) in enumerate(zip(rbox.tolist(), poly.tolist())):
            self.jdict.append({
                'image_id': image_id,
                'category_id': self.class_map[int(predn[i, 5].item())],
                'score': round(predn[i, 4].item(), 5),
                'rbox': [round(x, 3) for x in r],
                'poly': [round(x, 3) for x in b]})

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls, angle in predn.tolist():
            xywha = torch.tensor([*xyxy, angle]).view(1, 5)
            xywha[:, :4] /= gn
            xyxyxyxy = ops.xywhr2xyxyxyxy(xywha).view(-1).tolist()  # normalized xywh
            line = (cls, *xyxyxyxy, conf) if save_conf else (cls, *xyxyxyxy)  # label format
            with open(file, 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

    def eval_json(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        if self.args.save_json and self.is_dota and len(self.jdict):
            import json
            import re
            from collections import defaultdict
            pred_json = self.save_dir / 'predictions.json'  # predictions
            pred_txt = self.save_dir / 'predictions_txt'  # predictions
            pred_txt.mkdir(parents=True, exist_ok=True)
            data = json.load(open(pred_json))
            # Save split results
            LOGGER.info(f'Saving predictions with DOTA format to {str(pred_txt)}...')
            for d in data:
                image_id = d['image_id']
                score = d['score']
                classname = self.names[d['category_id']].replace(' ', '-')

                lines = '{} {} {} {} {} {} {} {} {} {}\n'.format(
                    image_id,
                    score,
                    d['poly'][0],
                    d['poly'][1],
                    d['poly'][2],
                    d['poly'][3],
                    d['poly'][4],
                    d['poly'][5],
                    d['poly'][6],
                    d['poly'][7],
                )
                with open(str(pred_txt / f'Task1_{classname}') + '.txt', 'a') as f:
                    f.writelines(lines)
            # Save merged results, this could result slightly lower map than using official merging script,
            # because of the probiou calculation.
            pred_merged_txt = self.save_dir / 'predictions_merged_txt'  # predictions
            pred_merged_txt.mkdir(parents=True, exist_ok=True)
            merged_results = defaultdict(list)
            LOGGER.info(f'Saving merged predictions with DOTA format to {str(pred_merged_txt)}...')
            for d in data:
                image_id = d['image_id'].split('__')[0]
                pattern = re.compile(r'\d+___\d+')
                x, y = (int(c) for c in re.findall(pattern, d['image_id'])[0].split('___'))
                bbox, score, cls = d['rbox'], d['score'], d['category_id']
                bbox[0] += x
                bbox[1] += y
                bbox.extend([score, cls])
                merged_results[image_id].append(bbox)
            for image_id, bbox in merged_results.items():
                bbox = torch.tensor(bbox)
                max_wh = torch.max(bbox[:, :2]).item() * 2
                c = bbox[:, 6:7] * max_wh  # classes
                scores = bbox[:, 5]  # scores
                b = bbox[:, :5].clone()
                b[:, :2] += c
                # 0.3 could get results close to the ones from official merging script, even slightly better.
                i = ops.nms_rotated(b, scores, 0.3)
                bbox = bbox[i]

                b = ops.xywhr2xyxyxyxy(bbox[:, :5]).view(-1, 8)
                for x in torch.cat([b, bbox[:, 5:7]], dim=-1).tolist():
                    classname = self.names[int(x[-1])].replace(' ', '-')
                    poly = [round(i, 3) for i in x[:-2]]
                    score = round(x[-2], 3)

                    lines = '{} {} {} {} {} {} {} {} {} {}\n'.format(
                        image_id,
                        score,
                        poly[0],
                        poly[1],
                        poly[2],
                        poly[3],
                        poly[4],
                        poly[5],
                        poly[6],
                        poly[7],
                    )
                    with open(str(pred_merged_txt / f'Task1_{classname}') + '.txt', 'a') as f:
                        f.writelines(lines)

        return stats

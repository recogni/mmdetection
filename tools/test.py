# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
from copy import deepcopy

import racon.torch as racon_torch
import recogni.torch as rtorch
import torch
from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.structures import DetDataSample
from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner
from recogni.torch.conversion.prepare import PrepareStrategy
from recogni.torch.conversion.prepare.mapping import DEFAULT_MAPPING_INCL_CLS, Fn2FactoryMapping


@HOOKS.register_module()
class ConvertModelHook(Hook):
    """Hook to convert model to recogni-compatible format."""

    def before_test(self, runner) -> None:
        """All subclasses should override this method, if they need any operations before testing."""
        runner.model = self._convert_model(runner.model.eval())
        print(runner.model)

    def _get_data_sample(self, img_shape, pad_shape):
        """Return an object of DetDataSample type."""
        data_sample = DetDataSample()
        img_meta = dict(img_shape=img_shape, pad_shape=pad_shape)
        data_sample.set_metainfo(img_meta)
        # t_instances = InstanceData(metainfo=img_meta)
        # gt_instances.bboxes = torch.rand((5, 4))
        # gt_instances.labels = torch.rand((5,))
        # data_sample.gt_instances = gt_instances
        # print(data_sample)
        # print(data_sample.gt_instances.metainfo_keys())
        return data_sample

    def _prepare_model(self, model, data_sample):
        """Model preparation step."""
        sample_kwargs = {
            "inputs": torch.randn((1, 3, 1088, 1632), dtype=torch.float32, device="cuda:0"),
            "data_samples": [
                data_sample,
            ],
        }

        prepared_model = rtorch.conversion.prepare(
            module=model.eval(),
            concrete_args=(sample_kwargs,),
            mapping=Fn2FactoryMapping(DEFAULT_MAPPING_INCL_CLS),
            strategy=PrepareStrategy.TRACE_DYNAMO,
        )
        return prepared_model

    def _convert_model(self, model):
        """Convert the model."""
        # transform the model
        data_sample = self._get_data_sample(img_shape=(1088, 1632), pad_shape=(1088, 1632))
        prepared_model = self._prepare_model(model, data_sample)

        model_api = racon_torch.api.Model(
            ops=racon_torch.api.Op(add=racon_torch.api.ops.AddV2()),
            batchnorm_folding=racon_torch.api.Batchnorm(enabled=True),
        )
        transformed_model, _ = racon_torch.utils.transform_model(prepared_model, model_api)
        return transformed_model

# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--tta', action='store_true')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # testing speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:

        if 'tta_model' not in cfg:
            warnings.warn('Cannot find ``tta_model`` in config, '
                          'we will set it as default.')
            cfg.tta_model = dict(
                type='DetTTAModel',
                tta_cfg=dict(
                    nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
        if 'tta_pipeline' not in cfg:
            warnings.warn('Cannot find ``tta_pipeline`` in config, '
                          'we will set it as default.')
            test_data_cfg = cfg.test_dataloader.dataset
            while 'dataset' in test_data_cfg:
                test_data_cfg = test_data_cfg['dataset']
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='RandomFlip', prob=1.),
                        dict(type='RandomFlip', prob=0.)
                    ],
                    [
                        dict(
                            type='PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'flip',
                                       'flip_direction'))
                    ],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpDetResults(out_file_path=args.out))

    # start testing
    runner.register_custom_hooks([dict(type="ConvertModelHook")])
    runner.test()


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""命令行入口：python run_all.py --stage all [--smoke]

stage: main | kernels | ablation | n | noise | ttests | all
--smoke: 仅用 1:2 子集快速验证管线可运行（不用于论文结果）
"""
import argparse
import config as cfg
import experiments as E


STAGES = ['main', 'kernels', 'ablation', 'n', 'noise', 'ttests']


def run(stage, smoke):
    ratios = [2] if smoke else cfg.RATIOS
    if stage == 'main':
        E.experiment_main(ratios=ratios)
    elif stage == 'kernels':
        E.experiment_kernels(ratio=2 if smoke else 16)
    elif stage == 'ablation':
        E.experiment_ablation(ratio=2 if smoke else 16)
    elif stage == 'n':
        E.experiment_N(ratio=2 if smoke else 16)
    elif stage == 'noise':
        E.experiment_noise(ratio=2 if smoke else 16)
    elif stage == 'ttests':
        E.experiment_ttests(ratios=ratios)
    else:
        raise ValueError(stage)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all', choices=STAGES + ['all'])
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.stage == 'all':
        for s in STAGES:
            print('\n' + '=' * 70 + f'\nSTAGE: {s}\n' + '=' * 70)
            run(s, args.smoke)
    else:
        run(args.stage, args.smoke)
    print('\n全部完成，结果在:', cfg.RESULTS_DIR)

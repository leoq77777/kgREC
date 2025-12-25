"""
Main training script for KGRec
Adapted from HKUDS/KGRec reference implementation
"""
try:
    import setproctitle
    setproctitle.setproctitle('EXP@KGRec')
except ImportError:
    pass  # setproctitle is optional

import random
from tqdm import tqdm
import torch
import numpy as np
import os
from time import time
try:
    from prettytable import PrettyTable
except ImportError:
    # Fallback if prettytable is not available
    class PrettyTable:
        def __init__(self, *args, **kwargs):
            self.field_names = []
        def add_row(self, row):
            print(" | ".join(str(x) for x in row))
        def __str__(self):
            return ""

import datetime
from utils.parser import parse_args_kgsr
from utils.data_loader import load_data
from modules.KGRec import KGRec
from utils.evaluate_kgsr import test
from utils.helper import early_stopping, init_logger, ensureDir
from logging import getLogger
from utils.sampler import UniformSampler
from collections import defaultdict

seed = 2020
n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0

# Try to use C++ sampler, fallback to Python if not available
try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    path = join(dirname(__file__), "utils/ext/sampling.cpp")
    sampling = imp_from_filepath(path)
    sampling.seed(seed)
except:
    sampling = UniformSampler(seed)

def neg_sampling_cpp(train_cf_pairs, train_user_dict):
    time1 = time()
    train_cf_negs = sampling.sample_negative(train_cf_pairs[:, 0], n_items, train_user_dict, 1)
    train_cf_negs = np.asarray(train_cf_negs)
    # 确保维度匹配：如果train_cf_negs是1D，需要reshape为2D
    if train_cf_negs.ndim == 1:
        train_cf_negs = train_cf_negs.reshape(-1, 1)
    elif train_cf_negs.ndim == 2 and train_cf_negs.shape[1] == 1:
        pass  # 已经是正确形状
    else:
        # 如果是列表的列表，需要处理
        train_cf_negs = np.array([item[0] if isinstance(item, (list, np.ndarray)) and len(item) > 0 else item 
                                  for item in train_cf_negs]).reshape(-1, 1)
    train_cf_triples = np.concatenate([train_cf_pairs, train_cf_negs], axis=1)
    time2 = time()
    logger.info('neg_sampling_cpp time: %.2fs', time2 - time1)
    logger.info('train_cf_triples shape: {}'.format(train_cf_triples.shape))
    return train_cf_triples

def get_feed_dict(train_cf_with_neg, start, end):
    feed_dict = {}
    entity_pairs = torch.from_numpy(train_cf_with_neg[start:end]).to(device).long()
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = entity_pairs[:, 2]
    feed_dict['batch_start'] = start
    return feed_dict

if __name__ == '__main__':
    try:
        """fix the random seed"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        """read args"""
        global args, device
        args = parse_args_kgsr()
        
        log_fn = init_logger(args)
        logger = getLogger()
        
        # 智能设备选择：如果cuda=1但GPU不可用，自动降级到CPU
        if args.cuda and torch.cuda.is_available():
            device = torch.device("cuda:"+str(args.gpu_id))
            gpu_name = torch.cuda.get_device_name(args.gpu_id)
            gpu_memory = torch.cuda.get_device_properties(args.gpu_id).total_memory / 1024**3
            logger.info(f"使用GPU训练: {gpu_name} ({gpu_memory:.2f} GB)")
            # 检查是否是ROCm
            if hasattr(torch.version, 'hip') or 'rocm' in torch.__version__.lower():
                logger.info(f"ROCm版本: {getattr(torch.version, 'hip', '未知')}")
        else:
            if args.cuda and not torch.cuda.is_available():
                logger.warning("请求使用GPU但未检测到CUDA/ROCm支持，将使用CPU训练")
                logger.warning("提示: 在WSL中可以使用 'python train_with_rocm.py --auto_setup' 自动安装ROCm")
            device = torch.device("cpu")
            args.cuda = 0  # 更新参数以反映实际使用的设备
            logger.info("使用CPU训练")
        
        logger.info('PID: %d', os.getpid())
        logger.info(f"DESC: {args.desc}\n")

        """build dataset"""
        train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)
        adj_mat_list, norm_mat_list, mean_mat_list = mat_list

        n_users = n_params['n_users']
        n_items = n_params['n_items']
        n_entities = n_params['n_entities']
        n_relations = n_params['n_relations']
        n_nodes = n_params['n_nodes']

        """define model"""
        model_dict = {
            'KGSR': KGRec,
        }
        model = model_dict[args.model]
        model = model(n_params, args, graph, mean_mat_list[0]).to(device)
        model.print_shapes()
        """define optimizer"""
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        test_interval = 10 if args.dataset == 'last-fm' else 1
        early_stop_step = 5 if args.dataset == 'last-fm' else 10

        cur_best_pre_0 = 0
        cur_stopping_step = 0
        should_stop = False
        start_epoch = 0

        # 从检查点恢复训练
        if args.resume is not None:
            logger.info(f"尝试从检查点恢复训练: {args.resume}")
            try:
                checkpoint = torch.load(args.resume, map_location=device)
                
                # 加载模型权重
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info("已加载模型权重")
                else:
                    # 兼容旧格式（只有state_dict）
                    model.load_state_dict(checkpoint)
                    logger.info("已加载模型权重（旧格式）")
                
                # 加载优化器状态
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logger.info("已加载优化器状态")
                
                # 加载训练状态
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                    logger.info(f"从 Epoch {start_epoch} 继续训练")
                
                if 'cur_best_pre_0' in checkpoint:
                    cur_best_pre_0 = checkpoint['cur_best_pre_0']
                    logger.info(f"最佳 Recall@20: {cur_best_pre_0:.4f}")
                
                if 'cur_stopping_step' in checkpoint:
                    cur_stopping_step = checkpoint['cur_stopping_step']
                
                logger.info("成功从检查点恢复训练状态")
            except Exception as e:
                logger.error(f"加载检查点失败: {e}")
                logger.error("将从头开始训练")
                start_epoch = 0

        logger.info("start training ...")
        logger.info(f"总训练轮数: {args.epoch}, 从 Epoch {start_epoch} 开始")
        logger.info("=" * 80)
        
        for epoch in range(start_epoch, args.epoch):
            """training CF"""
            """cf data"""
            train_cf_with_neg = neg_sampling_cpp(train_cf, user_dict['train_user_set'])
            # shuffle training data
            index = np.arange(len(train_cf))
            np.random.shuffle(index)
            train_cf_with_neg = train_cf_with_neg[index]

            """training"""
            model.train()
            add_loss_dict, s = defaultdict(float), 0
            train_s_t = time()
            
            # 改进的进度条显示
            total_batches = len(train_cf) // args.batch_size
            epoch_progress = f"Epoch {epoch+1}/{args.epoch}"
            pbar = tqdm(
                total=total_batches,
                desc=epoch_progress,
                unit='batch',
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
            
            while s + args.batch_size <= len(train_cf):
                batch = get_feed_dict(train_cf_with_neg,
                                    s, s + args.batch_size)
                batch_loss, batch_loss_dict = model(batch)

                optimizer.zero_grad(set_to_none=True)
                batch_loss.backward()
                optimizer.step()

                for k, v in batch_loss_dict.items():
                    add_loss_dict[k] += v
                s += args.batch_size
                
                # 更新进度条，显示当前loss
                avg_loss = sum(add_loss_dict.values()) / (s // args.batch_size) if s > 0 else 0
                loss_str = ', '.join([f"{k}: {v/(s//args.batch_size):.4f}" if s > 0 else f"{k}: 0.0000" 
                                     for k, v in add_loss_dict.items()])
                pbar.set_postfix({'loss': loss_str[:50]})  # 限制长度避免过长
                pbar.update(1)

            pbar.close()
            train_e_t = time()
            
            # 显示epoch训练摘要
            epoch_time = train_e_t - train_s_t
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch+1}/{args.epoch} 训练完成")
            logger.info(f"训练时间: {epoch_time:.2f}秒 ({epoch_time/60:.2f}分钟)")
            logger.info(f"平均Loss: {loss_str}")
            logger.info(f"进度: [{epoch+1}/{args.epoch}] ({100*(epoch+1)/args.epoch:.1f}%)")
            if start_epoch > 0:
                remaining_epochs = args.epoch - (epoch + 1)
                avg_time_per_epoch = epoch_time
                estimated_remaining = remaining_epochs * avg_time_per_epoch
                logger.info(f"预计剩余时间: {estimated_remaining/60:.1f}分钟 ({estimated_remaining/3600:.2f}小时)")
            logger.info(f"{'='*80}\n")

            if epoch % test_interval == 0 and epoch >= 1:
                """testing"""
                test_s_t = time()
                model.eval()
                with torch.no_grad():
                    ret = test(model, user_dict, n_params)
                test_e_t = time()

                # 改进的测试结果显示
                train_res = PrettyTable()
                train_res.field_names = ["Epoch", "训练时间", "测试时间", "Loss", "Recall@20", "NDCG@20", "Precision@20", "Hit Ratio"]
                train_res.add_row(
                    [
                        f"{epoch+1}/{args.epoch}",
                        f"{train_e_t - train_s_t:.2f}s",
                        f"{test_e_t - test_s_t:.2f}s",
                        f"{sum(add_loss_dict.values()):.4f}",
                        f"{ret['recall'][0]:.4f}",
                        f"{ret['ndcg'][0]:.4f}",
                        f"{ret['precision'][0]:.4f}",
                        f"{ret['hit_ratio'][0]:.4f}"
                    ]
                )
                logger.info("\n" + "="*80)
                logger.info("测试结果:")
                logger.info(train_res)
                logger.info("="*80)
                
                # 显示最佳指标对比
                if cur_best_pre_0 > 0:
                    improvement = ret['recall'][0] - cur_best_pre_0
                    if improvement > 0:
                        logger.info(f"✨ 发现更好的模型! Recall@20 提升: {improvement:.4f} (当前: {ret['recall'][0]:.4f}, 之前最佳: {cur_best_pre_0:.4f})")
                    else:
                        logger.info(f"当前 Recall@20: {ret['recall'][0]:.4f}, 最佳: {cur_best_pre_0:.4f} (未提升)")

                # *********************************************************
                # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
                cur_best_pre_0, cur_stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,cur_stopping_step, expected_order='acc', flag_step=early_stop_step)
                if cur_stopping_step == 0:
                    logger.info("###find better!")
                elif should_stop:
                    break

                """save weight"""
                # 保存最佳模型
                if ret['recall'][0] == cur_best_pre_0 and args.save:
                    save_path = args.out_dir + log_fn + '.ckpt'
                    # 确保输出目录存在
                    ensureDir(args.out_dir)
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'cur_best_pre_0': cur_best_pre_0,
                        'cur_stopping_step': cur_stopping_step,
                        'recall': ret['recall'],
                        'ndcg': ret['ndcg'],
                        'args': args.__dict__,
                    }
                    torch.save(checkpoint, save_path)
                    logger.info('save better model at epoch %d to path %s' % (epoch, save_path))
                
                # 定期保存检查点（用于恢复训练）
                if args.save and args.save_interval > 0 and (epoch + 1) % args.save_interval == 0:
                    periodic_save_path = args.out_dir + log_fn + f'_epoch_{epoch+1}.ckpt'
                    # 确保输出目录存在
                    ensureDir(args.out_dir)
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'cur_best_pre_0': cur_best_pre_0,
                        'cur_stopping_step': cur_stopping_step,
                        'recall': ret['recall'],
                        'ndcg': ret['ndcg'],
                        'args': args.__dict__,
                    }
                    torch.save(checkpoint, periodic_save_path)
                    logger.info('save periodic checkpoint at epoch %d to path %s' % (epoch + 1, periodic_save_path))

            else:
                # 非测试epoch的日志
                loss_str = ', '.join([f"{k}: {v:.4f}" for k, v in add_loss_dict.items()])
                logger.info('{} | Epoch {}/{} | 训练时间: {:.2f}s | Loss: {}'.format(
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    epoch+1, args.epoch,
                    train_e_t - train_s_t,
                    loss_str
                ))
                
                # 即使不测试，也定期保存检查点
                if args.save and args.save_interval > 0 and (epoch + 1) % args.save_interval == 0:
                    periodic_save_path = args.out_dir + log_fn + f'_epoch_{epoch+1}.ckpt'
                    # 确保输出目录存在
                    ensureDir(args.out_dir)
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'cur_best_pre_0': cur_best_pre_0,
                        'cur_stopping_step': cur_stopping_step,
                        'args': args.__dict__,
                    }
                    torch.save(checkpoint, periodic_save_path)
                    logger.info('save periodic checkpoint at epoch %d to path %s' % (epoch + 1, periodic_save_path))

        logger.info("\n" + "="*80)
        logger.info("训练完成!")
        logger.info(f"最终 Epoch: {epoch+1}")
        logger.info(f"最佳 Recall@20: {cur_best_pre_0:.4f}")
        logger.info("="*80)

    except Exception as e:
        logger.exception(e)


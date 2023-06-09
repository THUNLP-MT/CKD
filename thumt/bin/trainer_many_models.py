# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import glob
import logging
import os
import re
import six
import socket
import time
import torch

import thumt.data as data
import torch.distributed as dist
import thumt.models as models
import thumt.optimizers as optimizers
import thumt.utils as utils
import thumt.utils.summary as summary


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Train a neural machine translation model.",
        usage="trainer.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--teacher",type=str)
    parser.add_argument("--teacher_path",type=str)
    parser.add_argument("--input", type=str, nargs=2,
                        help="Path to source and target corpus.")
    parser.add_argument("--output", type=str, default="train",
                        help="Path to load/store checkpoints.")
    parser.add_argument("--vocabulary", type=str, nargs=2,
                        help="Path to source and target vocabulary.")
    parser.add_argument("--validation", type=str,
                        help="Path to validation file.")
    parser.add_argument("--references", type=str,
                        help="Pattern to reference files.")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to pre-trained checkpoint.")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed training.")
    parser.add_argument("--local_rank", type=int,
                        help="Local rank of this process.")
    parser.add_argument("--half", action="store_true",
                        help="Enable mixed-precision training.")
    parser.add_argument("--hparam_set", type=str,
                        help="Name of pre-defined hyper-parameter set.")

    # model and configuration
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model.")
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper-parameters.")

    return parser.parse_args(args)


def default_params():
    params = utils.HParams(
        input=["", ""],
        output="",
        model="transformer",
        vocab=["", ""],
        pad="<pad>",
        bos="<eos>",
        eos="<eos>",
        unk="<unk>",
        # Dataset
        batch_size=4096,
        fixed_batch_size=False,
        min_length=1,
        max_length=256,
        buffer_size=10000,
        # Initialization
        initializer_gain=1.0,
        initializer="uniform_unit_scaling",
        # Regularization
        scale_l1=0.0,
        scale_l2=0.0,
        # Training
        initial_step=0,
        warmup_steps=4000,
        train_steps=100000,
        update_cycle=1,
        optimizer="Adam",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        adadelta_rho=0.95,
        adadelta_epsilon=1e-7,
        pattern="",
        clipping="global_norm",
        clip_grad_norm=5.0,
        learning_rate=1.0,
        initial_learning_rate=0.0,
        learning_rate_schedule="linear_warmup_rsqrt_decay",
        learning_rate_boundaries=[0],
        learning_rate_values=[0.0],
        device_list=[0],
        # Checkpoint Saving
        keep_checkpoint_max=20,
        keep_top_checkpoint_max=5,
        save_summary=True,
        save_checkpoint_secs=0,
        save_checkpoint_steps=1000,
        # Validation
        eval_steps=2000,
        eval_secs=0,
        top_beams=1,
        beam_size=4,
        decode_batch_size=32,
        decode_alpha=0.6,
        decode_ratio=1.0,
        decode_length=50,
        validation="",
        references="",
    )

    return params


def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    p_name = os.path.join(model_dir, "params.json")
    m_name = os.path.join(model_dir, model_name + ".json")

    if os.path.exists(p_name):
        with open(p_name) as fd:
            logging.info("Restoring hyper parameters from %s" % p_name)
            json_str = fd.readline()
            params.parse_json(json_str)

    if os.path.exists(m_name):
        with open(m_name) as fd:
            logging.info("Restoring model parameters from %s" % m_name)
            json_str = fd.readline()
            params.parse_json(json_str)

    return params

def export_params(output_dir, name, params):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save params as params.json
    filename = os.path.join(output_dir, name)

    with open(filename, "w") as fd:
        fd.write(params.to_json())


def merge_params(params1, params2):
    params = utils.HParams()

    for (k, v) in six.iteritems(params1.values()):
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in six.iteritems(params2.values()):
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params


def override_params(params, args):
    params.model = args.model or params.model
    params.input = args.input or params.input
    params.output = args.output or params.output
    params.vocab = args.vocabulary or params.vocab
    params.validation = args.validation or params.validation
    params.references = args.references or params.references
    params.parse(args.parameters.lower())

    params.vocabulary = {
        "source": data.Vocabulary(params.vocab[0]),
        "target": data.Vocabulary(params.vocab[1])
    }

    return params


def collect_params(all_params, params):
    collected = utils.HParams()

    for k in six.iterkeys(params.values()):
        collected.add_hparam(k, getattr(all_params, k))

    return collected


def print_variables(model, pattern, log=True):
    flags = []

    for (name, var) in model.named_parameters():
        if re.search(pattern, name):
            flags.append(True)
        else:
            flags.append(False)

    weights = {v[0]: v[1] for v in model.named_parameters()}
    total_size = 0

    for name in sorted(list(weights)):
        if re.search(pattern, name):
            v = weights[name]
            total_size += v.nelement()

            if log:
                print("%s %s" % (name.ljust(60), str(list(v.shape)).rjust(15)))

    if log:
        print("Total trainable variables size: %d" % total_size)

    return flags


def exclude_variables(flags, grads_and_vars):
    idx = 0
    new_grads = []
    new_vars = []

    for grad, (name, var) in grads_and_vars:
        if flags[idx]:
            new_grads.append(grad)
            new_vars.append((name, var))

        idx += 1

    return zip(new_grads, new_vars)


def save_checkpoint(step, epoch, model, optimizer, params):
    if dist.get_rank() == 0:
        state = {
            "step": step,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        utils.save(state, params.output, params.keep_checkpoint_max)


def infer_gpu_num(param_str):
    result = re.match(r".*device_list=\[(.*?)\].*", param_str)

    if not result:
        return 1
    else:
        dev_str = result.groups()[-1]
        return len(dev_str.split(","))


def broadcast(model):
    for var in model.parameters():
        dist.broadcast(var.data, 0)


def get_learning_rate_schedule(params):
    if params.learning_rate_schedule == "linear_warmup_rsqrt_decay":
        schedule = optimizers.LinearWarmupRsqrtDecay(
            params.learning_rate, params.warmup_steps,
            initial_learning_rate=params.initial_learning_rate,
            summary=params.save_summary)
    elif params.learning_rate_schedule == "piecewise_constant_decay":
        schedule = optimizers.PiecewiseConstantDecay(
            params.learning_rate_boundaries, params.learning_rate_values,
            summary=params.save_summary)
    elif params.learning_rate_schedule == "linear_exponential_decay":
        schedule = optimizers.LinearExponentialDecay(
            params.learning_rate, params.warmup_steps,
            params.start_decay_step, params.end_decay_step,
            dist.get_world_size(), summary=params.save_summary)
    elif params.learning_rate_schedule == "constant":
        schedule = params.learning_rate
    else:
        raise ValueError("Unknown schedule %s" % params.learning_rate_schedule)

    return schedule


def get_clipper(params):
    if params.clipping.lower() == "none":
        clipper = None
    elif params.clipping.lower() == "adaptive":
        clipper = optimizers.adaptive_clipper(0.95)
    elif params.clipping.lower() == "global_norm":
        clipper = optimizers.global_norm_clipper(params.clip_grad_norm)
    else:
        raise ValueError("Unknown clipper %s" % params.clipping)

    return clipper


def get_optimizer(params, schedule, clipper):
    if params.optimizer.lower() == "adam":
        optimizer = optimizers.AdamOptimizer(learning_rate=schedule,
                                             beta_1=params.adam_beta1,
                                             beta_2=params.adam_beta2,
                                             epsilon=params.adam_epsilon,
                                             clipper=clipper,
                                             summaries=params.save_summary)
    elif params.optimizer.lower() == "adadelta":
        optimizer = optimizers.AdadeltaOptimizer(
            learning_rate=schedule, rho=params.adadelta_rho,
            epsilon=params.adadelta_epsilon, clipper=clipper,
            summaries=params.save_summary)
    elif params.optimizer.lower() == "sgd":
        optimizer = optimizers.SGDOptimizer(
            learning_rate=schedule, clipper=clipper,
            summaries=params.save_summary)
    else:
        raise ValueError("Unknown optimizer %s" % params.optimizer)

    return optimizer


def load_references(pattern):
    if not pattern:
        return None

    files = glob.glob(pattern)
    references = []

    for name in files:
        ref = []
        with open(name, "rb") as fd:
            for line in fd:
                items = line.strip().split()
                ref.append(items)
        references.append(ref)

    return list(zip(*references))


def main(args):
    # 初始化单个模型的子函数
    def model_init(mode="train",output_path=None,optimizer_resume=True):
        # 获取模型的默认参数
        model_cls = models.get_model(args.model)
        params = default_params()
        params = merge_params(params, model_cls.default_params(args.hparam_set))
        # 尝试从文件中载入上次的参数
        params = import_params(args.output, args.model, params)
        # 用本次输入覆盖上次输入
        params = override_params(params, args)

        # 用给定的输出路径覆盖args中的参数
        if output_path is not None:
            params.output=output_path

        # 分布式训练，还没实现
        if args.distributed:
            exit(865)
        else:
            params.device = params.device_list[args.local_rank]
            dist.init_process_group("nccl", init_method=args.url,
                                    rank=args.local_rank,
                                    world_size=len(params.device_list))
            # 设置本进程rank，在外围被赋值
            torch.cuda.set_device(params.device_list[args.local_rank])
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

        # 本次运行输出本次参数
        if dist.get_rank() == 0:
            export_params(params.output, "params.json", params)
            export_params(params.output, "%s.json" % params.model,
                          collect_params(params, model_cls.default_params()))

        # 模型实例化
        model = model_cls(params).cuda()

        # 半精度
        if args.half:
            model = model.half()
            torch.set_default_dtype(torch.half)
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

        # 模型模式
        if mode =="train":
            model.train()
        elif mode =="eval":
            model.eval()
        else:
            exit(866)

        # 优化器策略
        schedule = get_learning_rate_schedule(params)
        # 优化器裁剪
        clipper = get_clipper(params)
        # 优化器
        optimizer = get_optimizer(params, schedule, clipper)

        # 优化器半精度
        if args.half:
            optimizer = optimizers.LossScalingOptimizer(optimizer)

        # 优化器实例化
        optimizer = optimizers.MultiStepOptimizer(optimizer, params.update_cycle)

        # 参数表
        if dist.get_rank() == 0:
            print("*"*15,"MODEL %s" % params.model,"*"*15)
        trainable_flags = print_variables(model, params.pattern,
                                          dist.get_rank() == 0)
        if dist.get_rank() == 0:
            print("*"*30)

        # 载入checkpoint
        checkpoint = utils.latest_checkpoint(params.output)

        # 从args中读取checkpoint
        if args.checkpoint is not None:
            # Load pre-trained models
            state = torch.load(args.checkpoint, map_location="cpu")
            model.load_state_dict(state["model"])
            step = params.initial_step
            epoch = 0
            broadcast(model)
        # 从文件中读取checkpoint
        elif checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu")
            step = state["step"]
            epoch = state["epoch"]
            model.load_state_dict(state["model"])

            if "optimizer" in state and optimizer_resume:
                optimizer.load_state_dict(state["optimizer"])
        # 从头开始
        else:
            step = 0
            epoch = 0
            broadcast(model)

        # tensorboard 初始化
        global summary
        summary.init(params.output, params.save_summary)

        # 打印模型参数
        if dist.get_rank() == 0:
            print("Model %s" % args.model)
            for i in six.iterkeys(params.values()):
                print(i, getattr(params, i))
            print("*" * 30)

        # 每个模型自己的验证集吧
        if params.validation:
            sorted_key, eval_dataset = data.MTPipeline.get_infer_dataset(
                params.validation, params)
            references = load_references(params.references)
        else:
            sorted_key = None
            eval_dataset = None
            references = None

        return model,params,step,epoch,optimizer,trainable_flags,summary,sorted_key, eval_dataset,references

    # 训练一趟
    def gradient_des(loss, optimizer, model, trainable_flags, step, epoch, counter, params, sorted_key,
                     eval_dataset, references,alias):
        t = time.time()
        # 计算损失和梯度
        # loss = train_fn(features)
        gradients = optimizer.compute_gradients(loss,
                                                list(model.parameters()))
        # 梯度下降
        grads_and_vars = exclude_variables(
            trainable_flags,
            zip(gradients, list(model.named_parameters())))
        optimizer.apply_gradients(grads_and_vars)

        # 记录
        t = time.time() - t
        summary.scalar("loss", loss, step, write_every_n_steps=1)
        summary.scalar("global_step/sec", t, step)
        print(alias, ": epoch = %d, step = %d, loss = %.3f (%.3f sec)" %
              (epoch + 1, step, float(loss), t))

        if counter % params.update_cycle == 0:
            # 训练结束退出
            if step >= params.train_steps:
                utils.evaluate(model, sorted_key, eval_dataset,
                               params.output, references, params)
                save_checkpoint(step, epoch, model, optimizer, params)

                if dist.get_rank() == 0:
                    summary.close()

                exit(867)

            # 在验证集上评估
            if step % params.eval_steps == 0:
                utils.evaluate(model, sorted_key, eval_dataset,
                               params.output, references, params)

            # 保存checkpoint
            if step % params.save_checkpoint_steps == 0:
                save_checkpoint(step, epoch, model, optimizer, params)

    # 实例化模型
    model,params,step,epoch,optimizer,trainable_flags,summary,sorted_key, eval_dataset,references=model_init()

    # 载入数据集和验证集
    dataset = data.MTPipeline.get_train_dataset(params.input, params)

    # 训练计数
    counter = 0

    while True:
        # 取数据循环
        for features in dataset:
            # 设置step
            if counter % params.update_cycle == 0:
                step += 1
                utils.set_global_step(step)
            counter += 1

            feature,label=features
            loss=model(feature,label)
            gradient_des(loss,optimizer,model,trainable_flags,step,epoch,counter,params,sorted_key,eval_dataset,references,"Model 1")

        # 一个epoch结束
        epoch += 1


# Wrap main function
def process_fn(rank, args):
    local_args = copy.copy(args)
    local_args.local_rank = rank
    main(local_args)


def cli_main():
    parsed_args = parse_args()

    if parsed_args.distributed:
        main(parsed_args)
    else:
        # Pick a free port
        with socket.socket() as s:
            s.bind(("localhost", 0))
            port = s.getsockname()[1]
            url = "tcp://localhost:" + str(port)
            parsed_args.url = url

        world_size = infer_gpu_num(parsed_args.parameters)

        if world_size > 1:
            torch.multiprocessing.spawn(process_fn, args=(parsed_args,),
                                        nprocs=world_size)
        else:
            process_fn(0, parsed_args)


if __name__ == "__main__":
    cli_main()

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import os
import re
import time
import torch
import logging

from torch import Tensor
import torch.distributed as dist

from tutel.impls.communicate import get_world_size
import subprocess

def list_all_to_all(input, input_splits, output_splits, group=None, background=False):
    world_size = get_world_size(group)
    if world_size == 1:
        return input if not background else (input, lambda *args: None)
    list_all_to_all._use_builtins = True
    input = input.contiguous()
    input = list(torch.split(input, input_splits, dim=0))
    # input = [i.contiguous() for i in input]
    output = [torch.empty([i] + list(input[0].shape[1:]), dtype=input[0].dtype, device=input[0].device, requires_grad=input[0].requires_grad) for i in output_splits]
    if background:
        future_op = dist.all_to_all(output, input, group=group, async_op=True)
        return output, future_op.wait
    dist.all_to_all(output, input, group=group)
    output = torch.cat(output, dim=0)
    return output

class ListAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, input_splits, output_splits, group=None):
        ctx.group = group
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        return list_all_to_all(input, input_splits, output_splits, group)

    @staticmethod
    def backward(ctx, grad_output):
        return (list_all_to_all(grad_output, ctx.output_splits, ctx.input_splits, ctx.group), None, None, None)
        # return (ListAllToAll.apply(grad_output, ctx.output_splits, ctx.input_splits, ctx.group), None, None, None)

    @staticmethod
    def single(input, input_splits, output_splits, group=None):
        return ListAllToAll.apply(input, input_splits, output_splits, group)

list_all_to_all_single = ListAllToAll.single


from tutel.impls.communicate import TUTEL_GROUPING_CACHE

def create_groups_from_world_slurm(group_count, include_init=None):
    backend = TUTEL_GROUPING_CACHE.get('', include_init)
    if include_init:
        assert backend == include_init, "Only 1 backend type is allowed, get: %s v.s. %s" % (backend, include_init)
        TUTEL_GROUPING_CACHE[''] = backend

    if group_count in TUTEL_GROUPING_CACHE:
        return TUTEL_GROUPING_CACHE[group_count]

    def dist_init(host_addr, rank, local_rank, world_size, port=23456):
        host_addr_full = 'tcp://' + host_addr + ':' + str(port)
        torch.distributed.init_process_group(backend, init_method=host_addr_full,
                                            rank=rank, world_size=world_size)
        assert torch.distributed.is_initialized()

    try:
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        iplist = os.environ['SLURM_JOB_NODELIST']
        ip = subprocess.getoutput(f"scontrol show hostname {iplist} | head -n1")

        dist_init(ip, rank, local_rank, world_size, port=os.environ.get('MASTER_PORT', '23456'))
        dist_local_rank = local_rank

        glob_world_size, glob_world_rank = dist.get_world_size(), dist.get_rank()
        is_distributed = True

        def dist_print(*args):
            if glob_world_rank == 0:
                print(*args)
        
        # debug
        logging.info('successfully inin dist')

    except ValueError:
        glob_world_size, glob_world_rank, dist_local_rank = 1, 0, 0
        is_distributed = False
        dist_print = print

    assert glob_world_size % group_count == 0, f"Expected to evenly divide devices into {group_count} groups, while the world size of current sesion is {glob_world_size}."

    dist_group_size = group_count
    dist_world_size = glob_world_size // dist_group_size
    dist_world_rank = glob_world_rank % dist_world_size
    dist_group_rank = glob_world_rank // dist_world_size

    if is_distributed:
        global_group = model_group = data_group = dist.group.WORLD

        if dist_world_size != glob_world_size:
            groups, inner_ranks = [], []
            for gr in range(dist_group_size):
                group_ranks = [x for x in range(gr * dist_world_size, (gr + 1) * dist_world_size)]
                groups += [dist.new_group(ranks=group_ranks)]
                inner_ranks += [group_ranks]
            model_group = groups[dist_group_rank]

        if dist_group_size != glob_world_size:
            groups, outer_ranks = [], []
            for gr in range(dist_world_size):
                group_ranks = [x for x in range(gr, dist_world_size * dist_group_size, dist_world_size)]
                groups += [dist.new_group(ranks=group_ranks)]
                outer_ranks += [group_ranks]
            data_group = groups[dist_world_rank]
    else:
        model_group, data_group, global_group = None, None, None

    class ParallelPropStorage:
        pass

    result = ParallelPropStorage()

    result.global_size = glob_world_size
    result.global_rank = glob_world_rank

    result.group_count = dist_group_size
    result.data_rank = dist_group_rank

    result.model_size = dist_world_size
    result.model_rank = dist_world_rank

    if backend == 'nccl':
        result.local_device = torch.device('cuda', dist_local_rank)
        torch.cuda.set_device(result.local_device)
    elif backend == 'gloo':
        result.local_device = torch.device('cpu')
    elif backend is None:
        result.local_device = None
    else:
        raise Exception('Unsupported backend type: %s' % backend)

    result.data_group = data_group
    result.model_group = model_group
    result.global_group = global_group

    result.is_distributed = is_distributed
    result.dist_print = dist_print

    TUTEL_GROUPING_CACHE[group_count] = result
    return result



def create_groups_from_world(group_count, include_init=None, timeout=None):
    backend = TUTEL_GROUPING_CACHE.get('', include_init)
    if include_init:
        assert backend == include_init, "Only 1 backend type is allowed, get: %s v.s. %s" % (backend, include_init)
        TUTEL_GROUPING_CACHE[''] = backend

    if group_count in TUTEL_GROUPING_CACHE:
        return TUTEL_GROUPING_CACHE[group_count]

    try:
        if timeout is not None:
            if ('LOCAL_RANK' not in os.environ) and ('OMPI_COMM_WORLD_SIZE' in os.environ):
                if include_init:
                    dist.init_process_group(backend=backend,
                        init_method='tcp://%s:%s' % (os.environ['MASTER_ADDR'], os.environ.get('MASTER_PORT', '23456')),
                        rank=int(os.environ['OMPI_COMM_WORLD_RANK']), world_size=int(os.environ['OMPI_COMM_WORLD_SIZE']), timeout=timeout)
                dist_local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
            else:
                if include_init:
                    dist.init_process_group(backend=backend, timeout=timeout)
                dist_local_rank = min(int(os.environ.get('LOCAL_RANK', 0)), torch.cuda.device_count() - 1)
        else:
            if ('LOCAL_RANK' not in os.environ) and ('OMPI_COMM_WORLD_SIZE' in os.environ):
                if include_init:
                    dist.init_process_group(backend=backend,
                        init_method='tcp://%s:%s' % (os.environ['MASTER_ADDR'], os.environ.get('MASTER_PORT', '23456')),
                        rank=int(os.environ['OMPI_COMM_WORLD_RANK']), world_size=int(os.environ['OMPI_COMM_WORLD_SIZE']))
                dist_local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
            else:
                if include_init:
                    dist.init_process_group(backend=backend)
                dist_local_rank = min(int(os.environ.get('LOCAL_RANK', 0)), torch.cuda.device_count() - 1)
        glob_world_size, glob_world_rank = dist.get_world_size(), dist.get_rank()
        is_distributed = True

        def dist_print(*args):
            if glob_world_rank == 0:
                print(*args)
    except ValueError:
        glob_world_size, glob_world_rank, dist_local_rank = 1, 0, 0
        is_distributed = False
        dist_print = print

    assert glob_world_size % group_count == 0, f"Expected to evenly divide devices into {group_count} groups, while the world size of current sesion is {glob_world_size}."

    dist_group_size = group_count
    dist_world_size = glob_world_size // dist_group_size
    dist_world_rank = glob_world_rank % dist_world_size
    dist_group_rank = glob_world_rank // dist_world_size

    if is_distributed:
        global_group = model_group = data_group = dist.group.WORLD

        if dist_world_size != glob_world_size:
            groups, inner_ranks = [], []
            for gr in range(dist_group_size):
                group_ranks = [x for x in range(gr * dist_world_size, (gr + 1) * dist_world_size)]
                groups += [dist.new_group(ranks=group_ranks)]
                inner_ranks += [group_ranks]
            model_group = groups[dist_group_rank]

        if dist_group_size != glob_world_size:
            groups, outer_ranks = [], []
            for gr in range(dist_world_size):
                group_ranks = [x for x in range(gr, dist_world_size * dist_group_size, dist_world_size)]
                groups += [dist.new_group(ranks=group_ranks)]
                outer_ranks += [group_ranks]
            data_group = groups[dist_world_rank]
    else:
        model_group, data_group, global_group = None, None, None

    class ParallelPropStorage:
        pass

    result = ParallelPropStorage()

    result.global_size = glob_world_size
    result.global_rank = glob_world_rank

    result.group_count = dist_group_size
    result.data_rank = dist_group_rank

    result.model_size = dist_world_size
    result.model_rank = dist_world_rank

    if backend == 'nccl':
        result.local_device = torch.device('cuda', dist_local_rank)
        torch.cuda.set_device(result.local_device)
    elif backend == 'gloo':
        result.local_device = torch.device('cpu')
    elif backend is None:
        result.local_device = None
    else:
        raise Exception('Unsupported backend type: %s' % backend)

    result.data_group = data_group
    result.model_group = model_group
    result.global_group = global_group

    result.is_distributed = is_distributed
    result.dist_print = dist_print

    TUTEL_GROUPING_CACHE[group_count] = result
    return result
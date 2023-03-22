# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast, List

import torch
from torch import Tensor

from tutel.impls.jit_compiler import IS_HIP_EXTENSION
from tutel.jit_kernels.gating import fast_cumsum_sub_one, torch_cumsum_sub_one
from tutel.impls.communicate import simple_all_reduce
from . import tutel_sparse_nobatch as jit_kernel
import math
from torch.distributions.normal import Normal

class GatingEncoder(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, config: Any, reshaped_input: Tensor, *gates_):
        ctx.reshaped_input = reshaped_input
        ctx.config = config
        if gates_:
          ctx.gates_h2 = [x.view(-1, 1).repeat(1, 2) if x.dtype == torch.float16 else x for x in gates_]
        else:
          ctx.gates_h2 = [ctx.config.ones_helper] * len(ctx.config.indices_)

        sample_num = ctx.config.indices_[0].size(0)
        expert_input_nums = ctx.config.expert_input_nums
        expert_input_cumsum_exclusive = torch.cumsum(expert_input_nums, dim=0) - expert_input_nums
        expert_locations_begin = expert_input_cumsum_exclusive.to(expert_input_nums.dtype).contiguous()
        top_k = len(ctx.config.indices_)
        dispatched_input_numel = sum(expert_input_nums)
        assert sample_num * top_k == dispatched_input_numel

        dispatched_input = torch.zeros([dispatched_input_numel, ctx.config.model_dim], dtype=reshaped_input.dtype, device=reshaped_input.device)
        for g, i, l in zip(ctx.gates_h2, ctx.config.indices_, ctx.config.locations_):
          ctx.config.func_fwd(g, i, l, expert_locations_begin, reshaped_input, dispatched_input, extra=[ctx.config.indices_[0].size(0), ctx.config.aligned_dim, ctx.config.capacity])
        return dispatched_input

    @staticmethod
    def backward(ctx: Any, dispatched_input: Tensor):
        dispatched_input = dispatched_input.contiguous()
        last_result = None

        expert_input_nums = ctx.config.expert_input_nums
        expert_input_cumsum_exclusive = torch.cumsum(expert_input_nums, dim=0) - expert_input_nums
        expert_locations_begin = expert_input_cumsum_exclusive.to(expert_input_nums.dtype).contiguous()

        for g, i, l in zip(ctx.gates_h2, ctx.config.indices_, ctx.config.locations_):
          grad_data = torch.empty(ctx.reshaped_input.shape, dtype=dispatched_input.dtype, device=dispatched_input.device)
          ctx.config.func_bwd_data(g, i, l, expert_locations_begin, grad_data, dispatched_input, extra=[ctx.config.indices_[0].size(0), ctx.config.aligned_dim, ctx.config.capacity])
          last_result = grad_data if last_result is None else last_result + grad_data

        grad_gates = []
        if id(ctx.gates_h2[0]) != id(ctx.config.ones_helper):
          for i, l in zip(ctx.config.indices_, ctx.config.locations_):
            grad_gates1_s = torch.empty([ctx.config.sample_size,], dtype=dispatched_input.dtype, device=dispatched_input.device)
            ctx.config.func_bwd_gate(grad_gates1_s, i, l, expert_locations_begin, ctx.reshaped_input, dispatched_input, extra=[ctx.config.indices_[0].size(0), ctx.config.aligned_dim, ctx.config.capacity])
            grad_gates.append(grad_gates1_s)
        return (None, last_result, *grad_gates)


class GatingDecoder(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, config: Any, expert_output: Tensor, *gates_):
        ctx.expert_output = expert_output
        ctx.config = config
        if gates_:
          ctx.gates_h2 = [x.view(-1, 1).repeat(1, 2) if x.dtype == torch.float16 else x for x in gates_]
        else:
          ctx.gates_h2 = [ctx.config.ones_helper] * len(ctx.config.indices_)
        
        expert_input_nums = ctx.config.expert_input_nums
        expert_input_cumsum_exclusive = torch.cumsum(expert_input_nums, dim=0) - expert_input_nums
        expert_locations_begin = expert_input_cumsum_exclusive.to(expert_input_nums.dtype).contiguous()
        last_result = None
        for g, i, l in zip(ctx.gates_h2, ctx.config.indices_, ctx.config.locations_):
          single_output = torch.empty([config.sample_size, config.model_dim], dtype=expert_output.dtype, device=expert_output.device)
          config.func_bwd_data(g, i, l, expert_locations_begin, single_output, expert_output, extra=[ctx.config.indices_[0].size(0), ctx.config.aligned_dim, ctx.config.capacity])
          last_result = single_output if last_result is None else last_result + single_output
        return last_result

    @staticmethod
    def backward(ctx: Any, combined_output: Tensor):
        expert_input_nums = ctx.config.expert_input_nums
        expert_input_cumsum_exclusive = torch.cumsum(expert_input_nums, dim=0) - expert_input_nums
        expert_locations_begin = expert_input_cumsum_exclusive.to(expert_input_nums.dtype).contiguous()

        combined_output = combined_output.contiguous()
        grad_expert_output = torch.zeros(ctx.expert_output.shape, dtype=combined_output.dtype, device=combined_output.device)
        for g, i, l in zip(ctx.gates_h2, ctx.config.indices_, ctx.config.locations_):
          ctx.config.func_fwd(g, i, l, expert_locations_begin, combined_output, grad_expert_output, extra=[ctx.config.indices_[0].size(0), ctx.config.aligned_dim, ctx.config.capacity])

        grad_gates = []
        if id(ctx.gates_h2[0]) != id(ctx.config.ones_helper):
          for i, l in zip(ctx.config.indices_, ctx.config.locations_):
            grad_gates1_s = torch.empty([ctx.config.sample_size,], dtype=combined_output.dtype, device=combined_output.device)
            ctx.config.func_bwd_gate(grad_gates1_s, i, l, expert_locations_begin, combined_output, ctx.expert_output, extra=[ctx.config.indices_[0].size(0), ctx.config.aligned_dim, ctx.config.capacity])
            grad_gates.append(grad_gates1_s)
        return (None, grad_expert_output, *grad_gates)


class TutelMoeFastDispatcher:

    kernel_pool = dict()

    def __init__(self, num_global_experts, capacity, model_dim, dispatch_dtype):
        self.num_global_experts = int(num_global_experts)
        self.capacity = int(capacity)
        self.model_dim = int(model_dim)
        self.dtype = dispatch_dtype
        if IS_HIP_EXTENSION or dispatch_dtype != torch.float16:
            self.dtype = torch.float32
        self.original_dtype = dispatch_dtype
        self.aligned_dim = model_dim // (2 if self.dtype == torch.float16 else 1)
        self.is_cuda = None

    def update(self, indices_, locations_, gates_, expert_input_nums, capacity=None, is_postscore=True, dispatcher_no_score=False):
        self.indices_ = [x.to(torch.int32).view(-1) for x in indices_]
        self.locations_ = [x.to(torch.int32) for x in locations_]
        self.gates_ = [x.to(self.dtype) for x in gates_]
        self.is_postscore = is_postscore
        self.sample_size, self.capacity = int(self.indices_[0].size(0)), int(capacity) or self.capacity
        self.expert_input_nums = expert_input_nums.to(torch.int32)
        self.dispatcher_no_score = dispatcher_no_score

        if self.is_cuda != indices_[0].is_cuda:
            self.is_cuda = indices_[0].is_cuda
            if self.is_cuda not in TutelMoeFastDispatcher.kernel_pool:
                self.func_fwd = jit_kernel.create_forward(self.dtype, indices_[0].is_cuda)
                self.func_bwd_data = jit_kernel.create_backward_data(self.dtype, indices_[0].is_cuda)
                self.func_bwd_gate = jit_kernel.create_backward_gate(self.dtype, indices_[0].is_cuda)
                self.ones_helper = torch.ones([self.sample_size, 2], dtype=self.dtype, device=self.indices_[0].device)
                TutelMoeFastDispatcher.kernel_pool[self.is_cuda] = self.func_fwd, self.func_bwd_data, self.func_bwd_gate, self.ones_helper
            else:
                self.func_fwd, self.func_bwd_data, self.func_bwd_gate, self.ones_helper = TutelMoeFastDispatcher.kernel_pool[self.is_cuda]
                if self.ones_helper.shape[0] < self.sample_size:
                    self.ones_helper = torch.ones([self.sample_size, 2], dtype=self.dtype, device=self.indices_[0].device)
                    TutelMoeFastDispatcher.kernel_pool[self.is_cuda] = self.func_fwd, self.func_bwd_data, self.func_bwd_gate, self.ones_helper

    def encode(self, data):
        if self.dispatcher_no_score:
            return GatingEncoder.apply(self, data.to(self.dtype)).to(self.original_dtype)
        else:
            if self.is_postscore:
                return GatingEncoder.apply(self, data.to(self.dtype)).to(self.original_dtype)
            else:
                return GatingEncoder.apply(self, data.to(self.dtype), *self.gates_).to(self.original_dtype)

    def decode(self, data):
        if self.dispatcher_no_score:
            return GatingDecoder.apply(self, data.to(self.dtype)).to(self.original_dtype)
        else:
            if self.is_postscore:
                return GatingDecoder.apply(self, data.to(self.dtype), *self.gates_).to(self.original_dtype)
            else:
                return GatingDecoder.apply(self, data.to(self.dtype)).to(self.original_dtype)

fast_dispatcher = TutelMoeFastDispatcher

def one_hot_with_dtype(data, num_classes, dtype):
    result = torch.zeros([data.size(0), num_classes], device=data.device, dtype=dtype)
    result.scatter_(1, data.unsqueeze(-1), 1)
    return result

def compute_sorted_location(x, importance_scores):
    sorted_x = x[importance_scores.argsort(dim=0)]
    sorted_cumsum = fast_cumsum_sub_one(sorted_x) * sorted_x
    return sorted_cumsum[importance_scores.argsort(dim=0).argsort(dim=0)]

def load_balance(gates, mask1, num_global_experts, fp32_gate):
    if gates.dtype == torch.float32 or fp32_gate:
        me = torch.sum(gates.float(), dim=0)
        ce = torch.sum(mask1.to(me.dtype), dim=0)
        l_loss = torch.sum(me * ce) * (num_global_experts / (gates.size(0) * gates.size(0)))
    else:
        me = torch.mean(gates, dim=0)
        ce = torch.mean(mask1.to(gates.dtype), dim=0)
        l_loss = torch.sum(me * ce) * num_global_experts
    return l_loss

def load_importance_loss(scores_wo_noise, topk_logits, num_global_experts, gate_noise):
    def load_loss(scores_wo_noise, topk_logits, num_global_experts, gate_noise):
        assert gate_noise > 0, "`gate_noise` must be > 0 for normalization in load_importance_loss()."
        normal = Normal(
            torch.tensor([0.0], device=scores_wo_noise.device),
            torch.tensor([gate_noise / num_global_experts], device=scores_wo_noise.device),
        )
        threshold = topk_logits[:, -1].view(-1, 1).float()
        diff = scores_wo_noise.float() - threshold.float()
        prob = normal.cdf(diff)
        Load = prob.sum(0)
        l_load = Load.float().var() / (Load.float().mean() ** 2 + 1e-10)
        return l_load

    def importance_loss(scores_wo_noise):
        Impi = scores_wo_noise.float().sum(0)
        l_imp = Impi.float().var() / (Impi.float().mean() ** 2 + 1e-10)

        return l_imp

    l_imp = importance_loss(scores_wo_noise)
    l_load = load_loss(scores_wo_noise, topk_logits, num_global_experts, gate_noise)
    return (l_imp + l_load) / 2.0

def extract_critical(gates, top_k, capacity_factor=1.0, fp32_gate=False, batch_prioritized_routing=False):
    topk_indices = torch.topk(gates, top_k, dim=1).indices
    num_global_experts = gates.size(1)

    indices_s = [x.view(-1) for x in topk_indices.chunk(top_k, dim=1)]
    masks_se = [one_hot_with_dtype(x, num_classes=num_global_experts, dtype=x.dtype) for x in indices_s]
    gates_s = [(gates * x).sum(dim=1) for x in masks_se]

    l_loss = load_balance(gates, masks_se[0], num_global_experts, fp32_gate)

    if batch_prioritized_routing:
        importance_scores = -1 * gates.max(dim=1)[0]
        compute_location = lambda x: compute_sorted_location(x, importance_scores)
    else:
        compute_location = fast_cumsum_sub_one

    locations1 = compute_location(masks_se[0])

    locations_s = [torch.sum(locations1 * masks_se[0], dim=1).to(torch.int32)]

    expert_input_nums = locations1[-1, :] + 1

    if top_k > 1:
        acc_base = None
        for k in range(1, top_k):
            acc_base = torch.sum(masks_se[k - 1], dim=0, keepdim=True) if acc_base is None else acc_base + torch.sum(masks_se[k - 1], dim=0, keepdim=True)
            locations2 = compute_location(masks_se[k])
            locations2 += acc_base
            locations_s.append(torch.sum(locations2 * masks_se[k], dim=1).to(torch.int32))

            # added by mizhenxing
            expert_input_nums = locations2[-1, :] + 1

        # Normalize Gate
        denom_s = torch.clamp(sum(gates_s), min=torch.finfo(gates_s[0].dtype).eps)
        gates_s = [x / denom_s for x in gates_s]

    indices_s = [x.to(torch.int32) for x in indices_s]

    if capacity_factor > 0:
        capacity = top_k * int(capacity_factor * ((int(gates.size(0)) + num_global_experts - 1) // num_global_experts))
    else:
        capacity = torch.max(torch.concat(locations_s, dim=0))
        capacity = int(simple_all_reduce(capacity, op=torch.distributed.ReduceOp.MAX)) + 1
        if capacity_factor < 0:
            capacity = min(capacity, top_k * int(-capacity_factor * ((int(gates.size(0)) + num_global_experts - 1) // num_global_experts)))
    return [num_global_experts, indices_s, locations_s, gates_s, expert_input_nums, capacity], l_loss


def extract_critical_load_importance(gates, gates_wo_noise, logits_w_noise, top_k, gate_noise, capacity_factor=1.0, fp32_gate=False, batch_prioritized_routing=False, compute_balance_loss=False):
    # gates is from logits_w_noise
    topk_indices = torch.topk(gates, top_k, dim=1).indices
    num_global_experts = gates.size(1)

    indices_s = [x.view(-1) for x in topk_indices.chunk(top_k, dim=1)]
    masks_se = [one_hot_with_dtype(x, num_classes=num_global_experts, dtype=x.dtype) for x in indices_s]
    gates_s = [(gates * x).sum(dim=1) for x in masks_se]

    if compute_balance_loss:
        l_balance_loss = load_balance(gates, masks_se[0], num_global_experts, fp32_gate)
    else:
        l_balance_loss = torch.tensor(0.0, gates.device)
    l_loss = load_importance_loss(gates_wo_noise, logits_w_noise.gather(index=topk_indices, dim=1), num_global_experts, gate_noise)

    if batch_prioritized_routing:
        importance_scores = -1 * gates.max(dim=1)[0]
        compute_location = lambda x: compute_sorted_location(x, importance_scores)
    else:
        compute_location = fast_cumsum_sub_one

    locations1 = compute_location(masks_se[0])

    locations_s = [torch.sum(locations1 * masks_se[0], dim=1).to(torch.int32)]

    expert_input_nums = locations1[-1, :] + 1

    if top_k > 1:
        acc_base = None
        for k in range(1, top_k):
            acc_base = torch.sum(masks_se[k - 1], dim=0, keepdim=True) if acc_base is None else acc_base + torch.sum(masks_se[k - 1], dim=0, keepdim=True)
            locations2 = compute_location(masks_se[k])
            locations2 += acc_base
            locations_s.append(torch.sum(locations2 * masks_se[k], dim=1).to(torch.int32))

            # added by mizhenxing
            expert_input_nums = locations2[-1, :] + 1

        # Normalize Gate
        denom_s = torch.clamp(sum(gates_s), min=torch.finfo(gates_s[0].dtype).eps)
        gates_s = [x / denom_s for x in gates_s]

    indices_s = [x.to(torch.int32) for x in indices_s]

    if capacity_factor > 0:
        capacity = top_k * int(capacity_factor * ((int(gates.size(0)) + num_global_experts - 1) // num_global_experts))
    else:
        capacity = torch.max(torch.concat(locations_s, dim=0))
        capacity = int(simple_all_reduce(capacity, op=torch.distributed.ReduceOp.MAX)) + 1
        if capacity_factor < 0:
            capacity = min(capacity, top_k * int(-capacity_factor * ((int(gates.size(0)) + num_global_experts - 1) // num_global_experts)))
    return [num_global_experts, indices_s, locations_s, gates_s, expert_input_nums, capacity], l_loss, l_balance_loss


def fast_encode(data, critial_data, is_postscore=True):
    num_global_experts = critial_data[0]
    dispatcher = TutelMoeFastDispatcher(num_global_experts, 0, data.size(-1), data.dtype)
    dispatcher.update(*critial_data[1:], is_postscore=is_postscore)
    return dispatcher.encode(data).view(num_global_experts, -1, data.size(-1))

def fast_decode(data, critial_data, is_postscore=True):
    num_global_experts = critial_data[0]
    dispatcher = TutelMoeFastDispatcher(num_global_experts, 0, data.size(-1), data.dtype)
    dispatcher.update(*critial_data[1:], is_postscore=is_postscore)
    return dispatcher.decode(data).view(-1, data.size(-1))
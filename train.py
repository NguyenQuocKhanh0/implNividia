import os
import time
import argparse
import math
from numpy import finfo

import torch
# from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from hparams import create_hparams


import torch
import torch.distributed as dist
from torch.nn.modules import Module
from torch.autograd import Variable

def _flatten_dense_tensors(tensors):
    """Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.
    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.
    Returns:
        A contiguous 1D buffer containing input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].contiguous().view(-1)
    flat = torch.cat([t.contiguous().view(-1) for t in tensors], dim=0)
    return flat

def _unflatten_dense_tensors(flat, tensors):
    """View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by _flatten_dense_tensors.
    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
          unflatten flat.
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)


'''
This version of DistributedDataParallel is designed to be used in conjunction with the multiproc.py
launcher included with this example. It assumes that your run is using multiprocess with 1
GPU/process, that the model is on the correct device, and that torch.set_device has been
used to set the device.

Parameters are broadcasted to the other processes on initialization of DistributedDataParallel,
and will be allreduced at the finish of the backward pass.
'''
class DistributedDataParallel(Module):

    def __init__(self, module):
        super(DistributedDataParallel, self).__init__()
        #fallback for PyTorch 0.3
        if not hasattr(dist, '_backend'):
            self.warn_on_half = True
        else:
            self.warn_on_half = True if dist._backend == dist.dist_backend.GLOO else False

        self.module = module

        for p in self.module.state_dict().values():
            if not torch.is_tensor(p):
                continue
            dist.broadcast(p, 0)

        def allreduce_params():
            if(self.needs_reduction):
                self.needs_reduction = False
                buckets = {}
                for param in self.module.parameters():
                    if param.requires_grad and param.grad is not None:
                        tp = type(param.data)
                        if tp not in buckets:
                            buckets[tp] = []
                        buckets[tp].append(param)
                if self.warn_on_half:
                    if torch.cuda.HalfTensor in buckets:
                        print("WARNING: gloo dist backend for half parameters may be extremely slow." +
                              " It is recommended to use the NCCL backend in this case. This currently requires" +
                              "PyTorch built from top of tree master.")
                        self.warn_on_half = False

                for tp in buckets:
                    bucket = buckets[tp]
                    grads = [param.grad.data for param in bucket]
                    coalesced = _flatten_dense_tensors(grads)
                    dist.all_reduce(coalesced)
                    coalesced /= dist.get_world_size()
                    for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                        buf.copy_(synced)

        for param in list(self.module.parameters()):
            def allreduce_hook(*unused):
                param._execution_engine.queue_callback(allreduce_params)
            if param.requires_grad:
                param.register_hook(allreduce_hook)

    def forward(self, *inputs, **kwargs):
        self.needs_reduction = True
        return self.module(*inputs, **kwargs)

    '''
    def _sync_buffers(self):
        buffers = list(self.module._all_buffers())
        if len(buffers) > 0:
            # cross-node buffer sync
            flat_buffers = _flatten_dense_tensors(buffers)
            dist.broadcast(flat_buffers, 0)
            for buf, synced in zip(buffers, _unflatten_dense_tensors(flat_buffers, buffers)):
                buf.copy_(synced)
     def train(self, mode=True):
        # Clear NCCL communicator and CUDA event cache of the default group ID,
        # These cache will be recreated at the later call. This is currently a
        # work-around for a potential NCCL deadlock.
        if dist._backend == dist.dist_backend.NCCL:
            dist._clear_group_cache()
        super(DistributedDataParallel, self).train(mode)
        self.module.train(mode)
    '''
'''
Modifies existing model to do gradient allreduce, but doesn't change class
so you don't need "module"
'''
def apply_gradient_allreduce(module):
        if not hasattr(dist, '_backend'):
            module.warn_on_half = True
        else:
            module.warn_on_half = True if dist._backend == dist.dist_backend.GLOO else False

        for p in module.state_dict().values():
            if not torch.is_tensor(p):
                continue
            dist.broadcast(p, 0)

        def allreduce_params():
            if(module.needs_reduction):
                module.needs_reduction = False
                buckets = {}
                for param in module.parameters():
                    if param.requires_grad and param.grad is not None:
                        tp = param.data.dtype
                        if tp not in buckets:
                            buckets[tp] = []
                        buckets[tp].append(param)
                if module.warn_on_half:
                    if torch.cuda.HalfTensor in buckets:
                        print("WARNING: gloo dist backend for half parameters may be extremely slow." +
                              " It is recommended to use the NCCL backend in this case. This currently requires" +
                              "PyTorch built from top of tree master.")
                        module.warn_on_half = False

                for tp in buckets:
                    bucket = buckets[tp]
                    grads = [param.grad.data for param in bucket]
                    coalesced = _flatten_dense_tensors(grads)
                    dist.all_reduce(coalesced)
                    coalesced /= dist.get_world_size()
                    for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                        buf.copy_(synced)

        for param in list(module.parameters()):
            def allreduce_hook(*unused):
                Variable._execution_engine.queue_callback(allreduce_params)
            if param.requires_grad:
                param.register_hook(allreduce_hook)

        def set_needs_reduction(self, input, output):
            self.needs_reduction = True

        module.register_forward_hook(set_needs_reduction)
        return module

def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    model.train()
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
        logger.log_validation(val_loss, model, y, y_pred, iteration)


def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    criterion = Tacotron2Loss()

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    model.train()
    is_overflow = False
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            model.zero_grad()
            x, y = model.parse_batch(batch)
            y_pred = model(x)

            loss = criterion(y_pred, y)
            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()
            if hparams.fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if hparams.fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), hparams.grad_clip_thresh)
                is_overflow = math.isnan(grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)

            optimizer.step()

            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    iteration, reduced_loss, grad_norm, duration))
                logger.log_training(
                    reduced_loss, grad_norm, learning_rate, duration, iteration)

            if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
                validate(model, criterion, valset, iteration,
                         hparams.batch_size, n_gpus, collate_fn, logger,
                         hparams.distributed_run, rank)
                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)

            iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams()
    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    train(args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)

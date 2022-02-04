from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import os
import time
import json
import random
import wandb
import logging
from collections import OrderedDict

from torch.utils.data import Sampler
import torch.distributed as dist

import torch
from apex.parallel import DistributedDataParallel as ApexDDP
import argparse

from log import setup_default_logging
from models.resnet import ResNet50

_logger = logging.getLogger('train')

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt
    

class OrderedDistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


def train(model, dataloader, criterion, optimizer, log_interval, accumulation_steps=1):   
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc_m = AverageMeter()
    losses_m = AverageMeter()
    
    end = time.time()
    
    model.train()
    optimizer.zero_grad()
    for idx, (inputs, targets) in enumerate(dataloader):
        # optimizer condition
        opt_cond = (idx + 1) % accumulation_steps == 0

        if opt_cond or idx == 0:
            data_time_m.update(time.time() - end)
        
        inputs, targets = inputs.cuda(), targets.cuda()

        # predict
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # loss for accumulation steps
        loss /= accumulation_steps        
        loss.backward()

        if opt_cond:
            # loss update
            optimizer.step()
            optimizer.zero_grad()

            if args.distributed:
                loss = reduce_tensor(loss.data, args.world_size)
            losses_m.update(loss.item()*accumulation_steps)

            # accuracy
            preds = outputs.argmax(dim=1) 
            acc_m.update(targets.eq(preds).sum().item()/targets.size(0), n=targets.size(0))
            
            batch_time_m.update(time.time() - end)
        
            if (idx // accumulation_steps) % log_interval == 0 and idx != 0: 
                _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                        'Acc: {acc.avg:.3%} '
                        'LR: {lr:.3e} '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        (idx+1)//accumulation_steps, len(dataloader)//accumulation_steps, 
                        loss       = losses_m, 
                        acc        = acc_m, 
                        lr         = optimizer.param_groups[0]['lr'],
                        batch_time = batch_time_m,
                        rate       = inputs.size(0) / batch_time_m.val,
                        rate_avg   = inputs.size(0) / batch_time_m.avg,
                        data_time  = data_time_m))
   
        end = time.time()
    
    return OrderedDict([('acc',acc_m.avg), ('loss',losses_m.avg)])
        
def test(model, dataloader, criterion, log_interval):
    correct = 0
    total = 0
    total_loss = 0
    
    model.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            
            # predict
            outputs = model(inputs)
            
            # loss 
            loss = criterion(outputs, targets)
            
            # total loss and acc
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += targets.eq(preds).sum().item()
            total += targets.size(0)
            
            if idx % log_interval == 0 and idx != 0: 
                _logger.info('TEST [%d/%d]: Loss: %.3f | Acc: %.3f%% [%d/%d]' % 
                            (idx+1, len(dataloader), total_loss/(idx+1), 100.*correct/total, correct, total))
                
    return OrderedDict([('acc',correct/total), ('loss',total_loss/len(dataloader))])
                
def fit(
    exp_name, model, epochs, trainloader, testloader, criterion, optimizer, scheduler, 
    savedir, log_interval, args, accumulation_steps=1
):
    savedir = os.path.join(savedir,exp_name)
    os.makedirs(savedir, exist_ok=True)
    wandb.init(name=exp_name, project='Accumulation Steps', config=args)
    
    best_acc = 0

    for epoch in range(epochs):
        if args.distributed and hasattr(trainloader.sampler, 'set_epoch'):
            trainloader.sampler.set_epoch(epoch)

        _logger.info(f'\nEpoch: {epoch+1}/{epochs}')
        train_metrics = train(model, trainloader, criterion, optimizer, log_interval, accumulation_steps)
        eval_metrics = test(model, testloader, criterion, log_interval)

        scheduler.step()

        # wandb
        metrics = OrderedDict(epoch=epoch)
        metrics.update([('train_' + k, v) for k, v in train_metrics.items()])
        metrics.update([('eval_' + k, v) for k, v in eval_metrics.items()])
        wandb.log(metrics)
    
        # checkpoint
        if best_acc < eval_metrics['acc']:
            state = {'best_epoch':epoch, 'best_acc':eval_metrics['acc']}
            json.dump(state, open(os.path.join(savedir, f'{exp_name}.json'),'w'), indent=4)

            weights = {'model':model.state_dict()}
            torch.save(weights, os.path.join(savedir, f'{exp_name}.pt'))
            
            _logger.info('Best Accuracy {0:.3%} to {1:.3%}'.format(best_acc, eval_metrics['acc']))

            best_acc = eval_metrics['acc']

    _logger.info('Best Metric: {0:.3%} (epoch {1:})'.format(state['best_acc'], state['best_epoch']))


def run(args):
    setup_default_logging()
    torch_seed(args.seed)

    # distributed
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = f'cuda:{args.device_num}'
    args.world_size = 1
    args.rank = args.device_num  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        _logger.info(f'Training with a single process on 1 GPUs. (device: {args.device})')
    assert args.rank >= 0


    # Load Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR100(os.path.join(args.datadir,'CIFAR100'), train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(os.path.join(args.datadir,'CIFAR100'), train=False, download=True, transform=transform_test)


    # distributed
    train_sampler = None
    test_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        
        # This will add extra duplicate entries to result in equal num
        # of samples per-process, will slightly alter validation results
        test_sampler = OrderedDistributedSampler(testset)


    # data loader
    trainloader = DataLoader(
        trainset, 
        batch_size  = args.batch_size, 
        shuffle     = train_sampler is None, 
        num_workers = args.num_workers, 
        sampler     = train_sampler
    )

    testloader = DataLoader(
        testset, 
        batch_size  = args.batch_size, 
        shuffle     = False, 
        num_workers = args.num_workers, 
        sampler     = test_sampler
    )

    # Build Model
    model = ResNet50(num_classes=100)
    model.cuda()
    _logger.info('# of params: {}'.format(np.sum([p.numel() for p in model.parameters()])))

    # setup distributed training
    if args.distributed:
        # Apex DDP preferred unless native amp is activated
        if args.local_rank == 0:
            _logger.info("Using NVIDIA APEX DistributedDataParallel.")
        model = ApexDDP(model, delay_allreduce=True)


    # Set training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Fitting model
    fit(exp_name           = args.exp_name,
        model              = model, 
        epochs             = args.epochs, 
        trainloader        = trainloader, 
        testloader         = testloader, 
        criterion          = criterion, 
        optimizer          = optimizer, 
        scheduler          = scheduler,
        savedir            = args.savedir,
        log_interval       = args.log_interval,
        accumulation_steps = args.accumulation_steps,
        args               = args)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Tootouch's AMP Experiments")
    parser.add_argument('--exp-name',type=str,help='experiment name')
    parser.add_argument('--datadir',type=str,default='/datasets',help='data directory')
    parser.add_argument('--savedir',type=str,default='./saved_model',help='saved model directory')
    parser.add_argument('--epochs',type=int,default=100,help='the number of epochs')
    parser.add_argument('--lr',type=float,default=0.1,help='learning_rate')
    parser.add_argument('--batch-size',type=int,default=128,help='batch size')
    parser.add_argument('--num-workers',type=int,default=8,help='the number of workers (threads)')
    parser.add_argument('--apex',action='store_true',default=False)
    parser.add_argument('--log-interval',type=int,default=10,help='log interval')
    parser.add_argument('--seed',type=int,default=223,help='223 is my birthday')
    parser.add_argument('--accumulation-steps',type=int,default=1,help='accumulation step size')
    parser.add_argument('--local_rank',type=str,default=0,help='local rank')
    parser.add_argument('--device_num',type=int,default=0, help='gpu device number')

    args = parser.parse_args()

    run(args)
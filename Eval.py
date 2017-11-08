"""Training and validation code for bddmodelcar."""
import sys
import traceback
import logging
import time
import os
import importlib
import numpy as np

from Config import config
from Dataset import Dataset

import Utils

from torch.autograd import Variable
import torch.nn.utils as nnutils
import torch
Net = importlib.import_module(config['model']['py_path']).Net

iter_num = {'i': 0}

def iterate(net, loss_func, optimizer=None, input=None, truth=None, train=False):
    """
    Encapsulates a training or validation iteration.

    :param net: <nn.Module>: network to train
    :param optimizer: <torch.optim>: optimizer to use
    :param input: <tuple>: tuple of np.array or tensors to pass into net. Should contain data for this iteration
    :param truth: <np.array | tensor>: tuple of np.array to pass into optimizer. Should contain data for this iteration
\    :return: loss
    """

    net.eval()

    # Transform inputs into Variables for pytorch and run forward prop
    input = tuple([Variable(tensor).cuda() for tensor in input])
    outputs = net(*input).cuda()
    truth = Variable(truth).cuda()
    loss = loss_func(outputs, truth)

    print('------------------')
    print([int(i * 1000) / 1000. for i in
           np.ndarray.tolist(outputs.cpu()[0].data.transpose(0,1).contiguous().view(-1).numpy())])
    print([int(i * 1000) / 1000. for i in
           np.ndarray.tolist(truth.cpu()[0].data.transpose(0,1).contiguous().view(-1).numpy())])
    iter_num['i'] = 1 + iter_num['i']
    return loss.cpu().data[0]

def main():
    # Configure logging
    logging.basicConfig(filename=config['logging']['path'], level=logging.DEBUG)
    logging.debug(config)

    # Set Up PyTorch Environment
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.set_device(config['hardware']['gpu'])
    torch.cuda.device(config['hardware']['gpu'])

    # Define basic training and network parameters
    net, loss_func = Net(n_steps=config['model']['future_frames'],
                        n_frames=config['model']['past_frames']).cuda(), \
                    torch.nn.MSELoss().cuda()

    # Iterate over all epochs

    torch.cuda.set_device(config['hardware']['gpu'])
    torch.cuda.device(config['hardware']['gpu'])
    if not config['training']['start_epoch'] == 0:
        print("Resuming")
        save_data = torch.load(
            os.path.join(config['model']['save_path'], config['model']['name'] + "epoch%04d.weights" % (config['training']['start_epoch'] - 1,)))
        net.load_state_dict(save_data)
        net.cuda()

    for epoch in range(config['training']['start_epoch'], config['training']['num_epochs']):
        try:
            start = time.time()
            logging.debug('Starting validation epoch #{}'.format(epoch))

            val_dataset = Dataset(config['validation']['dataset']['path'],
                                  require_one=config['dataset']['include_labels'],
                                  ignore_list=config['dataset']['ignore_labels'],
                                  stride=config['model']['frame_stride'],
                                  seed=config['validation']['rand_seed'],
                                  nframes=config['model']['past_frames'],
                                  train_ratio=config['validation']['dataset']['train_ratio'],
                                  nsteps=config['model']['future_frames'],
                                  separate_frames=config['model']['separate_frames'],
                                  metadata_shape=config['model']['metadata_shape'],
                                  cache_file=('partition_cache' in config['training'] and config['training']['partition_cache'])
                                             or config['model']['save_path'] + config['model']['name'] + '.cache'
                                  )

            val_data_loader = val_dataset.get_val_loader(batch_size=1,
                                                               shuffle=config['validation']['dataset']['shuffle'],
                                                               pin_memory=False)
            val_loss = Utils.LossLog()

            net.eval()

            for batch_idx, (camera, meta, truth) in enumerate(val_data_loader):
                # Cuda everything
                camera, meta, truth = camera.cuda(), meta.cuda(), truth.cuda()

                loss = iterate(net, loss_func=loss_func, truth=truth, input=(camera, meta), train=False)

                # Logging Loss
                val_loss.add(loss)

                print('Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                      .format(epoch, batch_idx * len(camera), len(val_data_loader.dataset.val_part),
                              100. * batch_idx / len(val_data_loader), loss))

            logging.debug('Finished validation epoch #{}'.format(epoch))

        except Exception:
            logging.error(traceback.format_exc())  # Log exception
            sys.exit(1)

if __name__ == '__main__':
    main()

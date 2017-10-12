"""Training and validation code for bddmodelcar."""
import sys
import traceback
import logging
import time
import os
import importlib

from Config import config
from Dataset import Dataset

import Utils

from torch.autograd import Variable
import torch.nn.utils as nnutils
import torch
Net = importlib.import_module('nets.' + config['model']['name'] + '.Net').Net

def iterate(net, loss_func, optimizer=None, input=None, truth=None, mask=None, train=True):
    """
    Encapsulates a training or validation iteration.

    :param net: <nn.Module>: network to train
    :param optimizer: <torch.optim>: optimizer to use
    :param input: <tuple>: tuple of np.array or tensors to pass into net. Should contain data for this iteration
    :param truth: <np.array | tensor>: tuple of np.array to pass into optimizer. Should contain data for this iteration
    :param mask: <np.array | tensor>: mask to ignore unnecessary outputs.
    :return: loss
    """

    if train:
        net.train()
        optimizer.zero_grad()
    else:
        net.eval()

    # Transform inputs into Variables for pytorch and run forward prop
    input = tuple([Variable(tensor) for tensor in input])
    outputs = net(*input).cuda() * Variable(mask)
    loss = loss_func(outputs, Variable(truth))

    if not train:
        return loss.data[0]

    # Run backprop, gradient clipping
    loss.backward()
    nnutils.clip_grad_norm(net.parameters(), 1.0)

    # Apply backprop gradients
    optimizer.step()

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
    net, loss_func= Net().cuda(), torch.nn.MSELoss().cuda()
    optimizer = torch.optim.Adagrad(net.parameters())

    # Iterate over all epochs
    for epoch in range(config['training']['start_epoch'], config['training']['num_epochs']):
        try:
            if not epoch == 0:
                print("Resuming")
                save_data = torch.load(os.path.join(config['model']['save_path'], "epoch%02d.weights" % (epoch - 1,)))
                net.load_state_dict(save_data)

            logging.debug('Starting training epoch #{}'.format(epoch))

            train_dataset = Dataset(config['training']['dataset']['path'],
                                    require_one=config['dataset']['include_labels'],
                                    ignore_list=config['dataset']['ignore_labels'],
                                    stride=config['model']['frame_stride'],
                                    seed=config['training']['rand_seed'],
                                    nframes=config['model']['past_frames'],
                                    train_ratio=config['training']['dataset']['train_ratio'],
                                    separate_frames=config['model']['separate_frames'])

            train_data_loader = train_dataset.get_train_loader(batch_size=config['training']['dataset']['batch_size'],
                                                               shuffle=config['training']['dataset']['shuffle'],
                                                               p_subsample=config['training']['dataset']['p_subsample'],
                                                               pin_memory=False)

            train_loss = Utils.LossLog()
            start = time.time()

            for batch_idx, (camera, meta, truth, mask) in enumerate(train_data_loader):
                # Cuda everything
                camera, meta, truth, mask = camera.cuda(), meta.cuda(), mask.cuda(), truth.cuda()
                truth = truth * mask

                loss = iterate(net, loss_func, optimizer, (camera, meta), truth, mask)

                # Logging Loss
                train_loss.add(loss)

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(camera), len(train_data_loader.dataset.train_part),
                100. * batch_idx / len(train_data_loader), loss.data[0]))

                cur = time.time()
                print('{} Hz'.format(250./(cur - start)))
                start = cur


            Utils.csvwrite('trainloss.csv', [train_loss.average()])

            logging.debug('Finished training epoch #{}'.format(epoch))
            logging.debug('Starting validation epoch #{}'.format(epoch))

            val_dataset = Dataset(config['validation']['dataset']['path'],
                                    require_one=config['dataset']['include_labels'],
                                    ignore_list=config['dataset']['ignore_labels'],
                                    stride=config['model']['frame_stride'],
                                    seed=config['validation']['rand_seed'],
                                    nframes=config['model']['past_frames'],
                                    train_ratio=config['validation']['dataset']['train_ratio'],
                                    separate_frames=config['model']['separate_frames'])

            val_data_loader = val_dataset.get_val_loader(batch_size=config['validation']['dataset']['batch_size'],
                                                               shuffle=config['validation']['dataset']['shuffle'],
                                                               pin_memory=False)
            val_loss = Utils.LossLog()

            net.eval()

            for batch_idx, (camera, meta, truth, mask) in enumerate(val_data_loader):
                # Cuda everything
                camera, meta, mask, truth = camera.cuda(), meta.cuda(), mask.cuda(), truth.cuda()
                truth = truth * mask

                loss = iterate(net, loss_func, truth=truth, train=False)

                # Logging Loss
                val_loss.add(loss)

                print('Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                      .format(epoch, batch_idx * len(camera), len(val_data_loader.dataset.val_part),
                              100. * batch_idx / len(val_data_loader), loss.data[0]))

            Utils.csvwrite('valloss.csv', [val_loss.average()])
            logging.debug('Finished validation epoch #{}'.format(epoch))
            Utils.save_net("epoch%02d" % (epoch,), net)

        except Exception:
            logging.error(traceback.format_exc())  # Log exception
            sys.exit(1)

if __name__ == '__main__':
    main()

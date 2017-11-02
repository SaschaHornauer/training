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
mse_loss, linear_loss = torch.nn.MSELoss(), torch.nn.L1Loss()
nll1, nll2 = torch.nn.NLLLoss(), torch.nn.NLLLoss()

def iterate(net, loss_func, optimizer=None, input=None, truth=None, train=True):
    """
    Encapsulates a training or validation iteration.

    :param net: <nn.Module>: network to train
    :param optimizer: <torch.optim>: optimizer to use
    :param input: <tuple>: tuple of np.array or tensors to pass into net. Should contain data for this iteration
    :param truth: <np.array | tensor>: tuple of np.array to pass into optimizer. Should contain data for this iteration
\    :return: loss
    """

    if train:
        net.train()
        optimizer.zero_grad()
        input = tuple([Variable(tensor).cuda() for tensor in input] + ([truth] if ('requires_controls' in dir(net)
                                                                         and net.requires_controls) else []))
    else:
        net.eval()
        input = tuple([Variable(tensor).cuda() for tensor in input])


    # Transform inputs into Variables for pytorch and run forward prop
    outputs = [_.cuda() for _ in net(*input)]
    truth = Variable(truth).cuda() * 9
    truths = torch.unbind(torch.squeeze(truth, 1), 1)
    loss = nll1(outputs[0], truths[0].long()) + nll2(outputs[1], truths[1].long())
    # loss = (mse_loss(outputs, truth) + linear_loss(outputs, truth)) / 2

    if iter_num['i'] % 5 == 0:
        print('------------------')
        steering, controls = outputs[0][0].cpu().data.view(-1), outputs[0][1].cpu().data.view(-1)
        steering, controls = [int(i * 1000) / 1000. for i in
                              np.ndarray.tolist(steering.numpy())], \
                             [int(i * 1000) / 1000. for i in
                              np.ndarray.tolist(controls.numpy())]
        true_steering, true_controls = truths[0][0].cpu().data.view(-1), truths[1][0].cpu().data.view(-1)
        print('Predicted steering: ' + str(steering.index(max(steering))))
        print('Predicted motor: ' + str(controls.index(max(controls))))
        print('Actual steering: ' + str(true_steering))
        print('Actual motor: ' + str(true_controls))
    iter_num['i'] = 1 + iter_num['i']

    if not train:
        return loss.cpu().data[0]

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
    net, loss_func = Net(n_steps=config['model']['future_frames'],
                        n_frames=config['model']['past_frames']).cuda(), \
                    torch.nn.NLLLoss().cuda()

    # Iterate over all epochs

    torch.cuda.set_device(config['hardware']['gpu'])
    torch.cuda.device(config['hardware']['gpu'])
    if not config['training']['start_epoch'] == 0:
        print("Resuming")
        save_data = torch.load(
            os.path.join(config['model']['save_path'], config['model']['name'] + "epoch%02d.weights" % (config['training']['start_epoch'] - 1,)))
        net.load_state_dict(save_data)
        net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=config['training']['learning_rate']) \
        if config['training']['learning_rate'] else torch.optim.Adam(net.parameters())

    for epoch in range(config['training']['start_epoch'], config['training']['num_epochs']):
        try:
            logging.debug('Starting training epoch #{}'.format(epoch))
            train_dataset = Dataset(config['training']['dataset']['path'],
                                    require_one=config['dataset']['include_labels'],
                                    ignore_list=config['dataset']['ignore_labels'],
                                    stride=config['model']['step_stride'],
                                    frame_stride=config['model']['frame_stride'],
                                    seed=config['training']['rand_seed'],
                                    nframes=config['model']['past_frames'],
                                    nsteps=config['model']['future_frames'],
                                    train_ratio=config['training']['dataset']['train_ratio'],
                                    separate_frames=config['model']['separate_frames'],
                                    metadata_shape=config['model']['metadata_shape'],
                                    p_exclude_run=config['training']['p_exclude_run'],
                                    cache_file=('partition_cache' in config['training'] and config['training']['partition_cache'])
                                               or config['model']['save_path'] + config['model']['name'] + '.cache')
            train_data_loader = train_dataset.get_train_loader(batch_size=config['training']['dataset']['batch_size'],
                                                               shuffle=config['training']['dataset']['shuffle'],
                                                               p_subsample=config['training']['dataset']['p_subsample'],
                                                               seed=(epoch+config['training']['rand_seed']),
                                                               pin_memory=False)

            train_loss = Utils.LossLog()
            start = time.time()

            for batch_idx, (camera, meta, truth) in enumerate(train_data_loader):
                # Cuda everything
                camera, meta, truth = camera.cuda(), meta.cuda(), truth.cuda()

                loss = iterate(net, loss_func=loss_func, optimizer=optimizer,
                               input=(camera, meta), truth=truth)

                # Logging Loss
                train_loss.add(loss)

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(camera), len(train_data_loader.dataset.subsampled_train_part),
                100. * batch_idx / len(train_data_loader), loss))

                cur = time.time()
                print('{} Hz'.format(float(len(camera))/(cur - start)))
                start = cur


            Utils.csvwrite(config['logging']['training_loss'], [train_loss.average()])

            logging.debug('Finished training epoch #{}'.format(epoch))
            logging.debug('Starting validation epoch #{}'.format(epoch))

            val_dataset = Dataset(config['validation']['dataset']['path'],
                                  require_one=config['dataset']['include_labels'],
                                  ignore_list=config['dataset']['ignore_labels'],
                                  stride=config['model']['step_stride'],
                                  frame_stride=config['model']['frame_stride'],
                                  seed=config['validation']['rand_seed'],
                                  nframes=config['model']['past_frames'],
                                  # train_ratio=config['validation']['dataset']['train_ratio'],
                                  train_ratio=0,
                                  nsteps=config['model']['future_frames'],
                                  separate_frames=config['model']['separate_frames'],
                                  metadata_shape=config['model']['metadata_shape'],
                                  cache_file=('partition_cache' in config['training'] and config['training']['partition_cache'])
                                             or config['model']['save_path'] + config['model']['name'] + '.cache'
                                  )

            val_data_loader = val_dataset.get_val_loader(batch_size=config['validation']['dataset']['batch_size'],
                                                               shuffle=config['validation']['dataset']['shuffle'],
                                                               pin_memory=False,
                                                               seed=config['validation']['rand_seed'],
                                                               p_subsample=1.-config['validation']['dataset']['train_ratio'])
            val_loss = Utils.LossLog()

            net.eval()

            for batch_idx, (camera, meta, truth) in enumerate(val_data_loader):
                # Cuda everything
                camera, meta, truth = camera.cuda(), meta.cuda(), truth.cuda()

                loss = iterate(net, loss_func=loss_func, truth=truth, input=(camera, meta), train=False)

                # Logging Loss
                val_loss.add(loss)

                print('Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                      .format(epoch, batch_idx * len(camera), len(val_data_loader.dataset.subsampled_val_part),
                              100. * batch_idx / len(val_data_loader), loss))

            Utils.csvwrite(config['logging']['validation_loss'], [val_loss.average()])
            logging.debug('Finished validation epoch #{}'.format(epoch))
            Utils.save_net(config['model']['save_path'], config['model']['name'] + "epoch%02d" % (epoch,), net)

        except Exception:
            logging.error(traceback.format_exc())  # Log exception
            sys.exit(1)

if __name__ == '__main__':
    main()
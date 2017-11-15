import numpy as np
import math

loss_file = open('loss', 'r')

losses = [float(_) for _ in loss_file]
# losses = [losses[i] for i in range(len(losses)) if i % 4 == 0]
max_loss = max(losses)
min_loss = min(losses)
min_loss_index = losses.index(min_loss)


loss_normalized = [(loss - min_loss) / max_loss for loss in losses]
loss_ratios = [losses[i+1] / (losses[i] + 1e-30) for i in range(len(losses) - 1)]
r = reduce(lambda a, b: a * b, loss_ratios[:min_loss_index-1], 1.) ** (1. / (min_loss_index - 1))
sigma_r = math.e ** (math.sqrt(sum([math.log(r_hat / r) ** 2 for r_hat in loss_ratios[:min_loss_index-1]])) / (min_loss_index-2))
r_div = reduce(lambda a, b: a * b, loss_ratios[min_loss_index:], 1.) ** (1. / (len(loss_ratios)-min_loss_index))
sigma_r_div = math.e ** (math.sqrt(sum([math.log(r_div_hat / r) ** 2 for r_div_hat in loss_ratios[min_loss_index:]])) / (len(loss_ratios) - min_loss_index - 1))

print('Min loss: {} at index {}'.format(min_loss, min_loss_index))
print('Geometric average of convergence rate: {}'.format(r ** 4))
print('Geometric SD of convergence rate: {}'.format(sigma_r ** 4))
print('Geometric average of divergence rate: {}'.format(r_div ** 4))
print('Geometric SD of divergence rate: {}'.format(sigma_r_div ** 4))

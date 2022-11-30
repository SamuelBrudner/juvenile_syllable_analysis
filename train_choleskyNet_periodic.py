import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.metrics import Loss
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from python_src.choleskyNet_periodic import CHOLESKY
from python_src.devoDataset_periodic import DevoDataset

MAX_EPOCHS = 500  # never train for more than this many epochs
PATIENCE = 3  # after this many epochs without test set improvement, stop training
DATA_DICTIONARY = {'animal_name': os.path.join('path', 'to', 'pytorch_training', 'parent_directory')}
MODEL_OUTPUT_SUBDIRECTORY = 'choleskyModels_periodic'  # Created inside data-dictionary directory. Holds trained models 


def get_ffnn_data_loader(pc_data, age_data, batch_size=512, num_workers=4, random_seed=42, validation_split=0.2):
    dataset_size = len(age_data)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating pytorch data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    overall_dataset = DevoDataset(pc_data, age_data, transform=numpy_to_tensor)
    train_dataloader = DataLoader(overall_dataset, batch_size=batch_size, num_workers=num_workers,
                                  sampler=train_sampler)
    val_dataloader = DataLoader(overall_dataset, batch_size=batch_size, num_workers=num_workers, sampler=val_sampler)

    return train_dataloader, val_dataloader


def numpy_to_tensor(x):
    """Transform a numpy array into a torch.FloatTensor."""
    return torch.from_numpy(x).type(torch.FloatTensor)


def mv_gauss_nll_loss(prediction, target, sigma):
    mv_distribution = MultivariateNormal(prediction, sigma)
    log_probs = mv_distribution.log_prob(target)
    loss = torch.mean(-log_probs)
    return loss


def mv_gauss_ll_score(prediction, target, sigma):
    mv_distribution = MultivariateNormal(prediction, sigma)
    log_probs = mv_distribution.log_prob(target)
    score = torch.mean(log_probs)
    return score


def output_transform(batch_output_dict):
    mu = batch_output_dict['mu']
    sigma = batch_output_dict['sigma']
    pc_data = batch_output_dict['pc_data']
    return (mu, pc_data, {'sigma': sigma})


# Even though this function calculates and uses batch loss for updating,
# it also must output the my_gauss_lnn_loss inputs, so that training_step
# can trigger downstream evaluation of the Ignite helper "metric" and get epoch loss
def train_step(engine, batch):
    pc_data, age_data = batch
    pc_data = pc_data.to(ch_net.device)
    age_data = age_data.to(ch_net.device)
    ch_net.train()
    mu, sigma = ch_net.calculate_fit(age_data)
    loss = mv_gauss_nll_loss(mu, pc_data, sigma)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {'mu': mu, 'sigma': sigma, 'pc_data': pc_data}


def validation_step(engine, batch):
    ch_net.eval()
    with torch.no_grad():
        pc_data, age_data = batch[0], batch[1]
        pc_data = pc_data.to(ch_net.device)
        age_data = age_data.to(ch_net.device)
        mu, sigma = ch_net.calculate_fit(age_data)
    return {'mu': mu, 'sigma': sigma, 'pc_data': pc_data}


def score_function(engine):
    val_loss = engine.state.metrics['Loss']
    return -val_loss


for animal in DATA_DICTIONARY:
    datafolder = DATA_DICTIONARY[animal]
    animal_prefix = animal
    file_pattern = '_'.join((animal_prefix, '*', 'pcs.npy'))
    file_pattern = os.path.join(datafolder, 'pytorch_training', file_pattern)
    print(file_pattern)
    for pc_datafile in glob.glob(file_pattern):
        pc_datafile_fn = os.path.basename(pc_datafile)
        syllPattern = pc_datafile_fn.split('_')[1]
        print(syllPattern)
        age_datafile_fn = '_'.join((animal_prefix, syllPattern, 'age.npy'))
        age_datafile = os.path.join(datafolder, 'pytorch_training', age_datafile_fn)

        pcs = np.load(pc_datafile)
        n_pcs = pcs.shape[1]
        print('Number of components to use: ' + str(n_pcs))
        age = np.load(age_datafile)

        metric_loss = Loss(mv_gauss_nll_loss, output_transform=output_transform)
        metric_score = Loss(mv_gauss_ll_score, output_transform=output_transform)

        trainer = Engine(train_step)
        metric_loss.attach(trainer, "Loss")
        evaluator = Engine(validation_step)
        metric_loss.attach(evaluator, "Loss")
        metric_score.attach(evaluator, "Score")

        train_dl, val_dl = get_ffnn_data_loader(pcs, age)

        ch_net = CHOLESKY(latent_size=n_pcs)
        optimizer = Adam(ch_net.parameters(), lr=ch_net.lr)

        stopper = EarlyStopping(patience=PATIENCE, score_function=score_function, trainer=trainer)
        evaluator.add_event_handler(Events.COMPLETED, stopper)
        to_save = {'trainer': trainer, 'evaluator': evaluator, 'model': ch_net, 'optimizer': optimizer}
        save_dir = os.path.join(datafolder, MODEL_OUTPUT_SUBDIRECTORY, syllPattern)
        saver = ModelCheckpoint(save_dir, filename_prefix=syllPattern, create_dir=True, score_function=score_function,
                                score_name='neg_loss', n_saved=1, require_empty=False)
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, saver, to_save)


        @trainer.on(Events.STARTED)
        def start_message():
            print("Start training!")


        @trainer.on(Events.EPOCH_COMPLETED)
        def run_validation():
            print('Epoch ' + str(trainer.state.epoch) + ' complete')
            print('training loss: %.2f' % (trainer.state.metrics['Loss']))
            evaluator.run(val_dl)  # run current model over val_dl and store loss in evaluator.state.output
            print('validation loss: %.2f' % evaluator.state.metrics['Loss'])


        trainer.run(train_dl, max_epochs=MAX_EPOCHS)

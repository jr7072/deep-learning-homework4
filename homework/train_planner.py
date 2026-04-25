import argparse
from datetime import datetime
import torch
import numpy as np
from homework.models import load_model, save_model
from homework.datasets.road_dataset import load_data
import torch.utils.tensorboard as tb
from homework.metrics import PlannerMetric
from collections import defaultdict

"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

def get_device() -> torch.DeviceObjType:
    '''
        loads the device to use for training
    '''

    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")

    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    return device

def reset_metrics(metric_store: dict):

    for metric in metric_store.values():

        metric.reset()


def transformer_training(
    model: torch.nn.Module,
    train_data,
    val_data,
    num_epochs,
    logger,
    device,
    log_dir,
    lr
):
    
    global_step = 0

    metric_store = defaultdict(dict)

    for mode in ['train', 'val']:
        
        metric_store[mode] = PlannerMetric()

    loss_func = torch.nn.functional.l1_loss

    # load optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=5)
    
    print('started training loop...')
    for epoch in range(num_epochs):

        reset_metrics(metric_store)

        model.train()
        logger.add_scalar('meta/epoch', epoch, global_step=global_step)

        for train_package in train_data:

            left_tracks = train_package['track_left'].to(device)
            right_tracks = train_package['track_right'].to(device)
            waypoints = train_package['waypoints'].to(device)
            waypoints_mask = train_package['waypoints_mask'].to(device)
            batched_waypoints_mask = waypoints_mask[..., None]
            clean_waypoints = waypoints * batched_waypoints_mask

            # forward pass
            waypoints_pred = model(left_tracks, right_tracks)
            clean_waypoints_pred = waypoints_pred * batched_waypoints_mask
            loss = loss_func(clean_waypoints_pred, clean_waypoints)

            optim.zero_grad()
            loss.backward()
            optim.step()

            # log the loss
            logger.add_scalar(
                'train/loss',
                loss,
                global_step=global_step
            )

            metric_store['train'].add(
                waypoints_pred,
                waypoints,
                waypoints_mask
            )

            global_step += 1
        
        with torch.no_grad():
            
            model.eval()
            for val_package in val_data:

                left_tracks = val_package['track_left'].to(device)
                right_tracks = val_package['track_right'].to(device)
                waypoints = val_package['waypoints'].to(device)
                waypoints_mask = val_package['waypoints_mask'].to(device)
                batched_waypoints_mask = waypoints_mask[..., None]
                clean_waypoints = waypoints * batched_waypoints_mask

                waypoints_pred = model(left_tracks, right_tracks)
                clean_waypoints_pred = waypoints_pred * batched_waypoints_mask
                val_loss = loss_func(clean_waypoints_pred, clean_waypoints)

                # log the loss here
                logger.add_scalar(
                    f'val/loss',
                    val_loss,
                    global_step=global_step
                )

                metric_store['val'].add(
                    waypoints_pred,
                    waypoints,
                    waypoints_mask
                )
            
            sched.step(val_loss)

        # calculate all the metrics
        for mode, metrics in metric_store.items():


            metric_results = metrics.compute()
            long_error = metric_results['longitudinal_error']
            lat_error = metric_results['lateral_error']
            
            # log the errors here per mode
            logger.add_scalar(
                f'{mode}/longitudinal_error',
                long_error,
                global_step=global_step
            )

            logger.add_scalar(
                f'{mode}/lateral_error',
                lat_error,
                global_step=global_step
            )

        # save model just in case I want to early stop
        if ((epoch + 1) % 10 == 0) or (epoch == num_epochs - 1):
            torch.save(model.state_dict(), log_dir + f'/mlp_planner_{epoch}.th')



def cnn_training(
    model: torch.nn.Module,
    train_data,
    val_data,
    num_epochs,
    logger,
    device,
    log_dir,
    lr
):

    global_step = 0
    metric_store = defaultdict(dict)

    for mode in ['train', 'val']:
        metric_store[mode] = PlannerMetric()

    loss_func = torch.nn.functional.l1_loss

    # load optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=5)
    
    print('started training loop...')
    for epoch in range(num_epochs):

        reset_metrics(metric_store)

        model.train()
        logger.add_scalar('meta/epoch', epoch, global_step=global_step)

        for train_package in train_data:

            images = train_package['image'].to(device)
            waypoints = train_package['waypoints'].to(device)
            waypoints_mask = train_package['waypoints_mask'].to(device)
            batched_waypoints_mask = waypoints_mask[..., None]
            clean_waypoints = waypoints * batched_waypoints_mask

            # forward pass
            waypoints_pred = model(images)
            clean_waypoints_pred = waypoints_pred * batched_waypoints_mask
            loss = loss_func(clean_waypoints_pred, clean_waypoints)

            optim.zero_grad()
            loss.backward()
            optim.step()

            # log the loss
            logger.add_scalar(
                'train/loss',
                loss,
                global_step=global_step
            )

            metric_store['train'].add(
                waypoints_pred,
                waypoints,
                waypoints_mask
            )

            global_step += 1
        
        with torch.no_grad():
            
            model.eval()
            for val_package in val_data:

                images = val_package['image'].to(device)
                waypoints = val_package['waypoints'].to(device)
                waypoints_mask = val_package['waypoints_mask'].to(device)
                batched_waypoints_mask = waypoints_mask[..., None]
                clean_waypoints = waypoints * batched_waypoints_mask

                waypoints_pred = model(images)
                clean_waypoints_pred = waypoints_pred * batched_waypoints_mask
                val_loss = loss_func(clean_waypoints_pred, clean_waypoints)

                # log the loss here
                logger.add_scalar(
                    f'val/loss',
                    val_loss,
                    global_step=global_step
                )

                metric_store['val'].add(
                    waypoints_pred,
                    waypoints,
                    waypoints_mask
                )
            
            sched.step(val_loss)

        # calculate all the metrics
        for mode, metrics in metric_store.items():

            metric_results = metrics.compute()
            long_error = metric_results['longitudinal_error']
            lat_error = metric_results['lateral_error']
            
            # log the errors here per mode
            logger.add_scalar(
                f'{mode}/longitudinal_error',
                long_error,
                global_step=global_step
            )

            logger.add_scalar(
                f'{mode}/lateral_error',
                lat_error,
                global_step=global_step
            )

        # save model just in case I want to early stop
        if ((epoch + 1) % 10 == 0) or (epoch == num_epochs - 1):
            torch.save(model.state_dict(), log_dir + f'/cnn_planner_{epoch}.th')


def mlp_training(
    model: torch.nn.Module,
    train_data,
    val_data,
    num_epochs,
    logger,
    device,
    log_dir,
    lr
):

    global_step = 0

    metric_store = defaultdict(dict)

    for mode in ['train', 'val']:
        
        metric_store[mode] = PlannerMetric()

    loss_func = torch.nn.functional.l1_loss

    # load optimizer
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=5)
    
    print('started training loop...')
    for epoch in range(num_epochs):

        reset_metrics(metric_store)

        model.train()
        logger.add_scalar('meta/epoch', epoch, global_step=global_step)

        for train_package in train_data:

            left_tracks = train_package['track_left'].to(device)
            right_tracks = train_package['track_right'].to(device)
            waypoints = train_package['waypoints'].to(device)
            waypoints_mask = train_package['waypoints_mask'].to(device)
            batched_waypoints_mask = waypoints_mask[..., None]
            clean_waypoints = waypoints * batched_waypoints_mask

            # forward pass
            waypoints_pred = model(left_tracks, right_tracks)
            clean_waypoints_pred = waypoints_pred * batched_waypoints_mask
            loss = loss_func(clean_waypoints_pred, clean_waypoints)

            optim.zero_grad()
            loss.backward()
            optim.step()

            # log the loss
            logger.add_scalar(
                'train/loss',
                loss,
                global_step=global_step
            )

            metric_store['train'].add(
                waypoints_pred,
                waypoints,
                waypoints_mask
            )

            global_step += 1
        
        with torch.no_grad():
            
            model.eval()
            for val_package in val_data:

                left_tracks = val_package['track_left'].to(device)
                right_tracks = val_package['track_right'].to(device)
                waypoints = val_package['waypoints'].to(device)
                waypoints_mask = val_package['waypoints_mask'].to(device)
                batched_waypoints_mask = waypoints_mask[..., None]
                clean_waypoints = waypoints * batched_waypoints_mask

                waypoints_pred = model(left_tracks, right_tracks)
                clean_waypoints_pred = waypoints_pred * batched_waypoints_mask
                val_loss = loss_func(clean_waypoints_pred, clean_waypoints)

                # log the loss here
                logger.add_scalar(
                    f'val/loss',
                    val_loss,
                    global_step=global_step
                )

                metric_store['val'].add(
                    waypoints_pred,
                    waypoints,
                    waypoints_mask
                )
            
            sched.step(val_loss)

        # calculate all the metrics
        for mode, metrics in metric_store.items():


            metric_results = metrics.compute()
            long_error = metric_results['longitudinal_error']
            lat_error = metric_results['lateral_error']
            
            # log the errors here per mode
            logger.add_scalar(
                f'{mode}/longitudinal_error',
                long_error,
                global_step=global_step
            )

            logger.add_scalar(
                f'{mode}/lateral_error',
                lat_error,
                global_step=global_step
            )

        # save model just in case I want to early stop
        if ((epoch + 1) % 10 == 0) or (epoch == num_epochs - 1):
            torch.save(model.state_dict(), log_dir + f'/mlp_planner_{epoch}.th')


trainer_factory = {
    'mlp_planner': mlp_training,
    'cnn_planner': cnn_training,
    'transformer_planner': transformer_training
}

def train(
    model_name: str,
    num_epochs: int,
    lr: float=1e-3,
    batch_size: int=128,
    seed: int=42,
    **kwargs
):
    '''
        trains the models for this homework and saves it to model folder

        model_name: str the name of the model to train see model factories
        num_epochs: int the number of epochs for the training run
        in models.py for acceptable names
        lr: float the learning rate for the ADAM optimizer
        batch_size: (int) the batch size for the training run
        seed: (int) the seed to use for the training run
        nas: (bool) whether to use network architecture search
    '''

    # get the device
    device = get_device()

    # set the seed for torch and numpy 
    torch.manual_seed(seed)
    np.random.seed(seed)

    # load tensorboard and set log dir
    log_dir = f'{model_name}/{model_name}_{datetime.now().strftime("%m%d_%H%M%S")}'
    logger = tb.SummaryWriter(log_dir)
    
    # load the model
    model = load_model(model_name, **kwargs).to(device)

    # load the data
    print('loading data...')
    train_package = load_data('drive_data/train', batch_size=batch_size, shuffle=True)
    val_package = load_data('drive_data/val', batch_size=batch_size)

    # TODO: add evaluator
    trainer = trainer_factory[model_name]
    trainer(
        model,
        train_package,
        val_package,
        num_epochs,
        logger,
        device,
        log_dir,
        lr
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=2)

    args = vars(parser.parse_args())

    train(**args)

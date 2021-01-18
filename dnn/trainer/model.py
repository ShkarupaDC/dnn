import torch
from torch import nn
from torch.utils.data import DataLoader

import psutil
from tqdm import tqdm, trange

from dnn.utils import AverageMeter
from dnn.enums import ModelState, TrainingState
from dnn.callback import CallbackRunner

class Model(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.train_loader = None
        self.valid_loader = None
        self.optimizer = None
        self.scheduler = None
        self.scheduler_step_after = 'batch'
        self.train_step = 0
        self.valid_step = 0
        self.epoch = 0
        self._train_state = None
        self._model_state = None
        self.device = 'cuda'
        self.fp16 = False
        self.metrics = {}
        self.metrics['train'] = {}
        self.metrics['valid'] = {}
        self.metrics['test'] = {}
        self.callback_runner = None

    @property
    def model_state(self):
        return self._model_state

    @model_state.setter
    def model_state(self, value):
        self._model_state = value

    @property
    def train_state(self):
        return self._train_state

    @train_state.setter
    def train_state(self, value):
        self._train_state = value
        
        if self.callback_runner is not None:
            self.callback_runner(value)

    def init_model(self, device, train_dataset, valid_dataset, train_sampler, valid_sampler, 
        train_bs, valid_bs, workers, callbacks, fp16):

        if next(self.parameters()).device != device:
            self.to(device)
        
        self.device = device

        if not callbacks: self.callbacks = []

        if workers is None: 
            workers = psutil.cpu_count()

        if self.train_loader is None:
                
            self.train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=train_bs,
                sampler=train_sampler,
                num_workers=workers,
                shuffle=True
            )

        if self.valid_loader is None:
            if valid_dataset is not None:

                self.valid_loader = DataLoader(
                    dataset=valid_dataset,
                    batch_size=valid_bs,
                    sampler=valid_sampler,
                    num_workers=workers
                )

        if self.optimizer is None:
            self.optimizer = self.fetch_optimizer()

        if self.scheduler is None:
            self.scheduler = self.fetch_scheduler()

        self.callback_runner = CallbackRunner(self, callbacks)
        self.fp16 = fp16

        if self.fp16 is True:
            self.scaler = torch.cuda.amp.GradScaler()

    def loss(self, *args, **kwargs):
        return

    def fetch_optimizer(self, *args, **kwargs):
        return 

    def fetch_scheduler(self, *args, **kwargs):
        return   
    
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def model_output(self, data):
        
        for key, value in data.items():
            data[key] = value.to(self.device)
        
        if self.fp16 is True:
            with torch.cuda.amp.autocast():
                out, loss, metrics = self(data)
        else:
            out, loss, metrics = self(data)

        return out, loss, metrics

    def train_one_step(self, data):
        
        self.optimizer.zero_grad()
        _, loss, metrics = self.model_output(data)

        if self.fp16 is True:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:     
            loss.backward()
            self.optimizer.step()

        if self.scheduler is not None:
            if self.scheduler_step_after == 'batch':
                 self.scheduler.step()

        return loss, metrics

    def valid_one_step(self, data):
        
        _, loss, metrics = self.model_output(data)
        return loss, metrics

    def predict_one_step(self, data):

        out, _, _ self.model_output(data)
        return out

    def update_metrics(self, losses, monitor):
        
        self.metrics[self._model_state].update(monitor)
        self.metrics[self._model_state]['loss'] = losses.average

    def train_one_epoch(self, data_loader):
        self.train()
        
        self.model_state = ModelState.TRAIN
        losses = AverageMeter()
        
        loader = tqdm(data_loader, total=len(data_loader))
        for batch_idx, data in enumerate(loader):
            
            self.model_state = TrainingState.TRAIN_EPOCH_START
            loss, metrics = self.train_one_step(data)
            
            self.model_state = TrainingState.TRAIN_STEP_END
            losses.update(loss.item(), len(data))
            
            if not batch_idx:
                metrics_meters = {name: AverageMeter() for name in metrics}
            
            monitor = {}
            for name in metrics:
                metrics_meters[name].update(metrics[name], len(data))
                monitor[name] = metrics_meters[name].average
            
            loader.set_postfix(monitor)
            self.train_step += 1
        
        loader.close()

        self.update_metrics(losses, monitor)
        return losses.average

    def valid_one_epoch(self, data_loader):
        self.eval()

        self.model_state = ModelState.VALID
        losses = AverageMeter()

        loader = tqdm(data_loader, total=len(data_loader))
        for batch_idx, data in enumerate(loader):
            
            with torch.no_grad():
                self.model_state = TrainingState.VALID_STEP_START
                loss, metrics = self.valid_one_step(data)
                
                self.model_state = TrainingState.VALID_STEP_END
                losses.update(loss.item(), len(data))
            
            if not batch_idx:
                metrics_meters = {name: AverageMeter() for name in metrics}
            
            monitor = {}
            for name in metrics:
                metrics_meters[name].update(metrics[name], len(data))
                monitor[name] = metrics_meters[name].average
            
            loader.set_postfix(monitor)
            self.valid_step += 1
        
        loader.close()
        
        self.update_metrics(losses, monitor)
        return losses.average 

    def process_output(self, out):
        return out.cpu().detach().numpy()

    def predict_one_step(self, dataset, sampler, bs, workers):

        if workers is None:
            workers = psutil.cpu_count()
        
        test_loader = DataLoader(
            dataset=dataset,
            batch_size=bs,
            sampler=sampler,
            num_workers=workers
        )

        self.eval()
        self.model_state = ModelState.TEST
        
        loader = tqdm(test_loader, total=len(test_loader))
        for batch_idx, data in enumerate(loader):
            
            with torch.no_grad():
                out = self.predict_one_step(data)
                out = self.process_output(out)
            
            yield out

        loader.close()

    def fit(self, device='cuda', train_dataset=None, valid_dataset=None, train_sampler=None, valid_sampler=None,
        train_bs=64, valid_bs=64, epochs=10, workers=None, callbacks=None, fp16=False):
        
        self.init_model(device, train_dataset, valid_dataset, train_sampler, valid_sampler,
            train_bs, valid_bs, workers, callbacks, fp16)

        self.model_state = TrainingState.TRAIN_START
        for epoch in trange(epochs):
            self.model_state = TrainingState.EPOCH_START

            self.model_state = TrainingState.TRAIN_EPOCH_START
            train_loss = self.train_one_epoch(self.train_loader)
            self.model_state = TrainingState.TRAIN_EPOCH_END
            
            if self.valid_loader:
                self.model_state = TrainingState.VALID_EPOCH_START
                valid_loss = self.valid_one_epoch(self.valid_loader)
                self.model_state = TrainingState.VALID_EPOCH_END
            
            if self.scheduler is not None:
                if self.scheduler_step_after = 'epoch':
                    self.scheduler.step()

            self.epoch += 1
            self.model_state = TrainingState.EPOCH_END

        self.model_state = TrainingState.TRAIN_END

            
            


            



import numpy as np
import torch


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 notebook: bool = False
                 ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []
        self.validation_iou = []

    def run_trainer(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        progressbar = trange(self.epochs, desc='Progress')
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            self._train()

            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate()

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.batch(self.validation_loss[i])  # learning rate scheduler step with validation loss
                else:
                    self.lr_scheduler.batch()  # learning rate scheduler step
                    
        return self.training_loss, self.validation_loss, self.learning_rate, self.validation_iou

    def _train(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
            self.optimizer.zero_grad()  # zerograd the parameters
            out = self.model(input)  # one forward pass
            loss = self.criterion(out, target)  # calculate loss
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar

        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])


        batch_iter.close()

    def dice_loss(self, inputs, target):
        intersection = 2.0 * (target * inputs).sum()
        union = target.sum() + inputs.sum()
        if target.sum() == 0 and inputs.sum() == 0:
            return 1.0

        return intersection / union

    def mean_IOU(self, target, predicted):

        if target.dim() == 3:
            target = target.reshape(-1,1, target.shape[1], target.shape[2])
            print("reshapping", target.shape)
            # return

        if target.dim() != 4:
            print("target has dim", target.dim(), ", Must be 4.")
            return

        if target.shape != predicted.shape:
            print("target has dimension", target.shape, ", predicted values have shape", predicted.shape)
            return
        
        iousum = 0
        for i in range(target.shape[0]):
            target_arr = target[i, :, :, :].clone().detach().cpu().numpy()
            predicted_arr = predicted[i, :, :, :].clone().detach().cpu().numpy()
            intersection = np.logical_and(target_arr, predicted_arr).sum()
            print('intersection',intersection)
            union = np.logical_or(target_arr, predicted_arr).sum()
            print('union',union)
            if union == 0:
                iou_score = 0
            else :
                iou_score = intersection / union
            iousum +=iou_score
            
        miou = iousum/target.shape[0]
        return miou
        
    def _validate(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                          leave=False)
        
        # num_correct = 0
        # num_pixels = 0
        meanIou_list = []

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)

            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(out, target)
                loss_value = loss.item()
                valid_losses.append(loss_value)

                # finding the accuracy
                preds = torch.sigmoid(out)
                preds = (preds > 0.5).float()
                # num_correct += (preds == target).sum()
                # num_pixels += torch.numel(preds)
                # accuracy = num_correct / num_pixels * 100
                # accuracy_list.append(accuracy.cpu().numpy())
                meanIou = self.mean_IOU(target,preds)
                meanIou_list.append(meanIou)

                batch_iter.set_description(f'Validation: (loss {loss_value:.4f}, iou {meanIou:.2f})')

        self.validation_loss.append(np.mean(valid_losses))
        self.validation_iou.append(np.mean(meanIou_list))

        print(f'EPOCH: {self.epoch}, Validation Loss: {np.mean(valid_losses)}, Accuracy: {np.mean(meanIou_list)}')
        batch_iter.close()

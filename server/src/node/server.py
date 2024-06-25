import os
import torch
import torch.nn.functional as F

from utils.file_actions import mkdir_save

from abc import ABC, abstractmethod

class Server(ABC):
    def __init__(self, args, config, model, save_interval=50):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.experiment_name = args.experiment_name
        self.save_path = os.path.join("results", config.EXP_NAME, args.experiment_name)
        self.save_interval = save_interval
        self.allocated_incentive_pool = config.ALLOCATED_INCENTIVE_POOL
        self.args = args

        self.model = model.to(self.device)
        self.model.train()
        
        mkdir_save(self.model, os.path.join(self.save_path, "init_model.pt"))

        self.indices = None
        
        self.W1 = config.w1
        self.W2 = config.w2
        
        self.server_training_time = None
        
        self.prev_ps_mean = None

        self.test_loader = None
        self.init_test_loader()
        self.save_exp_config(self.config, self.args)

    @abstractmethod
    def init_test_loader(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_exp_config(self, *args, **kwargs):
        pass
    
    def model_evaluate(self):
        test_loss = 0
        n_correct = 0

        self.model.eval()
        for inputs, labels in self.test_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            test_loss += F.binary_cross_entropy_with_logits(outputs, labels, reduction='sum').item()

            labels_predicted = outputs.argmax(dim=1, keepdim=True)
            if labels.dim() == 2:
                labels = torch.argmax(labels, dim=1)

            n_correct += labels_predicted.eq(labels.view_as(labels_predicted)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        acc = n_correct / len(self.test_loader.dataset)
        
        self.model.train()
        
        return test_loss, acc
    
    def main(self, idx, list_cm, lr, list_loss, list_acc):
    
        list_sd = [x.state_dict() for x in list_cm]
        # Aggregating the client models
        server_dict = self.model.state_dict()
        for key in server_dict.keys():
            server_dict[key] = torch.stack([list_sd[i][key].float() for i in range(len(list_sd))], 0).mean(0)
                
        self.model.load_state_dict(server_dict)

        if idx % self.config.EVAL_DISP_INTERVAL == 0:
            loss, acc = self.model_evaluate()
            list_loss.append(loss)
            list_acc.append(acc)

            print("Round #{} (Experiment = {}).".format(idx, self.experiment_name))
            print("Loss/acc (at round #{}) = {}/{}".format((len(list_loss) - 1) * self.config.EVAL_DISP_INTERVAL, loss, acc))
            print("Current lr = {}".format(lr))

        if idx % self.save_interval == 0:
            mkdir_save(list_loss, os.path.join(self.save_path, "loss.pt"))
            mkdir_save(list_acc, os.path.join(self.save_path, "accuracy.pt"))
            mkdir_save(self.model, os.path.join(self.save_path, "model.pt"))

        return [self.model.state_dict() for _ in range(self.config.NUM_CLIENTS)]
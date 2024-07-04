# selecionar o modelo
# selecionar o dataset
# rodar o treino cross val

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from sklearn.metrics import accuracy_score
import time

class Experiments():
    def __init__(self,model,loss_func,optimizer,fold=""):
        self.model = model
        
        if torch.cuda.device_count() > 1:
            self.model= torch.nn.DataParallel(model)
            
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.fold = fold
        
        if fold == "":
            self.writer = SummaryWriter()
        else:
            path_log_dir = "runs/" + fold
            self.writer = SummaryWriter(log_dir=path_log_dir)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)

    def eval_loss(self,data_loader):    
        with torch.no_grad():
            loss, cnt = 0, 0
            pred_list, target_list = [], []
            for images, labels, _, img_id in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss += self.loss_func(outputs, labels).item()
                cnt += 1

                _, pred = torch.max(outputs, axis=1)
                pred_list.append(pred.cpu().numpy())
                target_list.append(labels.cpu().numpy())

            
            pred_list  = np.concatenate(pred_list).ravel()
            target_list  = np.concatenate(target_list).ravel()
            eval_acc = accuracy_score(pred_list, target_list)
        
        return loss/cnt , eval_acc

    def train(self,train_dataloader,val_dataloader,num_epochs=100):
        self.model.to(self.device)
        best_loss = np.inf

        print('--------------------------------')
        print('| Epoch | Train Loss | Train Acc | Validation Loss | Validation Acc | Time |')
    
        for epoch in range(num_epochs):
            start = time.time()
            loss_epoch, cnt = 0, 0
            pred_list, target_list = [], []
            for k, (batch_images, batch_labels,_ , id_img) in enumerate(train_dataloader):  
                # Aplicando um flatten na imagem e movendo ela para o device alvo
                batch_images, batch_labels = batch_images.to(self.device), batch_labels.to(self.device)
                
                # Fazendo a forward pass
                # observe que o modelo é agnóstico ao batch size
                outputs = self.model(batch_images)
                loss = self.loss_func(outputs, batch_labels)
                loss_epoch += loss.item()

                _, pred = torch.max(outputs, axis=1)
                pred_list.append(pred.cpu().numpy())
                target_list.append(batch_labels.cpu().numpy())
                
                # Fazendo a otimização
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()    
                cnt += 1
                
                
            loss_epoch = loss_epoch / cnt
            loss_val_epoch,eval_acc = self.eval_loss(val_dataloader)
            
            
            self.writer.add_scalar("LossTrain", loss_epoch, epoch)
            self.writer.add_scalar("LossVal", loss_val_epoch, epoch)
            
            temp = {
                "Train": loss_epoch,
                "Val": loss_val_epoch
            }
            self.writer.add_scalars("Loss", temp, epoch)
            
                
            # Salvando o checkpoint da última época
            checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.loss_func,
                    'loss_val': loss_epoch
            }
            checkpoint_name = "last_checkpoint"+self.fold+".pth"
            torch.save(checkpoint, checkpoint_name)
        
            # Salvando a mellhor execução    
            if loss_val_epoch < best_loss:        
                best_loss = loss
                best_checkpoint_name = "best_checkpoint"+self.fold+".pth"
                torch.save(checkpoint, best_checkpoint_name)
                
            pred_list  = np.concatenate(pred_list).ravel()
            target_list  = np.concatenate(target_list).ravel()
            train_acc = accuracy_score(pred_list, target_list)

            self.writer.add_scalar("AccTrain", train_acc, epoch)
            self.writer.add_scalar("AccVal", eval_acc, epoch)
            
            end = time.time()
            #print (f"- Epoch [{epoch+1}/{num_epochs}] | Loss: {loss_epoch:.4f} | Loss Val: {loss_val_epoch:.4f}")
            print(f'|  {epoch:03.0f}  |   {loss_epoch:.5f}  |    {train_acc*100:02.0f}%    |     {loss_val_epoch:.5f}     |       {eval_acc*100:02.0f}%      | {end-start:.2f} |')

    def test(self,test_dataloader):
        with torch.no_grad():
            correct, total = 0, 0
            for images, labels, _, img_id in test_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
            print(f"Accuracy: {100 * correct / total}%")
import matplotlib.pyplot as plt
import torch


class ModelTrainer:
    def __init__(self, model, criterion, optimizer, device, train_loader, valid_loader, epoch):
        '''
        :param model: torch.nn.Module类型，需要训练的神经网络模型
        :param criterion: torch.nn.modules.loss._Loss类型，损失函数，用于计算模型输出与实际值之间的误差
        :param optimizer: torch.optim.Optimizer类型，优化器，用于根据损失函数的梯度更新模型的权重
        :param device: str类型，设备标识符，指定模型在CPU或GPU上运行，例如'cpu'或'cuda:0'
        :param train_loader: torch.utils.data.DataLoader类型，训练数据加载器，用于迭代训练数据集
        :param valid_loader: torch.utils.data.DataLoader类型，评估数据加载器，用于迭代测试数据集
        :param epoch: int 轮数
        '''
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.epoch = epoch
        self.num_print = 1104

    def train_model(self):
        self.model.to(self.device)
        self.model.train()

        train_loss_list = []
        valid_loss_list = []
        train_acc_list = []
        valid_acc_list = []

        print("Training started...\n")
        for epoch in range(self.epoch):
            print(f'Epoch [{epoch + 1}/{self.epoch}]')
            running_loss = 0.0
            correct = 0
            total = 0

            for i, (inputs, labels) in enumerate(self.train_loader, 0):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                predicted = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if (i + 1) % self.num_print == 0:  # 假设self.num_print是打印频率
                    print(
                        f'[{epoch + 1} epoch,{i + 1}/{len(self.train_loader)}]  Loss: {running_loss / self.num_print:.6f}')
                    print(f'Accuracy: {100 * correct / total:.2f}%')
                    print(f'Learning rate: {self.optimizer.param_groups[0]["lr"]:.15f}')
                    running_loss = 0.0

            train_accuracy = 100 * correct / total
            train_loss_list.append(running_loss / len(self.train_loader))
            train_acc_list.append(train_accuracy)
            print(f'Training Accuracy of the model after {epoch + 1} epochs: {train_accuracy:.2f}%')

            # Validation phase
            self.model.eval()
            valid_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in self.valid_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    valid_loss += loss.item()

                    predicted = outputs.argmax(dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            valid_loss /= len(self.valid_loader)
            valid_accuracy = 100 * correct / total
            valid_loss_list.append(valid_loss)
            valid_acc_list.append(valid_accuracy)
            print(f'Validation Loss after {epoch + 1} epochs: {valid_loss:.6f}')
            print(f'Validation Accuracy after {epoch + 1} epochs: {valid_accuracy:.2f}%\n')

            self.model.train()

        print("Training finished!")

        # Plotting
        epochs = range(1, self.epoch + 1)

        # Loss plot
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss_list, 'b-', label='Training Loss')
        plt.plot(epochs, valid_loss_list, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_acc_list, 'b-', label='Training Accuracy')
        plt.plot(epochs, valid_acc_list, 'r-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_validation_plots.png')
        plt.show()

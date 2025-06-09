import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


#%% General CNN architecture base class


class CNN_base(nn.Module):
    """
    Base class for a Convolutional Neural Network (CNN).
    
    This class provides basic structure and utility methods for training,
    validation, evaluation, and feature extraction. It does not define
    layers or the forward method, which should be implemented in subclasses.
    """

    def __init__(self, lr=0.001):
        """
        Initialize the CNN base class.

        Args:
            lr (float): Learning rate for the optimizer.
        """
        super().__init__()

        self.lr = lr
        # Adam optimizer; parameters are expected to be defined in subclasses
        self.optim = optim.Adam(self.parameters(), self.lr)
        # Binary Cross Entropy with Logits Loss: combines sigmoid layer and BCELoss for stability
        self.criterion = nn.BCEWithLogitsLoss()

        # Lists to store metrics during training
        self.loss_during_training = []
        self.valid_loss_during_training = []
        self.accuracy_during_training = []
        self.valid_accuracy_during_training = []

        print('## - CNN base model created')
        print(f'## - Number of parameters: {sum(p.numel() for p in self.parameters())}')

        # Device setup: prefer CUDA, fallback to Apple MPS or CPU
        import platform
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and platform.system() != 'Darwin' else
            "mps" if torch.backends.mps.is_available() and platform.system() == 'Darwin' else
            "cpu"
        )
        self.to(self.device)
        print(f'Network allocated on device: {self.device}')

    def trainloop(self, trainloader, validloader, epochs, info_frequency=1):
        """
        Train the CNN for a specified number of epochs.

        Args:
            trainloader (DataLoader): DataLoader for training data.
            validloader (DataLoader or None): DataLoader for validation data (optional).
            epochs (int): Number of training epochs.
            info_frequency (int): Frequency (in epochs) to print training info and plot losses.
                - If 0, no info is printed during training.
                - If -1, info is printed only at the last epoch.
        """
        # Adjust info_frequency to handle special cases
        info_frequency = 999999 if info_frequency == 0 else (epochs - 1 if info_frequency == -1 else info_frequency)

        for epoch in range(int(epochs)):
            start_time = time.time()
            running_loss = 0.0
            correct_predictions_train = 0
            total_predictions_train = 0

            self.train()
            for images, labels in trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optim.zero_grad()
                outputs = self.forward(images)
                loss = self.criterion(outputs.squeeze(), labels.float())
                running_loss += loss.item()
                loss.backward()
                self.optim.step()

                # Apply sigmoid to outputs to get probabilities
                probabilities = torch.sigmoid(outputs)
                # Threshold probabilities to get binary predictions
                predictions = (probabilities > 0.5).squeeze().float()
                correct_predictions_train += (predictions == labels).sum().item()
                total_predictions_train += labels.size(0)

            # Store average training loss and accuracy for this epoch
            self.loss_during_training.append(running_loss / len(trainloader))
            self.accuracy_during_training.append(correct_predictions_train / total_predictions_train)

            # Evaluate on validation set if provided
            if validloader is not None:
                validation_loss, validation_accuracy = self.evaluate(validloader)
                self.valid_loss_during_training.append(validation_loss)
                self.valid_accuracy_during_training.append(validation_accuracy)

            # Print and plot info based on frequency
            if epoch % info_frequency == 0:
                elapsed_minutes = (time.time() - start_time) / 60
                print(
                    f"EPOCH {epoch + 1} - "
                    f"Training Loss: {self.loss_during_training[-1]:.6f} "
                    f"{'Validation Loss: %.6f' % validation_loss if validloader else ''} / "
                    f"Training Accuracy: {self.accuracy_during_training[-1]:.6f} "
                    f"{'Validation Accuracy: %.6f' % validation_accuracy if validloader else ''} / "
                    f"Time: {int(elapsed_minutes)}m {int((elapsed_minutes - int(elapsed_minutes)) * 60)}s"
                )
                self.plot_training_loss()

    def find_best_epochs(self, trainloader, validloader, max_epochs=100, patience=5,
                         min_epochs=1, criterium='accuracy', info_frequency=1):
        """
        Finds the best number of training epochs using early stopping.

        Args:
            trainloader (DataLoader): Training data loader.
            validloader (DataLoader): Validation data loader.
            max_epochs (int): Maximum number of epochs to train.
            patience (int): Number of epochs to wait without improvement before stopping.
            min_epochs (int): Minimum number of epochs to train before early stopping is considered.
            criterium (str): Criterion to monitor ('loss' or 'accuracy').
            info_frequency (int): Frequency to print info.

        Returns:
            int: Best epoch number based on validation performance.
        """
        info_frequency = 9999999 if info_frequency in (-1, 0) else info_frequency

        best_score = float('inf')
        best_epoch = 0
        epochs_no_improve = 0

        print('#### FINDING BEST NUMBER OF EPOCHS FOR TRAINING ####')
        print('####################################################')

        for epoch in range(1, max_epochs + 1):
            show_plot = epoch % info_frequency == 0
            self.trainloop(trainloader, validloader, epochs=1, info_frequency=show_plot)

            if criterium == 'loss':
                current_score = self.valid_loss_during_training[-1]
            elif criterium in ('accuracy', 'acc'):
                # For accuracy, lower is worse, so negate to get a score to minimize
                current_score = -self.valid_accuracy_during_training[-1]
            else:
                raise ValueError("criterium must be 'loss' or 'accuracy'")

            if current_score < best_score and epoch >= min_epochs:
                best_score = current_score
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience and best_epoch >= min_epochs:
                print(f'Early stopping at epoch {epoch}')
                print('####################################################')
                self.plot_training_loss()
                break

        print(f'Best epoch: {best_epoch} with validation score: {best_score}')
        return best_epoch

    def evaluate(self, dataloader):
        """
        Evaluate model on a dataset.

        Args:
            dataloader (DataLoader): DataLoader to evaluate.

        Returns:
            tuple: (average_loss, accuracy)
        """
        self.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.forward(images)
                loss = self.criterion(outputs.squeeze(), labels.float())
                running_loss += loss.item()

                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).squeeze().float()
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

        self.train()
        average_loss = running_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        return average_loss, accuracy

    def extract_features(self, dataloader):
        """
        Extract features from images using the forward method with feature extraction mode.

        Args:
            dataloader (DataLoader): DataLoader providing images.

        Returns:
            tuple: (features tensor, labels tensor)
        """
        features = None
        labels_all = torch.tensor([], dtype=torch.long, device=self.device)

        self.eval()
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                out = self.forward(images, extract_features=True)
                labels_all = torch.cat([labels_all, labels])
                if features is None:
                    features = out
                else:
                    features = torch.cat([features, out])
        self.train()
        return features, labels_all

    def plot_training_loss(self, show_accuracy=True, xlim=None, show=True):
        """
        Plot training and validation loss and accuracy curves.

        Args:
            show_accuracy (bool): Whether to plot accuracy curves.
            xlim (tuple or None): Limits for the x-axis.
            show (bool): Whether to display the plot or return the figure and axes.
        
        Returns:
            tuple: (fig, ax1, ax2) if show is False
        """
        plt.ioff()
        fig, ax1 = plt.subplots(dpi=300)
        ax2 = ax1.twinx()

        ax1.plot(range(1, len(self.loss_during_training) + 1), self.loss_during_training,
                 color='r', linewidth=2, label='Training Loss')
        ax1.plot(range(1, len(self.valid_loss_during_training) + 1), self.valid_loss_during_training,
                 'r--', label='Validation Loss')
        ax1.set_ylabel('Loss', color='tab:red')
        ax1.tick_params(axis='y', labelcolor='tab:red')

        if show_accuracy:
            ax2.plot(range(1, len(self.accuracy_during_training) + 1), self.accuracy_during_training,
                     'b', linewidth=2, label='Training Accuracy')
            ax2.plot(range(1, len(self.valid_accuracy_during_training) + 1), self.valid_accuracy_during_training,
                     'b--', label='Validation Accuracy')
            ax2.set_ylabel('Accuracy', color='tab:blue')
            ax2.tick_params(axis='y', labelcolor='tab:blue')

        ax1.set_xlabel('Epochs')
        ax1.set_title('Loss and Accuracy during Training')

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='center right')

        if xlim:
            ax1.set_xlim(xlim)
        if show:
            plt.show()
        else:
            return fig, ax1, ax2

    def eval_performance(self, loader):
        """
        Calculate accuracy of the model on the given DataLoader.

        Args:
            loader (DataLoader): DataLoader for evaluation.

        Returns:
            float: Accuracy score.
        """
        self.eval()
        accuracy = 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.forward(images)
                # Get predicted class (top-1)
                top_p, top_class = outputs.topk(1, dim=1)
                equals = (top_class == labels.view(images.shape[0], 1))
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        self.train()
        return (accuracy / len(loader)).item()

    def summary(self, loader=None, im_dimensions=None):
        """
        Print the model summary using torchsummary package.

        Args:
            loader (DataLoader or None): DataLoader to extract image dimensions from.
            im_dimensions (tuple or None): Image dimensions as (channels, height, width).
        """
        if im_dimensions is None:
            if loader is None:
                raise ValueError("Either loader or im_dimensions must be provided")
            batch = next(iter(loader))
            batch_dimensions = batch[0].size()
            im_dimensions = tuple(batch_dimensions[1:])

        from torchsummary import summary
        summary(self, im_dimensions)

    def predict(self, loader):
        """
        Generate predictions and true labels for a dataset.

        Args:
            loader (DataLoader): DataLoader for input data.

        Returns:
            tuple: (list of predicted classes, list of true labels)
        """
        predictions = []
        true_labels = []

        self.eval()
        with torch.no_grad():
            for images, labels_batch in loader:
                images = images.to(self.device)
                logits = self.forward(images)
                probabilities = torch.sigmoid(logits).squeeze()
                predicted_classes = (probabilities > 0.5).long()
                predictions.extend(predicted_classes.cpu().tolist())
                true_labels.extend(labels_batch.tolist())

        self.train()
        return predictions, true_labels




#%% Specific implementation of CNN

class CNN_5layers_AgeSex(CNN_base):
    """
    Convolutional Neural Network with 5 convolutional layers followed by a deep multilayer perceptron (MLP).
    
    Architecture:
        - Conv layers progressively increase channels: 16 -> 64 -> 128 -> 256 -> 512
        - Batch Normalization follows each convolution to stabilize training.
        - ReLU activation applied after each batch norm.
        - MaxPooling layers reduce spatial dimensions at various stages.
        - Flatten output from convolutional layers to feed fully connected layers.
        - MLP with layers of sizes 1024, 256, 64 and final output layer of size 1.
        
    Parameters:
        nlabels (int): Number of output labels (currently only output size 1 supported).
        lr (float): Learning rate for optimizer (default 0.001).
        dropout (float): Dropout probability applied after each fully connected layer (default 0.5).
        in_channels (int): Number of input channels (default 3).
    """

    def __init__(self, nlabels, lr=0.001, dropout=0.5, in_channels=3):
        super().__init__(lr)

        # Convolutional layers with increasing number of filters
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=13, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=1, padding=0)

        # Batch normalization layers for each conv output
        self.batch_norm_16 = nn.BatchNorm2d(num_features=16)
        self.batch_norm_64 = nn.BatchNorm2d(num_features=64)
        self.batch_norm_128 = nn.BatchNorm2d(num_features=128)
        self.batch_norm_256 = nn.BatchNorm2d(num_features=256)
        self.batch_norm_512 = nn.BatchNorm2d(num_features=512)

        # Activation function
        self.relu = nn.ReLU()

        # MaxPooling layers with different kernel sizes and strides to reduce spatial dimensions
        self.maxpool_2_2_0 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.maxpool_3_2_0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.maxpool_4_1_0 = nn.MaxPool2d(kernel_size=4, stride=1, padding=0)

        # Size of the flattened feature vector after convolution and pooling
        self.size_after_convs = 512 * 2 * 2

        # Fully connected layers sizes
        self.n_features = 1024
        self.dropout = nn.Dropout(p=dropout)

        # Fully connected layers (MLP)
        self.linear1 = nn.Linear(self.size_after_convs, self.n_features)
        self.linear2 = nn.Linear(self.n_features, 256)
        self.linear3 = nn.Linear(256, 64)
        self.linear4 = nn.Linear(64, 1)  # Final output layer

    def forward(self, x, extract_features=False):
        """
        Perform forward propagation through the network.

        Args:
            x (Tensor): Input tensor with shape (batch_size, in_channels, height, width).
            extract_features (bool): If True, return features from the first fully connected layer
                                     instead of the final output. Useful for feature extraction.

        Returns:
            Tensor: If extract_features=False, returns a tensor of shape (batch_size, 1) with
                    regression or classification output.
                    If extract_features=True, returns tensor of shape (batch_size, 1024) with
                    intermediate features.
        """

        # Convolutional block 1: conv -> batch norm -> ReLU -> maxpool
        x = self.conv1(x)
        x = self.batch_norm_16(x)
        x = self.relu(x)
        x = self.maxpool_3_2_0(x)

        # Convolutional block 2
        x = self.conv2(x)
        x = self.batch_norm_64(x)
        x = self.relu(x)
        x = self.maxpool_3_2_0(x)

        # Convolutional block 3
        x = self.conv3(x)
        x = self.batch_norm_128(x)
        x = self.relu(x)
        x = self.maxpool_3_2_0(x)

        # Convolutional block 4
        x = self.conv4(x)
        x = self.batch_norm_256(x)
        x = self.relu(x)
        x = self.maxpool_2_2_0(x)

        # Convolutional block 5
        x = self.conv5(x)
        x = self.batch_norm_512(x)
        x = self.relu(x)
        x = self.maxpool_4_1_0(x)

        # Flatten the output of conv layers to feed the fully connected layers
        x = x.view(x.shape[0], self.size_after_convs)

        # Fully connected layers with ReLU and dropout
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)

        if not extract_features:
            x = self.linear2(x)
            x = self.relu(x)
            x = self.dropout(x)

            x = self.linear3(x)
            x = self.relu(x)
            x = self.dropout(x)

            x = self.linear4(x)  # Output layer

        return x


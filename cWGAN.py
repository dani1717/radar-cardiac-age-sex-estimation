import os
import time
import random
import datetime
import platform

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid

from utils.GANmetrics import GAN_metrics


#%% cWGAN

class Generator(nn.Module):
    """
    Conditional Generator for cWGAN.

    This generator takes as input a random noise vector (z) and a class label, and outputs
    a synthetic image conditioned on the label. It uses a stack of transposed convolutional
    layers (a.k.a. deconvolutions) to upsample the latent space into image space.

    Attributes:
        nlabels (int): Number of distinct class labels.
        z_dim (int): Dimension of the latent noise vector.
        im_shape (tuple): Shape of the output image (channels, height, width).
        hidden_dim (int): Base number of channels for the hidden layers.
        label_emb (nn.Embedding): Embedding layer for label conditioning.
        gen (nn.Sequential): Generator network built with transposed convolutions.
        lr (float): Learning rate for the optimizer.
        optim (torch.optim.Optimizer): Adam optimizer used for training.
        device (torch.device): Device on which the model is allocated.
    """

    def __init__(self, nlabels, z_dim=10, im_shape=(1, 28, 28), hidden_dim=64, lr=0.001, device=None):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(nlabels, nlabels)
        self.im_shape = im_shape
        self.im_chan = im_shape[0]
        self.z_dim = z_dim
        self.nlabels = nlabels

        self.gen = nn.Sequential(
            self.get_generator_block(z_dim + nlabels, hidden_dim * 4, kernel_size=3, stride=2),
            self.get_generator_block(hidden_dim * 4, hidden_dim * 2, kernel_size=3, stride=2),
            self.get_generator_block(hidden_dim * 2, hidden_dim, kernel_size=5, stride=3),
            self.get_generator_block(hidden_dim, hidden_dim // 2, kernel_size=3, stride=3),
            self.get_generator_block(hidden_dim // 2, hidden_dim // 4, kernel_size=3, stride=2),
            self.get_generator_final_block(hidden_dim // 4, self.im_chan, kernel_size=3, stride=2)
        )

        self.lr = lr
        self.optim = optim.Adam(self.parameters(), lr=lr, betas=(0.0, 0.9))

        self.apply(weights_init)

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() and platform.system() != 'Darwin' else
                "mps" if torch.backends.mps.is_available() and platform.system() == 'Darwin' else
                "cpu"
            )
        self.device = device
        self.to(device)
        print('Generator allocated in', self.device)

    def get_generator_block(self, input_channel, output_channel, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

    def get_generator_final_block(self, input_channel, output_channel, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        noise = noise.view(len(noise), self.z_dim)
        gen_input = torch.cat((self.label_emb(labels), noise), 1)
        gen_input = gen_input.unsqueeze(-1).unsqueeze(-1)
        imgs = self.gen(gen_input)
        imgs = F.interpolate(imgs, size=self.im_shape[1:], mode='bilinear', align_corners=False)
        imgs = imgs.view(imgs.size(0), *self.im_shape)
        return imgs


class Discriminator(nn.Module):
    """
    Conditional Critic for cWGAN.

    This discriminator (also known as a critic in WGANs) evaluates whether an input image 
    is real or generated, conditioned on its associated class label. It combines features 
    extracted from the image with an embedding of the label, and passes them through a 
    final classifier network.

    Attributes:
        nlabels (int): Number of class labels.
        label_emb (nn.Embedding): Embedding layer for label conditioning.
        im_shape (tuple): Shape of the input images (channels, height, width).
        im_chan (int): Number of image channels (e.g., 1 for grayscale).
        feature_extractor (nn.Sequential): Convolutional layers to extract image features.
        n_features (int): Number of features after flattening the output of feature_extractor.
        final_block (nn.Sequential): Fully connected layers for binary classification.
        lr (float): Learning rate for the RMSprop optimizer.
        optim (torch.optim.Optimizer): Optimizer used to update model parameters.
        device (torch.device): Device on which the model runs.
    """

    def __init__(self, nlabels, im_shape=(1, 28, 28), hidden_dim=16, lr=0.001, device=None):
        super(Discriminator, self).__init__()

        self.nlabels = nlabels
        self.label_emb = nn.Embedding(nlabels, nlabels)
        self.im_shape = im_shape
        self.im_chan = im_shape[0]

        self.feature_extractor = nn.Sequential(
            self.get_critic_block(self.im_chan, hidden_dim * 4, kernel_size=4, stride=2),
            self.get_critic_block(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2),
            nn.Flatten()
        )

        self.n_features = get_n_features(self.feature_extractor, im_shape)

        self.final_block = nn.Sequential(
            nn.Linear(self.n_features + nlabels, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, 1)
        )

        self.lr = lr
        self.optim = optim.RMSprop(self.parameters(), lr=lr)

        self.apply(weights_init)

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() and platform.system() != 'Darwin' else
                "mps" if torch.backends.mps.is_available() and platform.system() == 'Darwin' else
                "cpu"
            )
        self.device = device
        self.to(device)
        print('Discriminator allocated in', self.device)

    def get_critic_block(self, input_channel, output_channel, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, image, label):
        features = self.feature_extractor(image)
        features_label = torch.cat((features, self.label_emb(label)), 1)
        return self.final_block(features_label)
    


class cWGAN_GP:
    """
    Conditional Wasserstein GAN with Gradient Penalty for classification tasks.
    """

    def __init__(self, nlabels, lr_G=0.0002, lr_D=0.0002, z_dim=10, im_shape=(1, 28, 28),
                 gen_hidden_dim=64, disc_hidden_dim=16, device=None, dataloader=None):
        """
        Initialize the cWGAN model with basic parameters and settings.
        
        Args:
            nlabels (int): Number of classes/labels.
            lr_G (float): Learning rate for the generator.
            lr_D (float): Learning rate for the discriminator.
            z_dim (int): Dimension of the latent noise vector.
            im_shape (tuple): Shape of input/output images (channels, height, width).
            gen_hidden_dim (int): Hidden layer size for generator.
            disc_hidden_dim (int): Hidden layer size for discriminator.
            device (torch.device or None): Device to run the model on (cpu, cuda, mps).
            dataloader (torch.utils.data.DataLoader or None): Data loader to infer image shape.
        """
        if dataloader is not None:
            im_shape = tuple(next(iter(dataloader))[0].shape[1:])

        self.nlabels = nlabels
        self.lr_G = lr_G
        self.lr_D = lr_D
        self.z_dim = z_dim
        self.im_shape = im_shape
        self.im_chan = im_shape[0]
        self.gen_hidden_dim = gen_hidden_dim
        self.disc_hidden_dim = disc_hidden_dim
        self.metrics = pd.DataFrame(columns=['Epoch', 'Inception', 'FID', 'Disc-Precision',
                                             'Disc-Recall', 'MSE', 'Gen-Loss', 'Disc-Loss', 'Date'])

        # Device selection logic: CUDA > MPS (Mac) > CPU
        if device is None:
            if torch.cuda.is_available() and platform.system() != 'Darwin':
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available() and platform.system() == 'Darwin':
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.gen_losses = []
        self.disc_losses = []
        self.gen_losses_by_batch = []
        self.disc_losses_by_batch = []

    def _continue_training(self, n_epochs, trainloader):
        """
        Private method to continue training for a given number of epochs.

        Args:
            n_epochs (int): Number of epochs to train.
            trainloader (DataLoader): DataLoader for training data.
        """
        gen = self.gen
        disc = self.disc
        device = self.device
        batch_size = self.batch_size
        disc_repeats = self.disc_repeats
        c_lambda = self.c_lambda
        seed = self.seed

        total_steps = 0
        for epoch in range(n_epochs):
            start_time = time.time()
            cur_step = 0
            epoch_critic_loss = 0

            for real_imgs, real_labels in trainloader:
                cur_batch_size = len(real_imgs)
                real_imgs = real_imgs.to(device)
                real_labels = real_labels.to(device)

                # Train discriminator multiple times per batch
                for _ in range(disc_repeats):
                    disc.optim.zero_grad()
                    noise = get_noise(cur_batch_size, self.z_dim, device=device)
                    fake_labels = torch.randint(0, self.nlabels, (cur_batch_size,), device=device)
                    fake_imgs = gen(noise, fake_labels)

                    pred_fake = disc(fake_imgs.detach(), fake_labels)
                    pred_real = disc(real_imgs, real_labels)

                    epsilon = torch.rand(cur_batch_size, 1, 1, 1, device=device, requires_grad=True)
                    gradient = get_gradient_Conditional(disc, real_imgs, fake_imgs.detach(), real_labels, fake_labels, epsilon)
                    gp = gradient_penalty(gradient)

                    critic_loss = get_crit_loss(pred_fake, pred_real, gp, c_lambda)
                    epoch_critic_loss += critic_loss.item() / disc_repeats
                    critic_loss.backward(retain_graph=True)
                    disc.optim.step()

                self.disc_losses_by_batch.append(epoch_critic_loss)

                # Train generator
                gen.optim.zero_grad()
                noise = get_noise(cur_batch_size, self.z_dim, device=device)
                fake_labels = torch.randint(0, self.nlabels, (cur_batch_size,), device=device)
                fake_imgs = gen(noise, fake_labels)
                pred_fake = disc(fake_imgs, fake_labels)

                gen_loss = get_gen_loss(pred_fake)
                gen_loss.backward()
                gen.optim.step()
                self.ema.update()

                self.gen_losses_by_batch.append(gen_loss.item())

                cur_step += 1
                total_steps += 1
                torch.cuda.empty_cache()

            # Average losses per epoch
            gen_mean = sum(self.gen_losses_by_batch[-cur_step:]) / cur_step
            disc_mean = sum(self.disc_losses_by_batch[-cur_step:]) / cur_step

            self.gen_losses.append(gen_mean)
            self.disc_losses.append(disc_mean)
            self.n_epochs += 1

            elapsed = time.time() - start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            print(f"Epoch: {epoch + 1}/{n_epochs} - Time elapsed: {minutes}m {seconds}s")

            if (epoch + 1) % 5 == 0:
                gan_metrics = GAN_metrics(GAN=self, real_data_loader=trainloader,
                                         num_samples=batch_size * 6, batch_size=batch_size, device=device, threshold=0.0)
                metrics_row = {'Epoch': epoch + self.n_epochs - n_epochs, 'Gen-Loss': gen_mean,
                               'Disc-Loss': disc_mean, 'Date': datetime.datetime.now()}
                metrics_row.update(gan_metrics)
                self.metrics = pd.concat([self.metrics, pd.DataFrame([metrics_row])], ignore_index=True)

                self.show_samples_per_class(num_samples_per_class=10, seed=seed)
                self.plot_loss()

        self.plot_loss()
        self.plot_loss(g_loss=self.gen_losses_by_batch, d_loss=self.disc_losses_by_batch, xlabel='Iteration')
        self.plot_metrics()

    def trainloop(self, n_epochs, batch_size, disc_repeats, trainloader, c_lambda=10):
        """
        Full training loop from scratch.

        Args:
            n_epochs (int): Number of epochs to train.
            batch_size (int): Batch size for training.
            disc_repeats (int): Number of discriminator updates per generator update.
            trainloader (DataLoader): Training data loader.
            c_lambda (float): Gradient penalty coefficient.
        """
        self.n_epochs = 0
        self.batch_size = batch_size
        self.disc_repeats = disc_repeats
        self.c_lambda = c_lambda

        self.gen = Generator(self.nlabels, z_dim=self.z_dim, im_shape=self.im_shape,
                             hidden_dim=self.gen_hidden_dim, lr=self.lr_G, device=self.device)
        self.disc = Discriminator(self.nlabels, im_shape=self.im_shape,
                                  hidden_dim=self.disc_hidden_dim, lr=self.lr_D, device=self.device)

        self.gen_losses = []
        self.disc_losses = []
        self.gen_losses_by_batch = []
        self.disc_losses_by_batch = []

        self.ema = EMA(self.gen, decay=0.999)

        self.seed = random.randint(0, 99999)
        self._continue_training(n_epochs, trainloader)

    def continue_trainloop(self, additional_epochs, trainloader):
        """
        Continue training preserving previous states.

        Args:
            additional_epochs (int): Number of additional epochs.
            trainloader (DataLoader): Training data loader.
        """
        self._continue_training(additional_epochs, trainloader)

    def show_new_gen_images(self, num_imgs):
        """
        Show newly generated images.

        Args:
            num_imgs (int): Number of images to generate and show.
        """
        noise = get_noise(num_imgs, self.z_dim, device=self.device)
        with torch.no_grad():
            fake_imgs = self.gen(noise)
        show_new_gen_images_AUX(fake_imgs.reshape(num_imgs, *self.im_shape))

    def generate_images_per_class(self, gen, z_dim, num_samples_per_class, n_classes, seed=None, device='cpu'):
        """
        Generate images for each class label.

        Args:
            gen (torch.nn.Module): Generator model.
            z_dim (int): Dimension of latent noise vector.
            num_samples_per_class (int): Number of images to generate per class.
            n_classes (int): Number of classes.
            seed (int or None): Random seed for reproducibility.
            device (str or torch.device): Device for tensor computations.

        Returns:
            List of tensors containing generated images per class.
        """
        all_images = []
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

        for label in range(n_classes):
            noise = get_noise(num_samples_per_class, z_dim, device=device)
            labels = torch.full((num_samples_per_class,), label, dtype=torch.long, device=device)
            with torch.no_grad():
                fake_images = gen(noise, labels)
            fake_images = (fake_images + 1) / 2  # Normalize to [0, 1]
            all_images.append(fake_images.cpu())
        return all_images

    def generate_new_images(self, n_images=None, labels=None, folder_path=None, batch_size=64):
        """
        Generate new images, optionally saving them to disk.

        Args:
            n_images (int or None): Number of images to generate.
            labels (list or int or None): Labels for generated images.
            folder_path (str or None): Directory path to save images.
            batch_size (int): Batch size for generation.

        Returns:
            Tuple: (Tensor with generated images, list of labels)
        """
        n_images = len(labels) if n_images is None and labels is not None else n_images

        if isinstance(labels, int):
            labels = [labels] * n_images
        elif labels is None:
            labels = [random.randint(0, self.nlabels - 1) for _ in range(n_images)]

        all_fake_images = []
        all_labels = []

        for start in range(0, n_images, batch_size):
            end = min(start + batch_size, n_images)
            current_batch_size = end - start

            noise = torch.randn(current_batch_size, self.z_dim, device=self.device)
            labels_tensor = torch.tensor(labels[start:end], dtype=torch.long, device=self.device)

            with torch.no_grad():
                fake_images = self.gen(noise, labels_tensor)
            fake_images = (fake_images + 1) / 2  # Normalize to [0, 1]

            if folder_path is not None:
                os.makedirs(folder_path, exist_ok=True)
                for i in range(current_batch_size):
                    class_label = labels[start + i]
                    filename = f'newScalogram_{start + i:05d}_class_{class_label}.png'
                    save_image(fake_images[i], os.path.join(folder_path, filename))

            all_fake_images.append(fake_images.cpu())
            all_labels.extend(labels[start:end])

        all_fake_images = torch.cat(all_fake_images, dim=0)
        return all_fake_images, all_labels
    
    
    def show_samples_per_class(self, num_samples_per_class, seed=None):
        """
        Display a grid of generated images for each class.

        Args:
            num_samples_per_class (int): Number of images to generate per class.
            seed (int or None): Random seed for reproducibility.
        """
        # Generate images for each class using the generator model
        all_images = self.generate_images_per_class(
            self.gen,
            self.z_dim,
            num_samples_per_class,
            self.nlabels,
            seed,
            self.device
        )

        # Create a figure with a grid of subplots
        fig, axes = plt.subplots(self.nlabels, num_samples_per_class,
                                figsize=(num_samples_per_class, self.nlabels))

        for i, images in enumerate(all_images):
            for j, img in enumerate(images):
                ax = axes[i, j]

                # Reorder image dimensions for visualization: (C, H, W) -> (H, W, C)
                img_permuted = img.permute(1, 2, 0)

                if self.im_chan == 1:
                    # For grayscale images, remove channel dimension and use grayscale colormap
                    img_permuted = img_permuted.squeeze(-1)
                    ax.imshow(img_permuted.cpu(), cmap='gray')
                else:
                    # For color images, show as is
                    ax.imshow(img_permuted.cpu())

                ax.axis('off')  # Hide axis ticks and labels

        plt.tight_layout()
        plt.show()
        
        
    def plot_loss(self, g_loss=None, d_loss=None, xlabel=None):
        """
        Plot generator and discriminator losses over training epochs.

        Args:
            g_loss (list or None): Generator loss values. Uses self.gen_losses if None.
            d_loss (list or None): Discriminator loss values. Uses self.disc_losses if None.
            xlabel (str or None): Label for the x-axis. Defaults to 'epoch'.
        """
    
        g_loss = self.gen_losses if g_loss is None else g_loss
        d_loss = self.disc_losses if d_loss is None else d_loss
        xlabel = 'epoch' if xlabel is None else xlabel

        plt.figure(figsize=(10, 5), dpi=300)
        title = f'lr: {self.gen.lr}, {self.disc.lr} / disc_repeats: {self.disc_repeats}'
        plt.title(title)
        plt.plot(g_loss, label="Generator Loss")
        plt.plot(d_loss, label="Discriminator Loss")
        plt.xlabel(xlabel)
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


    def plot_metrics(self, metrics=None):
        """
        Plot multiple training metrics with multiple y-axes.

        Args:
            metrics (dict or None): Dictionary containing metric lists indexed by metric names.
                                    Uses self.metrics if None.
        """
        metrics = self.metrics if metrics is None else metrics
        epochs = metrics['Epoch']

        fig, ax1 = plt.subplots(figsize=(15, 10), dpi=300)

        # Colors for different metrics
        colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:brown', 'tab:purple']

        # Plot Inception Score
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Inception Score', color=colors[0])
        ax1.plot(epochs, metrics['Inception'], color=colors[0], label='Inception Score')
        ax1.tick_params(axis='y', labelcolor=colors[0])

        # Second y-axis for FID
        ax2 = ax1.twinx()
        ax2.set_ylabel('FID', color=colors[1])
        ax2.plot(epochs, metrics['FID'], color=colors[1], linestyle='-', label='FID')
        ax2.tick_params(axis='y', labelcolor=colors[1])

        # Third y-axis for Discriminator Precision and Recall
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.set_ylabel('Disc-Precision/Recall', color=colors[2])
        ax3.plot(epochs, metrics['Disc-Precision'], color=colors[2], linestyle='dashed', label='Disc-Precision')
        ax3.plot(epochs, metrics['Disc-Recall'], color=colors[3], linestyle='dotted', label='Disc-Recall')
        ax3.tick_params(axis='y', labelcolor=colors[2])
        ax3.set_ylim(0, 1)

        # Fourth y-axis for MSE
        ax4 = ax1.twinx()
        ax4.spines['right'].set_position(('outward', 120))
        ax4.set_ylabel('MSE', color=colors[4])
        ax4.plot(epochs, metrics['MSE'], color=colors[4], linestyle='-', label='MSE')
        ax4.tick_params(axis='y', labelcolor=colors[4])

        # Combine legends from all axes
        fig.tight_layout()
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines4, labels4 = ax4.get_legend_handles_labels()
        ax1.legend(lines + lines2 + lines3 + lines4, labels + labels2 + labels3 + labels4, loc='upper left')

        plt.title('CWGAN Training Metrics Over Epochs')
        plt.show()






#%% LOSSES

def get_gen_loss(critic_fake_pred):
    """
    Calculate generator loss for WGAN.

    Args:
        critic_fake_pred (Tensor): Discriminator's prediction for generated (fake) images.

    Returns:
        Tensor: Generator loss (to be minimized).
    
    Explanation:
        The generator aims to maximize the discriminator's output for fake images,
        so the loss is the negative mean of the critic's prediction on fake images.
    """
    gen_loss = -torch.mean(critic_fake_pred)
    return gen_loss


def get_critic_loss(critic_fake_pred, critic_real_pred, gradient_penalty, lambda_gp):
    """
    Calculate critic (discriminator) loss for WGAN with gradient penalty.

    Args:
        critic_fake_pred (Tensor): Discriminator's prediction for fake images.
        critic_real_pred (Tensor): Discriminator's prediction for real images.
        gradient_penalty (Tensor): Gradient penalty term.
        lambda_gp (float): Gradient penalty coefficient.

    Returns:
        Tensor: Critic loss (to be minimized).
    
    Explanation:
        Critic tries to maximize difference between real and fake predictions,
        plus the gradient penalty term weighted by lambda.
    """
    crit_loss = torch.mean(critic_fake_pred) - torch.mean(critic_real_pred) + lambda_gp * gradient_penalty
    return crit_loss


#%% AUXILIARY FUNCTIONS


def get_noise(n_samples, z_dim, seed=None, device='cpu'):
    """
    Generate random noise vectors for the generator input.

    Args:
        n_samples (int): Number of noise vectors to generate.
        z_dim (int): Dimensionality of the noise vector.
        seed (int, optional): Random seed for reproducibility.
        device (str or torch.device): Device to create the tensor on.

    Returns:
        Tensor: Random noise tensor of shape (n_samples, z_dim).
    """
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn(n_samples, z_dim, device=device)


def weights_init(m):
    """
    Custom weights initialization as per DCGAN paper.

    Initializes all convolutional, transpose convolutional, and batch normalization
    layers with a normal distribution:
      - mean = 0, std = 0.02 for weights
      - constant 0 for batch norm biases

    Args:
        m (nn.Module): PyTorch module to initialize.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def gradient_penalty(gradient):
    """
    Compute gradient penalty used to enforce Lipschitz constraint in WGAN-GP.

    Args:
        gradient (Tensor): Gradient of the critic output w.r.t. interpolated inputs.

    Returns:
        Tensor: Scalar gradient penalty.
    """
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty


def get_gradient_conditional(critic, real_imgs, fake_imgs, real_labels, fake_labels, epsilon):
    """
    Compute gradients of the critic output with respect to interpolated images and labels
    for conditional WGAN gradient penalty.

    Args:
        critic (nn.Module): Critic (discriminator) model.
        real_imgs (Tensor): Real images batch.
        fake_imgs (Tensor): Fake images batch generated by the generator.
        real_labels (Tensor): Labels for real images.
        fake_labels (Tensor): Labels for fake images.
        epsilon (Tensor): Random interpolation factor between 0 and 1.

    Returns:
        Tensor: Gradient of critic output w.r.t. interpolated images.
    """
    # Interpolate images
    mixed_images = real_imgs * epsilon + fake_imgs * (1 - epsilon)

    # Interpolate labels (weighted average)
    mixed_labels = real_labels * epsilon.squeeze() + fake_labels * (1 - epsilon.squeeze())
    mixed_labels = mixed_labels.long()

    # Forward pass through critic
    mixed_scores = critic(mixed_images, mixed_labels)

    # Compute gradients of scores w.r.t. interpolated images
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    return gradient



def get_n_features(model, im_shape):
    """
    Compute the number of features output by a model given an input shape.

    Args:
        model (torch.nn.Module): Model to analyze.
        im_shape (tuple): Shape of the input tensor (channels, height, width).

    Returns:
        int: Number of features in the flattened output.
    """
    test_input = torch.randn(1, *im_shape)
    with torch.no_grad():
        test_output = model(test_input)
    return test_output.view(1, -1).shape[1]


def show_new_gen_images_AUX(tensor_img, num_img=25):
    """
    Display generated images from a tensor after normalizing from [-1,1] to [0,1].

    Args:
        tensor_img (torch.Tensor): Tensor containing generated images.
        num_img (int): Number of images to display.
    """
    # Normalize images from [-1, 1] to [0, 1]
    normalized_img = (tensor_img + 1) / 2
    unflat_img = normalized_img.detach().cpu()
    img_grid = make_grid(unflat_img[:num_img], nrow=5)
    plt.imshow(img_grid.permute(1, 2, 0).squeeze(), cmap='gray')
    plt.axis('off')
    plt.show()


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), show_fig=False, epoch=0):
    """
    Visualize images stored in a tensor as a grid.

    Args:
        image_tensor (torch.Tensor): Tensor with image data.
        num_images (int): Number of images to display.
        size (tuple): Size to reshape each image (channels, height, width).
        show_fig (bool): Whether to save the figure as a PNG file.
        epoch (int): Epoch number for the filename if saving.
    """
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.axis('off')
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if show_fig:
        plt.savefig(f'image_at_epoch_{epoch:04d}.png')
    plt.show()


class EMA:
    """
    Exponential Moving Average (EMA) of model parameters.

    Maintains a moving average of model parameters to stabilize training.

    Args:
        model (torch.nn.Module): Model to apply EMA to.
        decay (float): Decay rate for EMA, typically close to 1 (e.g., 0.999).
    """
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        # Create a copy of model parameters as shadow weights
        self.shadow = {name: param.clone() for name, param in model.named_parameters()}

    def update(self):
        """Update the shadow weights with current model parameters."""
        for name, param in self.model.named_parameters():
            self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param

    def apply_shadow(self):
        """Copy shadow weights back to the model parameters."""
        for name, param in self.model.named_parameters():
            param.data.copy_(self.shadow[name])
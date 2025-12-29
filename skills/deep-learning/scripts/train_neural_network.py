#!/usr/bin/env python3
"""
Deep Learning Training Script
Supports PyTorch and TensorFlow/Keras frameworks
"""

import os
import yaml
import argparse
from pathlib import Path
from datetime import datetime


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_pytorch_model(config: dict):
    """Create PyTorch neural network from config."""
    import torch
    import torch.nn as nn

    class DynamicNet(nn.Module):
        def __init__(self, layer_configs):
            super().__init__()
            self.layers = nn.ModuleList()

            for layer_cfg in layer_configs:
                layer_type = layer_cfg['type']

                if layer_type == 'Conv2D':
                    self.layers.append(nn.Conv2d(
                        in_channels=layer_cfg.get('in_channels', 3),
                        out_channels=layer_cfg['filters'],
                        kernel_size=tuple(layer_cfg['kernel_size']),
                        padding=layer_cfg.get('padding', 'same')
                    ))
                    if layer_cfg.get('activation') == 'relu':
                        self.layers.append(nn.ReLU())

                elif layer_type == 'BatchNormalization':
                    # Batch norm channels inferred from previous conv
                    self.layers.append(nn.BatchNorm2d(layer_cfg.get('num_features', 64)))

                elif layer_type == 'MaxPooling2D':
                    self.layers.append(nn.MaxPool2d(
                        kernel_size=tuple(layer_cfg['pool_size'])
                    ))

                elif layer_type == 'Dropout':
                    self.layers.append(nn.Dropout(layer_cfg['rate']))

                elif layer_type == 'Flatten':
                    self.layers.append(nn.Flatten())

                elif layer_type == 'Dense':
                    self.layers.append(nn.Linear(
                        in_features=layer_cfg.get('in_features', 512),
                        out_features=layer_cfg['units']
                    ))
                    if layer_cfg.get('activation') == 'relu':
                        self.layers.append(nn.ReLU())
                    elif layer_cfg.get('activation') == 'softmax':
                        self.layers.append(nn.Softmax(dim=1))

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    return DynamicNet(config['model']['layers'])


def create_keras_model(config: dict):
    """Create Keras/TensorFlow model from config."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential()
    input_shape = tuple(config['model']['input']['shape'])

    first_layer = True
    for layer_cfg in config['model']['layers']:
        layer_type = layer_cfg['type']

        if layer_type == 'Conv2D':
            kwargs = {
                'filters': layer_cfg['filters'],
                'kernel_size': tuple(layer_cfg['kernel_size']),
                'activation': layer_cfg.get('activation'),
                'padding': layer_cfg.get('padding', 'valid')
            }
            if first_layer:
                kwargs['input_shape'] = input_shape
                first_layer = False
            model.add(layers.Conv2D(**kwargs))

        elif layer_type == 'BatchNormalization':
            model.add(layers.BatchNormalization())

        elif layer_type == 'MaxPooling2D':
            model.add(layers.MaxPooling2D(pool_size=tuple(layer_cfg['pool_size'])))

        elif layer_type == 'Dropout':
            model.add(layers.Dropout(layer_cfg['rate']))

        elif layer_type == 'Flatten':
            model.add(layers.Flatten())

        elif layer_type == 'Dense':
            model.add(layers.Dense(
                units=layer_cfg['units'],
                activation=layer_cfg.get('activation')
            ))

    return model


def train_pytorch(model, train_loader, val_loader, config: dict):
    """Train PyTorch model."""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Setup optimizer
    opt_config = config['training']['optimizer']
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt_config['learning_rate'],
        betas=(opt_config['beta_1'], opt_config['beta_2']),
        eps=opt_config['epsilon']
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    epochs = config['training']['epochs']
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(val_loader)
        accuracy = 100. * correct / len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{epochs}')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= config['training']['early_stopping']['patience']:
                print(f'Early stopping at epoch {epoch+1}')
                break

    return model


def train_keras(model, train_data, val_data, config: dict):
    """Train Keras/TensorFlow model."""
    from tensorflow import keras

    # Compile model
    opt_config = config['training']['optimizer']
    optimizer = keras.optimizers.Adam(
        learning_rate=opt_config['learning_rate'],
        beta_1=opt_config['beta_1'],
        beta_2=opt_config['beta_2'],
        epsilon=opt_config['epsilon']
    )

    model.compile(
        optimizer=optimizer,
        loss=config['training']['loss'],
        metrics=config['training']['metrics'][:2]  # accuracy, precision
    )

    # Callbacks
    callbacks = []

    # Early stopping
    es_config = config['training']['early_stopping']
    callbacks.append(keras.callbacks.EarlyStopping(
        monitor=es_config['monitor'],
        patience=es_config['patience'],
        restore_best_weights=es_config['restore_best_weights']
    ))

    # Learning rate reduction
    lr_config = config['training']['lr_schedule']
    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=lr_config['factor'],
        patience=lr_config['patience'],
        min_lr=lr_config['min_lr']
    ))

    # Model checkpoint
    ckpt_config = config['training']['checkpoint']
    os.makedirs(os.path.dirname(ckpt_config['filepath']), exist_ok=True)
    callbacks.append(keras.callbacks.ModelCheckpoint(
        filepath=ckpt_config['filepath'],
        monitor=ckpt_config['monitor'],
        save_best_only=ckpt_config['save_best_only'],
        mode=ckpt_config['mode']
    ))

    # Train
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        callbacks=callbacks
    )

    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train Deep Learning Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--framework', type=str, default='keras',
                       choices=['pytorch', 'keras'], help='DL framework')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Deep Learning Training Pipeline")
    print(f"Framework: {args.framework.upper()}")
    print(f"Model: {config['model']['name']}")
    print(f"Output: {output_dir}")
    print(f"{'='*50}\n")

    if args.framework == 'pytorch':
        model = create_pytorch_model(config)
        print(f"Created PyTorch model")
        # Note: Add data loading logic for your specific use case
        print("Training with PyTorch...")
    else:
        model = create_keras_model(config)
        model.summary()
        print("Training with Keras/TensorFlow...")

    print("\n[SUCCESS] Training pipeline initialized!")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Learning Rate: {config['training']['optimizer']['learning_rate']}")


if __name__ == '__main__':
    main()

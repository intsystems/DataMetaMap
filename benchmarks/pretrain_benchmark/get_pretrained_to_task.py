from data_meta_map.task2vec import task2vec
from data_meta_map.models import get_model
from data_meta_map import datasets
from data_meta_map.task2vec import plot_distance_matrix

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf


@torch.inference_mode()
def evaluate(model, loader, device):

    model.eval()
    correct = 0
    total = 0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        pred = logits.argmax(1)

        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / total


def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          optimizer,
          criterion,
          best_path: str,
          num_epochs: int,
          patience: int,
          device: str):

    best_acc = 0
    best_epoch = 0
    epochs_without_improvement = 0
    logs = {}

    for epoch in range(num_epochs):

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        val_acc = evaluate(model, val_loader, device=device)

        # improvement
        if val_acc > best_acc:

            best_acc = val_acc
            epochs_without_improvement = 0
            best_epoch = epoch

            torch.save(model.state_dict(), best_path)
        else:
            epochs_without_improvement += 1

        # early stopping condition
        if epochs_without_improvement >= patience:

            break

    model.load_state_dict(torch.load(best_path))
    model.eval()
    logs = {
        'best_val_accuracy': best_acc,
        'best_val_epoch': best_epoch
    }
    return model, logs


@torch.inference_mode()
def extract_embeddings(model, loader):

    embeddings = []
    labels = []

    model.eval()

    for x, y in tqdm(loader):

        x = x.cuda()

        feat = model(x)
        feat = feat.view(feat.size(0), -1)

        embeddings.append(feat.cpu())
        labels.append(y)

    embeddings = torch.cat(embeddings).numpy()
    labels = torch.cat(labels).numpy()

    return embeddings, labels


def save_parquet(embeds, labels, path):

    df = pd.DataFrame({
        "hidden_state": embeds.tolist(),
        "target": labels
    })

    df.to_parquet(path, index=False)


class MLP(nn.Module):

    def __init__(self, input_dim=512, hidden_dim=256, num_classes=10):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),

            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def estimate_task_performance(config):
    train_df = pd.read_parquet(config['paths']['train_parquet'])
    test_df = pd.read_parquet(config['paths']['test_parquet'])

    X_train = torch.tensor(train_df.hidden_state.tolist(), dtype=torch.float32)
    y_train = torch.tensor(train_df.target.values, dtype=torch.long)

    X_test = torch.tensor(test_df.hidden_state.tolist(), dtype=torch.float32)
    y_test = torch.tensor(test_df.target.values, dtype=torch.long)

    VAL_RATIO = config['training_task']['val_split']

    dataset = TensorDataset(X_train, y_train)

    train_size = int((1 - VAL_RATIO) * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    BATCH_SIZE = config['training_task']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(
        X_test, y_test), batch_size=BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()

    device = torch.device(config.device)
    model = MLP(input_dim=512,
                hidden_dim=config['estimate_network_params']['hidden_dim'],
                num_classes=config['estimate_network_params']['num_classes']).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['training']['lr'])

    model, logs = train(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        best_path=config['paths']['checkpoint_task'],
        num_epochs=config['training_task']['epochs'],
        patience=config['training_task']['early_stopping_epochs'],
        device=device
    )
    test_acc = evaluate(model, test_loader, device)
    logs['test_acc'] = test_acc
    return logs


def apply_pretrain(config):
    device = torch.device(config.device)

    model = get_model(config['model']['name'],
                      pretrained=config['model']['pretrained']).to(device)
    train_dataset, test_dataset = datasets.__dict__[
        config['pretrain_dataset_name']](root=config['paths']['data_dir'])

    device = "cuda"

    VAL_RATIO = config['training']['val_split']
    BATCH_SIZE = config['training']['batch_size']

    train_size = int((1 - VAL_RATIO) * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_ds, val_ds = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['training']['lr'])

    model, logs = train(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        best_path=config['paths']['checkpoint'],
        num_epochs=config['training']['epochs'],
        patience=config['training']['early_stopping_epochs'],
        device=device
    )
    return logs


def inference_task(config):
    best_path = config['paths']['checkpoint']
    device = torch.device(config.device)

    model = get_model(config['model']['name'],
                      pretrained=config['model']['pretrained']).to(device)
    model.load_state_dict(torch.load(best_path))
    model.eval()

    embedder = torch.nn.Sequential(*list(model.children())[:-1]).to(device)

    train_dataset, test_dataset = datasets.__dict__[
        config['task_dataset_name']](root=config['paths']['data_dir'])

    BATCH_SIZE = config['training']['batch_size']
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    train_full_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train_embeds, train_labels = extract_embeddings(
        embedder, train_full_loader)
    test_embeds, test_labels = extract_embeddings(embedder, test_loader)

    save_parquet(train_embeds, train_labels, config['paths']['train_parquet'])
    save_parquet(test_embeds, test_labels, config['paths']['test_parquet'])


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config: DictConfig):
    _, pretrain_logs = apply_pretrain(
        config) if config['need_pretrain'] else {}
    task_logs = {}
    if not config['pretrain_only']:
        inference_task(config)
        task_logs = estimate_task_performance(config)
    logs = {
        'pretrain': pretrain_logs,
        'task': task_logs,
        'pretrain_to_task_config': OmegaConf.to_container(config, resolve=True)
    }
    with open(config['paths']['save_logs'], 'w') as f:
        yaml.safe_dump(logs, f)


if __name__ == "__main__":
    main()

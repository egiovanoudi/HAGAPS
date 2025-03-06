import argparse
import pandas as pd
import os
from tqdm import tqdm
import torch

from data import prepare_data, create_dataloader
from model import HAGAPS
from utils import *


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default=f'Data/train_sample.txt')
    parser.add_argument('--val_path', default=f'Data/val_sample.txt')
    parser.add_argument('--test_path', default=f'Data/test_sample.txt')
    parser.add_argument('--structure_path', default='Data/Structures')
    parser.add_argument('--model_path', default='ckpts')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--lamda', type=float, default=0.3)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    train_data = pd.read_csv(args.train_path, sep='\t')
    val_data = pd.read_csv(args.val_path, sep='\t')
    test_data = pd.read_csv(args.test_path, sep='\t')
    len_train = len(train_data)
    len_val = len(val_data)
    dataset = pd.concat([train_data, val_data, test_data])
    dataset['sequence'] = dataset['sequence'].str.upper()
    os.makedirs(args.model_path, exist_ok=True)

    print('Data preprocessing...')

    dataset, max_len = prepare_data(dataset, args.structure_path)
    train_data = dataset.iloc[:len_train]
    val_data = dataset.iloc[len_train:len_train + len_val].reset_index()
    test_data = dataset.iloc[len_train + len_val:].reset_index()

    train_loader = create_dataloader(train_data, args.batch_size, True, device)
    val_loader = create_dataloader(val_data, args.batch_size, False, device)
    test_loader = create_dataloader(test_data, args.batch_size, False, device)

    # Initialize the model
    model = HAGAPS(args.hidden_size, max_len)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print('Model training...')
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    best_epoch = 0
    for i in range(args.epochs):
        total_loss = []
        model.train()
        for batch in tqdm(train_loader):
            gene_loss = []
            optimizer.zero_grad()
            for gene in batch:
                output = model(gene)
                gene_loss.append(loss_function(gene.y, output, args.lamda, args.gamma))
            loss = sum(gene_loss) / len(gene_loss)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
        train_loss = sum(total_loss) / len(total_loss)
        print(f'Epoch {i + 1}: Training Loss =', train_loss)
        train_losses.append(train_loss)
        val_loss = evaluate(model, val_loader, args.lamda, args.gamma, False)
        val_losses.append(val_loss)
        ckpt = (model.state_dict(), optimizer.state_dict())
        torch.save(ckpt, os.path.join(args.model_path, f"model{i + 1}"))
        print(f'Epoch {i + 1}: Validation Loss =', val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = i + 1

    best_ckpt = os.path.join(args.model_path, f"model{best_epoch}")
    model.load_state_dict(torch.load(best_ckpt)[0])
    print('Testing Loss =', evaluate(model, test_loader, args.lamda, args.gamma, True))
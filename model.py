import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import config

def create_model_components():
    """Initialize the Transformer architecture components."""
    input_net = nn.Sequential(
        nn.Linear(config.FEATURE_SIZE, config.D_MODEL),
        nn.LayerNorm(config.D_MODEL),
        nn.ReLU()
    ).to(config.DEVICE)
    
    pos_embedding = nn.Parameter(torch.randn(1, config.SEQUENCE_LENGTH, config.D_MODEL, device=config.DEVICE))
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=config.D_MODEL, nhead=config.NHEAD, 
        dim_feedforward=config.D_MODEL*2, dropout=config.DROPOUT, batch_first=True
    )
    transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.NUM_LAYERS).to(config.DEVICE)
    decoder_net = nn.Sequential(
        nn.Linear(config.D_MODEL, config.D_MODEL // 2), 
        nn.ReLU(), 
        nn.Linear(config.D_MODEL // 2, config.FEATURE_SIZE)
    ).to(config.DEVICE)
    
    return input_net, pos_embedding, transformer, decoder_net

def forward_pass(src, input_net, pos_embedding, transformer, decoder_net):
    """Executes a single forward pass through the model."""
    x = input_net(src) + pos_embedding
    memory = transformer(x)
    embedding = torch.mean(memory, dim=1) 
    recon = decoder_net(memory)
    return recon, embedding

def train_model(features_dict, valid_tickers, start_date, end_date):
    """Trains the autoencoder and returns the trained components."""
    slice_sequences = []
    for t in valid_tickers:
        pkg = features_dict[t]
        data, idx = pkg['data'], pkg['index']
        mask = (idx >= start_date) & (idx < end_date)
        if mask.sum() < config.SEQUENCE_LENGTH + 5: continue
        
        subset = data[mask]
        n_samples = len(subset) - config.SEQUENCE_LENGTH
        if n_samples > 0:
            seqs = [subset[i : i + config.SEQUENCE_LENGTH] for i in range(n_samples)]
            slice_sequences.append(np.array(seqs))
            
    if not slice_sequences: return None
    
    tensor_data = torch.FloatTensor(np.concatenate(slice_sequences, axis=0))
    dataset = TensorDataset(tensor_data, tensor_data)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    
    components = create_model_components()
    input_net, pos_embedding, transformer, decoder_net = components
    params = list(input_net.parameters()) + [pos_embedding] + list(transformer.parameters()) + list(decoder_net.parameters())
    optimizer = optim.AdamW(params, lr=1e-3)
    criterion = nn.MSELoss()
    
    input_net.train(); transformer.train(); decoder_net.train()
    
    for _ in range(config.EPOCHS):
        for bx, by in dataloader:
            bx, by = bx.to(config.DEVICE), by.to(config.DEVICE)
            optimizer.zero_grad()
            recon, _ = forward_pass(bx, input_net, pos_embedding, transformer, decoder_net)
            loss = criterion(recon, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
    return components

def extract_embeddings_slice(features_dict, tickers, components, end_date):
    """Extract latent representations (embeddings) using the trained model."""
    input_net, pos_embedding, transformer, decoder_net = components
    input_net.eval(); transformer.eval(); decoder_net.eval()
    embeddings = {}
    
    with torch.no_grad():
        for t in tickers:
            pkg = features_dict[t]
            data, idx = pkg['data'], pkg['index']
            locs = np.where(idx < end_date)[0]
            if len(locs) < config.SEQUENCE_LENGTH: continue
            
            last_idx = locs[-1]
            seq = data[last_idx - config.SEQUENCE_LENGTH + 1 : last_idx + 1]
            tensor_in = torch.FloatTensor(seq).unsqueeze(0).to(config.DEVICE)
            _, emb = forward_pass(tensor_in, input_net, pos_embedding, transformer, decoder_net)
            embeddings[t] = emb.cpu().numpy().flatten()
            
    return pd.DataFrame(embeddings).T

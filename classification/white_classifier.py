from celeb_classifier import MLPClassifierWrapper, train_routine, test_routine
import argparse
import pytorch_lightning as pl
import torch
from pathlib import Path
import torch.utils.data as torchdata

import sys
sys.path.append('..')
from src.data_loader import EmbeddingData

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description = 'Trains a MLP classifier on Caucasians using vggface2 embeddings')
    parser.add_argument('-root', default = '..', help = 'Root path')
    
    args = parser.parse_args()
    
    # seed for reproducibility
    pl.seed_everything(42)
        
    root = Path(args.root)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Using {device} for inference')

    # celeb results
    
    results_dir = root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    fface_dir = results_dir / f'fface_results_vggface2'
    fface_dir.mkdir(exist_ok=True)
    
    # load white training data
    white_data = EmbeddingData(data_dir_name=f'fface_embeddings_vggface2', root = str(root), device= device, sample = False, prefix = 'white')
    
    # split white_data
    train_len = int(.8 * len(white_data))
    val_len = int(.10 * len(white_data))
    test_len = len(white_data) - train_len - val_len
    train_data, val_data, test_data = torchdata.random_split(white_data, lengths = [train_len, val_len, test_len])
    
    # loaders
    train_loader = torchdata.DataLoader(train_data, batch_size = 128, shuffle = True)
    val_loader = torchdata.DataLoader(val_data, batch_size = 256, shuffle = False)
    white_loader = torchdata.DataLoader(test_data, batch_size = 256, shuffle = False)
    # load other ethnicities
    black_loader = torchdata.DataLoader(EmbeddingData(data_dir_name=f'fface_embeddings_vggface2', root = str(root), device= device, sample = False, prefix = 'black'), batch_size = 256, shuffle=False)
    ea_loader = torchdata.DataLoader(EmbeddingData(data_dir_name=f'fface_embeddings_vggface2', root = str(root), device= device, sample = False, prefix = 'ea'), batch_size = 256, shuffle=False)
    ind_loader = torchdata.DataLoader(EmbeddingData(data_dir_name=f'fface_embeddings_vggface2', root = str(root), device= device, sample = False, prefix = 'ind'), batch_size = 256, shuffle=False)
    lh_loader = torchdata.DataLoader(EmbeddingData(data_dir_name=f'fface_embeddings_vggface2', root = str(root), device= device, sample = False, prefix = 'lh'), batch_size = 256, shuffle=False)
    me_loader = torchdata.DataLoader(EmbeddingData(data_dir_name=f'fface_embeddings_vggface2', root = str(root), device= device, sample = False, prefix = 'me'), batch_size = 256, shuffle=False)
    se_loader = torchdata.DataLoader(EmbeddingData(data_dir_name=f'fface_embeddings_vggface2', root = str(root), device= device, sample = False, prefix = 'se'), batch_size = 256, shuffle=False)
    
    # model 
    model = MLPClassifierWrapper(input_features = 512)
    
    test_loaders = [white_loader, black_loader, ea_loader, ind_loader, lh_loader, me_loader, se_loader]
    # train and test
    trainer = train_routine(model, train_loader, val_loader, test_loaders, save_dir = fface_dir)
    test_routine(model, trainer, test_loaders, ckpt_path = 'best')
    
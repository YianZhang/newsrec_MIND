def evaluate(dataset, model, ratio=1):
    dataset.encode_all_titles(model.news_encoder)
    print('finish encoding all the titles', flush = True)
    dataset.load_data_for_evaluation()
    labels, preds = valid.get_predictions(model, ratio)
    print(labels[:5], preds[:5])

if __name__ == '__main__':
    import torch
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_name', default = 'model.pt')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: ', device, flush = True)
    
    # get data
    from data_loading import MINDDataset
    from torch.utils.data import RandomSampler
    from torch.utils.data import DataLoader
    from utils import Config, save_checkpoint, load_checkpoint
    from os import path
    DATA_SIZE = "demo" # demo, small, complete

    valid = MINDDataset(path.join(DATA_SIZE,'valid/news.tsv'), path.join(DATA_SIZE,'valid/behaviors.tsv'), subset='valid')
    valid.load_data()
    valid_sampler = RandomSampler(valid)
    valid_dataloader = DataLoader(
    valid,
    sampler=valid_sampler,
    batch_size=valid.batch_size,
    collate_fn=valid.collate_fn
    )
    print('finish loading data', flush = True)

    # build the model
    from model import NewsRec
    self_attention_hyperparameters = {'num_attention_heads' : 16, 'hidden_size' : 768, 'attention_probs_dropout_prob': 0.2, 'max_position_embeddings': 4, 'is_decoder': False, 'position_embedding_type' : None}
    assert self_attention_hyperparameters['hidden_size'] % self_attention_hyperparameters['num_attention_heads'] == 0
    self_attention_config = Config(self_attention_hyperparameters)
    model = NewsRec(self_attention_config).to(device)
    print('finish building the model', flush = True)

    try:
        load_checkpoint(model, optimizer, device, args.checkpoint_name)
        print('checkpoint loaded', flush = True)
    except:
        print('failed to load any checkpoints.', flush = True)
    
    evaluate(valid, model, 0.1)
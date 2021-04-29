from utils import cal_metric

def evaluate(dataset, model, ratio=1):
    dataset.encode_all_titles(model.news_encoder)
    print('finish encoding all the titles', flush = True)
    dataset.load_data_for_evaluation()
    labels, preds = dataset.get_predictions(model, ratio)
    metrics = ['group_auc', 'mean_mrr', 'ndcg@5', 'ndcg@10']
    return cal_metric(labels, preds, metrics)
    #print(labels[:5], preds[:5])



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
    DATA_SIZE = "demo" # demo, small, large

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

    if args.pretrained_model == 'bert-base-uncased':
        HIDDEN_SIZE = 768
    elif args.pretrained_model == 'distilbert-base-uncased':
        HIDDEN_SIZE = 768
    elif args.pretrained_model == 'prajjwal1/bert-tiny':
        HIDDEN_SIZE = 128
    self_attention_hyperparameters = {'num_attention_heads' : 16, 'hidden_size' : HIDDEN_SIZE, 'attention_probs_dropout_prob': 0.2, 'max_position_embeddings': 4, 'is_decoder': False, 'position_embedding_type' : None}
    assert self_attention_hyperparameters['hidden_size'] % self_attention_hyperparameters['num_attention_heads'] == 0
    self_attention_config = Config(self_attention_hyperparameters)
    model = NewsRec(self_attention_config).to(device)
    print('finish building the model', flush = True)

    try:
        print(args.checkpoint_name)
        load_checkpoint(model, map_location = device, path = args.checkpoint_name)
        print('checkpoint loaded', flush = True)
    except:
        print('WARNING: failed to load any checkpoints.', flush = True)
    
    evaluate(valid, model, 1)

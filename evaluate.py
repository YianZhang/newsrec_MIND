from utils import cal_metric

def evaluate(dataset, model, ratio=1, subset='valid'):
    dataset.encode_all_news(model.news_encoder)
    print('finish encoding all the titles', flush = True)
    dataset.load_data_for_evaluation()

    if subset == 'valid':
        labels, preds = dataset.get_predictions(model, ratio)
        metrics = ['group_auc', 'mean_mrr', 'ndcg@5', 'ndcg@10']
        return cal_metric(labels, preds, metrics)
    else: # test
        preds = dataset.get_predictions(model, ratio)
        with open('predictions.txt','w') as f:
            f.write(str(preds))

    #print(labels[:5], preds[:5])



if __name__ == '__main__':
    import torch

    torch.manual_seed(42) # for reproducibility
    random.seed(42) # for reproducibility

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_name', default = 'model.pt')
    parser.add_argument('--pretrained_model', default = 'bert-base-uncased')
    parser.add_argument('--datasize', default = 'large')
    parser.add_argument('--attn_dropout', type = float, default = 0.2)
    parser.add_argument('--scorer', default = 'dot_product')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device, flush = True)
        
    self_attention_hyperparameters = {'num_attention_heads' : 20, 'attention_probs_dropout_prob': args.attn_dropout, 'max_position_embeddings': 4, 'is_decoder': False, 'position_embedding_type' : None,}
    # assert self_attention_hyperparameters['hidden_size'] % self_attention_hyperparameters['num_attention_heads'] == 0
    # get data
    from data_loading import MINDDataset
    from torch.utils.data import RandomSampler
    from torch.utils.data import DataLoader
    from utils import Config, save_checkpoint, load_checkpoint
    from os import path

    DATA_SIZE = args.datasize # demo, small, large

    test = MINDDataset(path.join(DATA_SIZE,'valid/news.tsv'), path.join(DATA_SIZE,'valid/behaviors.tsv'), 'all_embeddings.vec', 'large', batch_size=BATCH_SIZE, model=args.pretrained_model, subset='valid')
    test.init_news()
    test_sampler = RandomSampler(valid)
    test_dataloader = DataLoader(
    test,
    sampler=test_sampler,
    batch_size=test.batch_size,
    collate_fn=test.collate_fn
    )

    print('checking the class2id matrices:', train._class2id == valid._class2id, train._subclass2id == valid._subclass2id)
    print('finish loading data', flush = True)

    # build the model
    news_encoder_parameters = {'n_classes': len(train._class2id), 'n_subclasses': len(train._subclass2id), 'class_embedding_dim': 50, 'subclass_embedding_dim': 30, 'news_repr_dim': 400, 'distil_dropout': 0.1, 'class_dropout': 0, 'entity_embedding_dim': 100}
    self_attention_hyperparameters['hidden_size'] = news_encoder_parameters['news_repr_dim']
    print(self_attention_hyperparameters)
    self_attention_config = Config(self_attention_hyperparameters)
    model = NewsRec(self_attention_config, news_encoder_parameters, args.pretrained_model, args.scorer).to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    print('finish building the model', flush = True)

    try:
        print(args.checkpoint_name)
        load_checkpoint(model, map_location = device, path = args.checkpoint_name)
        print('checkpoint loaded', flush = True)
    except:
        print('WARNING: failed to load any checkpoints.', flush = True)

    evaluate(valid, model, 1, subset='test')

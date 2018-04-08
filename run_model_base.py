from trainer_base import Trainer

data_params = {
    'batch_size': 100,
    'vacab_size': 20002, # diff 2
    'word_dim': 300,
    'max_n_context': 150,
    'max_n_response': 20,

    'word2vec': './word2vec/word2vec.bin',
    'dataset':'./data/Dataset.dict.pkl',
    'train_data':'./data/Training.dialogues.pkl',
    'valid_data':'./data/Validation.dialogues.pkl',
    'test_data':'./data/Test.dialogues.pkl',
    'ground_valid':'data/ResponseContextPairs/raw_validation_responses.txt',
    'ground_test':'data/ResponseContextPairs/raw_testing_responses.txt',


    'learning_rate': 1e-3,
    'lr_decay_n_iters': 10000,
    'lr_decay_rate': 0.8,
    'max_epoches': 1000,
    'early_stopping': 40,
    'cache_dir': './results/base',
    'summary_dir': './summary/model',
    'output_dir':'./output_rl/',
    'display_batch_interval': 20,
    'summary_interval': 10,
    'evaluate_interval': 5,
    'saving_interval': 10,
    'epoch_reshuffle': True,

    'model_name': 'qvec_all_att_model',
    'lstm_dim': 512,
    'ref_dim':256,
    'ques_embed_dim': 100,  # original: 300
    'attention_dim': 256,
    'regularization_beta': 1e-7,
    'dropout_prob': 0.6

}

if __name__ == '__main__':
    trainer = Trainer(data_params)
    trainer.train()

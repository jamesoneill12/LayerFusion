import argparse
import os


# flickr_image_dir = "D:/data/flickr30k/flickr30k-images/"
# rnn_type = 'lstm' removed because dec already specifies this
def get_args(enc = 'resnet',
                dec = 'lstm',
                data_source = 'coco',
                training_method='standard',
                critic_reward='',
                loss = 'ce',
                reward = None,
                mod_folder = None,
                save_results_path = None,
                encoder_paths = None,
                decoder_paths =None,
    ):
    print(data_source)
    if data_source == 'coco':
        root = 'D:/data/coco/'
        images_path = root+'images/'
        annotation_path = root+'annotations/'
        vocab_path = annotation_path+'vocab.pkl'
        image_dir = images_path+'train2014/resized2014/'
        val_image_dir = images_path+'val2014/resized2014/'
        test_image_dir = images_path+'test2014/resized2014/'
        train_val_path = annotation_path+'annotations_trainval2014/annotations/'
        test_path = annotation_path+'image_info_test2014/annotations/'
        caption_path = train_val_path+'captions_train2014.json'
        val_caption_path = train_val_path+'captions_val2014.json'
        test_caption_path = test_path+'image_info_test2014.json'

    elif 'flickr' in data_source:
        root = 'D:/data/flickr30k/'
        vocab_path = root + 'vocab.pkl'
        image_dir = root + 'flickr30k-resized-images/'
        val_image_dir, test_image_dir = image_dir, image_dir # None, None
        caption_path = root + "flickr30k_train.json" # dataset_flickr30k.json
        val_caption_path = root + "flickr30k_val.json"
        test_caption_path = root + "flickr30k_test.json" # None, None

    model_root = root + 'pretrained_models/'
    results_root = root + 'results/'
    if mod_folder is None: mod_folder = enc + '_' + dec + '_' + training_method + '_' + loss + '' + critic_reward
    if reward is not None: mod_folder += '_' + reward
    save_model_path = model_root + mod_folder
    save_results_path = results_root + mod_folder

    if not os.path.exists(results_root): os.mkdir(results_root)
    if not os.path.exists(save_results_path): os.mkdir(save_results_path)
    if not os.path.exists(model_root): os.mkdir(model_root)
    if not os.path.exists(save_model_path): os.mkdir(save_model_path)

    if encoder_paths is None: encoder_path = save_model_path + '/encoder.ckpt'
    if decoder_paths is None: decoder_path = save_results_path + '/decoder.ckpt'

    parser = argparse.ArgumentParser()
    # 'C:/Users/jamesoneill/Projects/NLP/GOLM/golm/golm/golm_hil/save_models/image_caption/coco/'
    parser.add_argument('--model_path', type=str, default=save_model_path,
                        help='path for saving trained mods')
    # 'C:/Users/jamesoneill/Projects/NLP/GOLM/golm/golm/golm_hil/save_models/image_caption/coco/pretrained_model/'
    parser.add_argument("--save_results_path", type=str, default=save_results_path + "/results.pkl")
    parser.add_argument('--load_encoder_path', type=str,
                        default=encoder_path,
                        help='path for loading pretrained mods, none by default')
    parser.add_argument('--load_decoder_path', type=str,
                        default=decoder_path,
                        help='path for loading pretrained mods, none by default')

    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')

    parser.add_argument('--vocab_path', type=str, default=vocab_path,
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default=image_dir,
                        help='directory for resized images')
    parser.add_argument('--val_image_dir', type=str, default=val_image_dir)
    parser.add_argument('--test_image_dir', type=str, default=test_image_dir)
    parser.add_argument('--caption_path', type=str, default=caption_path,
                        help='path for train annotation json file')
    parser.add_argument('--val_caption_path', type=str, default=val_caption_path)
    parser.add_argument('--test_caption_path', type=str, default=test_caption_path)

    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained mods')
    parser.add_argument('--save', type=bool, default=True, help='whether to save the model or not')

    # Model parameters
    parser.add_argument('--train_method', type=str, default='standard', help='standard, raml, etc.')
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    # decoder params (rnn_type is not strictly rnns, can be transformers also)
    parser.add_argument('--rnn_type', type=str, default=dec)
    parser.add_argument('--max_seq_len', type=int, default=20)
    parser.add_argument('--drop_rate', type=float, default=0.15)
    parser.add_argument('--dropout_method', type=str, default='standard')
    parser.add_argument('--training_method', type=str, default='standard', help='standard (max likelihood (ML)), '
                                                                                'RAML,'
                                                                                'SPIDEr,'
                                                                                'latent'
                                                                                'any above with _ss extension uses scheduled samp')
    parser.add_argument('--loss', type=str, default='ce', help='see settings.py loss for full list')
    parser.add_argument('--sinkhorn_iter', type=int, default=100, help='sinkhorn iterations')
    parser.add_argument('--sinkhorn_eps', type=float, default=0.1, help='sinkhorn entropy regularization')

    # evaluation params
    parser.add_argument('--bleu_num', type=int, default=4, help='BLEU4 by default')
    parser.add_argument('--wmd', type=bool, default=True, help='whether to use wmd for evaluation')
    parser.add_argument('--cos', type=bool, default=True, help='whether to use average cos similarity for evaluation')
    parser.add_argument('--rouge', type=bool, default=True, help='whether to use average cos similarity for evaluation')
    parser.add_argument('--rouge_num', type=int, default=2, help='rouge-n')
    parser.add_argument('--cider', type=bool, default=True, help='whether to use average cos similarity for evaluation')

    # raml params (if training_method = raml)
    parser.add_argument('--mode', choices=['sample_from_model', 'sample_ngram_adapt', 'sample_ngram', None], default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--reward', choices=['bleu', 'edit_dist', 'wmd',  'cos',
                                             'infersent', 'uniskip', 'biskip, gpt,'
                                             'gpt2, transformer, transformerxl', None],
                        default=reward, help="wmd and cos added to the original here")
    parser.add_argument('--max_ngram_size', type=int, default=4)
    parser.add_argument('--temp', type=float, default=0.5)
    parser.add_argument('--smooth_bleu', default=False)
    parser.add_argument('--reg_weight', default=0.1, help='when using sent_sim as regularizationon defined weight in loss')

    args = parser.parse_args()
    return args
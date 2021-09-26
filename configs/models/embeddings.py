import argparse


def get_embedding_args(data_path, embeddings_path=None, num_iterations = 10000,
                       batch_size = 10, num_sample=5, cuda = True):
    if embeddings_path is None: embeddings_path = data_path
    parser = argparse.ArgumentParser(description='word2vec')
    parser.add_argument('--num-iterations', type=int, default=num_iterations, metavar='NI',
                        help='num iterations (default: 1000000)')
    parser.add_argument('--batch-size', type=int, default=batch_size, metavar='BS',
                        help='batch size (default: 10)')
    parser.add_argument('--num-sample', type=int, default=num_sample, metavar='NS',
                        help='num sample (default: 5)')
    parser.add_argument('--data_path', type=str, default=data_path, metavar='NS',
                        help='path to data')
    parser.add_argument('--embeddings_path', type=str, default=embeddings_path, metavar='NS',
                        help='path to data')
    parser.add_argument('--use-cuda', type=bool, default=cuda, metavar='CUDA_VISIBLE_DEVICE',
                        help='use cuda (default: True)')
    args = parser.parse_args()
    return args
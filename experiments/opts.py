import argparse


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_filters', type=int, default=100, help='num_filters')
    parser.add_argument('--filter_sizes', type=int, default=[3, 4, 5], help='filter_sizes')
    parser.add_argument('--hidden_dim', type=int, default=256, help='hidden_dim')
    parser.add_argument('--output_dim', type=int, default=11, help='output_dim')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased', help='bert_model')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')

    args = parser.parse_args(())

    return args



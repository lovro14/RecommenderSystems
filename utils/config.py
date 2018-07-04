import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Implementation and evaluation of"
                                                 " neural network based recommender systems")
    parser.add_argument('--dataset', type=str, default='ml-1m', choices=['ml-100k', 'ml-1m', 'ml-20m'])
    parser.add_argument('--model', type=str, default='latent-factor-model',
                        choices=['latent-factor-model', 'deep-neural-network-model', 'ensemble-no-transfer-learning',
                                 'ensemble-transfer-learning'], help='Four model implementations')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epoch for training algorithm')
    parser.add_argument('--batch_size', type=int, default=256, help='The size of batch')
    parser.add_argument('--dimension', type=int, default=8, help='shared latent space dimension')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='learning rate for model optimization')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'RMSProp', 'GradientDescent'])
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='dropout rate for dropout regularization')
    parser.add_argument('--regularization_factor', type=float, default=0.1, help='lambda for L2-regularization')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8,1]',
                        help="Size of each layer. Note that the first layer is the concatenation"
                             " of user and item embeddings. So layers[0]/2 is the embedding size.")

    return check_args(parser.parse_args())


# Checking arguments
def check_args(args):
    # --epoch
    try:
        assert args.epochs >= 1
    except:
        print('Number of epochs must be larger or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('Batch size must be larger than or equal to one')

    #--dimension
    try:
        assert args.dimension >= 1
    except:
        print('Latent factor space must be greater or equal to one')

    #--dropout_rate
    try:
        assert args.dropout_rate >= 0 and args.dropout_rate <= 1
    except:
        print('Dropout rate must be between 0 and 1')

    return args
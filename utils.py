import argparse
import torch

def get_arguments():
    parser = argparse.ArgumentParser(description='Transformer Parameters')

    parser.add_argument('--d_model', type=int, default=512, help='Dimension of word embeddings (default: 512)')
    parser.add_argument('--d_ff', type=int, default=2048, help='Dimension of feed-forward layer (default: 2048)')
    parser.add_argument('--d_k', type=int, default=64, help='Dimension of key and query vectors (default: 64)')
    parser.add_argument('--d_v', type=int, default=64, help='Dimension of value vectors (default: 64)')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of encoder and decoder layers (default: 6)')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads (default: 8)')
    parser.add_argument('--epochs',type=int,default=200,help='Number of epoch (default:200)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for computation (default: cuda if available, else cpu)')
    parser.add_argument('--lr',type=float,default=1e-3,help='Learning rate')
    args = parser.parse_args()
    return args



args = get_arguments()

d_model = args.d_model
d_ff = args.d_ff
d_k = args.d_k
d_v = args.d_v
n_layers = args.n_layers
n_heads = args.n_heads
epochs = args.epochs
device = args.device
lr = args.lr
# 打印参数
print("d_model:", d_model)
print("d_ff:", d_ff)
print("d_k:", d_k)
print("d_v:", d_v)
print("n_layers:", n_layers)
print("n_heads:", n_heads)
print("epochs:",epochs)
print("device:",device)
print("lr:",lr)
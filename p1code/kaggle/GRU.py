import numpy as np
import pandas as pd
import os
import argparse
import time
from utility import pad_history, calculate_hit
import torch.nn as nn
import torch as th

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=30,
                        help='Number of max epochs.')
    parser.add_argument('--epoch_size', type=int, default=-1,
                        help='Each epoch include how many samples')
    parser.add_argument('--data', nargs='?', default='data',
                        help='data directory')
    # parser.add_argument('--pretrain', type=int, default=1,
    #                     help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--gru_layer', type=int, default=1,
                        help='GRU layer number.')
    parser.add_argument('--r_click', type=float, default=0.2,
                        help='reward for the click behavior.')
    parser.add_argument('--r_buy', type=float, default=1.0,
                        help='reward for the purchase behavior.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    return parser.parse_args()


class GRUnetwork(nn.Module):

    def __init__(self, item_num, embedding_size, num_layers = 1):
        super(GRUnetwork, self).__init__()

        self.embedding = nn.Embedding(item_num+1,
                                      embedding_size,
                                      padding_idx = item_num,
                                      max_norm=True)

        self.gru = nn.GRU(embedding_size, embedding_size, batch_first = True, num_layers = num_layers)

        self.fc = nn.Linear(embedding_size, item_num)

    def forward(self, input):
        x = self.embedding(input)
        x, hidden = self.gru(x)  # (N,L,H_in)  => (batch size, sequence length, input size)
        output = self.fc(th.squeeze(hidden) )
        return output

    def initHidden(self):
         pass

def check_device():
  return 'cuda' if th.cuda.is_available() else 'cpu'

def get_training_data(data_directory, batch_size):

    replay_buffer = pd.read_pickle(os.path.join(data_directory, 'replay_buffer.df'))

    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing state_size and item_num
    state_size = data_statis['state_size'][0]  # the length of history to define the state
    item_num = data_statis['item_num'][0]  # total number of items

    num_batches = int(replay_buffer.shape[0] / args.batch_size)

    return replay_buffer, num_batches, item_num, state_size


def create_model(item_num, args):

    model = GRUnetwork(item_num, args.hidden_factor, num_layers=args.gru_layer)
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)

    return model, loss_fn, optimizer


def get_batch_data_from_replay_buffer(replay_buffer, batch_size, device):

    batch = replay_buffer.sample(n=batch_size).to_dict()
    rc_state = th.tensor(np.array(list(batch['state'].values()), dtype=np.int32)).to(device)
    rc_action = th.tensor(np.array(list(batch['action'].values()), dtype=np.int64)).to(device)

    return rc_state, rc_action


def print_evaluate_msg(loss, step_cnt, total_step, start_time):

    if step_cnt % 100 == 0:

        cur_time = time.perf_counter()
        elapse = (cur_time - start_time) / 60
        left_time = ((total_step - step_cnt) / step_cnt) * elapse

        print("the loss in %dth batch is: %f" % (step_cnt, loss))
        print("Elapse %f mins, estimate %f mins left " % (elapse, left_time))

    if step_cnt % 2000 == 0:
        print("Evaluate at %d steps" % (step_cnt))
        pass

def get_train_msg(args, num_batches, total_step):

    msg = "==================================\n"  \
        + "Modle: Basic GRU on ReplayBuffer\n"  \
        + "GRU layer %d\n"      % (1) \
        + "Embedding size %d\n" % (args.hidden_factor) \
        + "Learn rate is %f\n"  % (args.lr) \
        + "Total epoch %d\n"    % (args.epoch) \
        + "Each epoch has %d batches, total %d batches\n" % (num_batches, total_step) \
        + "=================================="

    return msg


def train_model(arg):

    device = check_device()

    # get training data
    replay_buffer, num_batches, item_num, state_size = get_training_data(args.data, args.batch_size)

    # get model, loss function and optimizer
    model, loss_fn, optimizer = create_model(item_num, args)

    total_step = num_batches * args.epoch

    start_time = time.perf_counter()

    model.to(device)
    model.train()

    step_cnt = 0

    print(get_train_msg(args, num_batches, total_step))

    for i in range(args.epoch):

        print("\nStart of epoch %d" % (i,))

        for j in range(num_batches):

            model.zero_grad()

            states, actions = get_batch_data_from_replay_buffer(replay_buffer, args.batch_size, device)
            logits = model(states)

            loss = loss_fn(logits, actions)
            loss.backward()

            optimizer.step()

            # Debug Msg
            step_cnt += 1
            print_evaluate_msg(loss, step_cnt, total_step, start_time)


if __name__ == '__main__':

    args = parse_args()

    train_model(args)



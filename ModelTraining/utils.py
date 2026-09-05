import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Bangla emergency post classification')
    parser.add_argument('--task', default='social-media', type=str, help='Task name')
    parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'), help='use cuda if available')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate')
    parser.add_argument('--decay', default=0., type=float, help='weight decay')
    parser.add_argument('--model', default="bert-base-multilingual-cased", type=str, help='pretrained BERT model name')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size (default: 64)')
    parser.add_argument('--epoch', default=10, type=int, help='total epochs (default: 200)')
    parser.add_argument('--fine-tune', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to fine-tune embedding or not')
    parser.add_argument('--use_lstm', default=False,type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--use_last_4_hState', default=False,type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--save-path', default='out', type=str, help='output log/result directory')
    args = parser.parse_args()
    return args


class TaskConfig:
    num_class = None
    train_split = None
    val_split = None
    test_split = None
    sequence_len = None
    eval_interval = None
    patience = None
    balance = None


def get_task_config(task_name):
    config = TaskConfig()
    if task_name == 'social-media':
        config.num_class = 9
        config.train_split = 'train'
        config.val_split = 'validation'
        config.test_split = 'test'
        config.sequence_len = 100
        config.eval_interval = 100
        config.patience = 30
        config.balance = False
    else:
        raise ValueError('Task not supported')
    return config

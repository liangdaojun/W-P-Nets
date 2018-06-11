import argparse

from models.WPNet import WPNet
from models.PWNet import PWNet
from data_providers.utils import get_data_provider_by_name

train_params_cifar = {
    'batch_size': 64,
    'n_epochs': 300,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 150,  # epochs * 0.5
    'reduce_lr_epoch_2': 225,  # epochs * 0.75
    'validation_set': True,
    'validation_split': None,  # None or float
    'shuffle': 'every_epoch',  # None, once_prior_train, every_epoch
    'normalization': 'by_chanels',  # None, divide_256, divide_255, by_chanels
}

train_params_svhn = {
    'batch_size': 64,
    'n_epochs': 40,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 20,
    'reduce_lr_epoch_2': 30,
    'validation_set': True,
    'validation_split': None,  # you may set it 6000 as in the paper
    'shuffle': True,  # shuffle dataset every epoch or not
    'normalization': 'divide_255',
}


def get_train_params_by_name(name):
    if name in ['C10', 'C10+', 'C100', 'C100+']:
        return train_params_cifar
    if name == 'SVHN':
        return train_params_svhn

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--train' , default = True , action = 'store_true' ,
            help = 'Train the model')
    parser.add_argument(
            '--test' , default = True , action = 'store_true' ,
            help = 'Test model for required dataset if pretrained model exists.'
                   'If provided together with `--train` flag testing will be'
                   'performed right after training.')
    parser.add_argument(
            '--model_type' , '-m' , type = str , choices = ['WPNet' , 'PWNet'] ,
            default = 'WPNet' ,
            help = 'What type of model to use')
    parser.add_argument(
            '--dataset' , '-ds' , type = str ,
            choices = ['C10' , 'C10+' , 'C100' , 'C100+' , 'SVHN','ImageNet'] ,
            default = 'C10' ,
            help = 'What dataset should be used')
    parser.add_argument(
            '--initial_channel' , '-ic' , type = int , choices = [ 48,50,72,96 ,100, 120,150,180] ,
            default = 50 ,
            help = 'The number of channels in the initial layer.')
    parser.add_argument(
            '--total_blocks' , '-tb' , type = int , default = 3 , metavar = '' ,
            help = 'Total blocks of layers stack (default: %(default)s)')
    parser.add_argument(
            '--group_num' , '-gn' , type = int , choices = [10 , 25 , 40] ,
            default = 10 ,
            help = 'The number of groups in each block.')
    parser.add_argument(
            '--growth_rate' , '-gr' , type = int , choices = [0, 1 , 2 , 3] ,
            default = 1 ,
            help = 'The number of channels in each group is increased by R (growth rate) times.')
    parser.add_argument(
            '--compress_rate' , '-cr' , type = float , default = 1, metavar = '' ,
            help = 'reduction Theta at transition layer for DenseNets-BC models')
    parser.add_argument(
            '--keep_prob' , '-kp' , type = float , metavar = '' ,
            help = "Keep probability for dropout.")
    parser.add_argument(
            '--weight_decay' , '-wd' , type = float , default = 1e-4 , metavar = '' ,
            help = 'Weight decay for optimizer (default: %(default)s)')
    parser.add_argument(
            '--nesterov_momentum' , '-nm' , type = float , default = 0.9 , metavar = '' ,
            help = 'Nesterov momentum (default: %(default)s)')

    parser.add_argument(
            '--logs' , dest = 'should_save_logs' , action = 'store_true' ,
            help = 'Write tensorflow logs')
    parser.add_argument(
            '--no-logs' , dest = 'should_save_logs' , action = 'store_false' ,
            help = 'Do not write tensorflow logs')
    parser.set_defaults(should_save_logs = True)

    parser.add_argument(
            '--restore' , dest = 'should_restore_model' , action = 'store_true' ,
            help = 'Save model during training')
    parser.set_defaults(should_restore_model = True)
    parser.add_argument(
            '--saves' , dest = 'should_save_model' , action = 'store_true' ,
            help = 'Save model during training')
    parser.add_argument(
            '--no-saves' , dest = 'should_save_model' , action = 'store_false' ,
            help = 'Do not save model during training')
    parser.set_defaults(should_save_model = True)

    parser.add_argument(
            '--renew-logs' , dest = 'renew_logs' , action = 'store_true' ,
            help = 'Erase previous logs for model if exists.')
    parser.add_argument(
            '--not-renew-logs' , dest = 'renew_logs' , action = 'store_false' ,
            help = 'Do not erase previous logs for model if exists.')
    parser.set_defaults(renew_logs = True)

    return parser.parse_args()

if __name__ == '__main__':
# region
    args = parser()

    if not args.keep_prob:
        if args.dataset in ['C10', 'C100', 'SVHN']:
            args.keep_prob = 0.8
        else:
            args.keep_prob = 1.0

    model_params = vars(args)

    if not args.train and not args.test:
        print("You should train or test your network. Please check params.")
        exit()

    # some default params dataset/architecture related
    train_params = get_train_params_by_name(args.dataset)
    print("Params:")
    for k, v in model_params.items():
        print("\t%s: %s" % (k, v))
    print("Train params:")
    for k, v in train_params.items():
        print("\t%s: %s" % (k, v))
# endregion

    print("Prepare training data...")
    data_provider = get_data_provider_by_name(args.dataset , train_params)
    print("Initialize the model..")

    if args.model_type == 'WPNet':
        model = WPNet(data_provider = data_provider , **model_params)
    elif args.model_type == 'PWNet':
        model = PWNet(data_provider = data_provider , **model_params)

    if args.train:
        print("Data provider train images: ", data_provider.train.num_examples)
        model.train_all_epochs(train_params)
    # when the model have trained, we test for the model.
    if args.test:
        if not args.train:
            model.load_model()
        print("Data provider test images: ", data_provider.test.num_examples)
        print("Testing...")
        loss, accuracy = model.test(data_provider.test, batch_size=200)
        print("mean cross_entropy: %f, mean accuracy: %f" % (loss, accuracy))

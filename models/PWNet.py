import os
import time
import shutil
from datetime import timedelta

import numpy as np
import tensorflow as tf

TF_VERSION = float('.'.join(tf.__version__.split('.')[:2]))


class PWNet :
    def __init__(self , data_provider , initial_channel , total_blocks , group_num , growth_rate ,
                 compress_rate , keep_prob ,
                 weight_decay , nesterov_momentum , model_type , dataset ,
                 should_save_logs , should_restore_model , should_save_model ,
                 renew_logs = False ,
                 **kwargs) :
        self.data_provider = data_provider
        self.data_shape = data_provider.data_shape
        self.n_classes = data_provider.n_classes
        self.initial_channel = initial_channel
        self.total_blocks = total_blocks
        self.group_num = group_num
        self.growth_rate = growth_rate
        self.compress_rate = compress_rate
        self.keep_prob = keep_prob
        self.weight_decay = weight_decay
        self.nesterov_momentum = nesterov_momentum
        self.model_type = model_type
        self.dataset_name = dataset
        self.should_save_logs = should_save_logs
        self.should_restore_model = should_restore_model
        self.should_save_model = should_save_model
        self.renew_logs = renew_logs

        self.batches_step = 0
        self.epoch = 1

        self._define_inputs()
        self._build_graph()
        self._initialize_session()
        self._count_trainable_params()

    # region
    def _initialize_session(self) :
        """Initialize session, variables, saver"""
        config = tf.ConfigProto()
        # restrict model GPU memory utilization to min required
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        # tf_ver = int(tf.__version__.split('.')[1])
        if TF_VERSION <= 0.10 :
            self.sess.run(tf.initialize_all_variables())
            logswriter = tf.train.SummaryWriter
        else :
            self.sess.run(tf.global_variables_initializer())
            logswriter = tf.summary.FileWriter
        self.saver = tf.train.Saver()
        self.summary_writer = logswriter(self.logs_path)

    def _count_trainable_params(self) :
        total_parameters = 0
        for variable in tf.trainable_variables() :
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape :
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print(total_parameters , "Total training params: %.1fM" % (total_parameters / 1e6))

    @property
    def save_path(self) :
        try :
            save_path = self._save_path
        except AttributeError :
            save_path = 'saves/%s' % self.model_identifier
            os.makedirs(save_path)
            save_path = os.path.join(save_path , 'model.chkpt')
            self._save_path = save_path
        return save_path

    @property
    def logs_path(self) :
        try :
            logs_path = self._logs_path
        except AttributeError :
            logs_path = 'logs/%s' % self.model_identifier
            if self.renew_logs :
                shutil.rmtree(logs_path , ignore_errors = True)
            os.makedirs(logs_path)
            self._logs_path = logs_path
        return logs_path

    @property
    def model_identifier(self) :
        return "{}_G={}_R={}_{}".format(
                self.model_type , self.group_num , self.growth_rate , self.dataset_name)

    def save_model(self , global_step = None) :
        self.saver.save(self.sess , self.save_path , global_step = global_step)

    def load_model(self) :
        try :
            self.saver.restore(self.sess , self.save_path + 'something')
        except Exception as e :
            raise IOError("Failed to to load model "
                          "from save path: %s" % self.save_path)
        self.saver.restore(self.sess , self.save_path)
        print("Successfully load model from save path: %s" % self.save_path)

    def log_loss_accuracy(self , loss , accuracy , epoch , prefix ,
                          should_print = True) :
        if should_print :
            print("mean cross_entropy: %f, mean accuracy: %f" % (
                loss , accuracy))
            # summary = tf.Summary(value = [
            #     tf.Summary.Value(
            #             tag = 'loss_%s' % prefix , simple_value = float(loss)) ,
            #     tf.Summary.Value(
            #             tag = 'accuracy_%s' % prefix , simple_value = float(accuracy))
            # ])
            # self.summary_writer.add_summary(summary , epoch)

    def _define_inputs(self) :
        shape = [None]
        shape.extend(self.data_shape)
        self.images = tf.placeholder(
                tf.float32 ,
                shape = shape ,
                name = 'input_images')
        self.labels = tf.placeholder(
                tf.float32 ,
                shape = [None , self.n_classes] ,
                name = 'labels')
        self.learning_rate = tf.placeholder(
                tf.float32 ,
                shape = [] ,
                name = 'learning_rate')
        self.is_training = tf.placeholder(tf.bool , shape = [])

    # endregion

    def composite_function(self , _input , out_features , kernel_size = 3) :
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        with tf.variable_scope("composite_function") :
            # BN
            output = self.batch_norm(_input)
            # ReLU
            output = tf.nn.relu(output)
            # convolution
            output = self.conv2d(
                    output , out_features = out_features , kernel_size = kernel_size)
            # dropout(in case of training and in case it is no 1.0)
            output = self.dropout(output)
        return output

    def bottleneck(self , _input , out_features) :
        with tf.variable_scope("bottleneck") :
            output = self.batch_norm(_input)
            output = tf.nn.relu(output)
            output = self.conv2d(
                    output , out_features = out_features , kernel_size = 1 ,
                    padding = 'VALID')
            # output = self.dropout(output)
        return output

    def add_internal_layer(self , name , x , growth_rate) :
        """Perform H_l composite function for the layer and after concatenate
        input with output from composite function.
        """
        with tf.variable_scope(name) :
            b = self.composite_function(
                    x , out_features = growth_rate // 4 , kernel_size = 3)
            b = self.bottleneck(b , out_features = growth_rate)
        return b

    def add_block(self , x , group_num , growth_rate = 1) :
        """The block structure of PWNets can be divided into layers
         without increasing the number of channels or increasing the number of channels."""
        m = x.shape[-1].value
        growth_channel = (m // group_num) * growth_rate
        if growth_rate == 0 :
            for i in range(group_num) :
                with tf.variable_scope("plus_%d" % i) :
                    n = i * growth_channel
                    print("------------------------------" , n , '---' , growth_channel)
                    b = self.add_internal_layer('b_{}'.format(i) , x[: , : , : , n :n + growth_channel] , m)
                    x += b
        else :
            n = 0
            for i in range(group_num) :
                with tf.variable_scope("plus_%d" % i) :
                    b = self.add_internal_layer('b_{}'.format(i) , x[: , : , : , n :n + growth_channel] , m)
                    c = x[: , : , : , n + growth_channel :] + b[: , : , : , :-growth_channel]
                    x = tf.concat(axis = 3 ,
                                  values = (x[: , : , : , :n + growth_channel] , c , b[: , : , : , -growth_channel :]))
                    n += growth_channel
                    print(i + 1 , '-' * 5 , growth_channel , '-' * 5 , x.shape)
            print('=' * 30)
        return x

    def transition_layer(self , _input) :
        """Call H_l composite function with 1x1 kernel and after average
        pooling
        """
        # call composite function with 1x1 kernel
        out_features = int(int(_input.get_shape()[-1]))
        output = self.composite_function(
                _input , out_features = out_features , kernel_size = 1)
        # run average pooling
        output = self.avg_pool(output , k = 2)
        return output

    def trainsition_layer_to_classes(self , _input) :
        """This is last transition to get probabilities by classes. It perform:
        - batch normalization
        - ReLU nonlinearity
        - wide average pooling
        - FC layer multiplication
        """
        # BN
        output = self.batch_norm(_input)
        # ReLU
        output = tf.nn.relu(output)
        # average pooling
        last_pool_kernel = int(output.get_shape()[-2])
        output = self.avg_pool(output , k = last_pool_kernel)
        # FC
        features_total = int(output.get_shape()[-1])
        output = tf.reshape(output , [-1 , features_total])
        W = self.weight_variable_xavier(
                [features_total , self.n_classes] , name = 'W')
        bias = self.bias_variable([self.n_classes])
        logits = tf.matmul(output , W) + bias
        return logits

    # region

    def _display_params(self) :
        import platform , sys
        # if 'Linux' in platform.system() :
        param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
                tf.get_default_graph() ,
                tfprof_options = tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)

        sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

        tf.contrib.tfprof.model_analyzer.print_model_analysis(
                tf.get_default_graph() ,
                tfprof_options = tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

    def conv2d(self , _input , out_features , kernel_size ,
               strides = [1 , 1 , 1 , 1] , padding = 'SAME') :
        in_features = int(_input.get_shape()[-1])
        kernel = self.weight_variable_msra(
                [kernel_size , kernel_size , in_features , out_features] ,
                name = 'kernel')
        output = tf.nn.conv2d(_input , kernel , strides , padding)
        return output

    def avg_pool(self , _input , k) :
        ksize = [1 , k , k , 1]
        strides = [1 , k , k , 1]
        padding = 'VALID'
        output = tf.nn.avg_pool(_input , ksize , strides , padding)
        return output

    def batch_norm(self , _input) :
        output = tf.contrib.layers.batch_norm(
                _input , scale = True , is_training = self.is_training ,
                updates_collections = None)
        return output

    def dropout(self , _input) :
        if self.keep_prob < 1 :
            output = tf.cond(
                    self.is_training ,
                    lambda : tf.nn.dropout(_input , self.keep_prob) ,
                    lambda : _input
            )
        else :
            output = _input
        return output

    def weight_variable_msra(self , shape , name) :
        return tf.get_variable(
                name = name ,
                shape = shape ,
                initializer = tf.contrib.layers.variance_scaling_initializer())

    def weight_variable_xavier(self , shape , name) :
        return tf.get_variable(
                name ,
                shape = shape ,
                initializer = tf.contrib.layers.xavier_initializer())

    def bias_variable(self , shape , name = 'bias') :
        initial = tf.constant(0.0 , shape = shape)
        return tf.get_variable(name , initializer = initial)

    # endregion
    def _build_graph(self) :
        # first - initial 3 x 3 conv to first_output_features
        if self.dataset_name != 'ImageNet' :
            stages = [self.group_num , self.group_num , self.group_num]
            with tf.variable_scope("Initial_convolution") :
                output = self.conv2d(
                        self.images ,
                        out_features = self.initial_channel ,
                        kernel_size = 3)
        else :
            stages = [6 , 12 , 24 , 16]
            with tf.variable_scope("Initial_convolution") :
                output = self.conv2d(
                        self.images ,
                        out_features = self.initial_channel ,
                        kernel_size = 7 , strides = [1 , 2 , 2 , 1])
                print('begin1:' , output.shape)
                output = tf.nn.max_pool(output , ksize = [1 , 3 , 3 , 1] , strides = [1 , 2 , 2 , 1] , padding = 'SAME')
        print('begin2:' , output.shape)

        # add N required blocks
        for block in range(len(stages)) :
            with tf.variable_scope("Block_%d" % block) :
                output = self.add_block(output , stages[block] , self.growth_rate)
            # last block exist without transition layer
            if block != len(stages) - 1 :
                with tf.variable_scope("Transition_after_block_%d" % block) :
                    output = self.transition_layer(output)

        with tf.variable_scope("Transition_to_classes") :
            logits = self.trainsition_layer_to_classes(output)
        prediction = tf.nn.softmax(logits)

        # Losses
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits = logits , labels = self.labels))
        self.cross_entropy = cross_entropy
        l2_loss = tf.add_n(
                [tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        # optimizer and train step
        optimizer = tf.train.MomentumOptimizer(
                self.learning_rate , self.nesterov_momentum , use_nesterov = True)
        self.train_step = optimizer.minimize(
                cross_entropy + l2_loss * self.weight_decay)

        correct_prediction = tf.equal(
                tf.argmax(prediction , 1) ,
                tf.argmax(self.labels , 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction , tf.float32))

    def train_all_epochs(self , train_params) :
        n_epochs = train_params['n_epochs']
        learning_rate = train_params['initial_learning_rate']
        batch_size = train_params['batch_size']
        reduce_lr_epoch_1 = train_params['reduce_lr_epoch_1']
        reduce_lr_epoch_2 = train_params['reduce_lr_epoch_2']
        total_start_time = time.time()
        lowest_error = 0.3
        for epoch in range(1 , n_epochs + 1) :
            print('-' * 30 , "Train epoch: %d" % epoch , '-' * 30)
            start_time = time.time()
            if epoch == reduce_lr_epoch_1 or epoch == reduce_lr_epoch_2 :
                learning_rate = learning_rate / 10
                print("Decrease learning rate, new lr = %f" % learning_rate)

            print("Training...")
            loss , acc = self.train_one_epoch(
                    self.data_provider.train , batch_size , learning_rate)
            if self.should_save_logs :
                self.log_loss_accuracy(loss , acc , epoch , prefix = 'train')

            if train_params.get('validation_set' , False) :
                print("Validation...")
                loss , acc = self.test(
                        self.data_provider.validation , batch_size)
                if self.should_save_logs :
                    self.log_loss_accuracy(loss , acc , epoch , prefix = 'valid')
                if 1 - acc < lowest_error :
                    lowest_error = 1 - acc
                    print("lowest error = %f" % lowest_error)
            time_per_epoch = time.time() - start_time
            seconds_left = int((n_epochs - epoch) * time_per_epoch)
            print("Time per epoch: %s, Est. complete in: %s" % (
                str(timedelta(seconds = time_per_epoch)) ,
                str(timedelta(seconds = seconds_left))))
            # if self.should_save_model:
            #     self.save_model()
        total_training_time = time.time() - total_start_time
        print("\nTotal training time: %s" % str(timedelta(
                seconds = total_training_time)) , "lowest error = %f" % lowest_error)

    def train_one_epoch(self , data , batch_size , learning_rate) :
        num_examples = data.num_examples
        total_loss = []
        total_accuracy = []
        for i in range(num_examples // batch_size) :
            batch = data.next_batch(batch_size)
            images , labels = batch
            feed_dict = {
                self.images : images ,
                self.labels : labels ,
                self.learning_rate : learning_rate ,
                self.is_training : True ,
            }
            fetches = [self.train_step , self.cross_entropy , self.accuracy]
            result = self.sess.run(fetches , feed_dict = feed_dict)
            _ , loss , accuracy = result
            total_loss.append(loss)
            total_accuracy.append(accuracy)
            if self.should_save_logs :
                self.batches_step += 1
                self.log_loss_accuracy(
                        loss , accuracy , self.batches_step , prefix = 'per_batch' ,
                        should_print = False)
        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)
        return mean_loss , mean_accuracy

    def test(self , data , batch_size) :
        num_examples = data.num_examples
        total_loss = []
        total_accuracy = []
        for i in range(num_examples // batch_size) :
            batch = data.next_batch(batch_size)
            feed_dict = {
                self.images : batch[0] ,
                self.labels : batch[1] ,
                self.is_training : False ,
            }
            fetches = [self.cross_entropy , self.accuracy]
            loss , accuracy = self.sess.run(fetches , feed_dict = feed_dict)
            total_loss.append(loss)
            total_accuracy.append(accuracy)
        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)
        return mean_loss , mean_accuracy

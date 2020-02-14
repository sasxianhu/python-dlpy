#!/usr/bin/env python
# encoding: utf-8
#
# Copyright SAS Institute
#
#  Licensed under the Apache License, Version 2.0 (the License);
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

''' The GAN Model class adds training, evaluation, feature analysis routines for Generative Adversarial Network '''
import os
from random import random

import pandas as pd
from copy import deepcopy

from swat import CASTable

from dlpy import Model
from dlpy.layers import Input, OutputLayer, Keypoints, Reshape, Segmentation
from .model import DataSpec
from .network import WeightsTable
from dlpy.utils import DLPyError, random_name, caslibify_context
import time


def mix_weights_with_fedsql(conn, model_weights, model_old_weights, damping_factor):
    if damping_factor is None or damping_factor == 0.0:
        return model_weights

    conn.loadactionset('fedsql', _messagelevel='error')

    # save the tbl attr
    res = conn.retrieve('table.attribute', _messagelevel='error',
                        task='CONVERT',
                        name=model_weights.name,
                        attrtable=model_weights.name + '_attr')

    tbl1 = model_weights.name
    tbl2 = model_old_weights['name']
    query_join = 'select {}._layerid_, {}._weightid_, {}._weight_, {}._weight_ \
                           from {} left outer join  {} \
                           on {}._layerid_ = {}._layerid_ and {}._weightid_ = {}._weightid_'. \
        format(tbl1, tbl1, tbl1, tbl2, tbl1, tbl2, tbl1, tbl2, tbl1, tbl2)
    res = conn.retrieve('fedsql.execDirect', _messagelevel='error',
                        query=query_join,
                        casout=dict(name=model_weights.name, replace=True))

    res = conn.retrieve('table.alterTable', _messagelevel='error',
                        name=model_weights.name,
                        columns=[dict(name='_weight_', rename='_weight1_')])

    scale1 = 1 - damping_factor
    scale2 = damping_factor
    comp_weight = 'if missing(_Weight__2) then do; _weight_=_weight1_; ' \
                  'end; else do; _weight_=_weight1_*{}+_Weight__2*{};end;'.format(scale1, scale2)

    res = conn.retrieve('table.partition', _messagelevel='error',
                        table=dict(computedVars=['_Weight_'],
                                   computedvarsprogram=comp_weight,
                                   vars=['_layerid_', '_weightid_', '_weight_'],
                                   name=model_weights.name),
                        casout=dict(replace=True, name=model_weights.name))

    # attach the tbl attr
    res = conn.retrieve('table.attribute', _messagelevel='error',
                        task='ADD',
                        name=model_weights.name,
                        attrtable=model_weights.name + '_attr')

    return model_weights


def mix_weights(conn, model_weights, model_old_weights, damping_factor):
    if damping_factor is None or damping_factor == 0.0:
        return model_weights

    # save the tbl attr
    res = conn.retrieve('table.attribute', _messagelevel='error',
                        task='CONVERT',
                        name=model_weights.name,
                        attrtable=model_weights.name + '_attr')

    res = conn.retrieve('table.alterTable', _messagelevel='error',
                        name=model_weights.name,
                        columns=[dict(name='_weight_', rename='_weight1_')])

    res = conn.retrieve('table.alterTable', _messagelevel='error',
                        name=model_old_weights['name'],
                        columns=[dict(name='_weight_', rename='_weight0_')])

    keypgm = "length key $26; length k0 $12; length k1 $12; \
              k0=putn(_layerid_,'best12.');\
              k1=putn(_weightid_,'best12.');\
              key=catx('_', k0, k1);"

    res = conn.retrieve('deepLearn.dlJoin', _messagelevel='error',
                        casout=dict(name=model_weights.name, replace=True),
                        id='key',
                        left=dict(name=model_weights.name,
                                  computedvars='key',
                                  computedvarsprogram=keypgm),
                        right=dict(name=model_old_weights['name'],
                                   computedvars='key',
                                   computedvarsprogram=keypgm,
                                   vars=['_weight0_'])
                        )

    scale1 = 1 - damping_factor
    scale2 = damping_factor
    comp_weight = 'if missing(_weight0_) then do; _weight_=_weight1_; ' \
                  'end; else do; _weight_=_weight1_*{}+_weight0_*{};end;'.format(scale1, scale2)

    res = conn.retrieve('table.partition', _messagelevel='error',
                        table=dict(computedVars=['_Weight_'],
                                   computedvarsprogram=comp_weight,
                                   vars=['_layerid_', '_weightid_', '_weight_'],
                                   name=model_weights.name),
                        casout=dict(replace=True, name=model_weights.name))

    # attach the tbl attr
    res = conn.retrieve('table.attribute', _messagelevel='error',
                        task='ADD',
                        name=model_weights.name,
                        attrtable=model_weights.name + '_attr')

    return model_weights


class GANModel:
    input_layer_name_discriminator = 'InputLayer_Discriminator'
    input_layer_name_generator = 'InputLayer_Generator'
    output_layer_name = 'output_regression_layer'
    segmentation_layer_name = 'output_segmentation_layer'

    generated_tensor_layer_name = 'GAN_generation_layer'
    data_specs = None
    models = {}
    conn = None

    def __init__(self, generator, discriminator, encoder=None,
                 output_width=None, output_height=None, output_depth=None):
        '''

        Build an Generative Adversarial Network model based on a given generator and discriminator model branches

        Parameters
        ----------
        generator : Model
            Specifies the base model for generator.
        discriminator : Model
            Specifies the base model for discriminator.
        encoder : Model
            Specifies the base model for encoder to encode the generated images back to the latent space.
        output_width : int, optional
            Specifies the output tensor width from generator.
            When it is not given, the input width from discriminator will be used.
        output_height : int, optional
            Specifies the output tensor height from generator.
            When it is not given, the input height from discriminator will be used.
        output_depth : int, optional
            Specifies the output tensor depth from generator.
            When it is not given, the input depth from discriminator will be used.

        Returns
        -------
        :class: `GANModel`

        '''

        self.conn = generator.conn

        # check the model branch type
        if not isinstance(generator, Model):
            raise DLPyError('The generator option must contain a valid model')
        if not isinstance(discriminator, Model):
            raise DLPyError('The discriminator option must contain a valid model')

        # convert both into functional models
        if not hasattr(generator, 'output_layers'):
            print("NOTE: Convert the generator model into a functional model.")
            generator_tensor = generator.to_functional_model()
        else:
            generator_tensor = deepcopy(generator)

        if not hasattr(discriminator, 'output_layers'):
            print("NOTE: Convert the discriminator model into a functional model.")
            discriminator_tensor = discriminator.to_functional_model()
        else:
            discriminator_tensor = deepcopy(discriminator)

        if encoder:
            if not hasattr(encoder, 'output_layers'):
                print("NOTE: Convert the encoder model into a functional model.")
                encoder_tensor = encoder.to_functional_model()
            else:
                encoder_tensor = deepcopy(encoder)
            encoder_tensor.number_of_instances = 0
        else:
            encoder_tensor = None

        # add layer name prefix
        for layer in discriminator_tensor.layers:
            layer.name = 'discriminator_' + layer.name

        for layer in generator_tensor.layers:
            layer.name = 'generator_' + layer.name

        if encoder_tensor:
            for layer in encoder_tensor.layers:
                layer.name = 'encoder_' + layer.name

        generator_tensor.number_of_instances = 0
        discriminator_tensor.number_of_instances = 0

        # check the output layer for generator
        if len(generator_tensor.output_layers) != 1:
            raise DLPyError('The generator model cannot contain more than one output layer')
        elif generator_tensor.output_layers[0].type == OutputLayer.type or \
                generator_tensor.output_layers[0].type == Keypoints.type:
            print("NOTE: Remove the task layers from the generator model.")
            generator_tensor.layers.remove(generator_tensor.output_layers[0])
            generator_tensor.output_layers[0] = generator_tensor.layers[-1]
        elif generator_tensor.output_layers[0].can_be_last_layer:
            raise DLPyError('The generator model cannot contain task layer except output or keypoints layer.')

        # check the output layer for encoder
        if encoder_tensor:
            if len(encoder_tensor.output_layers) != 1:
                raise DLPyError('The encoder model cannot contain more than one output layer')
            elif encoder_tensor.output_layers[0].type == OutputLayer.type or \
                    encoder_tensor.output_layers[0].type == Keypoints.type:
                print("NOTE: Remove the task layers from the encoder model.")
                encoder_tensor.layers.remove(encoder_tensor.output_layers[0])
                encoder_tensor.output_layers[0] = encoder_tensor.layers[-1]
            elif encoder_tensor.output_layers[0].can_be_last_layer:
                raise DLPyError('The encoder model cannot contain task layer except output or keypoints layer.')

        # check the output layer for discriminator
        # note discriminator could have more task layers
        # for now, let us use the same restriction as generator
        if len(discriminator_tensor.output_layers) != 1:
            raise DLPyError('The discriminator model cannot contain more than one output layer')
        elif discriminator_tensor.output_layers[0].type == OutputLayer.type or \
                discriminator_tensor.output_layers[0].type == Keypoints.type:
            print("NOTE: Remove the task layers from the discriminator model.")
            discriminator_tensor.layers.remove(discriminator_tensor.output_layers[0])
            discriminator_tensor.output_layers[0] = discriminator_tensor.layers[-1]
        elif discriminator_tensor.output_layers[0].can_be_last_layer:
            raise DLPyError('The discriminator model cannot contain task layer except output or keypoints layer.')

        discriminator_tensor_for_generator = deepcopy(discriminator_tensor)

        # check the output size
        if output_width is None:
            output_width = discriminator_tensor.layers[0].output_size[0]

        if output_height is None:
            output_height = discriminator_tensor.layers[0].output_size[1]

        if output_depth is None:
            output_depth = discriminator_tensor.layers[0].output_size[2]

        # check whether the last tensor size from generator matches the above output size
        if isinstance(generator_tensor.layers[-1].tensor.shape, tuple):
            output_size = generator_tensor.layers[-1].tensor.shape[0] * generator_tensor.layers[-1].tensor.shape[1] * \
                          generator_tensor.layers[-1].tensor.shape[2]
        else:
            output_size = generator_tensor.layers[-1].tensor.shape
        if output_width * output_height * output_depth != output_size:
            raise DLPyError('The output size ({}) from the last layer of the generator model does not match '
                            'the required width ({}), height ({}), and depth({}).'.
                            format(output_size,
                                   output_width, output_height, output_depth))

        # construct the discriminator model
        # scale pixels to (-1, 1)
        discriminator_tensor.layers[0].config['scale'] = 1.0
        discriminator_tensor.layers[0].config['offsets'] = [255 / 2, 255 / 2, 255 / 2]
        discriminator_tensor.layers[0].config['norm_stds'] = [255 / 2, 255 / 2, 255 / 2]
        temp_input_layer = Input(**discriminator_tensor.layers[0].config, name=self.input_layer_name_discriminator)
        temp_branch_d = discriminator_tensor(temp_input_layer)  # return a list of tensors
        # add the regression output layer that output a number between 0 and 1
        temp_output = OutputLayer(n=1, act='sigmoid', error='normal', name=self.output_layer_name)(temp_branch_d)
        discriminator_model = Model(discriminator.conn, temp_input_layer, temp_output)
        discriminator_model.compile()

        self.models['discriminator'] = discriminator_model

        # construct the generator model
        discriminator_tensor_for_generator.number_of_instances = 0
        temp_input_layer = Input(**generator_tensor.layers[0].config, name=self.input_layer_name_generator)
        temp_branch = generator_tensor(temp_input_layer)  # return a list of tensors
        # add reshape layer for image generation
        # tanh generates pixels from -1 to 1
        temp_reshape_branch = Reshape(name=self.generated_tensor_layer_name, act='IDENTITY',
                                      width=output_width, height=output_height, depth=output_depth)(temp_branch)
        temp_branch = discriminator_tensor_for_generator(temp_reshape_branch)

        # add encoder part
        if encoder_tensor:
            auto_encoder_branch = encoder_tensor(temp_reshape_branch)
            auto_encoder_branch = Segmentation(name=self.segmentation_layer_name)(auto_encoder_branch)

        # add the regression output layer that output a number between 0 and 1
        temp_output = OutputLayer(n=1, act='sigmoid', error='normal', name=self.output_layer_name)(temp_branch)
        if encoder_tensor:
            generator_model = Model(generator.conn, temp_input_layer, [temp_output, auto_encoder_branch])
        else:
            generator_model = Model(generator.conn, temp_input_layer, temp_output)
        generator_model.compile()

        self.models['generator'] = generator_model

        # build the freeze layer list for the generator model
        self.__freeze_layer_list = None
        for layer in generator_model.layers:
            if layer.name.find('discriminator_') == 0:
                if self.__freeze_layer_list is None:
                    self.__freeze_layer_list = [layer.name]
                else:
                    self.__freeze_layer_list.append(layer.name)
            elif layer.name == self.generated_tensor_layer_name:
                self.generated_tensor_layer_id = layer.layer_id

        self.__freeze_layer_list.append(self.output_layer_name)

        # init some attrs
        self.current_data_iter = -1
        self.generator_data_cas_table = None
        self.discriminator_data_cas_table = None
        self.output_width = output_width
        self.output_height = output_height
        self.output_depth = output_depth
        self.output_size = output_width * output_height * output_depth

        self.image_file_list = None
        self.real_image_casout = None
        self.fake_image_casout = None

        # get the input image size
        self.generator_input_depth = self.models['generator'].layers[0].config['n_channels']
        self.generator_input_width = self.models['generator'].layers[0].config['width']
        self.generator_input_height = self.models['generator'].layers[0].config['height']
        self.generator_input_size = self.generator_input_depth * self.generator_input_width * self.generator_input_height

        # data_specs
        gen_vars = []
        for i in range(0, self.generator_input_size):
            gen_vars.append('x_' + str(i))

        self.generator_data_specs = [
            DataSpec(type_='NUMNOM', layer=self.input_layer_name_generator, data=gen_vars),
            DataSpec(type_='NUMNOM', layer=self.output_layer_name, data=['_target_'])]

        self.discriminator_data_specs = [
            DataSpec(type_='IMAGE', layer=self.input_layer_name_discriminator, data=['_image_']),
            DataSpec(type_='NUMNOM', layer=self.output_layer_name, data=['_target_'])]

        if encoder_tensor:
            self.generator_data_specs.append(DataSpec(type_='NUMNOM', layer=self.segmentation_layer_name,
                                                      data_layer=self.input_layer_name_generator))

    def __del__(self):
        if self.generator_data_cas_table is not None:
            self.generator_data_cas_table.droptable(quiet=True)
        if self.discriminator_data_cas_table is not None:
            self.discriminator_data_cas_table.droptable(quiet=True)
        if self.real_image_casout is not None:
            self.real_image_casout.droptable(quiet=True)

    def fit(self, optimizer_generator,
            optimizer_discriminator, path_discriminator,
            n_samples_generator=512,
            n_samples_discriminator=256,
            resize_width=None, resize_height=None,
            max_iter=1,
            gpu=None, seed=0, record_seed=0,
            save_best_weights=False, n_threads=None,
            train_from_scratch=None, path_generator=None,
            damping_factor=None):

        """
        Fitting a deep learning model for GAN.

        Parameters
        ----------

        optimizer_generator : :class:`Optimizer`
            Specifies the parameters for the optimizer for the generator model.
        optimizer_discriminator : :class:`Optimizer`
            Specifies the parameters for the optimizer for the discriminator model.
        path_discriminator : string
            The path to the image directory on the server that contains the real images for discriminator.
            Path may be absolute, or relative to the current caslib root.
            A new sample of data will be randomly generated after the number of epochs defined in Optimizer.
            max_iter defines how many iterations the random sample will be generated.
        path_generator : string
            The path to the image directory on the server that contains the real images for generator.
            When the string is empty, the data for generator will be generated automatically.
            Path may be absolute, or relative to the current caslib root.
            A new sample of data will be randomly generated after the number of epochs defined in Optimizer.
            max_iter defines how many iterations the random sample will be generated.
        n_samples_generator: int, optional
            Number of samples for generator.
            Default: 512
        n_samples_discriminator: int, optional
            Number of samples for discriminator.
            Default: 256
        resize_width : int, optional
            Specifies the image width that needs be resized to. When resize_width is not given, it will be reset to
            the specified resize_height.
        resize_height : int, optional
            Specifies the image height that needs be resized to. When resize_height is not given, it will be reset to
            the specified resize_width.
        max_iter : int, optional
            Hard limit on iterations when randomly generating data.
            Default: 1
        gpu : :class:`Gpu`, optional
            When specified, the action uses graphical processing unit hardware.
            The simplest way to use GPU processing is to specify "gpu=1".
            In this case, the default values of other GPU parameters are used.
            Setting gpu=1 enables all available GPU devices for use. Setting
            gpu=0 disables GPU processing.
        seed : double, optional
            specifies the random number seed for the random number generator
            in SGD. The default value, 0, and negative values indicate to use
            random number streams based on the computer clock. Specify a value
            that is greater than 0 for a reproducible random number sequence.
        record_seed : double, optional
            specifies the random number seed for the random record selection
            within a worker. The default value 0 disables random record selection.
            Records are read as they are laid out in memory.
            Negative values indicate to use random number streams based on the
            computer clock.
        save_best_weights : bool, optional
            When set to True, it keeps the weights that provide the smallest
            loss error.
        n_threads : int, optional
            Specifies the number of threads to use. If nothing is set then
            all of the cores available in the machine(s) will be used.
        train_from_scratch : bool, optional
            When set to True, it ignores the existing weights and trains the model from the scratch.
        damping_factor : double, optional
            Specifies the ratio to mix the new weights with the previous weights.
            Default: None

        Returns
        --------
        :a dict of `CASResults` that contains the optimization history for both generator and discriminator

        """

        self.conn.loadactionset('datastep', _messagelevel='error')

        # check options

        res = {'generator': [], 'discriminator': []}
        time_start = time.time()

        for data_iter in range(0, max_iter):
            self.current_data_iter = data_iter

            print("***************************************")
            print("NOTE: generator optimization at {}th iteration".format(data_iter))
            print("***************************************")

            # optimize generator first. If discriminator has weights, use them. And freeze all discriminator weights
            res_t = self.optimize_generator(self, optimizer=optimizer_generator,
                                            path=path_generator,
                                            n_samples=n_samples_generator,
                                            gpu=gpu, seed=seed, record_seed=record_seed,
                                            save_best_weights=save_best_weights, n_threads=n_threads,
                                            train_from_scratch=train_from_scratch,
                                            damping_factor=damping_factor)
            res['generator'].append(res_t)

            # optimize discriminator. load the data from path, append the data from generator.
            print("***************************************")
            print("NOTE: discriminator optimization at {}th iteration".format(data_iter))
            print("***************************************")

            res_t = self.optimize_discriminator(self, optimizer=optimizer_discriminator,
                                                path=path_discriminator,
                                                n_samples=n_samples_discriminator,
                                                resize_width=resize_width, resize_height=resize_height,
                                                gpu=gpu, seed=seed, record_seed=record_seed,
                                                save_best_weights=save_best_weights, n_threads=n_threads,
                                                train_from_scratch=train_from_scratch)
            res['discriminator'].append(res_t)

        print('Note: Training with data generation took {} (s)'.format(time.time() - time_start))

        return res

    @staticmethod
    def scale_images(conn, name, columns, width, height, depth):
        code_str = 'data ' + name + ';\n' + 'set ' + name + ';\n'
        if columns is None:
            for i in range(0, width * height * depth):
                code_str += 'x_' + str(i) + '=round(x_' + str(i) + '*128 + 128);\n'
        else:
            for col in columns:
                code_str += col + '=round(' + col + '*128 + 128);\n'
        code_str += 'run;'
        # print(code_str)
        conn.runcode(single='Yes', code=code_str)

    def predict(self, n_samples=512, seed=0, gpu=None, buffer_size=10, use_best_weights=False, n_threads=None,
                layer_image_type='jpg', log_level=0):
        """
        Generate the fake images using the generator model

        Parameters
        ----------

        n_samples : int, optional
            Specifies the number of samples that will be generated.
            Default: 512
        seed : double, optional
            specifies the random number seed for the random number generator.
            The default value, 0, and negative values indicate to use
            random number streams based on the computer clock. Specify a value
            that is greater than 0 for a reproducible random number sequence.
        gpu : :class:`Gpu`, optional
            When specified, the action uses graphical processing
            unit hardware. The simplest way to use GPU processing is
            to specify "gpu=1". In this case, the default values of
            other GPU parameters are used. Setting gpu=1 enables all
            available GPU devices for use. Setting gpu=0 disables GPU
            processing.
        buffer_size : int, optional
            Specifies the number of observations to score in a single
            batch. Larger values use more memory.
            Default: 10
        use_best_weights : bool, optional
            When set to True, the weights that provides the smallest loss
            error saved during a previous training is used while scoring
            input data rather than the final weights from the training.
            default: False
        n_threads : int, optional
            Specifies the number of threads to use. If nothing is set then
            all of the cores available in the machine(s) will be used.
        layer_image_type : string, optional
            Specifies the image type to store in the output layers table.
            JPG means a compressed image (e.g, jpg, png, and tiff)
            WIDE means a pixel per column
            Default: jpg
            Valid Values: JPG, WIDE
        log_level : int, optional
            specifies the reporting level for progress messages sent to the client.
            The default level 0 indicates that no messages are sent.
            Setting the value to 1 sends start and end messages.
            Setting the value to 2 adds the iteration history to the client messaging.
            default: 0
        Returns
        -------
        :class:`CASResults`

        """
        if self.generator_data_cas_table is not None:
            self.generator_data_cas_table.droptable(quiet=True)
            self.generator_data_cas_table = None

        if self.fake_image_casout is not None:
            self.fake_image_casout.droptable(quiet=True)
            self.fake_image_casout = None

        self.generator_data_cas_table = self.generate_random_images(self.conn, n_samples, seed,
                                                                    self.generator_input_width,
                                                                    self.generator_input_height,
                                                                    self.generator_input_depth)
        generator_model = self.models['generator']
        layer_out_temp = dict(name=random_name())
        res_pred = generator_model.predict(data=self.generator_data_cas_table, layer_out=layer_out_temp['name'],
                                           layers=self.generated_tensor_layer_name, gpu=gpu, buffer_size=buffer_size,
                                           use_best_weights=use_best_weights, n_threads=n_threads,
                                           layer_image_type='wide', log_level=log_level)
        layer_out_temp = CASTable(**layer_out_temp)
        layer_out_temp.set_connection(self.conn)

        if layer_image_type.lower() == 'wide':
            self.fake_image_casout = layer_out_temp
            return self.fake_image_casout

        # need scale values from (-1, 1) to (0, 256)
        res_cols = self.conn.retrieve('table.columnInfo', _messagelevel='error',
                                      table=layer_out_temp.name)
        inputs_list = []
        for col in res_cols['ColumnInfo']['Column'].tolist():
            if "_LayerAct_" in col:
                inputs_list.append(col)

        self.scale_images(self.conn, layer_out_temp.name, inputs_list,
                          self.output_width, self.output_height, self.output_depth)

        # condense images

        self.fake_image_casout = dict(name=random_name())
        res = self.conn.retrieve('image.condenseImages', _messagelevel='error',
                                 table=layer_out_temp,
                                 casout=self.fake_image_casout,
                                 numberofchannels=self.output_depth,
                                 groupedChannels=True, width=self.output_width, height=self.output_height,
                                 inputs=inputs_list,
                                 decode=dict(value=False, encodeType='jpg'))

        layer_out_temp.droptable(quiet=True)

        self.fake_image_casout = CASTable(**self.fake_image_casout)
        self.fake_image_casout.set_connection(self.conn)
        return res_pred

    def deploy(self, path, output_format='astore'):
        """
        Deploy the deep learning model to a data file

        Parameters
        ----------
        path : string
            Specifies the location to store the model files.
            If the output_format is set to castable, then the location has to be on the server-side.
            Otherwise, the location has to be on the client-side.
        output_format : string, optional
            Specifies the format of the deployed model.
            When astore is specified, the learned embedding features will be output as well.
            Valid Values: astore, castable, or onnx
            Default: astore

        Notes
        -----
        Currently, this function supports sashdat, astore, and onnx formats.

        More information about ONNX can be found at: https://onnx.ai/

        DLPy supports ONNX version >= 1.3.0, and Opset version 8.

        For ONNX format, currently supported layers are convo, pool,
        fc, batchnorm, residual, concat, reshape, and detection.

        If dropout is specified in the model, train the model using
        inverted dropout, which can be specified in :class:`Optimizer`.
        This will ensure the results are correct when running the
        model during test phase.

        Returns
        --------
        :class:`Model` for a branch model when model_type is 'branch'

        """
        # TODO: not implemented yet
        pass

    @staticmethod
    def generate_random_images(conn, n_obs, seed, width, height, depth):
        name = random_name()
        code_str = 'data ' + name + '; call streaminit(' + str(seed) + '); '
        code_str += 'do i=1 to ' + str(n_obs) + ';'
        for i in range(0, width * height * depth):
            code_str += 'x_' + str(i) + '=rand("UNIFORM")*2-1;'
        code_str += '_target_=1; output; end; drop i; run;'
        # code_str += '_target_=rand("UNIFORM")*0.1+0.9; output; end; drop i; run;'
        conn.retrieve('datastep.runcode', _messagelevel='error', single='Yes', code=code_str)
        temp_table = CASTable(name)
        temp_table.set_connection(conn)
        return temp_table

    @staticmethod
    def optimize_generator(self, optimizer, path, n_samples, gpu, seed, record_seed, save_best_weights, n_threads,
                           train_from_scratch, damping_factor):

        model = self.models['generator']

        temp_old_weights = None

        # generate random data
        if self.generator_data_cas_table is not None:
            self.generator_data_cas_table.droptable(quiet=True)
            self.generator_data_cas_table = None

        time_start = time.time()
        self.generator_data_cas_table = self.generate_random_images(self.conn, n_samples,
                                                                    random() * 100 +
                                                                    self.current_data_iter,
                                                                    self.generator_input_width,
                                                                    self.generator_input_height,
                                                                    self.generator_input_depth)
        # if discriminator has the weights, load them
        model_discriminator = self.models['discriminator']

        if self.conn.tableExists(model.model_weights).exists and \
                self.conn.tableExists(model_discriminator.model_weights).exists:
            # attach weights
            weight_tbl = WeightsTable(self.conn, model_discriminator.model_weights.name, model_discriminator.model_name)
            model.set_weights(weight_tbl, True)

        # fit but freeze discriminator weights
        optimizer_t = deepcopy(optimizer)
        optimizer_t.__setitem__('freeze_layers', self.__freeze_layer_list)

        if self.current_data_iter == 0:
            train_from_scratch_real = train_from_scratch
        else:
            train_from_scratch_real = False
            # store the old weights
            if damping_factor is not None and model.model_weights:
                temp_old_weights = dict(name=random_name())
                res = self.conn.retrieve('table.partition', _messagelevel='error',
                                         casout=temp_old_weights,
                                         table=dict(name=model.model_weights.name,
                                                    where='_layerid_ < {}'.format(self.generated_tensor_layer_id))
                                         )

        # do not optimize generator at the first iteration since discriminator has no weights yet.
        if (self.current_data_iter == 0 and self.conn.tableExists(model_discriminator.model_weights).exists == 0) or \
                train_from_scratch_real:
            optimizer_t.__setitem__('max_epochs', 1)
            optimizer_t.algorithm.__setitem__('learningrate', 0.0)

        print("NOTE: time for generating samples and loading weights to train generator: {} (s)".
              format(time.time() - time_start))

        # freeze all bn stats
        # if self.current_data_iter > 1:
        #     optimizer_t.__setitem__('freeze_batch_norm_stats', True)
        #     print("NOTE: all BN layers are frozen.")

        res = model.fit(self.generator_data_cas_table, inputs=None, target=None,
                        data_specs=self.generator_data_specs,
                        optimizer=optimizer_t,
                        valid_table=None, valid_freq=None, gpu=gpu,
                        seed=seed, record_seed=record_seed,
                        force_equal_padding=True,
                        save_best_weights=save_best_weights, n_threads=n_threads,
                        target_order=None, train_from_scratch=train_from_scratch_real)

        # mix the weights
        # new weights = weights * damping_factor + old_weights*(1-damping_factor)
        if temp_old_weights:
            time_start = time.time()
            model.model_weights = mix_weights_with_fedsql(self.conn, model.model_weights, temp_old_weights, damping_factor)
            self.conn.retrieve('table.dropTable', _messagelevel='error',
                               table=temp_old_weights['name'])
            print("NOTE: time for mixing weights: {} (s)".
                  format(time.time() - time_start))

        return res

    @staticmethod
    def optimize_discriminator(self, optimizer, path, n_samples, resize_width, resize_height,
                               gpu, seed, record_seed, save_best_weights, n_threads,
                               train_from_scratch):

        model = self.models['discriminator']

        if self.discriminator_data_cas_table is not None:
            self.discriminator_data_cas_table.droptable(quiet=True)
            self.discriminator_data_cas_table = None

        # load real images from path
        n_real_images = n_samples / 2
        if self.real_image_casout is not None:
            self.real_image_casout.droptable(quiet=True)
            self.real_image_casout = None

        time_start = time.time()
        self.real_image_casout = self.load_real_images(self, path, caslib=None, n_samples=n_real_images,
                                                       label_level=None,
                                                       resize_width=resize_width, resize_height=resize_height)

        # generate fake images using generator

        n_fake_images = n_samples - n_real_images
        self.predict(n_samples=n_fake_images, seed=seed, gpu=gpu, buffer_size=None, use_best_weights=None,
                     n_threads=n_threads, layer_image_type='jpg', log_level=None)

        # append the table to generate the final training data
        # generate _target_ with 1 for real images
        # generate _target_ with 0 for fake images
        real_image_casout_with_target = dict(name=self.real_image_casout.name,
                                             vars=['_image_', '_target_'],
                                             computedvars='_target_',
                                             computedvarsprogram='_target_ = 1;')
        fake_image_casout_with_target = dict(name=self.fake_image_casout.name,
                                             vars=['_image_', '_target_'],
                                             computedvars='_target_',
                                             computedvarsprogram='_target_ = 0;')
        # dlJoin cannot work with computedVars and copy the data first
        res = self.conn.retrieve('table.partition', _messagelevel='error',
                                 casout=dict(name=self.real_image_casout.name, replace=True),
                                 table=real_image_casout_with_target
                                 )
        res = self.conn.retrieve('table.partition', _messagelevel='error',
                                 casout=dict(name=self.fake_image_casout.name, replace=True),
                                 table=fake_image_casout_with_target
                                 )
        self.discriminator_data_cas_table = dict(name=random_name())
        res = self.conn.retrieve('deepLearn.dlJoin', _messagelevel='error',
                                 casout=self.discriminator_data_cas_table,
                                 joinType='append',
                                 left=self.real_image_casout,
                                 right=self.fake_image_casout
                                 )
        # shuffle this table
        res = self.conn.retrieve('table.shuffle', _messagelevel='error',
                                 table=self.discriminator_data_cas_table,
                                 casout=dict(replace=True, blocksize=32, **self.discriminator_data_cas_table))

        self.discriminator_data_cas_table = CASTable(self.discriminator_data_cas_table['name'])
        self.discriminator_data_cas_table.set_connection((self.conn))

        # fit. if discriminator has the weights, load them
        if self.current_data_iter == 0:
            train_from_scratch_real = train_from_scratch
        else:
            train_from_scratch_real = False

        print("NOTE: time for generating samples and loading images to train discriminator: {} (s)".
              format(time.time() - time_start))

        res = model.fit(self.discriminator_data_cas_table, inputs=None, target=None,
                        data_specs=self.discriminator_data_specs,
                        optimizer=optimizer,
                        valid_table=None, valid_freq=None, gpu=gpu,
                        seed=seed, record_seed=record_seed,
                        force_equal_padding=True,
                        save_best_weights=save_best_weights, n_threads=n_threads,
                        target_order=None, train_from_scratch=train_from_scratch_real)

        return res

    @staticmethod
    def load_real_images(self, path, caslib=None,
                         n_samples=512,
                         label_level=None, resize_width=None, resize_height=None):

        conn = self.conn
        conn.loadactionset('image', _messagelevel='error')
        conn.loadactionset('sampling', _messagelevel='error')
        conn.loadactionset('deepLearn', _messagelevel='error')

        if label_level is None:
            real_label_level = -2
        else:
            real_label_level = label_level

        # check resize options
        if resize_width is not None and resize_height is None:
            resize_height = resize_width
        if resize_width is None and resize_height is not None:
            resize_width = resize_height

        # ignore the unreasonable values for resize
        if resize_width is not None and resize_width <= 0:
            resize_width = None
            resize_height = None

        if resize_height is not None and resize_height <= 0:
            resize_width = None
            resize_height = None

        with caslibify_context(conn, path, task='load') as (caslib_created, path_created):

            if caslib is None:
                caslib = caslib_created
                path = path_created

            if caslib is None and path is None:
                print('Cannot create a caslib for the provided path. Please make sure that the path is accessible from'
                      'the CAS Server. Please also check if there is a subpath that is part of an existing caslib')

            # load the path information for all the files
            if self.image_file_list is None:
                castable_with_file_list = dict(name=random_name())
                conn.retrieve('table.loadTable', _messagelevel='error',
                              casout=castable_with_file_list,
                              importOptions=dict(fileType='image', contents=False, recurse=True),
                              path=path_created, caslib=caslib)
                # download all the file information to a dataframe
                n_obs = conn.retrieve('simple.numRows', _messagelevel='error',
                                      table=castable_with_file_list)

                res_fetch = conn.retrieve('table.fetch', _messagelevel='error', maxRows=n_obs['numrows'] + 100,
                                          fetchVars=['_path_'], to=n_obs['numrows'], table=castable_with_file_list)

                # this stores the entire file path information
                self.image_file_list = res_fetch['Fetch']['_path_']
                # generate the list using labels as keys
                self.image_file_list_with_labels = {}
                for file in self.image_file_list:
                    label = os.path.normpath(file).split(os.sep)[real_label_level]
                    if label in self.image_file_list_with_labels:
                        self.image_file_list_with_labels[label] = \
                            self.image_file_list_with_labels[label].append(pd.Series([file]))
                    else:
                        self.image_file_list_with_labels[label] = pd.Series([file])

                conn.retrieve('table.dropTable', _messagelevel='error',
                              quiet=True,
                              table=castable_with_file_list['name'])

                # check whether the image file contains the correct labels
                if len(self.image_file_list_with_labels) == 1 and label_level is not None:
                    raise DLPyError('Only one class {} is present in the image files. This could be caused by '
                                    'the wrong labels generated by label_level or the data '
                                    'is highly imbalanced.'.format(self.image_file_list_with_labels))

            # randomly select n_sample files
            # do stratified sampling
            real_image_file_list = []
            for label in self.image_file_list_with_labels:
                n_samples_per_label = \
                    round(n_samples * (len(self.image_file_list_with_labels[label]) / len(self.image_file_list)))
                if n_samples_per_label == 0:
                    n_samples_per_label = 1
                sample_list = self.image_file_list_with_labels[label].sample(n=n_samples_per_label, replace=True)
                sample_list = sample_list.tolist()
                real_image_file_list += sample_list

            real_images_casout = self.__load_images_with_the_file_list(conn, path, caslib, real_image_file_list,
                                                                       resize_width, resize_height)

        out = CASTable(**real_images_casout)
        out.set_connection(conn)

        return out

    @staticmethod
    def __load_images_with_the_file_list(conn, path, caslib, file_list, resize_width, resize_height):

        # change the message level
        # upload_frame somehow does not honor _messagelevel
        current_msg_level = conn.getSessOpt(name='messagelevel')
        conn.setSessOpt(messageLevel='ERROR')

        # upload file_list
        file_list_casout = dict(name=random_name())

        # use relative path with respect to caslib
        file_list_relative_path = pd.Series()
        i_tot = 0
        for index, value in enumerate(file_list):
            pos = value.find(path)
            # file_list_relative_path = file_list_relative_path.set_value(i_tot, value[pos:])
            file_list_relative_path.at[i_tot] = value[pos:]
            i_tot = i_tot + 1
        conn.upload_frame(file_list_relative_path.to_frame(), casout=file_list_casout, _messagelevel='error')

        # save the file list
        conn.retrieve('table.save', _messagelevel='error',
                      table=file_list_casout,
                      name=file_list_casout['name'] + '.csv', caslib=caslib)

        conn.retrieve('table.dropTable', _messagelevel='error',
                      quiet=True,
                      table=file_list_casout['name'])

        # load images based on the csv file
        # we need use the absolute path here since .csv stores the absolute file location
        images_casout = dict(name=random_name())
        # caslib_info = conn.retrieve('caslibinfo', _messagelevel='error',
        #                            caslib=caslib)
        # full_path = caslib_info['CASLibInfo']['Path'][0] + file_list_casout['name'] + '.csv'
        # relative to caslib
        csv_path = file_list_casout['name'] + '.csv'

        # use relative paths
        conn.retrieve('image.loadimages', _messagelevel='error',
                      casout=images_casout,
                      caslib=caslib,
                      recurse=True,
                      path=csv_path,
                      pathIsList=True)

        # remove the csv file
        conn.deleteSource(source=file_list_casout['name'] + '.csv', caslib=caslib, _messagelevel='error')

        # resize when it is specified
        if resize_width is not None:
            conn.retrieve('image.processImages', _messagelevel='error',
                          imagefunctions=[
                              {'options': {
                                  'functiontype': 'RESIZE',
                                  'width': resize_width,
                                  'height': resize_height
                              }}
                          ],
                          casout=dict(name=images_casout['name'], replace=True),
                          table=images_casout)

        # reset msg level
        conn.setSessOpt(messageLevel=current_msg_level['messageLevel'])

        return images_casout

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

from copy import deepcopy

from dlpy import Model, ImageTable
from dlpy.layers import Input, EmbeddingLoss, OutputLayer, Keypoints, Reshape
from .attribute_utils import create_extended_attributes
from .image_embedding import ImageEmbeddingTable
from .model import DataSpec
from .network import WeightsTable
from dlpy.utils import DLPyError


class GANModel:
    input_layer_name_discriminator = 'InputLayer_Discriminator'
    input_layer_name_generator = 'InputLayer_Generator'
    output_layer_name = 'output_regression_layer'

    generated_tensor_layer_name = 'GAN_generation_layer'
    data_specs = None
    models = {}

    @classmethod
    def build_gan_model(cls, generator, discriminator,
                        output_width=None, output_height=None, output_depth=None):
        '''

        Build an Generative Adversarial Network model based on a given generator and discriminator model branches

        Parameters
        ----------
        generator : Model
            Specifies the base model for generator.
        discriminator : Model
            Specifies the base model for discriminator.
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
        :a dict that contains a list of models. Keys are discriminator and generator

        '''

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

        # check the output size
        if output_width is None:
            output_width = discriminator_tensor.layers[0].output_size[0]

        if output_height is None:
            output_height = discriminator_tensor.layers[0].output_size[1]

        if output_depth is None:
            output_depth = discriminator_tensor.layers[0].output_size[2]

        # check whether the last tensor size from generator matches the above output size
        if output_width * output_height * output_depth != \
                (generator_tensor.layers[-1].output_size[0] * generator_tensor.layers[-1].output_size[1] *
                 generator_tensor.layers[-1].output_size[2]):
            raise DLPyError('The output size ({}, {}, {}) from the last layer of the generator model does not match '
                            'the required width ({}), height ({}), and depth({}).'.
                            format(generator_tensor.layers[-1].output_size[0],
                                   generator_tensor.layers[-1].output_size[1],
                                   generator_tensor.layers[-1].output_size[2],
                                   output_width, output_height, output_depth))

        # construct the discriminator model
        temp_input_layer = Input(**discriminator_tensor.layers[0].config, name=cls.input_layer_name_discriminator)
        temp_branch_d = discriminator_tensor(temp_input_layer)  # return a list of tensors
        # add the regression output layer that output a number between 0 and 1
        temp_output = OutputLayer(n=1, act='sigmoid', error='normal', name=cls.output_layer_name)(temp_branch_d)
        discriminator_model = Model(discriminator.conn, temp_input_layer, temp_output)
        discriminator_model.compile()

        cls.models['discriminator'] = discriminator_model

        # construct the generator model
        temp_input_layer = Input(**generator_tensor.layers[0].config, name=cls.input_layer_name_generator)
        temp_branch = generator_tensor(temp_input_layer)  # return a list of tensors
        # add reshape layer for image generation
        temp_branch = Reshape(name=cls.generated_tensor_layer_name,
                              width=output_width, height=output_height, depth=output_depth)(temp_branch)
        temp_branch = discriminator_tensor(temp_branch)
        # add the regression output layer that output a number between 0 and 1
        temp_output = OutputLayer(n=1, act='sigmoid', error='normal')(temp_branch)
        generator_model = Model(generator.conn, temp_input_layer, temp_output)
        generator_model.compile()

        cls.models['generator'] = generator_model

        # let the generator model use the weights from discriminator and fix them

        return cls()

    @classmethod
    def fit_gan_model(self, optimizer,
                      data=None, path=None, n_samples=512,
                      resize_width=None, resize_height=None,
                      max_iter=1,
                      gpu=None, seed=0, record_seed=0,
                      save_best_weights=False, n_threads=None,
                      train_from_scratch=None):

        """
        Fitting a deep learning model for GAN.

        Parameters
        ----------

        optimizer : :class:`Optimizer`
            Specifies the parameters for the optimizer.
        data : class:`ImageEmbeddingTable`, optional
            This is the input data. It muse be a ImageEmbeddingTable object. Either data or path has to be specified.
        path : string, optional
            The path to the image directory on the server.
            Path may be absolute, or relative to the current caslib root.
            when path is specified, the data option will be ignored.
            A new sample of data will be randomly generated after the number of epochs defined in Optimizer.
            max_iter defines how many iterations the random sample will be generated.
        n_samples : int, optional
            Number of samples to generate.
            Default: 512
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
            When set to True, it ignores the existing weights and trains the model from the scracth.

        Returns
        --------
        :class:`CASResults` or a list of `CASResults` when the path option is specified

        """

        # check options
        if data is None and path is None:
            raise DLPyError('Either the data option or path must be specified to generate the input data')

        if data is not None and path is not None:
            print('Note: the data option will be ignored and the path option will be used to generate the input '
                  'data')

        # check the data type
        if path is None:
            if not isinstance(data, ImageTable):
                raise DLPyError('The data option must contain a valid image table')

    @classmethod
    def deploy_gan_model(self, path, output_format='astore'):
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

        pass

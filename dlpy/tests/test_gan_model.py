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

# NOTE: This test requires a running CAS server.  You must use an ~/.authinfo
#       file to specify your username and password.  The CAS host and port must
#       be specified using the CASHOST and CASPORT environment variables.
#       A specific protocol ('cas', 'http', 'https', or 'auto') can be set using
#       the CASPROTOCOL environment variable.

import os
import unittest
import swat
import swat.utils.testing as tm

from dlpy.applications import ResNet18_Caffe, DLPyError, Model, Sequential
from dlpy.gan_model import GANModel
from dlpy.layers import Conv2D, Input, OutputLayer, InputLayer, Conv2d, Pooling, Dense
from dlpy.lr_scheduler import StepLR


class TestGANModel(unittest.TestCase):
    # Create a class attribute to hold the cas host type
    server_type = None
    s = None
    server_sep = '/'
    data_dir = None

    @classmethod
    def setUpClass(cls):
        swat.reset_option()
        swat.options.cas.print_messages = False
        swat.options.interactive_mode = False

        cls.s = swat.CAS()
        cls.server_type = tm.get_cas_host_type(cls.s)

        cls.server_sep = '\\'
        if cls.server_type.startswith("lin") or cls.server_type.startswith("osx"):
            cls.server_sep = '/'

        if 'DLPY_DATA_DIR' in os.environ:
            cls.data_dir = os.environ.get('DLPY_DATA_DIR')
            if cls.data_dir.endswith(cls.server_sep):
                cls.data_dir = cls.data_dir[:-1]
            cls.data_dir += cls.server_sep

        if "DLPY_DATA_DIR_LOCAL" in os.environ:
            cls.local_dir = os.environ.get("DLPY_DATA_DIR_LOCAL")

        # the server path that points to DLPY_DATA_DIR_LOCAL
        if "DLPY_DATA_DIR_SERVER" in os.environ:
            cls.server_dir = os.environ.get("DLPY_DATA_DIR_SERVER")
            if cls.server_dir.endswith(cls.server_sep):
                cls.server_dir = cls.server_dir[:-1]
            cls.server_dir += cls.server_sep

    @classmethod
    def tearDownClass(cls):
        # tear down tests
        try:
            cls.s.terminate()
        except swat.SWATError:
            pass
        del cls.s
        swat.reset_option()

    def test_build_gan_model(self):

        if self.server_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_SERVER is not set in the environment variables")

        # test default
        resnet18_model = ResNet18_Caffe(self.s,
                                        width=224,
                                        height=224,
                                        random_flip='HV',
                                        random_mutation='random'
                                        )
        branch = resnet18_model.to_functional_model(stop_layers=resnet18_model.layers[-1])

        # raise error
        self.assertRaises(DLPyError, lambda: GANModel(branch, branch))

        # change the output size for generator
        inp = Input(**branch.layers[0].config)
        generator = Conv2D(width=1, height=1, n_filters=224*224*3)(branch(inp))
        output = OutputLayer(n=1)(generator)
        generator = Model(self.s, inp, output)
        gan_model = GANModel(generator, branch)
        res = gan_model.models['generator'].print_summary()
        print(res)
        res = gan_model.models['discriminator'].print_summary()
        print(res)

    def test_build_gan_model_1(self):

        if self.server_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_SERVER is not set in the environment variables")

        discriminator = Sequential(self.s)
        discriminator.add(InputLayer(1, 28, 28))
        discriminator.add(Conv2d(3, 3))
        discriminator.add(Pooling(2))
        discriminator.add(Conv2d(3, 3))
        discriminator.add(Pooling(2))
        discriminator.add(Dense(16))
        discriminator.add(OutputLayer(n=1))

        generator = Sequential(self.s)
        generator.add(InputLayer(1, 28, 28))
        generator.add(Conv2d(3, 3))
        generator.add(Pooling(2))
        generator.add(Conv2d(3, 3))
        generator.add(Pooling(2))
        generator.add(Dense(16))
        generator.add(Dense(28 * 28, act='tanh'))
        generator.add(OutputLayer(n=1))

        gan_model = GANModel(generator, discriminator)

        res = gan_model.models['generator'].print_summary()
        print(res)
        res = gan_model.models['discriminator'].print_summary()
        print(res)

        from dlpy.model import Optimizer, MomentumSolver, AdamSolver
        solver = AdamSolver(lr_scheduler=StepLR(learning_rate=0.0001, step_size=4), clip_grad_max=100,
                            clip_grad_min=-100)
        optimizer = Optimizer(algorithm=solver, mini_batch_size=8, log_level=2, max_epochs=8, reg_l2=0.0001)

        res = gan_model.fit(optimizer, '', optimizer, self.server_dir+'mnist_validate',
                            n_samples=32, max_iter=2, n_threads=1)
        print(res)

    def test_build_gan_model_2(self):

        if self.server_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_SERVER is not set in the environment variables")

        discriminator = Sequential(self.s)
        discriminator.add(InputLayer(1, 28, 28))
        discriminator.add(Conv2d(3, 3))
        discriminator.add(Pooling(2))
        discriminator.add(Conv2d(3, 3))
        discriminator.add(Pooling(2))
        discriminator.add(Dense(16))
        discriminator.add(OutputLayer(n=1))

        generator = Sequential(self.s)
        generator.add(InputLayer(100, 1, 1))
        generator.add(Dense(256, act='relu'))
        generator.add(Dense(512, act='relu'))
        generator.add(Dense(1024, act='relu'))
        generator.add(Dense(28 * 28, act='tanh'))
        generator.add(OutputLayer(act='softmax', n=2))

        gan_model = GANModel(generator, discriminator)

        res = gan_model.models['generator'].print_summary()
        print(res)
        res = gan_model.models['discriminator'].print_summary()
        print(res)

        from dlpy.model import Optimizer, MomentumSolver, AdamSolver
        solver = AdamSolver(lr_scheduler=StepLR(learning_rate=0.0001, step_size=4), clip_grad_max=100,
                            clip_grad_min=-100)
        optimizer = Optimizer(algorithm=solver, mini_batch_size=8, log_level=2, max_epochs=4, reg_l2=0.0001)

        res = gan_model.fit(optimizer, '', optimizer, self.server_dir+'mnist_validate',
                            n_samples=32, max_iter=2, n_threads=1)
        print(res)

    def test_build_gan_model_3(self):

        if self.server_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_SERVER is not set in the environment variables")

        discriminator = Sequential(self.s)
        discriminator.add(InputLayer(1, 28, 28))
        discriminator.add(Conv2d(3, 3))
        discriminator.add(Pooling(2))
        discriminator.add(Conv2d(3, 3))
        discriminator.add(Pooling(2))
        discriminator.add(Dense(16))
        discriminator.add(OutputLayer(n=1))

        generator = Sequential(self.s)
        generator.add(InputLayer(1, 100, 1))
        generator.add(Dense(256, act='relu'))
        generator.add(Dense(512, act='relu'))
        generator.add(Dense(1024, act='relu'))
        generator.add(Dense(28 * 28, act='tanh'))
        generator.add(OutputLayer(act='softmax', n=2))

        encoder = Sequential(self.s)
        encoder.add(InputLayer(100, 1, 1))
        encoder.add(Dense(256, act='relu'))
        encoder.add(Dense(512, act='relu'))
        encoder.add(Dense(1024, act='relu'))
        encoder.add(Dense(100, act='tanh'))
        encoder.add(OutputLayer(act='softmax', n=2))

        gan_model = GANModel(generator, discriminator, encoder)

        res = gan_model.models['generator'].print_summary()
        print(res)

        res = gan_model.models['discriminator'].print_summary()
        print(res)

        from dlpy.model import Optimizer, MomentumSolver, AdamSolver
        solver = AdamSolver(lr_scheduler=StepLR(learning_rate=0.0001, step_size=4), clip_grad_max=100,
                             clip_grad_min=-100)
        optimizer = Optimizer(algorithm=solver, mini_batch_size=8, log_level=2, max_epochs=4, reg_l2=0.0001)

        res = gan_model.fit(optimizer, '', optimizer, self.server_dir+'mnist_validate',
                             n_samples=32, max_iter=2, n_threads=1)
        print(res)

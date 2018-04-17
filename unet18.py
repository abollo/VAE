
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os
import time
import matplotlib as mpl
import seaborn as sns

from matplotlib import pyplot as plt
import tensorflow.contrib.slim as slim
from collections import namedtuple
from .ops import conv2d, deconv2d, lrelu, fc, batch_norm, init_embedding, conditional_instance_norm
from .dataset import TrainDataProvider, InjectDataProvider, NeverEndingLoopingProvider
from .utils import scale_back, merge, save_concat_images
distributions = tf.distributions
# Auxiliary wrapper classes
# Used to save handles(important nodes in computation graph) for later evaluation
LossHandle = namedtuple("LossHandle", ["l1_loss","kl_loss","T_loss"])
InputHandle = namedtuple("InputHandle", ["real_data", "embedding_ids", "no_target_data", "no_target_ids"])
EvalHandle = namedtuple("EvalHandle", [ "generator2","target", "source", "embedding","gaussian_params"])
SummaryHandle = namedtuple("SummaryHandle", ["T_sum"])


class UNet(object):
    def __init__(self, experiment_dir=None, experiment_id=0, batch_size=16, input_width=256, output_width=256,
                 generator_dim=32, generator_dim2=32,discriminator_dim=64, L1_penalty=100, Lconst_penalty=15, Ltv_penalty=0.0,
                 Lcategory_penalty=1.0, embedding_num=40, embedding_dim=128, input_filters=3, output_filters=3,latent_dim=256):
        self.experiment_dir = experiment_dir
        self.experiment_id = experiment_id
        self.batch_size = batch_size
        self.input_width = input_width
        self.output_width = output_width
        self.generator_dim = generator_dim
        self.generator_dim2 = generator_dim2
        self.discriminator_dim = discriminator_dim
        self.L1_penalty = L1_penalty
        self.Lconst_penalty = Lconst_penalty
        self.Ltv_penalty = Ltv_penalty
        self.Lcategory_penalty = Lcategory_penalty
        self.embedding_num = embedding_num
        self.embedding_dim = embedding_dim
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.latent_dim = latent_dim
        # init all the directories
        self.sess = None
        # experiment_dir is needed for training
        if experiment_dir:
            self.data_dir = os.path.join(self.experiment_dir, "data")
            self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoint")
            self.sample_dir = os.path.join(self.experiment_dir, "sample")
            self.log_dir = os.path.join(self.experiment_dir, "logs")

            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
                print("create checkpoint directory")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
                print("create log directory")
            if not os.path.exists(self.sample_dir):
                os.makedirs(self.sample_dir)
                print("create sample directory")

    def gaussion_encoder(self, images, encoding_layers,is_training, reuse=False):
        with tf.variable_scope("generator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            def encode_layer(x, output_filters, layer,enc_layer):
                act = lrelu(x)
                conv = conv2d(act, output_filters=output_filters, scope="gaussion_e%d" % layer)
                enc = batch_norm(conv, is_training, scope="gaussion_e%d_bn" % layer)
                dec = tf.concat([enc, enc_layer], 3)
                #encode_layers["e%d" % layer] = enc
                return dec
            images1=lrelu(images)

            images2 = conv2d(images1, self.generator_dim2, scope="gaussion_e1")
            e11 = batch_norm(images2, is_training, scope="gaussion_images2")
            e1=tf.concat([e11,encoding_layers["e1"]],3)
            e2 = encode_layer(e1, self.generator_dim2 * 2, 2,enc_layer=encoding_layers["e2"])
            e3 = encode_layer(e2, self.generator_dim2 * 4, 3,enc_layer=encoding_layers["e3"])
            e3 = tf.nn.max_pool(e3,[1,2,2,1],[1,1,1,1],padding="SAME")
            e4 = encode_layer(e3, self.generator_dim2 * 8, 4,enc_layer=encoding_layers["e4"])
            e5 = encode_layer(e4, self.generator_dim2 * 8, 5,enc_layer=encoding_layers["e5"])
            e6 = encode_layer(e5, self.generator_dim2 * 8, 6,enc_layer=encoding_layers["e6"])
            e6 = tf.nn.max_pool(e6, [1, 2, 2, 1], [1, 1, 1, 1], padding="SAME")
            e7 = encode_layer(e6, self.generator_dim2 * 8, 7,enc_layer=encoding_layers["e7"])
            e8 = encode_layer(e7, self.generator_dim2 * 8, 8,enc_layer=encoding_layers["e8"])
            #print(e1.get_shape(),e2.get_shape(),e3.get_shape(),e4.get_shape(),e5.get_shape(),e6.get_shape(),e7.get_shape(),e8.get_shape())
            gaussian_shape = e8.get_shape().as_list()
            nodes = gaussian_shape[1] * gaussian_shape[2] * gaussian_shape[3]
            e9 = tf.reshape(e8, [-1, nodes])
            # print(e8.get_shape())
            gaussian_params = fc(e9, self.latent_dim * 2, stddev=0.02, scope="gaussion_fc")
            #gaussian_params_act=lrelu(gaussian_params)
            return gaussian_params






    def encoder(self, images, is_training, reuse=False):
        with tf.variable_scope("generator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            encode_layers = dict()

            def encode_layer(x, output_filters, layer):
                act = lrelu(x)
                conv = conv2d(act, output_filters=output_filters, scope="g_e%d_conv" % layer)
                enc = batch_norm(conv, is_training, scope="g_e%d_bn" % layer)
                encode_layers["e%d" % layer] = enc
                return enc

            e1 = conv2d(images, self.generator_dim, scope="g_e1_conv")
            encode_layers["e1"] = e1
            e2 = encode_layer(e1, self.generator_dim * 2, 2)
            e3 = encode_layer(e2, self.generator_dim * 4, 3)
            e4 = encode_layer(e3, self.generator_dim * 8, 4)
            e5 = encode_layer(e4, self.generator_dim * 8, 5)
            e6 = encode_layer(e5, self.generator_dim * 8, 6)
            e7 = encode_layer(e6, self.generator_dim * 8, 7)
            e8 = encode_layer(e7, self.generator_dim * 8, 8)
            #print(e1.get_shape(),e2.get_shape(),e3.get_shape(),e4.get_shape(),e5.get_shape(),e6.get_shape(),e7.get_shape(),e8.get_shape())
            return e8, encode_layers




    def decoder(self, encoded, encoding_layers, ids, inst_norm, is_training, reuse=False):
        with tf.variable_scope("generator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            s = self.output_width
            s2, s4, s8, s16, s32, s64, s128 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), int(
                s / 64), int(s / 128)

            def decode_layer(x, output_width, output_filters, layer, enc_layer, dropout=False, do_concat=True):
                # The mean parameter is unconstrained

                dec = deconv2d(x, [self.batch_size, output_width,
                                               output_width, output_filters], scope="gaussion_g_d%d_deconv" % layer)
                #print(dec.get_shape())
                if layer != 8:
                    # IMPORTANT: normalization for last layer
                    # Very important, otherwise GAN is unstable
                    # Trying conditional instance normalization to
                    # overcome the fact that batch normalization offers
                    # different train/test statistics
                    if inst_norm:
                        dec = conditional_instance_norm(dec, ids, self.embedding_num, scope="gaussion_g_d%d_inst_norm" % layer)
                    else:
                        dec = batch_norm(dec, is_training, scope="gaussion_g_d%d_bn" % layer)
                if dropout:
                    dec = tf.nn.dropout(dec, 0.5)
                if do_concat:
                    dec = tf.concat([dec, enc_layer], 3)
                    #print(dec.get_shape())
                return dec

            d1 = decode_layer(encoded, s128, self.generator_dim * 8, layer=1, enc_layer=encoding_layers["e7"],
                              dropout=True)
            d2 = decode_layer(d1, s64, self.generator_dim * 8, layer=2, enc_layer=encoding_layers["e6"], dropout=True)
            d3 = decode_layer(d2, s32, self.generator_dim * 8, layer=3, enc_layer=encoding_layers["e5"], dropout=True)
            d4 = decode_layer(d3, s16, self.generator_dim * 8, layer=4, enc_layer=encoding_layers["e4"])
            d5 = decode_layer(d4, s8, self.generator_dim * 4, layer=5, enc_layer=encoding_layers["e3"])
            d6 = decode_layer(d5, s4, self.generator_dim * 2, layer=6, enc_layer=encoding_layers["e2"])
            d7 = decode_layer(d6, s2, self.generator_dim, layer=7, enc_layer=encoding_layers["e1"])
            d8 = decode_layer(d7, s, self.output_filters, layer=8, enc_layer=None, do_concat=False)

            output = tf.nn.tanh(d8)  # scale to (-1, 1)
            #print(d1.get_shape(), d2.get_shape(), d3.get_shape(), d4.get_shape(), d5.get_shape(), d6.get_shape(),
            #      d7.get_shape(), d8.get_shape())
            return output


    def e_decoder(self,encoder,reuse=False):

        q_mu = encoder[:, :self.latent_dim]
        # The standard deviation must be positive. Parametrize with a softplus
        q_sigma = tf.nn.softplus(encoder[:, self.latent_dim:])
        q_z = distributions.Normal(loc=q_mu, scale=q_sigma)
        q_z_sample = q_z.sample()
        # print(q_z_sample.get_shape())
        q_z_decode = tf.reshape(q_z_sample, [-1, 1, 1, self.latent_dim])
        # q_z_decode2 = tf.nn.relu(q_z_decode)
        # print(q_z_decode.get_shape())
        return q_z_decode











    def generator_gaussian(self, enc_layers, images_target, embedding_ids, inst_norm, is_training, reuse=False):

        e8_gaussion = self.gaussion_encoder(images_target, enc_layers, is_training=is_training, reuse=reuse)
        e8_2 = self.e_decoder(e8_gaussion, reuse=reuse)
        # local_embeddings = tf.nn.embedding_lookup(embeddings, ids=embedding_ids)
        # local_embeddings = tf.reshape(local_embeddings, [self.batch_size, 1, 1, self.embedding_dim])
        embedded = tf.concat([enc_layers["e8"], e8_2], 3)
        # print(e8.get_shape())
        # print(embedded.get_shape().as_list())
        output = self.decoder(embedded, enc_layers, embedding_ids, inst_norm, is_training=is_training, reuse=reuse)
        return output, e8_gaussion, e8_2



    def discriminator(self, image, is_training, reuse=False):
        with tf.variable_scope("discriminator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            h0 = lrelu(conv2d(image, self.discriminator_dim, scope="d_h0_conv"))
            h1 = lrelu(batch_norm(conv2d(h0, self.discriminator_dim * 2, scope="d_h1_conv"),
                                  is_training, scope="d_bn_1"))
            h2 = lrelu(batch_norm(conv2d(h1, self.discriminator_dim * 4, scope="d_h2_conv"),
                                  is_training, scope="d_bn_2"))
            h3 = lrelu(batch_norm(conv2d(h2, self.discriminator_dim * 8, sh=1, sw=1, scope="d_h3_conv"),
                                  is_training, scope="d_bn_3"))
            # real or fake binary loss
            fc1 = fc(tf.reshape(h3, [self.batch_size, -1]), 1, scope="d_fc1")
            # category loss
            fc2 = fc(tf.reshape(h3, [self.batch_size, -1]), self.embedding_num, scope="d_fc2")

            return tf.nn.sigmoid(fc1), fc1, fc2




    def build_model(self, is_training=True, inst_norm=False, no_target_source=False):

        real_data = tf.placeholder(tf.float32,
                                   [self.batch_size, self.input_width, self.input_width,
                                    self.input_filters + self.output_filters],
                                   name='real_A_and_B_images')
        embedding_ids = tf.placeholder(tf.int64, shape=None, name="embedding_ids")
        no_target_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.input_width, self.input_width,
                                         self.input_filters + self.output_filters],
                                        name='no_target_A_and_B_images')
        no_target_ids = tf.placeholder(tf.int64, shape=None, name="no_target_embedding_ids")

        # target images
        real_A = real_data[:, :, :, :self.input_filters]
        # source images
        real_B = real_data[:, :, :, self.input_filters:self.input_filters + self.output_filters]
        real_A_shuffle=tf.random_shuffle(real_A)
        embedding = init_embedding(self.embedding_num, self.embedding_dim)

        #fake_target, target_gaussian1,target_gaussian2,source_e8,layers_source = self.generator(real_B, real_A, embedding_ids, is_training=is_training,inst_norm=inst_norm, reuse=False)

        source_e8, layers_source = self.encoder(real_B,is_training=is_training,reuse=False)



        target_gaussian1_shuffle = self.gaussion_encoder(real_A_shuffle, layers_source, is_training=is_training,
                                                         reuse=False)

        real_A_shuffle2=tf.random_shuffle(real_A_shuffle)
        target_gaussian2_shuffle = self.gaussion_encoder(real_A_shuffle2, layers_source, is_training=is_training,
                                                      reuse=True)

        real_A_shuffle3=tf.random_shuffle(real_A_shuffle2)
        target_gaussian3_shuffle = self.gaussion_encoder(real_A_shuffle3, layers_source, is_training=is_training,
                                                         reuse=True)

        real_A_shuffle4=tf.random_shuffle(real_A_shuffle3)

        target_gaussian4_shuffle = self.gaussion_encoder(real_A_shuffle4, layers_source, is_training=is_training,
                                                         reuse=True)

        real_A_shuffle5 = tf.random_shuffle(real_A_shuffle4)

        target_gaussian5_shuffle = self.gaussion_encoder(real_A_shuffle5, layers_source, is_training=is_training,
                                                         reuse=True)

        e8_gaussion_sum=tf.div(tf.add(tf.add(tf.add(tf.add(target_gaussian1_shuffle,target_gaussian2_shuffle),target_gaussian3_shuffle),target_gaussian4_shuffle),target_gaussian5_shuffle),5.0)

        #print(e8_gaussion_sum.get_shape())
        #print(target_gaussian1_shuffle.get_shape())

        e8_2_sum = self.e_decoder(e8_gaussion_sum, reuse=False)
        # local_embeddings = tf.nn.embedding_lookup(embeddings, ids=embedding_ids)
        # local_embeddings = tf.reshape(local_embeddings, [self.batch_size, 1, 1, self.embedding_dim])
        embedded_sum = tf.concat([layers_source["e8"], e8_2_sum], 3)
        # print(e8.get_shape())
        # print(embedded.get_shape().as_list())
        fake_target_shuffle = self.decoder(embedded_sum, layers_source, embedding_ids, inst_norm, is_training=is_training, reuse=False)




        source_fake_e8, layers_source_fake = self.encoder(fake_target_shuffle, is_training=is_training,reuse=True)


        target_gaussian1_fake=self.gaussion_encoder(fake_target_shuffle, layers_source,is_training=is_training,reuse=True)






        #real_AB = tf.concat([real_A, real_B], 3)
        #fake_AB = tf.concat([real_A, fake_B], 3)

        # Note it is not possible to set reuse flag back to False
        # initialize all variables before setting reuse to True
        #real_D, real_D_logits, real_category_logits = self.discriminator(real_AB, is_training=is_training, reuse=False)
        #fake_D, fake_D_logits, fake_category_logits = self.discriminator(fake_AB, is_training=is_training, reuse=True)

        # encoding constant loss
        # this loss assume that generated imaged and real image
        # should reside in the same space and close to each other
        #encoded_fake_B = self.encoder(fake_B, is_training, reuse=True)[0]
        #const_loss = (tf.reduce_mean(tf.square(encoded_real_A - encoded_fake_B))) * self.Lconst_penalty

        # category loss
        #true_labels = tf.reshape(tf.one_hot(indices=embedding_ids, depth=self.embedding_num),
        #                         shape=[self.batch_size, self.embedding_num])
        #real_category_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=target_gaussian1_fake,
        #                                                                            labels=target_gaussian1))

        real_category_loss=tf.reduce_sum(tf.abs(target_gaussian1_fake[:, :self.latent_dim] - target_gaussian1_shuffle[:, :self.latent_dim]))
        #fake_category_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_category_logits,
        #                                                                            labels=true_labels))
        #category_loss = self.Lcategory_penalty * (real_category_loss + fake_category_loss)

        # binary real/fake loss
        #d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_D_logits,
        #                                                                     labels=tf.ones_like(real_D)))
        #d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D_logits,
        #                                                                     labels=tf.zeros_like(fake_D)))
        # L1 loss between real and generated images
        l1_loss = self.L1_penalty * tf.reduce_mean(tf.abs(fake_target_shuffle - real_A),[1,2,3])


        # total variation loss
        #width = self.output_width
        #tv_loss = (tf.nn.l2_loss(fake_B[:, 1:, :, :] - fake_B[:, :width - 1, :, :]) / width
        #           + tf.nn.l2_loss(fake_B[:, :, 1:, :] - fake_B[:, :, :width - 1, :]) / width) * self.Ltv_penalty

        # maximize the chance generator fool the discriminator
        #cheat_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D_logits,
        #                                                                    labels=tf.ones_like(fake_D)))

        #d_loss = d_loss_real + d_loss_fake + category_loss / 2.0

        q_z = distributions.Normal(loc=target_gaussian1_shuffle[:, :self.latent_dim],
                                   scale=tf.nn.softplus(target_gaussian1_shuffle[:, self.latent_dim:]))



        # print(output.get_shape())
        p_z = distributions.Normal(loc=np.zeros(self.latent_dim, dtype=np.float32),
                                   scale=np.ones(self.latent_dim, dtype=np.float32))



        kl_loss = tf.reduce_sum(distributions.kl_divergence(q_z, p_z), 1)


        kl_loss_1 = tf.reduce_sum(tf.reduce_sum(distributions.kl_divergence(q_z, p_z), 1))


        #const_loss = (tf.reduce_mean(tf.reduce_sum(tf.square(lay_AB["e1"] - lay_A["e1"]))+tf.reduce_sum(tf.square(lay_AB["e2"] - lay_A["e2"]))+tf.reduce_sum(tf.square(lay_AB["e3"] - lay_A["e3"]))+tf.reduce_sum(tf.square(lay_AB["e4"] - lay_A["e4"]))+tf.reduce_sum(tf.square(lay_AB["e5"] - lay_A["e5"]))+tf.reduce_sum(tf.square(lay_AB["e6"] - lay_A["e6"]))+tf.reduce_sum(tf.square(lay_AB["e7"] - lay_A["e7"])))) * self.Lconst_penalty

        const_loss1 = tf.reduce_mean(tf.square(layers_source_fake["e8"] - layers_source["e8"]),[1,2,3])
        const_loss2 = 10.0*tf.reduce_mean(tf.square(layers_source_fake["e7"] - layers_source["e7"]),[1,2,3])
        const_loss3 = 5.0*tf.reduce_mean(tf.square(layers_source_fake["e6"] - layers_source["e6"]),[1,2,3])
        const_loss4 = tf.reduce_mean(tf.square(layers_source_fake["e5"] - layers_source["e5"]),[1,2,3])
        const_loss=tf.reduce_sum(const_loss1+const_loss2+const_loss3+const_loss4)*self.Lconst_penalty
        const_loss=(tf.reduce_sum(const_loss1))*self.Lconst_penalty
        T_loss =  tf.reduce_sum(l1_loss +kl_loss)







        #d_loss_real_summary = tf.summary.scalar("d_loss_real", d_loss_real)
        #d_loss_fake_summary = tf.summary.scalar("d_loss_fake", d_loss_fake)
        #category_loss_summary = tf.summary.scalar("category_loss", category_loss)
        #cheat_loss_summary = tf.summary.scalar("cheat_loss", cheat_loss)
        #l1_loss_summary = tf.summary.scalar("l1_loss", l1_loss)
        real_category_loss_summary = tf.summary.scalar("real_category_loss", tf.reduce_sum(real_category_loss))
        #const_loss_summary = tf.summary.scalar("const_loss", const_loss)
        #d_loss_summary = tf.summary.scalar("d_loss", d_loss)
        T_loss_summary = tf.summary.scalar("T_loss", T_loss)

        l1_loss_summary = tf.summary.scalar("l1_loss", tf.reduce_sum(l1_loss))


        kl_loss_summary = tf.summary.scalar("kl_loss", kl_loss_1)

        #kl_loss_fake_AB_summary = tf.summary.scalar("kl_loss_fake_AB", kl_loss_fake_AB_1)
        const_loss_summary = tf.summary.scalar("const_loss_loss", const_loss)
        All_merged_summary = tf.summary.merge([l1_loss_summary,T_loss_summary,kl_loss_summary,const_loss_summary,real_category_loss_summary])
        #tv_loss_summary = tf.summary.scalar("tv_loss", tv_loss)

        #d_merged_summary = tf.summary.merge([d_loss_real_summary, d_loss_fake_summary,
        #                                     category_loss_summary, d_loss_summary])
        #g_merged_summary = tf.summary.merge([cheat_loss_summary, l1_loss_summary,
        #                                     fake_category_loss_summary,
        #                                     const_loss_summary,
        #                                     g_loss_summary, tv_loss_summary])

        # expose useful nodes in the graph as handles globally
        input_handle = InputHandle(real_data=real_data,
                                   embedding_ids=embedding_ids,
                                   no_target_data=no_target_data,
                                   no_target_ids=no_target_ids)

        loss_handle = LossHandle(
                                 T_loss=T_loss,
                                 #const_loss=const_loss,
                                 kl_loss=kl_loss,
                                 l1_loss=l1_loss
                                 #category_loss=category_loss,
                                 #cheat_loss=cheat_loss,
                                 )

        eval_handle = EvalHandle(
                                 generator2=fake_target_shuffle,
                                 target=real_A,
                                 source=real_B,
                                 embedding=embedding,
                                 gaussian_params=target_gaussian1_shuffle)

        summary_handle = SummaryHandle(T_sum=All_merged_summary)

        # those operations will be shared, so we need
        # to make them visible globally
        setattr(self, "input_handle", input_handle)
        setattr(self, "loss_handle", loss_handle)
        setattr(self, "eval_handle", eval_handle)
        setattr(self, "summary_handle", summary_handle)

    def register_session(self, sess):
        self.sess = sess

    def retrieve_trainable_vars(self, freeze_encoder=False):
        all_vars = tf.global_variables()
        #t_vars = tf.trainable_variables()

        # d_vars = [var for var in t_vars if 'd_' in var.name]
        #g_vars = [var for var in t_vars if 'g_' in var.name]
        gaussian_vars = [var for var in all_vars if 'gaussion_' in var.name]
        # if freeze_encoder:
        #     # exclude encoder weights
        #     print("freeze encoder weights")
        #     g_vars = [var for var in g_vars if not ("g_e" in var.name)]
        #
        # return g_vars, d_vars
        return all_vars,gaussian_vars



    def retrieve_generator_vars(self):
        all_vars = tf.global_variables()
        #generate_vars = [var for var in all_vars if 'embedding' in var.name or "g_" in var.name]
        #return generate_vars
        return all_vars

    def retrieve_handles(self):
        input_handle = getattr(self, "input_handle")
        loss_handle = getattr(self, "loss_handle")
        eval_handle = getattr(self, "eval_handle")
        summary_handle = getattr(self, "summary_handle")

        return input_handle, loss_handle, eval_handle, summary_handle

    def get_model_id_and_dir(self):
        model_id = "experiment_%d_batch_%d" % (self.experiment_id, self.batch_size)
        model_dir = os.path.join(self.checkpoint_dir, model_id)
        return model_id, model_dir

    def checkpoint(self, saver, step):
        model_name = "unet.model"
        model_id, model_dir = self.get_model_id_and_dir()

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        saver.save(self.sess, os.path.join(model_dir, model_name), global_step=step)

    def restore_model(self, saver, model_dir):

        ckpt = tf.train.get_checkpoint_state(model_dir)

        if ckpt:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("restored model %s" % model_dir)
        else:
            print("fail to restore model %s" % model_dir)

    def generate_fake_samples(self, input_images, embedding_ids):
        input_handle, loss_handle, eval_handle, summary_handle = self.retrieve_handles()
        fake_images, real_images, target_images,\
        T_loss, kl_loss, l1_loss ,gaussian_params= self.sess.run([eval_handle.generator2,
                                                 eval_handle.source,
                                                 eval_handle.target,
                                                 loss_handle.T_loss,
                                                 loss_handle.kl_loss,
                                                 loss_handle.l1_loss,
                                                 eval_handle.gaussian_params],
                                                feed_dict={
                                                    input_handle.real_data: input_images,
                                                    input_handle.embedding_ids: embedding_ids,
                                                    input_handle.no_target_data: input_images,
                                                    input_handle.no_target_ids: embedding_ids
                                                })
        return fake_images, real_images, target_images,T_loss, kl_loss, l1_loss,gaussian_params

    def validate_model(self, val_iter, epoch, step):
        labels, images = next(val_iter)
        fake_imgs, real_imgs, target_imgs,T_loss, kl_loss, l1_loss,gaussian_params = self.generate_fake_samples(images, labels)
        print("Sample: T_loss: %.5f" % T_loss)
        #print(gaussian_params.get_shape())
        merged_real_images = merge(scale_back(real_imgs), [self.batch_size, 1])
        merged_fake_images = merge(scale_back(fake_imgs), [self.batch_size, 1])
        merged_target_images = merge(scale_back(target_imgs), [self.batch_size, 1])
        merged_pair = np.concatenate([ merged_fake_images,merged_real_images, merged_target_images], axis=1)

        model_id, _ = self.get_model_id_and_dir()

        model_sample_dir = os.path.join(self.sample_dir, model_id)
        if not os.path.exists(model_sample_dir):
            os.makedirs(model_sample_dir)

        sample_img_path = os.path.join(model_sample_dir, "sample_%02d_%04d.png" % (epoch, step))
        misc.imsave(sample_img_path, merged_pair)

    def export_generator(self, save_dir, model_dir, model_name="gen_model"):
        saver = tf.train.Saver()
        self.restore_model(saver, model_dir)

        gen_saver = tf.train.Saver(var_list=self.retrieve_generator_vars())
        gen_saver.save(self.sess, os.path.join(save_dir, model_name), global_step=0)

    def infer(self, source_obj, embedding_ids, model_dir, save_dir):
        source_provider = InjectDataProvider(source_obj)

        if isinstance(embedding_ids, int) or len(embedding_ids) == 1:
            embedding_id = embedding_ids if isinstance(embedding_ids, int) else embedding_ids[0]
            source_iter = source_provider.get_single_embedding_iter(self.batch_size, embedding_id)
        else:
            source_iter = source_provider.get_random_embedding_iter(self.batch_size, embedding_ids)

        tf.global_variables_initializer().run()
        saver = tf.train.Saver(var_list=self.retrieve_generator_vars())
        self.restore_model(saver, model_dir)

        def save_imgs(imgs, count):
            p = os.path.join(save_dir, "inferred_%04d.png" % count)
            save_concat_images(imgs, img_path=p)
            print("generated images saved at %s" % p)

        count = 0
        batch_buffer = list()
        for labels, source_imgs in source_iter:
            fake_imgs = self.generate_fake_samples(source_imgs, labels)[0]
            merged_fake_images = merge(scale_back(fake_imgs), [self.batch_size, 1])
            batch_buffer.append(merged_fake_images)
            if len(batch_buffer) == 10:
                save_imgs(batch_buffer, count)
                batch_buffer = list()
            count += 1
        if batch_buffer:
            # last batch
            save_imgs(batch_buffer, count)

    def interpolate(self, source_obj, between, model_dir, save_dir, steps):
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(var_list=self.retrieve_generator_vars())
        self.restore_model(saver, model_dir)
        # new interpolated dimension
        new_x_dim = steps + 1
        alphas = np.linspace(0.0, 1.0, new_x_dim)

        def _interpolate_tensor(_tensor):
            """
            Compute the interpolated tensor here
            """

            x = _tensor[between[0]]
            y = _tensor[between[1]]

            interpolated = list()
            for alpha in alphas:
                interpolated.append(x * (1. - alpha) + alpha * y)

            interpolated = np.asarray(interpolated, dtype=np.float32)
            return interpolated

        def filter_embedding_vars(var):
            var_name = var.name
            if var_name.find("embedding") != -1:
                return True
            if var_name.find("inst_norm/shift") != -1 or var_name.find("inst_norm/scale") != -1:
                return True
            return False

        embedding_vars = filter(filter_embedding_vars, tf.trainable_variables())
        # here comes the hack, we overwrite the original tensor
        # with interpolated ones. Note, the shape might differ

        # this is to restore the embedding at the end
        embedding_snapshot = list()
        for e_var in embedding_vars:
            val = e_var.eval(session=self.sess)
            self.append = embedding_snapshot.append((e_var, val))
            t = _interpolate_tensor(val)
            op = tf.assign(e_var, t, validate_shape=False)
            print("overwrite %s tensor" % e_var.name, "old_shape ->", e_var.get_shape(), "new shape ->", t.shape)
            self.sess.run(op)

        source_provider = InjectDataProvider(source_obj)
        input_handle, _, eval_handle, _ = self.retrieve_handles()
        for step_idx in range(len(alphas)):
            alpha = alphas[step_idx]
            print("interpolate %d -> %.4f + %d -> %.4f" % (between[0], 1. - alpha, between[1], alpha))
            source_iter = source_provider.get_single_embedding_iter(self.batch_size, 0)
            batch_buffer = list()
            count = 0
            for _, source_imgs in source_iter:
                count += 1
                labels = [step_idx] * self.batch_size
                generated, = self.sess.run([eval_handle.generator],
                                           feed_dict={
                                               input_handle.real_data: source_imgs,
                                               input_handle.embedding_ids: labels
                                           })
                merged_fake_images = merge(scale_back(generated), [self.batch_size, 1])
                batch_buffer.append(merged_fake_images)
            if len(batch_buffer):
                save_concat_images(batch_buffer,
                                   os.path.join(save_dir, "frame_%02d_%02d_step_%02d.png" % (
                                       between[0], between[1], step_idx)))
        # restore the embedding variables
        print("restore embedding values")
        for var, val in embedding_snapshot:
            op = tf.assign(var, val, validate_shape=False)
            self.sess.run(op)

    def train(self, lr=0.0002, epoch=100, schedule=10, resume=True, flip_labels=False,
              freeze_encoder=False, fine_tune=None, sample_steps=50, checkpoint_steps=500):
        #g_vars, d_vars = self.retrieve_trainable_vars(freeze_encoder=freeze_encoder)
        all_vars,_ = self.retrieve_trainable_vars(freeze_encoder=freeze_encoder)
        _, gaussian_vars =self.retrieve_trainable_vars(freeze_encoder=freeze_encoder)
        input_handle, loss_handle, eval_handle, summary_handle = self.retrieve_handles()

        if not self.sess:
            raise Exception("no session registered")

        learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        #d_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss_handle.d_loss, var_list=d_vars)
        #g_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss_handle.g_loss, var_list=g_vars)
        all_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss_handle.T_loss, var_list=all_vars)



        #all_optimizer= tf.train.RMSPropOptimizer(learning_rate).minimize(loss_handle.T_loss, var_list=all_vars)
        tf.global_variables_initializer().run()
        real_data = input_handle.real_data
        embedding_ids = input_handle.embedding_ids
        no_target_data = input_handle.no_target_data
        no_target_ids = input_handle.no_target_ids

        # filter by one type of labels
        data_provider = TrainDataProvider(self.data_dir, filter_by=fine_tune)
        total_batches = data_provider.compute_total_batch_num(self.batch_size)
        val_batch_iter = data_provider.get_val_iter(self.batch_size)

        saver = tf.train.Saver(max_to_keep=3)
        summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        if resume:
            _, model_dir = self.get_model_id_and_dir()
            self.restore_model(saver, model_dir)

        current_lr = lr
        counter = 0
        start_time = time.time()

        for ei in range(epoch):
            gaussian1=list()
            gaussian2=list()
            labels_plot=list()
            train_batch_iter = data_provider.get_train_iter(self.batch_size)

            if (ei + 1) % schedule == 0:
                update_lr = current_lr / 2.0
                # minimum learning rate guarantee
                update_lr = max(update_lr, 0.0002)
                print("decay learning rate from %.5f to %.5f" % (current_lr, update_lr))
                current_lr = update_lr

            for bid, batch in enumerate(train_batch_iter):
                counter += 1
                labels, batch_images = batch
                shuffled_ids = labels[:]
                if flip_labels:
                    np.random.shuffle(shuffled_ids)

                _, l1_loss_run, kl_loss_run, T_loss_run,T_sum_run,gaussian_params= self.sess.run([all_optimizer,loss_handle.l1_loss, loss_handle.kl_loss,loss_handle.T_loss,summary_handle.T_sum,eval_handle.gaussian_params],feed_dict={
                                                               real_data: batch_images,
                                                               embedding_ids: labels,
                                                               learning_rate: current_lr,
                                                               no_target_data: batch_images,
                                                               no_target_ids: shuffled_ids
                                                           })



                passed = time.time() - start_time
                gaussian1.extend(gaussian_params[:,0])
                gaussian2.extend(gaussian_params[:,1])
                labels_plot.extend(labels)

                #print(gaussian_params)
                log_format = "Epoch: [%2d], [%4d/%4d] time: %4.4f, T_loss: %.5f"
                print(log_format % (ei, bid, total_batches, passed, T_loss_run))
                #summary_writer.add_summary(l1_sum_run, counter)
                #summary_writer.add_summary(kl_sum_run, counter)
                summary_writer.add_summary(T_sum_run, counter)

                if counter % sample_steps == 0:
                    # sample the current model states with val data
                    cmap = mpl.colors.ListedColormap(sns.color_palette("husl"))
                    f, ax = plt.subplots(1, figsize=(6 * 1.1618, 6))
                    im = ax.scatter(gaussian1[:], gaussian2[:], c=labels_plot[:], cmap=cmap,
                                    alpha=0.7)
                    ax.set_xlabel('First dimension of sampled latent variable $z_1$')
                    ax.set_ylabel('Second dimension of sampled latent variable mean $z_2$')
                    ax.set_xlim([-5., 5.])
                    ax.set_ylim([-5., 5.])
                    f.colorbar(im, ax=ax, label='Digit class')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.log_dir,
                                             'experiment_%d_batch_%d_%2d_%d.png' % (self.experiment_id, self.batch_size,ei,counter)))
                    plt.close()



                    self.validate_model(val_batch_iter, ei, counter)

                if counter % checkpoint_steps == 0:
                    print("Checkpoint: save checkpoint step %d" % counter)
                    self.checkpoint(saver, counter)
        # save the last checkpoint
        print("Checkpoint: last checkpoint step %d" % counter)
        self.checkpoint(saver, counter)

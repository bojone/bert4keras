# -*- coding: utf-8 -*-
# 优化相关

import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, K, is_tf_keras
from bert4keras.snippets import is_string, string_matching
from bert4keras.snippets import is_one_of, insert_arguments
from bert4keras.backend import piecewise_linear
from bert4keras.backend import root_mean_square as rms
import re


class Adam(keras.optimizers.Optimizer):
    """重新定义Adam优化器，便于派生出新的优化器
    （tensorflow的optimizer_v2类）
    """
    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        bias_correction=True,
        **kwargs
    ):
        kwargs['name'] = kwargs.get('name') or 'Adam'
        super(Adam, self).__init__(**kwargs)
        self._set_hyper('learning_rate', learning_rate)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon or K.epislon()
        self.bias_correction = bias_correction

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
            self.add_slot(var, 'v')

    def _resource_apply(self, grad, var, indices=None):
        # 准备变量
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        epsilon_t = K.cast(self.epsilon, var_dtype)
        local_step = K.cast(self.iterations + 1, var_dtype)
        beta_1_t_power = K.pow(beta_1_t, local_step)
        beta_2_t_power = K.pow(beta_2_t, local_step)

        # 更新公式
        if indices is None:
            m_t = K.update(m, beta_1_t * m + (1 - beta_1_t) * grad)
            v_t = K.update(v, beta_2_t * v + (1 - beta_2_t) * K.square(grad))
        else:
            mv_ops = [K.update(m, beta_1_t * m), K.update(v, beta_2_t * v)]
            with tf.control_dependencies(mv_ops):
                m_t = self._resource_scatter_add(
                    m, indices, (1 - beta_1_t) * grad
                )
                v_t = self._resource_scatter_add(
                    v, indices, (1 - beta_2_t) * K.square(grad)
                )

        # 返回算子
        with tf.control_dependencies([m_t, v_t]):
            if self.bias_correction:
                m_t = m_t / (1.0 - beta_1_t_power)
                v_t = v_t / (1.0 - beta_2_t_power)
            var_t = var - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)
            return K.update(var, var_t)

    def _resource_apply_dense(self, grad, var):
        return self._resource_apply(grad, var)

    def _resource_apply_sparse(self, grad, var, indices):
        return self._resource_apply(grad, var, indices)

    def get_config(self):
        config = {
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'bias_correction': self.bias_correction,
        }
        base_config = super(Adam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AdaFactorBase(keras.optimizers.Optimizer):
    """AdaFactor优化器（基类）
    论文链接：https://arxiv.org/abs/1804.04235
    参考实现：https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/optimize.py
    """
    def __init__(
        self,
        learning_rate=1e-3,  # 可以为None
        beta1=0.0,
        beta2=None,
        epsilon1=1e-30,
        epsilon2=1e-3,
        multiply_by_parameter_scale=True,
        clipping_threshold=1.0,
        min_dim_size_to_factor=128,
        **kwargs
    ):
        super(AdaFactorBase, self).__init__(**kwargs)
        self._learning_rate = learning_rate
        self.beta1 = beta1
        self._beta2 = beta2
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.multiply_by_parameter_scale = multiply_by_parameter_scale
        self.clipping_threshold = clipping_threshold
        self.min_dim_size_to_factor = min_dim_size_to_factor

    @property
    def learning_rate(self):
        if self._learning_rate is None:
            iterations = K.cast(self.iterations + 1, K.floatx())
            learning_rate = K.minimum(1.0 / K.sqrt(iterations), 0.01)
            if self.multiply_by_parameter_scale:
                return learning_rate
            else:
                return learning_rate * 0.05
        else:
            if not hasattr(self, '__learning_rate'):
                with K.name_scope(self.__class__.__name__):
                    self.__learning_rate = K.variable(
                        self._learning_rate, name='learning_rate'
                    )
            return self.__learning_rate

    @property
    def beta2(self):
        if self._beta2 is None:
            iterations = K.cast(self.iterations + 1, K.floatx())
            return 1.0 - K.pow(iterations, -0.8)
        else:
            return self._beta2

    def factored_shape(self, shape):
        if len(shape) < 2:
            return None
        shape = np.array(shape)
        indices = shape.argpartition(-2)
        if shape[indices[-2]] < self.min_dim_size_to_factor:
            return None
        shape1, shape2 = np.array(shape), np.array(shape)
        shape1[indices[-1]] = 1
        shape2[indices[-2]] = 1
        return shape1, indices[-1], shape2, indices[-2]

    def get_config(self):
        config = {
            'learning_rate': self._learning_rate,
            'beta1': self.beta1,
            'beta2': self._beta2,
            'epsilon1': self.epsilon1,
            'epsilon2': self.epsilon2,
            'multiply_by_parameter_scale': self.multiply_by_parameter_scale,
            'clipping_threshold': self.clipping_threshold,
            'min_dim_size_to_factor': self.min_dim_size_to_factor,
        }
        base_config = super(AdaFactorBase, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AdaFactorV1(AdaFactorBase):
    """AdaFactor优化器（纯Keras版）
    论文链接：https://arxiv.org/abs/1804.04235
    参考实现：https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/optimize.py
    """
    def __init__(self, *args, **kwargs):
        super(AdaFactorV1, self).__init__(*args, **kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')

    @K.symbolic
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        self.weights = [self.iterations]
        lr = self.learning_rate

        for i, (p, g) in enumerate(zip(params, grads)):
            g2 = K.square(g) + self.epsilon1  # 如果换成g**2，在keras下Embedding层会报错
            shape, dtype = K.int_shape(p), K.dtype(p)
            factored_shape = self.factored_shape(shape)
            if factored_shape is None:
                # 定义参数
                v = K.zeros(shape, dtype=dtype, name='v_' + str(i))
                self.weights.append(v)
                # 定义更新
                v_t = self.beta2 * v + (1.0 - self.beta2) * g2
                self.updates.append(K.update(v, v_t))
            else:
                # 定义参数
                shape1, axis1, shape2, axis2 = factored_shape
                vr = K.zeros(shape1, dtype=dtype, name='vr_' + str(i))
                vc = K.zeros(shape2, dtype=dtype, name='vc_' + str(i))
                self.weights.extend([vr, vc])
                # 定义更新
                g2r = K.mean(g2, axis=axis1, keepdims=True)
                g2c = K.mean(g2, axis=axis2, keepdims=True)
                vr_t = self.beta2 * vr + (1.0 - self.beta2) * g2r
                vc_t = self.beta2 * vc + (1.0 - self.beta2) * g2c
                self.updates.extend([K.update(vr, vr_t), K.update(vc, vc_t)])
                # 合成矩阵
                v_t = vr_t * vc_t / K.mean(vr_t, axis=axis2, keepdims=True)
            # 增量主体
            u = g / K.sqrt(v_t + self.epsilon1)
            # 增量裁剪
            if self.clipping_threshold is not None:
                u = u / K.maximum(1.0, rms(u) / self.clipping_threshold)
            # 增量滑动
            if self.beta1 > 0.0:
                # 定义参数
                m = K.zeros(shape, dtype=dtype, name='m_' + str(i))
                self.weights.append(m)
                # 定义更新
                m_t = self.beta1 * m + (1.0 - self.beta1) * u
                self.updates.append(K.update(m, m_t))
                u = m_t
            # 增量调整
            if self.multiply_by_parameter_scale:
                u = u * K.maximum(rms(p), self.epsilon2)
            # 更新参数
            self.updates.append(K.update(p, p - lr * u))

        return self.updates


class AdaFactorV2(AdaFactorBase):
    """AdaFactor优化器（tf.keras版）
    论文链接：https://arxiv.org/abs/1804.04235
    参考实现：https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/optimize.py
    """
    def __init__(self, *args, **kwargs):
        kwargs['name'] = kwargs.get('name') or 'AdaFactor'
        super(AdaFactorV2, self).__init__(*args, **kwargs)

    def _create_slots(self, var_list):
        for var in var_list:
            if self.beta1 > 0.0:
                self.add_slot(var, 'm')
            shape = K.int_shape(var)
            factored_shape = self.factored_shape(shape)
            if factored_shape is None:
                self.add_slot(var, 'v')
            else:
                shape1, axis1, shape2, axis2 = factored_shape
                value1, value2 = np.zeros(shape1), np.zeros(shape2)
                self.add_slot(var, 'vr', value1)
                self.add_slot(var, 'vc', value2)

    def _decayed_lr(self, var_dtype):
        return self.learning_rate

    def _resource_apply(self, grad, var, indices=None):
        lr = self._decayed_lr(var.dtype.base_dtype)
        g2 = K.square(grad) + self.epsilon1
        shape = K.int_shape(var)
        factored_shape = self.factored_shape(shape)
        if factored_shape is None:
            v = self.get_slot(var, 'v')
            # 定义更新
            v_t = self.beta2 * v + (1.0 - self.beta2) * g2
            v_t = K.update(v, v_t)
        else:
            shape1, axis1, shape2, axis2 = factored_shape
            vr = self.get_slot(var, 'vr')
            vc = self.get_slot(var, 'vc')
            # 定义更新
            g2r = K.mean(g2, axis=axis1, keepdims=True)
            g2c = K.mean(g2, axis=axis2, keepdims=True)
            vr_t = self.beta2 * vr + (1.0 - self.beta2) * g2r
            vc_t = self.beta2 * vc + (1.0 - self.beta2) * g2c
            vr_t, vc_t = K.update(vr, vr_t), K.update(vc, vc_t)
            # 合成矩阵
            v_t = vr_t * vc_t / K.mean(vr_t, axis=axis2, keepdims=True)
        # 增量主体
        u = grad / K.sqrt(v_t + self.epsilon1)
        # 增量裁剪
        if self.clipping_threshold is not None:
            u = u / K.maximum(1.0, rms(u) / self.clipping_threshold)
        # 增量滑动
        if self.beta1 > 0.0:
            m = self.get_slot(var, 'm')
            # 定义更新
            m_t = self.beta1 * m + (1.0 - self.beta1) * u
            u = K.update(m, m_t)
        # 增量调整
        if self.multiply_by_parameter_scale:
            u = u * K.maximum(rms(var), self.epsilon2)
        # 更新参数
        return K.update(var, var - lr * u)

    def _resource_apply_dense(self, grad, var):
        return self._resource_apply(grad, var)

    def _resource_apply_sparse(self, grad, var, indices):
        grad = tf.IndexedSlices(grad, indices, K.shape(var))
        grad = tf.convert_to_tensor(grad)
        return self._resource_apply_dense(grad, var)


def export_to_custom_objects(base_extend_with):
    """装饰器，用来将优化器放到custom_objects中
    """
    def new_extend_with(BaseOptimizer, name=None):
        NewOptimizer = base_extend_with(BaseOptimizer)

        if is_string(name):
            NewOptimizer.__name__ = name

        name = NewOptimizer.__name__
        keras.utils.get_custom_objects()[name] = NewOptimizer

        return NewOptimizer

    return new_extend_with


@export_to_custom_objects
def extend_with_weight_decay(BaseOptimizer):
    """返回新的优化器类，加入权重衰减
    """
    class NewOptimizer(BaseOptimizer):
        """带有权重衰减的优化器
        """
        @insert_arguments(weight_decay_rate=0.01, exclude_from_weight_decay=[])
        def __init__(self, *args, **kwargs):
            super(NewOptimizer, self).__init__(*args, **kwargs)
            if not hasattr(self, 'learning_rate'):
                self.learning_rate = self.lr

        @K.symbolic
        def get_updates(self, loss, params):
            old_update = K.update

            def new_update(x, new_x):
                if is_one_of(x, params) and self._do_weight_decay(x):
                    new_x = new_x - self.learning_rate * self.weight_decay_rate * x
                return old_update(x, new_x)

            K.update = new_update
            updates = super(NewOptimizer, self).get_updates(loss, params)
            K.update = old_update

            return updates

        def _do_weight_decay(self, w):
            return (not string_matching(w.name, self.exclude_from_weight_decay))

        def get_config(self):
            config = {
                'weight_decay_rate': self.weight_decay_rate,
                'exclude_from_weight_decay': self.exclude_from_weight_decay,
            }
            base_config = super(NewOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return NewOptimizer


@export_to_custom_objects
def extend_with_weight_decay_v2(BaseOptimizer):
    """返回新的优化器类，加入权重衰减
    """
    class NewOptimizer(BaseOptimizer):
        """带有权重衰减的优化器
        """
        @insert_arguments(weight_decay_rate=0.01, exclude_from_weight_decay=[])
        def __init__(self, *args, **kwargs):
            super(NewOptimizer, self).__init__(*args, **kwargs)

        def _resource_apply(self, grad, var, indices=None):
            old_update = K.update

            def new_update(x, new_x):
                if x is var and self._do_weight_decay(x):
                    lr_t = self._decayed_lr(x.dtype.base_dtype)
                    new_x = new_x - lr_t * self.weight_decay_rate * x
                return old_update(x, new_x)

            K.update = new_update
            op = super(NewOptimizer, self)._resource_apply(grad, var, indices)
            K.update = old_update

            return op

        def _do_weight_decay(self, w):
            return (not string_matching(w.name, self.exclude_from_weight_decay))

        def get_config(self):
            config = {
                'weight_decay_rate': self.weight_decay_rate,
                'exclude_from_weight_decay': self.exclude_from_weight_decay,
            }
            base_config = super(NewOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return NewOptimizer


@export_to_custom_objects
def extend_with_layer_adaptation(BaseOptimizer):
    """返回新的优化器类，加入层自适应学习率
    """
    class NewOptimizer(BaseOptimizer):
        """带有层自适应学习率的优化器
        用每一层参数的模长来校正当前参数的学习率
        https://arxiv.org/abs/1904.00962
        """
        @insert_arguments(exclude_from_layer_adaptation=[])
        def __init__(self, *args, **kwargs):
            super(NewOptimizer, self).__init__(*args, **kwargs)
            if not hasattr(self, 'learning_rate'):
                self.learning_rate = self.lr

        @K.symbolic
        def get_updates(self, loss, params):
            old_update = K.update

            def new_update(x, new_x):
                if is_one_of(x, params) and self._do_layer_adaptation(x):
                    dx = new_x - x
                    lr_t = K.clip(self.learning_rate, K.epsilon(), 1e10)
                    x_norm = tf.norm(x)
                    g_norm = tf.norm(dx / lr_t)
                    ratio = K.switch(
                        x_norm > 0.0,
                        K.switch(g_norm > K.epsilon(), x_norm / g_norm, 1.0),
                        1.0
                    )
                    new_x = x + dx * ratio
                return old_update(x, new_x)

            K.update = new_update
            updates = super(NewOptimizer, self).get_updates(loss, params)
            K.update = old_update

            return updates

        def _do_layer_adaptation(self, w):
            return (
                not string_matching(w.name, self.exclude_from_layer_adaptation)
            )

        def get_config(self):
            config = {
                'exclude_from_layer_adaptation':
                    self.exclude_from_layer_adaptation,
            }
            base_config = super(NewOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return NewOptimizer


@export_to_custom_objects
def extend_with_layer_adaptation_v2(BaseOptimizer):
    """返回新的优化器类，加入层自适应学习率
    """
    class NewOptimizer(BaseOptimizer):
        """带有层自适应学习率的优化器
        用每一层参数的模长来校正当前参数的学习率
        https://arxiv.org/abs/1904.00962
        """
        @insert_arguments(exclude_from_layer_adaptation=[])
        def __init__(self, *args, **kwargs):
            super(NewOptimizer, self).__init__(*args, **kwargs)

        def _resource_apply(self, grad, var, indices=None):
            old_update = K.update

            def new_update(x, new_x):
                if x is var and self._do_layer_adaptation(x):
                    dx = new_x - x
                    lr_t = self._decayed_lr(x.dtype.base_dtype)
                    lr_t = K.clip(lr_t, K.epsilon(), 1e10)
                    x_norm = tf.norm(x)
                    g_norm = tf.norm(dx / lr_t)
                    ratio = K.switch(
                        x_norm > 0.0,
                        K.switch(g_norm > K.epsilon(), x_norm / g_norm, 1.0),
                        1.0
                    )
                    new_x = x + dx * ratio
                return old_update(x, new_x)

            K.update = new_update
            op = super(NewOptimizer, self)._resource_apply(grad, var, indices)
            K.update = old_update

            return op

        def _do_layer_adaptation(self, w):
            return (
                not string_matching(w.name, self.exclude_from_layer_adaptation)
            )

        def get_config(self):
            config = {
                'exclude_from_layer_adaptation':
                    self.exclude_from_layer_adaptation,
            }
            base_config = super(NewOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return NewOptimizer


@export_to_custom_objects
def extend_with_piecewise_linear_lr(BaseOptimizer):
    """返回新的优化器类，加入分段线性学习率
    """
    class NewOptimizer(BaseOptimizer):
        """带有分段线性学习率的优化器
        其中schedule是形如{1000: 1, 2000: 0.1}的字典，
        表示0～1000步内学习率线性地从零增加到100%，然后
        1000～2000步内线性地降到10%，2000步以后保持10%
        """
        @insert_arguments(lr_schedule={0: 1})
        def __init__(self, *args, **kwargs):
            super(NewOptimizer, self).__init__(*args, **kwargs)
            self.lr_schedule = {int(i): j for i, j in self.lr_schedule.items()}

        @K.symbolic
        def get_updates(self, loss, params):
            lr_multiplier = piecewise_linear(self.iterations, self.lr_schedule)

            old_update = K.update

            def new_update(x, new_x):
                if is_one_of(x, params):
                    new_x = x + (new_x - x) * lr_multiplier
                return old_update(x, new_x)

            K.update = new_update
            updates = super(NewOptimizer, self).get_updates(loss, params)
            K.update = old_update

            return updates

        def get_config(self):
            config = {
                'lr_schedule': self.lr_schedule,
            }
            base_config = super(NewOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return NewOptimizer


@export_to_custom_objects
def extend_with_piecewise_linear_lr_v2(BaseOptimizer):
    """返回新的优化器类，加入分段线性学习率
    """
    class NewOptimizer(BaseOptimizer):
        """带有分段线性学习率的优化器
        其中schedule是形如{1000: 1, 2000: 0.1}的字典，
        表示0～1000步内学习率线性地从零增加到100%，然后
        1000～2000步内线性地降到10%，2000步以后保持10%
        """
        @insert_arguments(lr_schedule={0: 1})
        def __init__(self, *args, **kwargs):
            super(NewOptimizer, self).__init__(*args, **kwargs)
            self.lr_schedule = {int(i): j for i, j in self.lr_schedule.items()}

        def _decayed_lr(self, var_dtype):
            lr_multiplier = piecewise_linear(self.iterations, self.lr_schedule)
            lr_t = super(NewOptimizer, self)._decayed_lr(var_dtype)
            return lr_t * K.cast(lr_multiplier, var_dtype)

        def get_config(self):
            config = {
                'lr_schedule': self.lr_schedule,
            }
            base_config = super(NewOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return NewOptimizer


@export_to_custom_objects
def extend_with_gradient_accumulation(BaseOptimizer):
    """返回新的优化器类，加入梯度累积
    """
    class NewOptimizer(BaseOptimizer):
        """带有梯度累积的优化器
        """
        @insert_arguments(grad_accum_steps=2)
        def __init__(self, *args, **kwargs):
            super(NewOptimizer, self).__init__(*args, **kwargs)
            self._first_get_gradients = True

        def get_gradients(self, loss, params):
            if self._first_get_gradients:
                self._first_get_gradients = False
                return super(NewOptimizer, self).get_gradients(loss, params)
            else:
                return [ag / self.grad_accum_steps for ag in self.accum_grads]

        @K.symbolic
        def get_updates(self, loss, params):
            # 更新判据
            cond = K.equal(self.iterations % self.grad_accum_steps, 0)
            cond = K.cast(cond, K.floatx())
            # 获取梯度
            grads = self.get_gradients(loss, params)
            self.accum_grads = [
                K.zeros(
                    K.int_shape(p), dtype=K.dtype(p), name='accum_grad_%s' % i
                ) for i, p in enumerate(params)
            ]

            old_update = K.update

            def new_update(x, new_x):
                new_x = cond * new_x + (1 - cond) * x
                return old_update(x, new_x)

            K.update = new_update
            updates = super(NewOptimizer, self).get_updates(loss, params)
            K.update = old_update

            # 累积梯度
            with tf.control_dependencies(updates):
                accum_updates = [
                    K.update(ag, g + (1 - cond) * ag)
                    for g, ag in zip(grads, self.accum_grads)
                ]

            return accum_updates

        def get_config(self):
            config = {
                'grad_accum_steps': self.grad_accum_steps,
            }
            base_config = super(NewOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return NewOptimizer


@export_to_custom_objects
def extend_with_gradient_accumulation_v2(BaseOptimizer):
    """返回新的优化器类，加入梯度累积
    """
    class NewOptimizer(BaseOptimizer):
        """带有梯度累积的优化器
        """
        @insert_arguments(grad_accum_steps=2)
        def __init__(self, *args, **kwargs):
            super(NewOptimizer, self).__init__(*args, **kwargs)

        def _create_slots(self, var_list):
            super(NewOptimizer, self)._create_slots(var_list)
            for var in var_list:
                self.add_slot(var, 'ag')

        def _resource_apply(self, grad, var, indices=None):
            # 更新判据
            cond = K.equal(self.iterations % self.grad_accum_steps, 0)
            # 获取梯度
            ag = self.get_slot(var, 'ag')

            old_update = K.update

            def new_update(x, new_x):
                new_x = K.switch(cond, new_x, x)
                return old_update(x, new_x)

            K.update = new_update
            ag_t = ag / self.grad_accum_steps
            op = super(NewOptimizer, self)._resource_apply(ag_t, var)
            K.update = old_update

            # 累积梯度
            with tf.control_dependencies([op]):
                ag_t = K.switch(cond, K.zeros_like(ag), ag)
                with tf.control_dependencies([K.update(ag, ag_t)]):
                    if indices is None:
                        ag_t = K.update(ag, ag + grad)
                    else:
                        ag_t = self._resource_scatter_add(ag, indices, grad)

            return ag_t

        def get_config(self):
            config = {
                'grad_accum_steps': self.grad_accum_steps,
            }
            base_config = super(NewOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return NewOptimizer


@export_to_custom_objects
def extend_with_lookahead(BaseOptimizer):
    """返回新的优化器类，加入look ahead
    """
    class NewOptimizer(BaseOptimizer):
        """带有look ahead的优化器
        https://arxiv.org/abs/1907.08610
        steps_per_slow_update: 即论文中的k；
        slow_step_size: 即论文中的alpha。
        """
        @insert_arguments(steps_per_slow_update=5, slow_step_size=0.5)
        def __init__(self, *args, **kwargs):
            super(NewOptimizer, self).__init__(*args, **kwargs)

        @K.symbolic
        def get_updates(self, loss, params):
            updates = super(NewOptimizer, self).get_updates(loss, params)

            k, alpha = self.steps_per_slow_update, self.slow_step_size
            cond = K.equal(self.iterations % k, 0)
            slow_vars = [
                K.zeros(
                    K.int_shape(p), dtype=K.dtype(p), name='slow_var_%s' % i
                ) for i, p in enumerate(params)
            ]

            with tf.control_dependencies(updates):
                slow_updates = [
                    K.update(q, K.switch(cond, q + alpha * (p - q), q))
                    for p, q in zip(params, slow_vars)
                ]
                with tf.control_dependencies(slow_updates):
                    copy_updates = [
                        K.update(p, K.switch(cond, q, p))
                        for p, q in zip(params, slow_vars)
                    ]

            return copy_updates

        def get_config(self):
            config = {
                'steps_per_slow_update': self.steps_per_slow_update,
                'slow_step_size': self.slow_step_size,
            }
            base_config = super(NewOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return NewOptimizer


@export_to_custom_objects
def extend_with_lookahead_v2(BaseOptimizer):
    """返回新的优化器类，加入look ahead
    """
    class NewOptimizer(BaseOptimizer):
        """带有look ahead的优化器
        https://arxiv.org/abs/1907.08610
        steps_per_slow_update: 即论文中的k；
        slow_step_size: 即论文中的alpha。
        """
        @insert_arguments(steps_per_slow_update=5, slow_step_size=0.5)
        def __init__(self, *args, **kwargs):
            super(NewOptimizer, self).__init__(*args, **kwargs)

        def _create_slots(self, var_list):
            super(NewOptimizer, self)._create_slots(var_list)
            for var in var_list:
                self.add_slot(var, 'slow_var')

        def _resource_apply(self, grad, var, indices=None):
            op = super(NewOptimizer, self)._resource_apply(grad, var, indices)

            k, alpha = self.steps_per_slow_update, self.slow_step_size
            cond = K.equal(self.iterations % k, 0)
            slow_var = self.get_slot(var, 'slow_var')
            slow_var_t = slow_var + alpha * (var - slow_var)

            with tf.control_dependencies([op]):
                slow_update = K.update(
                    slow_var, K.switch(cond, slow_var_t, slow_var)
                )
                with tf.control_dependencies([slow_update]):
                    copy_update = K.update(var, K.switch(cond, slow_var, var))

            return copy_update

        def get_config(self):
            config = {
                'steps_per_slow_update': self.steps_per_slow_update,
                'slow_step_size': self.slow_step_size,
            }
            base_config = super(NewOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return NewOptimizer


@export_to_custom_objects
def extend_with_lazy_optimization(BaseOptimizer):
    """返回新的优化器类，加入懒惰更新
    """
    class NewOptimizer(BaseOptimizer):
        """带有懒惰更新的优化器
        使得部分权重（尤其是embedding）只有在梯度不等于0时
        才发生更新。
        """
        @insert_arguments(include_in_lazy_optimization=[])
        def __init__(self, *args, **kwargs):
            super(NewOptimizer, self).__init__(*args, **kwargs)
            self._first_get_gradients = True

        def get_gradients(self, loss, params):
            if self._first_get_gradients:
                self._first_get_gradients = False
                return super(NewOptimizer, self).get_gradients(loss, params)
            else:
                return [self.grads[p] for p in params]

        @K.symbolic
        def get_updates(self, loss, params):
            self.grads = dict(zip(params, self.get_gradients(loss, params)))

            old_update = K.update

            def new_update(x, new_x):
                if is_one_of(x, params) and self._do_lazy_optimization(x):
                    g = self.grads[x]
                    r = K.any(K.not_equal(g, 0.0), axis=-1, keepdims=True)
                    new_x = x + (new_x - x) * K.cast(r, K.floatx())
                return old_update(x, new_x)

            K.update = new_update
            updates = super(NewOptimizer, self).get_updates(loss, params)
            K.update = old_update

            return updates

        def _do_lazy_optimization(self, w):
            return string_matching(w.name, self.include_in_lazy_optimization)

        def get_config(self):
            config = {
                'include_in_lazy_optimization':
                    self.include_in_lazy_optimization,
            }
            base_config = super(NewOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return NewOptimizer


@export_to_custom_objects
def extend_with_lazy_optimization_v2(BaseOptimizer):
    """返回新的优化器类，加入懒惰更新
    """
    class NewOptimizer(BaseOptimizer):
        """带有懒惰更新的优化器
        使得部分权重（尤其是embedding）只有在梯度不等于0时
        才发生更新。
        """
        @insert_arguments(include_in_lazy_optimization=[])
        def __init__(self, *args, **kwargs):
            super(NewOptimizer, self).__init__(*args, **kwargs)

        def _resource_apply(self, grad, var, indices=None):
            old_update = K.update

            def new_update(x, new_x):
                if x is var and self._do_lazy_optimization(x):
                    if indices is None:
                        r = K.any(
                            K.not_equal(grad, 0.0), axis=-1, keepdims=True
                        )
                        new_x = x + (new_x - x) * K.cast(r, K.floatx())
                        return old_update(x, new_x)
                    else:
                        return self._resource_scatter_add(
                            x, indices, K.gather(new_x - x, indices)
                        )
                return old_update(x, new_x)

            K.update = new_update
            op = super(NewOptimizer, self)._resource_apply(grad, var, indices)
            K.update = old_update

            return op

        def _do_lazy_optimization(self, w):
            return string_matching(w.name, self.include_in_lazy_optimization)

        def get_config(self):
            config = {
                'include_in_lazy_optimization':
                    self.include_in_lazy_optimization,
            }
            base_config = super(NewOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return NewOptimizer


@export_to_custom_objects
def extend_with_exponential_moving_average(BaseOptimizer):
    """返回新的优化器类，加入EMA（权重滑动平均）
    """
    class NewOptimizer(BaseOptimizer):
        """带EMA（权重滑动平均）的优化器
        """
        @insert_arguments(ema_momentum=0.999)
        def __init__(self, *args, **kwargs):
            super(NewOptimizer, self).__init__(*args, **kwargs)

        def get_updates(self, loss, params):
            updates = super(NewOptimizer, self).get_updates(loss, params)
            self.model_weights = params
            self.ema_weights = [K.zeros(K.shape(w)) for w in params]
            self.old_weights = K.batch_get_value(params)

            ema_updates, ema_momentum = [], self.ema_momentum
            with tf.control_dependencies(updates):
                for w1, w2 in zip(self.ema_weights, params):
                    new_w = ema_momentum * w1 + (1 - ema_momentum) * w2
                    ema_updates.append(K.update(w1, new_w))

            return ema_updates

        def get_config(self):
            config = {
                'ema_momentum': self.ema_momentum,
            }
            base_config = super(NewOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

        def apply_ema_weights(self, bias_correction=True):
            """备份原模型权重，然后将平均权重应用到模型上去。
            """
            self.old_weights = K.batch_get_value(self.model_weights)
            ema_weights = K.batch_get_value(self.ema_weights)

            if bias_correction:
                iterations = K.eval(self.iterations)
                scale = 1.0 - np.power(self.ema_momentum, iterations)
                ema_weights = [weight / scale for weight in ema_weights]

            K.batch_set_value(zip(self.model_weights, ema_weights))

        def reset_old_weights(self):
            """恢复模型到旧权重。
            """
            K.batch_set_value(zip(self.model_weights, self.old_weights))

    return NewOptimizer


@export_to_custom_objects
def extend_with_exponential_moving_average_v2(BaseOptimizer):
    """返回新的优化器类，加入EMA（权重滑动平均）
    """
    class NewOptimizer(BaseOptimizer):
        """带EMA（权重滑动平均）的优化器
        """
        @insert_arguments(ema_momentum=0.999)
        def __init__(self, *args, **kwargs):
            super(NewOptimizer, self).__init__(*args, **kwargs)

        def _create_slots(self, var_list):
            super(NewOptimizer, self)._create_slots(var_list)
            self.model_weights = var_list
            self.ema_weights = []
            for var in var_list:
                self.ema_weights.append(self.add_slot(var, 'ema'))

        def _resource_apply_dense(self, grad, var):
            op = super(NewOptimizer, self)._resource_apply_dense(grad, var)
            ema = self.get_slot(var, 'ema')
            ema_momentum = self.ema_momentum
            with tf.control_dependencies([op]):
                return K.update(
                    ema, ema * ema_momentum + var * (1.0 - ema_momentum)
                )

        def _resource_apply_sparse(self, grad, var, indices):
            op = super(NewOptimizer,
                       self)._resource_apply_sparse(grad, var, indices)
            ema = self.get_slot(var, 'ema')
            ema_momentum = self.ema_momentum
            with tf.control_dependencies([op]):
                return K.update(
                    ema, ema * ema_momentum + var * (1.0 - ema_momentum)
                )

        def get_config(self):
            config = {
                'ema_momentum': self.ema_momentum,
            }
            base_config = super(NewOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

        def apply_ema_weights(self, bias_correction=True):
            """备份原模型权重，然后将平均权重应用到模型上去。
            """
            self.old_weights = K.batch_get_value(self.model_weights)
            ema_weights = K.batch_get_value(self.ema_weights)

            if bias_correction:
                iterations = K.eval(self.iterations)
                scale = 1.0 - np.power(self.ema_momentum, iterations)
                ema_weights = [weight / scale for weight in ema_weights]

            K.batch_set_value(zip(self.model_weights, ema_weights))

        def reset_old_weights(self):
            """恢复模型到旧权重。
            """
            K.batch_set_value(zip(self.model_weights, self.old_weights))

    return NewOptimizer


@export_to_custom_objects
def extend_with_parameter_wise_lr(BaseOptimizer):
    """返回新的优化器类，加入分参数学习率
    主要场景就是给每层甚至每个参数设置不同的学习率。
    """
    class NewOptimizer(BaseOptimizer):
        """带有分参数学习率的优化器
        其中schedule是形如{name1: 2, name2: 0.1}的字典，
        其实name1、name2是字符串，表示变量名包含name1的
        参数学习率乘以2，变量名包含name2的参数学习率要
        乘以0.1。
        """
        @insert_arguments(paramwise_lr_schedule={})
        def __init__(self, *args, **kwargs):
            super(NewOptimizer, self).__init__(*args, **kwargs)

        @K.symbolic
        def get_updates(self, loss, params):
            old_update = K.update

            def new_update(x, new_x):
                if is_one_of(x, params):
                    lr_multiplier = 1
                    for k, v in self.paramwise_lr_schedule.items():
                        if k in x.name:
                            lr_multiplier *= v
                    if lr_multiplier != 1:
                        new_x = x + (new_x - x) * lr_multiplier
                return old_update(x, new_x)

            K.update = new_update
            updates = super(NewOptimizer, self).get_updates(loss, params)
            K.update = old_update

            return updates

        def get_config(self):
            config = {
                'paramwise_lr_schedule': self.paramwise_lr_schedule,
            }
            base_config = super(NewOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return NewOptimizer


@export_to_custom_objects
def extend_with_parameter_wise_lr_v2(BaseOptimizer):
    """返回新的优化器类，加入分参数学习率
    主要场景就是给每层甚至每个参数设置不同的学习率。
    """
    class NewOptimizer(BaseOptimizer):
        """带有分参数学习率的优化器
        其中schedule是形如{name1: 2, name2: 0.1}的字典，
        其实name1、name2是字符串，表示变量名包含name1的
        参数学习率乘以2，变量名包含name2的参数学习率要
        乘以0.1。
        """
        @insert_arguments(paramwise_lr_schedule={})
        def __init__(self, *args, **kwargs):
            super(NewOptimizer, self).__init__(*args, **kwargs)

        def _resource_apply(self, grad, var, indices=None):
            old_update = K.update

            def new_update(x, new_x):
                if x is var:
                    lr_multiplier = 1
                    for k, v in self.paramwise_lr_schedule.items():
                        if k in x.name:
                            lr_multiplier *= v
                    if lr_multiplier != 1:
                        new_x = x + (new_x - x) * lr_multiplier
                return old_update(x, new_x)

            K.update = new_update
            op = super(NewOptimizer, self)._resource_apply(grad, var, indices)
            K.update = old_update

            return op

        def get_config(self):
            config = {
                'paramwise_lr_schedule': self.paramwise_lr_schedule,
            }
            base_config = super(NewOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return NewOptimizer


if is_tf_keras:
    extend_with_weight_decay = extend_with_weight_decay_v2
    extend_with_layer_adaptation = extend_with_layer_adaptation_v2
    extend_with_piecewise_linear_lr = extend_with_piecewise_linear_lr_v2
    extend_with_gradient_accumulation = extend_with_gradient_accumulation_v2
    extend_with_lookahead = extend_with_lookahead_v2
    extend_with_lazy_optimization = extend_with_lazy_optimization_v2
    extend_with_exponential_moving_average = extend_with_exponential_moving_average_v2
    extend_with_parameter_wise_lr = extend_with_parameter_wise_lr_v2
    AdaFactor = AdaFactorV2
else:
    Adam = keras.optimizers.Adam
    AdaFactor = AdaFactorV1

AdaFactor.__name__ = 'AdaFactor'
custom_objects = {
    'Adam': Adam,
    'AdaFactor': AdaFactor,
}

keras.utils.get_custom_objects().update(custom_objects)

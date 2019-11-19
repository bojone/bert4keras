# -*- coding: utf-8 -*-
# 训练相关

import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.snippets import get_all_attributes
from bert4keras.backend import piecewise_linear
from tensorflow.python.ops import array_ops
import re


class OptimizerWrapper(keras.optimizers.Optimizer):
    """优化器包装，主要为了方便做一些修改原有优化器的工作
    （不适用于tf.keras中的OptimizerV2类优化器）
    """
    def __init__(self, optimizer, **kwargs):
        super(OptimizerWrapper, self).__init__(**kwargs)
        self.optimizer = optimizer
        self._optimizer_attributes = []
        for k, v in get_all_attributes(self.optimizer).items():
            if k not in dir(self):
                setattr(self, k, v)
                self._optimizer_attributes.append(k)

    def get_updates(self, loss, params):
        for k in self._optimizer_attributes:
            setattr(self.optimizer, k, getattr(self, k))
        self.updates = self.optimizer.get_updates(loss, params)
        self.weights = self.optimizer.weights
        return self.updates

    def get_config(self):
        config = {
            'optimizer': keras.optimizers.serialize(self.optimizer),
        }
        base_config = super(OptimizerWrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        optimizer = keras.optimizers.deserialize(config.pop('optimizer'))
        return cls(optimizer, **config)


class PiecewiseLinearLearningRate(OptimizerWrapper):
    """分段线性学习率优化器
    传入优化器，然后将优化器的学习率改为分段线性的。
    其中schedule是形如{1000: 1, 2000: 0.1}的字典，
    表示0～1000步内学习率线性地从零增加到100%，然后
    1000～2000步内线性地降到10%，2000步以后保持10%.
    """
    def __init__(self, optimizer, schedule=None, **kwargs):
        super(PiecewiseLinearLearningRate, self).__init__(optimizer, **kwargs)
        self.schedule = {int(i): j for i, j in schedule.items()}
        factor = piecewise_linear(self.iterations, self.schedule)
        self.learning_rate = self.learning_rate * factor

    def get_config(self):
        config = {'schedule': self.schedule}
        base_config = super(PiecewiseLinearLearningRate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GradientAccumulation(OptimizerWrapper):
    """梯度累积优化器
    将steps_per_update步的梯度平均起来，然后才更新模型。
    """
    def __init__(self, optimizer, steps_per_update=1, **kwargs):
        super(GradientAccumulation, self).__init__(optimizer, **kwargs)
        self.steps_per_update = steps_per_update
        # 判断是否要更新的标记
        self.cond = K.equal(self.iterations % self.steps_per_update, 0)
        # 用学习率来决定是否更新，不更新即学习率为0
        self.learning_rate = K.switch(self.cond, self.learning_rate, 0.)
        # 滑动平均量在非更新期内不要动
        for attr in ['momentum', 'rho', 'beta_1', 'beta_2']:
            if hasattr(self, attr):
                value = K.switch(self.cond, getattr(self, attr), 1. - 1e-7)
                setattr(self, attr, value)

    def get_gradients(self, loss, params):
        if hasattr(self, 'accum_grads'):
            return [ag / self.steps_per_update for ag in self.accum_grads]
        else:
            return super(GradientAccumulation,
                         self).get_gradients(loss, params)

    def get_updates(self, loss, params):
        # 获取梯度
        grads = self.get_gradients(loss, params)
        # 定义累积梯度
        self.accum_grads = [
            K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params
        ]
        accum_updates = [
            K.update(ag, K.switch(self.cond, ag * 0, ag + g))
            for g, ag in zip(grads, self.accum_grads)
        ]
        # 继承原更新
        self.optimizer.get_gradients = self.get_gradients
        super(GradientAccumulation, self).get_updates(loss, params)
        self.updates.extend(accum_updates)
        self.weights.extend(self.accum_grads)
        return self.updates

    def get_config(self):
        config = {'steps_per_update': self.steps_per_update}
        base_config = super(GradientAccumulation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def add_weight_decay_into(model, weight_decay_rate, exclude_from=None):
    """往模型加入权重衰减（权重衰减不等价于L2正则）
    """
    if exclude_from is None:
        exclude_from = []

    def need_to_do_weight_decay(w):
        for n in exclude_from:
            if re.search(n, w.name):
                return False
        return True

    weight_decay_updates = []
    factor = 1 - weight_decay_rate
    for w in model.trainable_weights:
        if need_to_do_weight_decay(w):
            weight_decay_updates.append(K.update(w, w * factor))

    model.add_update(weight_decay_updates)


class OptimizerV2(keras.optimizers.Optimizer):
    """修改优化器基类，主要是为了同时兼容tf 1.x和tf 2.0
    下面两个方法都是直接从tf 2.0的OptimizerV2里边抄过来的。
    """
    def _prepare_local(self, var_device, var_dtype, apply_state):
        if "learning_rate" in self._hyper:
            lr_t = array_ops.identity(self._decayed_lr(var_dtype))
            apply_state[(var_device, var_dtype)]["lr_t"] = lr_t

    def _fallback_apply_state(self, var_device, var_dtype):
        """Compatibility for subclasses that don't pass apply_state through."""
        apply_state = {(var_device, var_dtype): {}}
        self._prepare_local(var_device, var_dtype, apply_state)
        return apply_state[(var_device, var_dtype)]


class LAMB(OptimizerV2):
    """LAMB优化器，只支持tf.keras
    直接复制自：https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/lamb.py
    复制到此主要是免得大家要安装tensorflow-addons了。

    下面是原文：
    Optimizer that implements the LAMB (Layer-wise Adaptive Moments)
    optimizer as TF2 tf.keras.optimizers.
    See paper [Large Batch Optimization for Deep Learning: Training BERT
    in 76 minutes](https://arxiv.org/abs/1904.00962).
    """
    
    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 weight_decay_rate=0.0,
                 exclude_from_weight_decay=None,
                 exclude_from_layer_adaptation=None,
                 name='LAMB',
                 **kwargs):
        """
        learning_rate: A `Tensor` or a floating point value.
          The learning rate.
        beta_1: A `float` value or a constant `float` tensor.
          The exponential decay rate for the 1st moment estimates.
        beta_2: A `float` value or a constant `float` tensor.
          The exponential decay rate for the 2nd moment estimates.
        epsilon: A small constant for numerical stability.
        weight_decay_rate: weight decay rate.
        exclude_from_weight_decay: comma separated name patterns of variables
          excluded from weight decay. Variables whose name contain a substring
          matching the pattern will be excluded.
        exclude_from_layer_adaptation: comma separated name patterns of
          variables excluded from layer adaptation. Variables whose name
          contain a substring matching the pattern will be excluded.
        name: Optional name for the operations created when applying
          gradients. Defaults to "LAMB".
        **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`,
          `lr`, `decay`}. `clipnorm` is clip gradients by norm; `clipvalue`
          is clip gradients by value, `decay` is included for backward
          compatibility to allow time inverse decay of learning rate. `lr`
          is included for backward compatibility, recommended to use
          `learning_rate` instead.
        """
        super(LAMB, self).__init__(name, **kwargs)

        # Just adding the square of the weights to the loss function is *not*
        # the correct way of using L2 regularization/weight decay with Adam,
        # since that will interact with the m and v parameters in strange ways.
        #
        # Instead we want to decay the weights in a manner that doesn't interact
        # with the m/v parameters.
        self._set_hyper('weight_decay_rate', weight_decay_rate)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))

        # This is learning rate decay for using keras learning rate schedule.
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon or tf.backend_config.epsilon()
        self.exclude_from_weight_decay = exclude_from_weight_decay
        # exclude_from_layer_adaptation is set to exclude_from_weight_decay if
        # the arg is None.
        if exclude_from_layer_adaptation:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(LAMB, self)._prepare_local(var_device, var_dtype, apply_state)

        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_t = tf.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = tf.identity(self._get_hyper('beta_2', var_dtype))
        weight_decay_rate = tf.identity(
            self._get_hyper('weight_decay_rate', var_dtype))
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)
        apply_state[(var_device, var_dtype)].update(
            dict(
                weight_decay_rate=weight_decay_rate,
                epsilon=tf.convert_to_tensor(self.epsilon, var_dtype),
                beta_1_t=beta_1_t,
                beta_1_power=beta_1_power,
                one_minus_beta_1_t=1 - beta_1_t,
                beta_2_t=beta_2_t,
                beta_2_power=beta_2_power,
                one_minus_beta_2_t=1 - beta_2_t))

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, 'm')
        m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
        m_t = m * coefficients['beta_1_t'] + m_scaled_g_values
        m_t = m.assign(m_t, use_locking=self._use_locking)
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, 'v')
        v_scaled_g_values = (grad * grad) * coefficients['one_minus_beta_2_t']
        v_t = v * coefficients['beta_2_t'] + v_scaled_g_values
        v_t = v.assign(v_t, use_locking=self._use_locking)

        m_t_hat = m_t / (1. - coefficients['beta_1_power'])
        v_t_hat = v_t / (1. - coefficients['beta_2_power'])

        v_sqrt = tf.sqrt(v_t_hat)
        update = m_t_hat / (v_sqrt + coefficients['epsilon'])

        var_name = self._get_variable_name(var.name)
        if self._do_use_weight_decay(var_name):
            update += coefficients['weight_decay_rate'] * var

        ratio = 1.0
        if self._do_layer_adaptation(var_name):
            w_norm = tf.norm(var, ord=2)
            g_norm = tf.norm(update, ord=2)
            ratio = tf.where(
                tf.greater(w_norm, 0),
                tf.where(tf.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)

        var_update = var - ratio * coefficients['lr_t'] * update
        return var.assign(var_update, use_locking=self._use_locking).op

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, 'm')
        m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
        m_t = m.assign(
            m * coefficients['beta_1_t'], use_locking=self._use_locking)
        with tf.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, 'v')
        v_scaled_g_values = (grad * grad) * coefficients['one_minus_beta_2_t']
        v_t = v.assign(
            v * coefficients['beta_2_t'], use_locking=self._use_locking)
        with tf.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        m_t_hat = m_t / (1. - coefficients['beta_1_power'])
        v_t_hat = v_t / (1. - coefficients['beta_2_power'])

        v_sqrt = tf.sqrt(v_t_hat)
        update = m_t_hat / (v_sqrt + coefficients['epsilon'])

        var_name = self._get_variable_name(var.name)
        if self._do_use_weight_decay(var_name):
            update += coefficients['weight_decay_rate'] * var

        ratio = 1.0
        if self._do_layer_adaptation(var_name):
            w_norm = tf.norm(var, ord=2)
            g_norm = tf.norm(update, ord=2)
            ratio = tf.where(
                tf.greater(w_norm, 0),
                tf.where(tf.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)

        var_update = var.assign_sub(
            ratio * coefficients['lr_t'] * update,
            use_locking=self._use_locking)
        return tf.group(*[var_update, m_t, v_t])

    def get_config(self):
        config = super(LAMB, self).get_config()
        config.update({
            'learning_rate':
            self._serialize_hyperparameter('learning_rate'),
            'weight_decay_rate':
            self._serialize_hyperparameter('weight_decay_rate'),
            'decay':
            self._serialize_hyperparameter('decay'),
            'beta_1':
            self._serialize_hyperparameter('beta_1'),
            'beta_2':
            self._serialize_hyperparameter('beta_2'),
            'epsilon':
            self.epsilon,
        })
        return config

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _do_layer_adaptation(self, param_name):
        """Whether to do layer-wise learning rate adaptation for
        `param_name`."""
        if self.exclude_from_layer_adaptation:
            for r in self.exclude_from_layer_adaptation:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match('^(.*):\\d+$', param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name


custom_objects = {
    'PiecewiseLinearLearningRate': PiecewiseLinearLearningRate,
    'GradientAccumulation': GradientAccumulation,
    'LAMB': LAMB,
}

keras.utils.get_custom_objects().update(custom_objects)

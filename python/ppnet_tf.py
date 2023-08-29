import tensorflow as tf

class L2Convolution(tf.keras.layers.Layer):
    def __init__(self, config):
        super(L2Convolution, self).__init__()
        self.prototype_shape = [
            config['prototype_shape'][1],
            config['prototype_shape'][2],
            config['prototype_shape'][3],
            config['prototype_shape'][0]
        ]
        self.feature_shape = config['feature_shape']

    def build(self, input_shape):
        self.prototype_vectors = self.add_weight(
            'proto_vec',
            shape=self.prototype_shape,
            dtype='float32',
            initializer=tf.initializers.random_uniform(minval=0, maxval=1)
        )

        self.ones = self.add_weight(
            'ones',
            shape=self.prototype_shape,
            dtype='float32',
            initializer=tf.initializers.ones(),
            regularizer=None,
            trainable=False
        )

    def compute_output_shape(self, input_shape):
        return [None, self.feature_shape[0], self.feature_shape[1], self.feature_shape[3]]
    
    def call(self, inputs, *args, **kwargs):
        # x = tf.keras.layers.BatchNormalization()(inputs)
        
        x2 = tf.square(inputs)
        x2_patch_sum = tf.nn.conv2d(
            x2, self.ones, strides=1, padding='VALID'
        )

        p2 = tf.square(self.prototype_vectors)
        p2 = tf.reduce_sum(p2, axis=[0, 1, 2], keepdims=True)

        xp = tf.nn.conv2d(
            inputs, self.prototype_vectors, strides=1, padding='VALID'
        )
        xp = tf.multiply(xp, -2.0)

        intermediate_result = tf.add(xp, p2)
        distances = tf.nn.relu(tf.add(x2_patch_sum, intermediate_result))
        return distances
    
class Distance2Similarity(tf.keras.layers.Layer):
    def __init__(self, config):
        super(Distance2Similarity, self).__init__()

        self.epsilon = 1e-4

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, *args, **kwargs):
        
        similarity = tf.math.log(
            tf.divide(
                tf.add(inputs, 1.0),
                tf.add(inputs, self.epsilon)
            )
        )

        return similarity
    
class MinDistancePooling(tf.keras.layers.Layer):
    def __init__(self, config):
        super(MinDistancePooling, self).__init__()

        self.kernel_size = [config['feature_shape'][0], config['feature_shape'][1]]
        self.num_prototypes = config['prototype_shape'][0]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_prototypes) 

    def call(self, inputs, *args, **kwargs):
        distances = tf.negative(inputs)
        min_distances = tf.nn.pool(
            distances,
            window_shape=self.kernel_size,
            pooling_type='MAX',
            padding='VALID'
        )
        min_distances = tf.negative(min_distances)
        min_distances = tf.reshape(min_distances, (-1, self.num_prototypes))

        return min_distances

def proto_part_loss(cfg, proto_class_id):
    def loss_fn(y_true, y_pred):
        # print(y_true)
        # print(y_pred)
        labels = tf.argmax(y_true, axis=1)

        max_distance = cfg['prototype_shape'][1] * cfg['prototype_shape'][2] * cfg['prototype_shape'][3] * 1.0
        prototypes_of_correct_class = tf.transpose(tf.gather(proto_class_id, labels, axis=1))

        inverted_distances = tf.reduce_max(
            tf.multiply(tf.subtract(max_distance, y_pred), prototypes_of_correct_class),
            axis=1
        )

        cluster_cost = tf.reduce_mean(tf.subtract(max_distance, inverted_distances))

        # Separation cost
        prototypes_of_wrong_class = tf.subtract(1.0, prototypes_of_correct_class)
        inverted_distances_nontarget = tf.reduce_max(
            tf.multiply(tf.subtract(max_distance, y_pred), prototypes_of_wrong_class),
            axis=1
        )

        separation_cost = tf.reduce_mean(tf.subtract(max_distance, inverted_distances_nontarget))

        return tf.add_n([
            tf.multiply(tf.constant(0.8), cluster_cost),
            tf.multiply(tf.constant(-0.08), separation_cost)
        ])

    return loss_fn

def cross_entropy_loss(y_true, y_pred):
    tf.print(tf.argmax(y_pred, axis=1), summarize=-1)
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)
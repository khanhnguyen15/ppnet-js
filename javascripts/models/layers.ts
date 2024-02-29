import { Kwargs } from '@tensorflow/tfjs-layers/dist/types';
import * as tf from '@tensorflow/tfjs-node';
import exp from 'constants';
import { config } from 'process';

class L2Convolution_ extends tf.layers.Layer {
    private config: Object;
    private prototypeShape: number[];
    private featureShape: number[];

    private prototypeVectors: tf.LayerVariable;
    private ones: tf.LayerVariable;

    constructor(config: any) {
        super(config);

        this.config = Object.assign({ name: 'l2_convolution' }, config); 
        this.name = 'l2_convolution';
        this.prototypeShape = [config.prototypeShape[1], config.prototypeShape[2], config.prototypeShape[3], config.prototypeShape[0]];
        this.featureShape = config.featureShape;
    }

    build(inputShape: tf.Shape | tf.Shape[]): void {
        this.prototypeVectors = this.addWeight('proto_vec', this.prototypeShape, 'float32', tf.initializers.randomUniform({ minval: 0, maxval: 1 }));
        this.ones = this.addWeight('ones', this.prototypeShape, 'float32', tf.initializers.ones(), undefined, false);
    }

    computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
        return [null, this.featureShape[0], this.featureShape[1], this.prototypeShape[3]];
    }

    getConfig(): tf.serialization.ConfigDict {
        const config = super.getConfig();
        return Object.assign({}, config, this.config);
    }

    call(inputs: tf.Tensor<tf.Rank> | tf.Tensor<tf.Rank>[], kwargs: Kwargs): tf.Tensor<tf.Rank> | tf.Tensor<tf.Rank>[] {
        return tf.tidy(() => {
            if (Array.isArray(inputs)) {
                inputs = inputs[0];
            }
            this.invokeCallHook(inputs, kwargs);

            // B = batchSize, P = prototype, D = dimension, N = number
            const x2 = tf.square(inputs) as tf.Tensor4D;    // [B, 7, 7, PD]
            const x2_patch_sum = tf.conv2d(
                x2,
                this.ones.read() as tf.Tensor4D,
                1,
                'valid'
            );                                              // [B, 7, 7, PN]

            let p2 = tf.square(this.prototypeVectors.read());
            p2 = tf.sum(p2, [0, 1, 2], false);
            p2 = tf.reshape(p2, [1, 1, -1]);                // [PN]

            let xp = tf.conv2d(
                inputs as tf.Tensor4D,
                this.prototypeVectors.read() as tf.Tensor4D,
                1,
                'valid'
            );
            xp = tf.mul(xp, tf.scalar(-2));                 // [B, 7, 7, PN]

            const intermediate_result = tf.add(xp, p2);
            const distances = tf.relu(tf.add(x2_patch_sum, intermediate_result));

            return distances;
        })
    }

    static get className(): string {
        return 'L2Convolution';
    }
}
tf.serialization.registerClass(L2Convolution_);
export const L2Convolution = (config: any) => new L2Convolution_(config);

class Distance2Similarity_ extends tf.layers.Layer {
    private config: Object;
    private epsilon: number;
    private prototypeActivationFunction: string;

    constructor(config: any) {
        super(config);

        this.config = Object.assign({ name: 'distance_to_similarity' }, config); 
        this.name = 'distance_to_similarity';
        this.prototypeActivationFunction = config.prototypeActivationFunction;
        this.epsilon = 1e-4;
    }

    computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
        return inputShape;
    }

    call(inputs: tf.Tensor<tf.Rank> | tf.Tensor<tf.Rank>[], kwargs: Kwargs): tf.Tensor<tf.Rank> | tf.Tensor<tf.Rank>[] {
        return tf.tidy(() => {
            if (Array.isArray(inputs)) {
                inputs = inputs[0];
            }
            this.invokeCallHook(inputs, kwargs);
            return tf.log(
                tf.div(
                    tf.add(inputs, tf.scalar(1)),
                    tf.add(inputs, tf.scalar(this.epsilon))
                ) 
            )
        })
    }

    static get className(): string {
        return 'Distance2Similarity';
    }
}
tf.serialization.registerClass(Distance2Similarity_);
export const Distance2Similarity = (config: any) => new Distance2Similarity_(config);

class MinDistancesPooling_ extends tf.layers.Layer {
    private config: Object;
    private kernelSize: [number, number];
    private numPrototypes: number;

    constructor(config: any) {
        super(config);

        this.config = Object.assign({ name: 'min_distances' }, config); 
        this.name = 'min_distances';
        this.kernelSize = [config.featureShape[0], config.featureShape[1]];
        this.numPrototypes = config.prototypeShape[0];
    }

    computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
        return [null, this.numPrototypes];
    }
    
    getConfig(): tf.serialization.ConfigDict {
        const config = super.getConfig();
        return Object.assign({}, config, this.config);
    }

    call(inputs: tf.Tensor<tf.Rank> | tf.Tensor<tf.Rank>[], kwargs: Kwargs): tf.Tensor<tf.Rank> | tf.Tensor<tf.Rank>[] {
        return tf.tidy(() => {
            if (Array.isArray(inputs)) {
                inputs = inputs[0];
            }
            this.invokeCallHook(inputs, kwargs);

            let distances = tf.mul(inputs, tf.scalar(-1)) as tf.Tensor4D;
            let minDistances = tf.pool(
                distances,
                this.kernelSize,
                'max',
                'valid'
            ) as tf.Tensor;
            minDistances = tf.mul(minDistances, tf.scalar(-1));                 // [B, 1, 1, PN]
            minDistances = tf.reshape(minDistances, [-1, this.numPrototypes])   // [B, PN]

            return minDistances;
        })
    }

    static get className(): string {
        return 'MinDistancesPooling';
    }
}
tf.serialization.registerClass(MinDistancesPooling_);
export const MinDistancesPooling = (config: any) => new MinDistancesPooling_(config);
import * as tf from '@tensorflow/tfjs-node';
import { resnet50 } from './resnet';
import { L2Convolution, Distance2Similarity, MinDistancesPooling } from './layers';

export async function convFeatures (cfg: any): Promise<tf.LayersModel> {
    const config = Object.assign(
        {
            name: 'features',
        }, 
        cfg
    );
    const inputs = tf.input({ shape: [config.imgSize, config.imgSize, 3] });
    let x: tf.Tensor | tf.SymbolicTensor | tf.Tensor[] | tf.SymbolicTensor[];

    x = (await resnet50(config.pretrainedPath)).apply(inputs) as tf.SymbolicTensor;
    x = tf.layers.conv2d({
        name: 'add_on_layer/conv2_1',
        filters: config.prototypeShape[3],
        kernelSize: 1,
        kernelInitializer: 'glorotUniform',
        activation: 'relu'
    }).apply(x);
    
    x = tf.layers.conv2d({
        name: 'add_on_layer/conv2d_2',
        filters: config.prototypeShape[3],
        kernelSize: 1,
        kernelInitializer: 'glorotUniform',
        activation: 'sigmoid'
    }).apply(x) as tf.SymbolicTensor;
    
    return tf.model({ name: config.name, inputs: inputs, outputs: x });
}

export async function getProtoClassIdx (cfg: any): Promise<tf.Tensor> {
    const config = Object.assign({}, cfg);
    const numClasses = config.numClasses;
    const numPrototypes = config.prototypeShape[0];

    const numPrototypePerClasses = Math.floor(numPrototypes / numClasses);
    const protoClassId = tf.zeros([numPrototypes, numClasses]);
    let protoClassIdBuffer = tf.buffer(protoClassId.shape, protoClassId.dtype, protoClassId.dataSync())

    for (let j = 0; j < numPrototypes; j++) {
       protoClassIdBuffer.set(1, j, Math.floor(j / numPrototypePerClasses));
    }

    return protoClassIdBuffer.toTensor();
}

export async function PPNet (cfg: any): Promise<tf.LayersModel> {
    const configDefault: Object = {
        imgSize: 224,
        prototypeShape: [200, 1, 1, 128],
        prototypeActivationFunction: 'log'
    }

    const configModels: Object = {
        'resnet50': { 
            featureShape: [7, 7, 2048],
            pretrainedPath: './pretrained/js/resnet50v2/model.json'
        }
    }

    if (cfg.backbone) {
        if (!Object.keys(configModels).includes(cfg.backbone)) {
            throw new Error(`Invalid modelType: ${cfg.backbone}`);
        }
        const modelConfig = configModels[cfg.backbone as keyof Object];
        Object.assign(configDefault, modelConfig);
    }

    const config = Object.assign({}, configDefault, cfg);
    const featureLayers = await convFeatures(config);

    const inputs = tf.input({ shape: [config.imgSize, config.imgSize, 3] });
    const cnnFeatures = featureLayers.apply(inputs);
    const distances = L2Convolution(config).apply(cnnFeatures);

    const minDistances = MinDistancesPooling(config).apply(distances) as tf.SymbolicTensor;

    const prototype_activations = Distance2Similarity(config).apply(minDistances);

    const logits = tf.layers.dense({
        name: 'logits',
        units: config.numClasses
    }).apply(prototype_activations) as tf.SymbolicTensor;

    return tf.model({ inputs: inputs, outputs: [logits]});
}
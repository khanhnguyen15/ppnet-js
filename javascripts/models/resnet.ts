import * as tf from '@tensorflow/tfjs-node';

export async function resnet50 (
    pretrainedPath: string
): Promise<tf.LayersModel> {
    const handler = tf.io.fileSystem(pretrainedPath);
    const resnet50 = await tf.loadLayersModel(handler);
    
    return resnet50;
}
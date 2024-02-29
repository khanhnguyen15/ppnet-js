import * as tf from '@tensorflow/tfjs-node';

export function logitLoss(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Tensor {
    const loss = tf.losses.softmaxCrossEntropy(yTrue, yPred);
    // console.log(yTrue.argMax(1).print());
    // console.log(tf.softmax(yPred).argMax(1).print());
    return loss;
}

export function protoPartLoss (cfg: any, protoClassId: tf.Tensor) {
    return (yTrue: tf.Tensor, yPred: tf.Tensor): tf.Tensor => {
        const labels = yTrue.argMax(1);

        // cluster cost
        const maxDistance = cfg.prototypeShape[1] * cfg.prototypeShape[2] * cfg.prototypeShape[3];
        const prototypesOfCorrectClass = tf.transpose(protoClassId.gather(labels, 1));

        const invertedDistances = tf.max(
            tf.mul(tf.sub(maxDistance, yPred), prototypesOfCorrectClass),
            1
        );

        const clusterCost = tf.mean(tf.sub(maxDistance, invertedDistances)); 

        // separation cost
        const prototypesOfWrongClass = tf.sub(1, prototypesOfCorrectClass);
        const invertedDistancesNontarget = tf.max(
            tf.mul(tf.sub(maxDistance, yPred), prototypesOfWrongClass),
            1
        );

        const separationCost = tf.mean(tf.sub(maxDistance, invertedDistancesNontarget));

        return tf.addN([
            tf.mul(tf.scalar(0.8), clusterCost),
            tf.mul(tf.scalar(-0.08), separationCost)
        ]);
    }
}
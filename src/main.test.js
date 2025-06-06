const tf = require('@tensorflow/tfjs');

test('model compilation and training', async () => {
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));
    
    model.compile({
        optimizer: 'sgd',
        loss: 'meanSquaredError'
    });

    const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
    const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

    await model.fit(xs, ys, {epochs: 10});
    
    const output = model.predict(tf.tensor2d([5], [1, 1]));
    expect(output.dataSync()[0]).toBeGreaterThan(0);
});
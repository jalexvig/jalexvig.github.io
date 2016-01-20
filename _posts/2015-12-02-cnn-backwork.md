---
layout: post
title: "backworking strongly classified spurious inputs in CNNs"
categories: blog
excerpt: "Finding artifacts in convolutional neural networks"
tags: [tensorflow, ec2, python3]
---

# intro

Convolutional neural networks are full of artifacts. Properly calibrated images of static (to the human eye) can therefore be [strongly classified (>99.99% probability) as an image](http://arxiv.org/pdf/1412.1897.pdf). Not knowing this paper existed, I set out to figure out if I could fool a CNN. Described are some of the techniques I used to do so, and how I generalized the approach to retrain the original CNN for a potential increase in model accuracy.

All code is [freely available](https://github.com/jalexvig/cnn_backwork).

# background

I use [tensorflow](https://www.tensorflow.org/) (a symbolic programming language for Python) to build a convolutional neural network (CNN) for digit recognition. CNNs trained as pattern recognizers [has a long history](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) and the MNIST dataset is a canonical machine learning dataset. The training process is [even covered in the tensorflow tutorials](https://www.tensorflow.org/versions/0.6.0/tutorials/mnist/pros/index.html).

So once we have a fully trained CNN classifier, we have two questions:

1. How do we identify its artifacts? I.e. what are the intramodel relationships that results in spurious classifications?
2. Can we fix these artifacts?

# process

### overview

We first train a CNN on the MNIST dataset. Then we backwork inputs (using gradient descent on randomly intitialized inputs) to find strongly classified input images that look, to the human eye, nothing like the label as which they are classfied.

### setup

First we build our model. See the `build_model` function in the proc_mnist.py.

{% highlight python %}
params, x, y, layers = build_model(model_options, const_params=True)
{% endhighlight %}

Notice that `x` is now a variable that we will update and has dimensionality `(batch_size, x_size, y_size)`.

### cost

Next we need to build a cost to optimize in order to change `x`. To do this we will do two things simultaneously.

1. Maximize classification probability for some label `l`.
2. For each input/hidden layer, minimize the similarity between neuron activations at the layer (over the training data).

These two criteria suggest that we will get an input that is classified as label `l` but does not look like the other inputs the network was trained on.

##### getting values of neurons for training data

{% highlight python %}
def compute_layers_fixed_label(data, label, x, layers, sess):

    labels = data.train.labels.argmax(1)
    mask = labels == label

    data_label = data.train.images[mask]

    layers_label = sess.run(layers, feed_dict={x: data_label})

    return layers_label
{% endhighlight %}

We evaluate this over all layers (except the output softmax layer) of the net.

{% highlight python %}
layers_label = compute_layers_fixed_label(data, label, x, layers, sess)
layers_label = [tf.constant(a, name='layer{}_fixed_label'.format(idx)) for idx, a in enumerate(layers_label)]
{% endhighlight %}

##### building cost

Next we build the cost with respect to these neuronal values across the training data.

{% highlight python %}
costs = []
for layer, layer_label in zip(layers, layers_label):
    cost_tup = get_cost_from_layer(layer, layer_label, label, sim_coeff=sim_coeff, prev_coeff=prev_coeff)
    costs.append(cost_tup)
{% endhighlight %}

It would be nice to be able to process inputs in batches, but as the batchsize gets large, I found collisions in the artifacts found. In other words, different input images sometimes converged to similar values that minimize the cost. To avoid this phenomenon, we add a cost for similarities between the inputs at each layer. For each layer, the relative importance of maximizing difference between an input and the training data is encoded by `prev_coeff` and the relative importance of maximizing the difference between inputs in the batch is encoded by `sim_coeff`.

{% highlight python %}
def get_cost_from_layer(layer, layer_label, label, prev_coeff=1, sim_coeff=1):

    # For convolutional layers, layer has dimensionality (num_gen_examples, 28, 28, 1) and
    # layer_label has dimensionality (num_training_examples_label, 28, 28, 1)

    layer2 = tf.expand_dims(layer, 1)
    layer_label2 = tf.expand_dims(layer_label, 1)
    layer_name = layer.name.lower()

    prev_dist_term = tf.pow(layer - layer_label2, 2)
    similarity_dist_term = tf.pow(layer - layer2, 2)

    if 'softmax' in layer_name:
        # Average all probabilities of guessing the wrong answer
        cost = 1 - layer[:, label]
        cost = tf.reduce_mean(cost)

        return cost,

    elif 'inputs' in layer_name or 'conv' in layer_name:
        # prev_dist_term has dimensionality (num_training_examples_label, num_gen_examples, 28, 28, 1)
        # Get the mean of the minimum distances of generated inputs to training examples of label
        prev_dist_term = tf.reduce_mean(prev_dist_term, reduction_indices=[2, 3, 4])

    elif 'fc' in layer_name:
        pass

    else:
        raise ValueError('Unknown layer type on layer %s', layer_name)

    prev_dist_term = tf.reduce_min(prev_dist_term, reduction_indices=[0])
    prev_dist_term = tf.reduce_mean(prev_dist_term)

    similarity_dist_term = tf.reduce_mean(similarity_dist_term)

    prev_cost = prev_coeff / prev_dist_term
    sim_cost = sim_coeff / similarity_dist_term

    return prev_cost, sim_cost
{% endhighlight %}

Finally all the costs are weighted. This is enables costs at higher layers to be treated with greater importance than lower layers. Intuitively, this makes sense to do because features at higher layers represent greater levels of abstraction. If these can be combined in ways, not induced by the training set, to yield a high probability of classification, then our net has learned some high level features that have artifacts in them. Weighting cost contributions from higher layers more also means that transformations like rotations and translations are not penalized.

{% highlight python %}
costs = [f * c for f, cost_tup in zip(cost_factors, costs) for c in cost_tup]
{% endhighlight %}

### breaking out of local minima

Notice that in the softmax layer, we count the cost as the mean misclassification probability across the batch. This means that many of the inputs in the batch may be strongly classified, but a few may not be able to escape the local minima they are in. To combat this, we will replace the most consistently misclassified inputs. If an input's correct classification probability has been monotonically decreasing over `num_mono_dec_saves` saves (a save is just everytime the inputs are propagated forward to obtain classification probabilities) it is reinitialized.

So with a mask `m` to denote which inputs need to be reinitialized, we can do the following:

{% highlight python %}
input_vals[m] = np.random.rand(\*shape)
sess.run(x.assign(input_vals))
{% endhighlight %}

# results

Here are eight input images that were put through this process for each of the 10 labels.

![Generated misclassified]({{ site.url }}/images/misclass.png)

Here are eight images from the training set with associated probabilities of correct classification. These are the training examples that the net performs worst on.

![Training set]({{ site.url }}/images/worst_train_data.png)


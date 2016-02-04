---
layout: post
title: "lstm text modeling"
modified:
categories: blog
excerpt: "Text generation through neural network language models"
tags: [neural networks, lstm]
draft: true
image:
  feature:
date: 2015-09-14
---

Check out [the code](https://github.com/jalexvig/text_generator) corresponding to this post.

## Recurrent Neural Nets
[Recurrent neural networks](https://en.wikipedia.org/wiki/Recurrent_neural_network) (RNNs) are similar to feed forward neural networks except that they have connections backwards in time. They are called recurrent because neurons in hidden layers connect to themselves (propagation of the neuron's value to itself takes one timestep).

This property makes the RNN an ideal candidate for processing sequential data like text, audio, video, stock prices, etc. We could treat every character or word as a timestep with text files, every frame as a timestep with audio/visual files, and stock prices at every minute as a timestep for stock data. Interestingly, there are also many examples of people using RNNs to model non-temporal data by artificially creating sequences in the data.

## LSTMs
A common problem with RNNs is that information is either lost or becomes too important as it is [propagated backwards through time](https://en.wikipedia.org/wiki/Backpropagation_through_time). You can think of weight updates for a given timestep as a product of weight updates of successive timesteps. It's pretty easy to see how weight updates might either "explode" or "vanish" if they are similar to a multiplication. Check out [this paper](http://arxiv.org/pdf/1211.5063v2.pdf) for a thorough treatment of the subject.

Long-Short Term Memory Nets [LSTMs](https://en.wikipedia.org/wiki/Recurrent_neural_network#Long_short_term_memory_network) solve this problem by selectively protecting and flushing neuron values through a sequence of gates. There are many variants of LSTMs but the quintessential one has input, forget, and output gates. The values of each of these gates are determined by the previous output of the neuron, the current neuronal input, a bias of 1, and learned weights for all three of these. The gates can then control how the cell of the neuron receives input, retains value across time, and outputs its value. As such, the weight gradients are not as susceptible to the exploding/vanishing norms over time. For a treatment of LSTM's and the effects different gates have on the propagation of information through the net, check out [this paper](http://arxiv.org/pdf/1503.04069v1.pdf).

## Text
So how do we use LSTMs for modeling and generating text? The two most common approaches are word level language models and character level language models. Word level models are interesting, but character level models are even cooler. With character level models we learn things from the ground up... like a child would do as they learn the structure of written language. To do this we take some training text and we [one-hot](https://en.wikipedia.org/wiki/One-hot) encode it. So if we have a sentence of 20 characters. We would end up with 20 arrays. Each would have have the length of the number of unique characters in the training text. The entry in the position assigned to the character is a 1 and the rest 0s. This becomes our training data. Each array, save the last, is an input with the successive array being the label. So in the string "LSTMs are dope!":

* First training example is the one-hot encoded character 'L' with a label of the one-hot encoded character 'S' and a seed previous output value of 0 affecting the three gates.
* Second training example is the one-hot encoded character 'S' with a label of the one-hot encoded character 'T' and the output value of the neuron after the 'L' affecting the three gates.
* Third training example is the one-hot encoded character 'T' with a label of the one-hot encoded character 'M' and the output value of the neuron after the 'S' affecting the three gates.
* ...
* Fourteenth training example is the one-hot encoded character 'e' with a label of the one-hot encoded character '!' and the output value of the neuron after the 'p' affecting the three gates.

where:

* L might be one-hot encoded as `[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`
* S might be one-hot encoded as `[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]`
* T might be one-hot encoded as `[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]`
* M might be one-hot encoded as `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]`
* ...
* e might be one-hot encoded as `[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]`
* ! might be one-hot encoded as `[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`

Notice that there are 13 positions in these arrays because there are 13 unique characters in our ficticious single-sequence training data.

At the end of our net we add a [softmax layer](https://en.wikipedia.org/wiki/Softmax_function) such that each character class receives a probability. The probabilities across classes sum to 1. Then the net is trained by backpropagating error using the [negative log-likelihood](https://en.wikipedia.org/wiki/Likelihood_function) as the cost function.

## Implementation Notes

Using momentum to adjust the weights is a good way to avoid specious weight updates caused by outliers in the training data. The adaptive learning rate used in my code is [ADADELTA](http://arxiv.org/pdf/1212.5701v1.pdf).

In order to train the net we have to do lots of matrix multiplications. This can be greatly sped up by using a GPU. GPUs have tons of weak processors and are therefore really good at parallel operations (eg. matrix operations). GPU code is fairly low level, and kind of a royal pain. Luckily for us, the Python library [theano](http://deeplearning.net/software/theano/) can be used to define a net topology and compile it to either C or CUDA code. Even more luckily for us, we don't even need our own GPUs. We can [use Amazon's](https://www.kaggle.com/c/facial-keypoints-detection/details/deep-learning-tutorial).

## Shakespeare Results

Credit to [Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) for the idea of using Shakespeare's works and [Graves](http://arxiv.org/pdf/1308.0850v5.pdf) for details on text modeling with LSTMs (see the Wikipedia section of the paper).

With our training data and labels meaningfully encoded, we are ready to go. I downloaded the [entire works of Shakespeare](http://www.gutenberg.org/cache/epub/100/pg100.txt), cleaned it up, and then trained a 3-layer LSTM net on the one-hot encoded text.

After the net has been trained we can generate samples using the probability distribution over the set of possible characters given the previously predicted characters. Here is a random excerpt of text the net produced after training it for around 10 hours.

> And as think'st thou the better thus,  
> Fall yet he time of man's pidablit hands?  
> Say, their wisely had hast thou to Capiun?  
> QUEEN CARDIGALER:  
> Hearing the worth professions with thy grave himself  
> To whisper, with ill outless robes.  
> So donation seem so dangerously. There is encount'red.  
> Look not to whom, Hal, and with thy impatient much soft as oft too.  
> ELINOR:  
> He send me by the world;

or 

> FRROICK:  
> Soft, eble on suggestyers! Nothing come.  
> BRUTUS:  
> Indeed, even pound out! It poid nor your life to work  
> him, but he swondlly had brought one yet die.  
> CINNAG:  
> Sir, he loves me; the mould wert thou say this daring  
> And make it downd you my sons are none in my  
> thunder to see Rome resides. You have married me all go;  
> Under him, that is either prov'd forth to suck?  
> BRIDT:  
> I will! The mirror and growf'zenious argy awhile  
> The concorchice, and all our own dox!  
> But I will not be with him.  
> PLANTESS:  
> In this letters liv'd Cupid.  
> TITUNA:  
> If Cleopatra serve a verse is the wealth o'er-pater,  
> And we do make that kind thorns of France,  
> Bain wench! Puric, where it is as hearing lords,  
> He talking on men are never had I a brace.  
> Here is my sight; this boon, I de envious of heart  
> Till the colour of joint,  
> Or our excuse is done is nought.  
> A greey pleasure is finile- Baptista will  
> Is nearer your borth of ill-cured from time.  
> He meritle?  
> That strips not to have so drunkent of Cupid's  
> As mean and the flinty of Greeps.  
> We men hath me read as the King for me,  
> And I know not past a thousand times yet religiously  
> To the sun, Troy elses confis'd on. Ay, if I were  
> Thy thought which buys me to our de other purpose.  
> And shall bring you that? O humor? No, I will.  
> LAUNCELOT.S:  
> More previral winning at my horse that Authorsomamy to first  
> I shamed prediged or a duke.  
> TALBOT:  
> Keep mine own men. I have dine of much come in  
> confused with my hand written before, under Cassius, be butches,  
> And fear not. I sent thee answered with him;  
> Thou burds, the Great Montagues, shall require  
> That see now prevail'd me able times he,  
> And I am at the beard, well dear.

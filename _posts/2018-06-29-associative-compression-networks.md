---
layout: post
title: "associative compression networks"
categories: blog
excerpt: "Explaining and implementing associative compression networks"
tags: [machine learning, unsupervised learning]
---

# intro

Variational autoencoders (VAEs) are a type of generative model. Associative compression networks extends the idea of VAEs by ordering the training data.

# background

## vae

There are a couple equivalent ways of looking at VAEs.

1. Information theory
2. Variational lower bound

Terms:

* $p(z {\vert} x)$ the true optimal posterior distribution
* $q(z {\vert} x)$ given by the encoder (an approximation of the true posterior distribution)
* $p(z)$ the prior distribution over the latent space
* $p(x {\vert} z)$ reconstruction probability given by the decoder

#### information theory

We calculate the the total number of bits to transmit a message. The [minimum description length principle]() says that the best description of a dataset is the one that can compress the data the best. So if we can reduce the number of bits required to compress a dataset we can find a good model of the data.

Bits to reconstruct $x$ from $z$:

$$
\begin{align}
C_{recon} &= \int{q(z {\vert} x) \log{p(x {\vert} z)} dx} \\
          &= H(q, p)
\end{align}
$$

Bits coming from posterior not represented in prior:

$$
\begin{align}
C_{coding} &= \int{q(z {\vert} x) \log{q(z {\vert} x)} dx} - \int{q(z {\vert} x) \log{p(z) dx}} \\
           &= D_{KL}(q(z {\vert} x) || p(z))
\end{align}
$$

The sum of the $C_{recon}$ and $C_{coding}$ represents the total number of bits to transmit our dataset $X$ under our model posterior distribution $q(z {\vert} x)$ and model reconstructive distribution $p(x {\vert} z)$:

$$
C_{total} = H(q(z {\vert} x), p(x {\vert} z)) + D_{KL}(q(z {\vert} x) || p(z))
$$

#### variational lower bound

We want to come up with a distribution $q(z)$ that estimates our true posterior $p(z {\vert} x)$ (since it is difficult to compute this).

$$
\begin{align}
D_{KL}(q(z) || p(z {\vert} x)) &= E_{z {\sim} q(z)} [ \log{q(z)} - \log{p(z {\vert} x)} ] \\
                     &= E_{z {\sim} q(z)} [ \log{q(z)} - \log{p(x {\vert} z)} - \log{p(z)} + \log{p(x)} ] \\
                     &= D_{KL}(q(z {\vert} x) || p(z)) + H(q(z {\vert} x), p(x {\vert} z)) + \log{p(x)} \\
\end{align}
$$

There are a couple interpretations of this last form:

**$D_{KL}$**: Minimizing the KL divergence amounts to minimizing the first two terms (since $\log{p(x)}$ is constant under our optimization)

**ELBO**: Since $D_{KL}$ is non-negative we get:

$$
\log{p(x)} \ge -D_{KL}(q(z {\vert} x) || p(z)) + -H(q(z {\vert} x), p(x {\vert} z))
$$

The right hand side forms an "evidence based lower bound" (ELBO) and by maximizing it, we maximize the probability of our data.

## acn

One thing you may have noticed in the VAE is that the prior is just a distribution in the latent space ($p(z)$). This is typically modelled using a multivariate standard Gaussian distribution.

There's nothing that says the prior needs to be a marginal distribution. The key insight of this paper is to realize that we can order the data and then condition the prior distribution on the previous datum. The question then becomes how should we order the data? We compare similarity in the latent space (initialized randomly). When an input is encoded, can look at codes that are close to the encoded input and choose one. By choosing a unique code (and therefore datum) for each example in an epoch we choose an orderding for the data.

# code

The full code is available [on GitHub](https://github.com/jalexvig/associative_compression_networks).

## vae

The following code computes means and variances for a multivariate normal distribution (the `encode` function). It draws from that distribution and decodes the draw.

```python
def forward(self, inputs: torch.Tensor):

    u, logstd = self.encode(inputs)
    h2 = self.reparametrize(u, logstd)
    output = self.decode(h2)

    return output, u, logstd
```

## acn

This code in the prior network picks a unique code (that lives in latent space) for each code that we get from the VAE. In this way, the "close neighbor" code corresponds to the datum that immediately preceeds the current datum.

```python
def forward(self, codes: torch.Tensor):

    previous_codes = [self.pick_close_neighbor(c) for c in codes]
    previous_codes = torch.tensor(previous_codes)

    return self.encode(previous_codes)
```

In order to pick a close neighbor in latent space we can use a K nearest neighbors model and select randomly from the K results. Because each preceeding code must be unique (across the training epoch) we resize the KNN model as needed.

```python
def pick_close_neighbor(self, code: torch.Tensor) -> torch.Tensor:

    neighbor_idxs = self.knn.kneighbors([code.detach().numpy()], return_distance=False)[0]

    valid_idxs = [n for n in neighbor_idxs if n not in self.seen]

    if len(valid_idxs) < self.k:

        codes_new = [c for i, c in enumerate(self.codes) if i not in self.seen]
        self.fit_knn(codes_new)

        return self.pick_close_neighbor(code)

    neighbor_codes = [self.codes[idx] for idx in valid_idxs]

    if len(neighbor_codes) > self.k:
        code_np = code.detach().numpy()
        neighbor_codes = sorted(neighbor_codes, key=lambda n: ((code_np - n) ** 2).sum())[:self.k]

    neighbor = random.choice(neighbor_codes)

    return neighbor
```

# results

The following results correspond to two layer simple feedforward (512 dimensional hidden layer) architectures for both the VAE and the prior distribution on the MNIST dataset.

### loss

The following shows the average loss (in nats) for each training epoch. We see that conditioning the prior results in a steep decrease in the number of nats required.

{:refdef: style="text-align: center;"}
![Losses VAE ACN](/images/associative_compression_networks/loss_vae_acn.png){:height="80%" width="80%"}
{: refdef}

### daydreaming

As discussed in the original paper we can use the prior network to "daydream". The iterative procedure is as follows:

1. Encode image x using the VAE
2. Run the prior network on this encoding
3. Decode the results from the prior network using the VAE

Here is a sequence of images from a MNIST seeded daydream:

{:refdef: style="text-align: center;"}
![Daydreaming ACN](/images/associative_compression_networks/animation.gif){:height="50%" width="50%"}
{: refdef}

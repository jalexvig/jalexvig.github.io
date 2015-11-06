---
layout: post
title: "latent factors"
categories: blog
excerpt: "Collaborative filtering using latent factor models"
tags: [collaborative_filtering, latent_factors, theano]
---

# intro

Let's assume we have a bunch of users, a bunch of items that those users might use, and each user's ratings of a subset (usually a very small subset) of those items. How can we predict what the user might rate an item he/she has never used?

In this post we will talk about latent factor models: a technique commonly used in [collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering) methods for recommender systems. Latent factor models assume that both users and items are vectors that live in the same n-dimensional vector space. If a user and an item are close to each other in this space, then the user approves of the item. Inversely, if a user and an item are not close, then the user disapproves of the item. So how do we learn user/item representations in this vector space given a bunch of ratings?

Check out the [code on github](https://github.com/jalexvig/collaborative_filtering).

# setup

Let's say we choose a 20 dimensional latent factor space. So we have a user and an item whose elements are randomly intialized:

\\[ \mathbf{u} = <u_1, u_2, \ldots, u_{20}> \\]

\\[ \mathbf{i} = <i_1, i_2, \ldots, i_{20}> \\]

Then we predict the users rating with the inner product of (distance between) the two vectors i.e.

\\[ p = \mathbf{u} \cdot \mathbf{i} = \sum_{j=1}^{20}u_j i_j \\]

# learning

So how do we adjust the elements of a user vector or an item vector to make these predictions accurate? We start with a cost function to minimize over our set of ratings \\( R \\):

\\[ C = \sum_{r \in R} (p - r)^2 \\]

One simple way of minimizing a cost functino is [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent). We calculate the gradient of the cost function with respect to each element of the user/item vector. Then we adjust the element by a multiple, the learning rate \\( \eta \\), of this partial derivative.

\\[ u_i = u_i - \eta * \frac{\partial{C}}{\partial{u_i}} \\]

\\[ i_i = i_i - \eta * \frac{\partial{C}}{\partial{i_i}} \\]

Although the gradient of \\( C \\) is straightforward to calculate, more complicated cost functions may not be. If we use [theano](http://deeplearning.net/software/theano/) complicated cost functions come free of the pain-in-the-ass differentiation price tag. To create our theano model, first we define our inputs:

{% highlight python %}
ratings = T.vector('ratings')
var1_vector = T.vector('{}_vector'.format(var1_name))
var2_matrix = T.matrix('{}_matrix'.format(var2_name))
{% endhighlight %}

Notice that one of these inputs is a vector `(20,)` and the other a matrix `(n, 20)`. This is because each user has rated only a small subset of the items. If instead we used the matrix of users and the matrix of items, we would be multiplying two large matrices and their product would contain only a few usable elements. The memory and time penalties incurred with this approach are often times insurmountable. We can condition on the user and multiply each user vector by the matrix of items that user has rated. Similarly we could let the item be the variable that is conditioned on and multiply its vector by the matrix of users that have rated it.

The predictions and cost are calculated as discussed:

{% highlight python %}
predictions = T.dot(var2_matrix, var1_vector)
cost = ((ratings - predictions) ** 2).sum()
{% endhighlight %}

Lastly we calculate our gradients and define a callable theano function:

{% highlight python %}
var1_grad, var2_grad = T.grad(cost, [var1_vector, var2_matrix])
var1_grad /= var2_matrix.shape[0]
f = theano.function(inputs=[ratings, var1_vector, var2_matrix], outputs=[cost, var1_grad, var2_grad])
{% endhighlight %}

Note that the magnitude of the vector gradient (`var1_grad`) is normalized by `var2_matrix.shape[0]` to prevent large swings in variable conditioned on.

# regularization

We can prevent the learning process from [overfitting](https://en.wikipedia.org/wiki/Overfitting) by [regularizing](https://en.wikipedia.org/wiki/Regularization_(mathematics)) our model. There are many ways to regularize a model, but one common and useful way is through weight decay. We add the square of the l2 norm of the each vector processed to the cost function. This ensures that weights don't get arbitrarily big to fit noise in our data. In theano this looks like:

{% highlight python %}
prediction_error = ((ratings - predictions) ** 2).sum()
l2_penalty = (var1_vector ** 2).sum() + (var2_matrix ** 2).sum().sum()
cost = prediction_error + reg_constant * l2_penalty
{% endhighlight %}

# biases

Lastly we can imagine that some users might consistently rate items higher than other (grumpier) users. Additionally, some items might consistently receive higher ratings than other (crappier) items. We would like the dimensions of our vector space to correspond to attributes of our users/items rather than a quality inherent to either. To solve this problem we can add biases \\(u_0\\) and \\(i_0\\) to our user and item vectors respectively:

\\[ \mathbf{u} = <u_0, u_1, \ldots, u_{20}> \\]

\\[ \mathbf{i} = <i_0, i_1, \ldots, i_{20}> \\]

Our predicted ratings become:

\\[ p = u_0 + i_0 + \sum_{j=1}^{20}u_j i_j \\]

And our theano model becomes:

{% highlight python %}
predictions = T.dot(var2_matrix[:, 1:], var1_vector[1:]) + var2_matrix[:, 0] + var1_vector[0]
{% endhighlight %}

# experiment

Let's see how this method of collaborative filtering will work with some real data. First I downloaded the [MovieLens dataset](http://grouplens.org/datasets/movielens/latest/) - for details see the `data.py` module in the git repo. Then I found the latent factor representation for all the users and movies via the algorithm described above (with user and item biases). Over 1000 epochs through the data, the validation error (absolute value of difference between predictions and ratings) was around 0.75.

Lastly I created my own user, based on my own ratings of a few movies. Here were some ratings I entered:

| Movie | Rating |
| --- | --- |
| Toy Story (1995) | 4.0 |
| Jumanji (1995) | 3.5 |
| GoldenEye (1995) | 4.0 |
| Casino (1995) | 4.0 |
| Ace Ventura: When Nature Calls (1995) | 4.0 |
| Twelve Monkeys (a.k.a. 12 Monkeys) (1995) | 5.0 |
| Babe (1995) | 3.0 |
| Seven (a.k.a. Se7en) (1995) | 5.0 |
| Usual Suspects, The (1995) | 5.0 |
| Big Green, The (1995) | 3.0 |
| Don't Be a Menace to South Central While Drinking Your Juice in the Hood (1996) | 2.0 |
| Friday (1995) | 2.0 |
| Happy Gilmore (1996) | 2.5 |
| Braveheart (1995) | 3.5 |
| Taxi Driver (1976) | 4.5 |
| Bad Boys (1995) | 3.5 |
| Mallrats (1995) | 4.0 |
| Before Sunrise (1995) | 5.0 |
| Billy Madison (1995) | 3.0 |
| Clerks (1994) | 4.5 |
| Dumb & Dumber (Dumb and Dumber) (1994) | 3.0 |
| Interview with the Vampire: The Vampire Chronicles (1994) | 3.5 |
| Star Wars: Episode IV - A New Hope (1977) | 3.5 |
| Léon: The Professional (a.k.a. The Professional) (Léon) (1994) | 4.5 |
| Pulp Fiction (1994) | 5.0 |
| Shawshank Redemption, The (1994) | 5.0 |
| Tommy Boy (1995) | 2.5 |
| What's Eating Gilbert Grape (1993) | 5.0 |
| Ace Ventura: Pet Detective (1994) | 3.5 |
| Crow, The (1994) | 4.0 |
| Forrest Gump (1994) | 5.0 | 

# results

In the same way we figured out other user/movie vectors, we can figure out my latent factor representation. Then we can score this vector against the movie vectors:

{% highlight python %}
movie_scores = movies.dot(user)  # movies is a DataFrame of shape (n, 21) that corresponds to all movie representations in our vector space.
{% endhighlight %}

and look at the 20 highest predicted scores:

{% highlight python %}
>>> scores = movies.iloc[:, 1:].dot(user[1:]) + movies.iloc[:, 0] + user[0]
>>> scores.index = movie_titles[scores.index]
>>> scores.nlargest(20)
{% endhighlight %}

| Movie | Rating |
| --- | --- |
| Shawshank Redemption, The (1994) | 4.301845 |
| Godfather, The (1972) | 4.280729 |
| Schindler's List (1993) | 4.247809 |
| Usual Suspects, The (1995) | 4.237750 |
| Casablanca (1942) | 4.216967 |
| Godfather: Part II, The (1974) | 4.216394 |
| Seven Samurai (Shichinin no samurai) (1954) | 4.214890 |
| Sunset Blvd. (a.k.a. Sunset Boulevard) (1950) | 4.213665 |
| Third Man, The (1949) | 4.211904 |
| Fight Club (1999) | 4.208055 |
| Band of Brothers (2001) | 4.206785 |
| One Flew Over the Cuckoo's Nest (1975) | 4.205237 |
| Yojimbo (1961) | 4.205028 |
| Lives of Others, The (Das leben der Anderen) (2006) | 4.204302 |
| Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964) | 4.202918 |
| Rear Window (1954) | 4.197167 |
| Paths of Glory (1957) | 4.196674 |
| Black Mirror (2011) | 4.191165 |
| Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001) | 4.184165 |
| Dark Knight, The (2008) | 4.184119 |

With respect to the movies I've seen, I find myself agreeing very much with my 20-dimensional movie-preference representation. Feel free to test out the code and see what movie ratings you agree/disagree with ... or use it to discover new movies you might enjoy!

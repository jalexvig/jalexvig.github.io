---
layout: post
title: "subreddit important terms"
categories: blog
excerpt: "Collaborative filtering using latent factor models"
tags: [reddit, tfidf]
---

# intro

A reddit user [posted ~1.7 billion reddit comments](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/?st=j3vre6ad&sh=5dca21cf). There is also a one month subset of this data that we can work with on our local machines.

Check out the [code on github](https://github.com/jalexvig/reddit_comment_analysis).

Download the one month subset of Reddit comments. We start by getting word counts of the comments and grouping them by subreddit. Then we will look for words that "define" each subreddit.

# tokenization

The process of extracting words is called tokenization as a string is broken up into "tokens" that are used to characterize the string. In our case the tokens will, approximately, be each word of the comments. The tokenization specification is arbitrary. You could look at pairs of words (bigrams), triplets of words (trigrams), or even every other word. In our case we use every word individually.

What happens if we have different forms of the same word? We probably want to count "foot" and "feet" as the same token. We have two options: a stemmer which chops off the ends of words, or a lemmatizer which does a morphological analysis on each word to determine its lemma (a canonical version of the word).

In this project we can piggy back off of [nltk's implementation](http://www.nltk.org/api/nltk.stem.html#nltk.stem.wordnet.WordNetLemmatizer) of a [WordNet](https://en.wikipedia.org/wiki/WordNet) lemmatizer. We will also use [nltk's part of speech tagger](http://www.nltk.org/api/nltk.tag.html#nltk.tag.pos_tag) to pass the part of speech into the lemmatizer. Finally, the [word_tokenize function](http://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.word_tokenize) will do exactly what you think... it splits up the string into distinct words.

{% highlight python %}
class LemmaTokenizer(object):

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, doc):
        res = []
        for token, pos in pos_tag(word_tokenize(doc)):
            pos = pos.lower()

            if pos[0] in ['a', 'n', 'v']:
                res.append(self.lemmatizer.lemmatize(token, pos[0]))
            else:
                res.append(self.lemmatizer.lemmatize(token))

        return res
{% endhighlight %}

By passing text through an instance of this class we will get a list of words that approximate the original string.

# vectorization

We can use this lemmatizer in our pipeline to vectorize our text. This means we transform our comments from a string into a vector of word counts (token counts). Scikit-learn provides an [implementation of a vectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html).

{% highlight python %}
vect = CountVectorizer(tokenizer=tokenizer, preprocessor=preprocess, stop_words='english')
{% endhighlight %}

# tfidf

We mentiond in the intro section that we were going to look for words that "define" each subreddit. So now that we have a count of each word for each subreddit we need to rank them. One way of doing this is using Term Frequency Inverse Document Frequency (tfidf). The basic idea is to look at how frequently a word appears in a subreddit and offset it by the number of subreddits that word appears in. So if the word "kumquat" appears a lot of times in /r/fruit but doesn't appear in very many other subreddits it will be weighted strongly. If the word "banana" appears often in /r/fruit and it also appears in lots of other subreddits, then it will be weighted weakly. There are a number of variations on this but we will be using the default [one from sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html).

\\[ idf(d, t) = \log{\dfrac{n_d + 1}{df(d, t) + 1} + 1} \\]

where \\(n_d\\) is the number of documents and \\(df(d, t)\\) is the number of times term \\(t\\) appears in document \\(d\\). The resulting matrix is then normalized using the [l2 norm](http://mathworld.wolfram.com/L2-Norm.html).

{% highlight python %}
TfidfTransformer().fit_transform(counts)
{% endhighlight %}

# results

Let's look at some of the top words for varying subreddits (words with corresponding tfidf scores):

funny || aww || worldnews || todayilearned
--- | --- | --- | ---
mo | 0.04 | pup | 0.088 | hamas | 0.053 | muslim | 0.036
͡° | 0.037 | breed | 0.086 | palestinian | 0.05 | jew | 0.032
/r/funny | 0.036 | puppy | 0.084 | /r/worldnews | 0.045 | islam | 0.031
b | 0.029 | kitty | 0.08 | israel | 0.042 | religion | 0.03
r/funny | 0.028 | kitten | 0.079 | kurd | 0.041 | tax | 0.03
reposts | 0.028 | pet | 0.073 | gaza | 0.039 | rape | 0.029
gifs | 0.028 | dog | 0.072 | israeli | 0.038 | nazi | 0.029
repost | 0.028 | cat | 0.071 | ukraine | 0.037 | christian | 0.028
penis | 0.027 | adorable | 0.069 | iran | 0.037 | government | 0.028
ha | 0.027 | paw | 0.069 | palestine | 0.037 | slave | 0.028
toilet | 0.026 | breeder | 0.069 | muslim | 0.036 | gay | 0.028
racist | 0.026 | rescue | 0.069 | hezbollah | 0.036 | jewish | 0.027
karma | 0.026 | animal | 0.067 | ukrainian | 0.036 | hitler | 0.027
meme | 0.026 | shelter | 0.067 | arab | 0.035 | drug | 0.027
religion | 0.026 | husky | 0.065 | paywall | 0.035 | military | 0.027
gif | 0.026 | vet | 0.063 | islam | 0.035 | slavery | 0.027
sex | 0.026 | amp | 0.061 | assad | 0.035 | population | 0.027
muslim | 0.025 | adopt | 0.061 | sunni | 0.035 | wage | 0.027
gay | 0.025 | cute | 0.058 | nato | 0.034 | culture | 0.027
rape | 0.025 | cuddle | 0.057 | sharia | 0.034 | billion | 0.026

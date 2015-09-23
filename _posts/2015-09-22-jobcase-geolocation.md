---
layout: post
title: "jobcase geolocation exercise"
categories: jobcase
excerpt: "User generated place string matching"
tags: [python3, elasticsearch]
draft: true
---

# objective

We want to match user generated strings describing a place against documents stored on an elasticsearch server. Additionally, some strings are associated with an inferred state code that is derived contextually (e.g. using the site the user is browsing).

# setup

The `geocode` method of the `LocationFinder` class in the `process_locations` Python module queries an elasticsearch server. That server is assumed, by default, to be hosted locally on port 9200 with an index of geolocations named `geolocation_v1`.

### requirements

* python3
* python3 libraries
    * pandas
    * elasticsearch
    * nltk

# parsing

Nearly all of the place description strings can be separated into a city, state, and/or postal code. The ones that cannot are assumed to be garbage.

Here are some examples of strings that ought to be interpretable

* `"TAMPA, FL 33601"`
* `"52403"`
* `"Danville, IN"`
* `", AZ"`

Here are some examples of strings that ought to prompt the user to re-enter a value

* `"beijing 22"`
* `"271"`
* `"jobs.html"`

Recognizing that nearly all valid entries have their fields delimited either by a space, comma, or both, we can parse the strings after normalizing them. Note that normalization also takes care of url escape characters, so entries like `"Columbus%2B%2BOH"` are acceptable.

{% highlight python %}
s = urllib.parse.unquote(s)
s = s.replace('.', '')
s = s.lower()
l = re.split('[, ]+', s)
l = list(filter(None, l)) 
{% endhighlight %}

This turns `"TAMPA, FL 33601"` into `["tampa", "fl", "33601"]`.

1. As postal codes seem to always come last, we check to see if the last element in the list is composed of digits, and if so, consume it. We use the first five digits as the postal code (to distill a zip+4 code). If only two digits exist, we will ignore this string entirely as that is indicative of a place in another country. So `"Hyderabad 02"` will be ignored.
2. The next element(s) to be consumed represent(s) the state. This is either a state abbreviation (e.g. "nc") or the state name in full (e.g. "north carolina"). We look backwards through the list to see if a state or an abbreviation exists. A full state name is translated to its state code through a predefined dictionary.
3. The city is assumed to comprise the remaining elements.

So for example

1. `["washington", "district", "of", "columbia", "20500003"]` -> `["washington", "district", "of", "columbia"]` and a postal code of `"20500"`
2. `["washington", "district", "of", "columbia"]` -> `["washington"]` and a state code of `"dc"`
3. `["washington"]` -> `[]` and a city of `"washington"`

# querying

### postal code

As each document in the index has a unique postal code the most accurate matching technique is to filter on the postal code if it exists. Easy peasy.

### city/state

If no match is found using the postal code filter or the place description contains no postal code, we look at the city, state, and optionally at the inferred state. As the inferred state is obtained contextually, it is presumed to be better to use the user-supplied state if it exists.

If the state is provided in the place description, we use a filtered query with the state as a filter. Otherwise, if a state is inferred, we match the inferred state. This way, if the user does not provide a state, we can have a resulting document with a city that matches strongly but a state that is not the inferred state.

Next we match the city supplied by the user to the `city` and array `cityAliases` in the documents. This allows for partial matches to influence our overall match using TFIDF and term length.

If no city is supplied, then we sort on the most populated of the returned documents. This means that if the user only provides a state in a place description, the location returned will default to a highly populated place in that state.

##### boosting

In order to make the city field more relevant than city aliases, we boost the importance of a city match by 2. This means, roughly, that matching the city field of a document is twice as important as matching any city alias belonging to the document.

### edit distance

The [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance) can be used to correct misspellings in city names.

> ... the Levenshtein distance between two words is the minimum number of single-character edits (i.e. insertions, deletions or substitutions) required to change one word into the other.

* `'here'` has a distance of 1 unit from `'there'`
* `'here'` has a distance of 1 unit from `'mere'`
* `'here'` has a distance of 1 unit from `'ere'`
* `'here'` has a distance of 6 units from `'jobcase'`

This distance can be used to compare the city derived from a string to all of the cities and city aliases contained in the index. So instead of matching on the user-supplied city, we look at all of the cities/aliases existing in the index and match on the ones that have the lowest Levenshtein distance. The maximum distance is also capped (at 2 by default).

The algorithm admits three modes of edit distance usage:

1. Never change the city supplied.
2. Only use the edit distance to find replacement cities if no documents are found that match the original city.
3. Always use the edit distance to find replacement cities.

*Implementation note*: the three modes respectively correspond to the `'never'`, `'conditionally'`, and `'always'` values for the `use_edit_distance` keyword argument supplied to the `LocationFinder` constructor.

# reporting

After we get our response from the server, we take the highest scored document and return its latitude and longitude. If no documents are returned or no feasible place identifiers could be found in the original string, a sentinel `None` value is returned.

Results for new strings can be obtained using the following Python code.

{% highlight python %}
>>> from process_locations import LocationFinder
>>> lf = LocationFinder()
>>> lf.geocode('columbus, ohio', 'oh')
(40.020272, -83.017128)
{% endhighlight %}

The `geocode` method has the following signature:

{% highlight python %}
def geocode(self, user_string, inferred_state='', score_cutoff=0, test_entries=False):
    """
    Geocode an individual place
    :param user_string: user supplied place description to geocode
    :param inferred_state: inferred state (from website, context, etc.)
    :param score_cutoff: minimum score needed to report results from document - if 0, will not be used
    :param test_entries: boolean flag indicating whether to test document fields against user_string
    :return: latitude, longitude of place or None if place cannot be geolocated
    """
{% endhighlight %}

### score cutoff

As seen in the method signature, one can optionally provide a score cutoff. If the document result has a score below the cutoff, then it is not reported and the sentinel value `None` is returned.

{% highlight python %}
>>> lf.geocode('columbus, ohio', inferred_state='oh', score_cutoff=10) is None
True
{% endhighlight %}

# results

There are a few ways of discerning how well this algorithm works both qualitatively and quantitatively.

In the sections below, we give results across 5000 training examples for each of the 3 modes of edit distance usage. We use a score cutoff of 0.3. The numbering scheme corresponds to the mode as follows:

1. Never using edit distance to match cities  
2. Conditionally using edit distance to match cities  
3. Always using edit distance to match cities  

### pseudo-labeling

We would like to examine misclassified place descriptions. In order to do this, however, we need a labeled set. Since we don't have this we can use descriptions that contain postal codes and assume the postal code is correct. For such a place description, we find the document with the corresponding postal code and see if its city/cityAliases fields match the city provided by the user. Similarly we see if the state provided by the user and the inferred state match the state from the document.

*Implementation note*: to do this, we use the edit distance to see if the city is represented by the document selected. The distance cap for matches was set to 2.

Every place description containing a postal code was matched to a document also containing the city and state provided by the user. Four unique descriptions existed that contained a post code, but whose resulting document's state did not match the inferred state.

These results held across all 3 modes of edit distance usage.

### false positives and incorrect labels

Additionally we can check cities/states from descriptions that contain no postal code. The same implementation methodology was used to check for city matches.

1. 9 unique place descriptions that had misidentified cities:  
`'Newmarket, ON'`, `'Chattanoogacleveland tn, TN'`, `'City or Zipcode'`, `'us'`, `'ashlin blackstone'`, `'Location'`, `'Columbus%2B%2BOH'`, `'lake city florid'`, and `'Wey Mass'`.  
Of these we might hope that `'Chattanoogacleveland tn, TN'`, `'Columbus%2B%2BOH'`, `'lake city florid'`, and `'Wey Mass'` could be labeled correctly. The others are bogus and ought to ideally return a None value.

2. An additonal 5 unique misidentified place descriptions were found for a total of 14. All of these were not able to be geocoded in the first configuration (more on this in the next section).  The six are:  
`'Dagenham, A1'`, `'job.html'`, `'L3X2Z9'`, `'jobs.html'`, and `'L5G1J2'`

3. The results were the same as the second mode except for 1 unique description, `'us'` that was deemed correctly identified in the third mode.

No misidentified state codes (either provided or inferred) were found.

### false negatives and spurious examples

We can also look at the place descriptions that were not matched. There are some valid reasons for not matching: a description of a place in another country, text that's really not a place description at all, a postal code that doesn't exist, returning a document below the score cutoff, and misspellings. Not matching due to the last of those, along with other types of false negatives, is undesirable.

1. We found 30 unique place descriptions that were not geocodable:  
`'beijing 22'`, `'jobs.html'`, `'Ponce 00'`, `'27140'`, `'job.html'`, `'Guanajuato, 11'`, `'47077'`, `'Bangalore, 19'`, `'laure'`, `'dubai 03'`, `'271'`, `'northwestvegas nv'`, `'Amadora 14'`, `'39 648'`, `'waikele'`, `'Plainfie'`, `'L5G1J2'`, `'Hyderabad, 02'`, `'Aguas Buenas 00'`, `'Johannesburg, 06'`, `'marshal'`, `'L3X2Z9'`, `'Saint Petersburg, 66'`, `'Hyderabad 02'`, `'Carolina, 00'`, `'3190'`, `'Dagenham, A1'`, `'Beijing 22'`, `'Accra, 01'`, and `'idabe'`  
All of these except `'northwestvegas nv'`, `'waikele'`, `'Plainfie'`, `'marshal'`, and `'laure'` are correct failures (spurious examples). `'waikele'` as it turns out is a place in Hawaii and is not represented as a city in our index so there's not much to be done about that.
2. We found 19 unique place descriptions that were not geocodable. These 19 are a proper subset of the 30 found in the first mode. The 11 that were geocoded are:  
`'marshal'`, `'L3X2Z9'`, `'northwestvegas nv'`, `'jobs.html'`, `'waikele'`, `'Dagenham, A1'`, `'Plainfie'`, `'L5G1J2'`, `'laure'`, `'job.html'`, and `'idabe'`
3. This yielded the same results as the second edit distance mode.

### time

Timing results were obtained using a 2.9 GHz Intel Core i7 processor with 16 GB DDR3 RAM.

1.  6 sec = ~0.0012 sec/example
2.  129 sec = ~0.026 sec/example
3.  7348 sec = ~1.47 sec/example

# abbreviated thoughts

We see that queries that require computing the Levenshtein distance take a horribly long time. That said, there is a tradeoff when varying the use of edit distance to choose cities. Generally, the more the edit distance is used the more false positives and fewer false negatives we get.

On a real dataset we could optimize the score cutoff to lower the number of false positives (aka correctly identify spurious examples).

Another thing that might be useful to implement is partial matching - either through regular expressions/wildcards or a custom analyzer.

Lastly, an interesting method of increasing match accuracy would be to use past user-made corrections (after being required to re-enter a place description) and/or past searches to inform future matches.... or just do [this](https://www.youtube.com/watch?v=blB_X38YSxQ).

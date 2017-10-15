---
layout: post
title: "huffman coding in rust"
categories: blog
excerpt: "Learning rust by writing a simple compression algorithm"
tags: [huffman, compression, rust]
---

# intro

Huffman codes are intuitive and in many cases optimal ways of encoding data losslessly. This post covers an implementation of simple huffman codes in Rust.

Feel free to [checkout the code](https://github.com/jalexvig/comprustion).

# idea

David Huffman came up with this compression scheme while studying for an exam! Turns out this idea yields the most efficient lossless encoding when symbols are encoded separately. We'll see the intuitive explanation for that optimality in a bit. But first how does it work?

Let's say we have a string of only ASCII characters "deadbeef" and we want to compress this such that we use the smallest number of bits without losing any information. Huffman's idea was to create a tree using a collapsing symbol (character in this case) count. So our symbol count looks like:

d | e | a | b | f
:---: | :---: | :---: | :---: | :---:
2 | 3 | 1 | 1 | 1

Then we create a node such that the two symbols with the smallest character counts are children. This new node is representative of a stand-in symbol and has a count equal to the sum of the counts of its two children. Our table now looks like this:

d | e | \*(a, b) | f
:---: | :---: | :---: | :---:
2 | 3 | 2 | 1

\*(a, b) represents a node with children corresponding to the characters 'a' and 'b'.

We do this recursively until we end up with a single node. The next step is to combine \*(a, b) and f:

d | e | \*(\*(a, b), f)
:---: | :---: | :---:
2 | 3 | 3

Now we combine d with e (we could also combine d with the node we just created since it has the same count as e):

\*(d, e) | \*(\*(a, b), f)
:---: | :---:
5 | 3

Lastly we combine the two into a single node:

\*(\*(d, e), \*(\*(a, b), f)) |
:---: |
8 |

From this tree we can create a map from each of the original characters to bits by descending from the root node to leaves. As we descend we will use a 0 when we take a left branch and a 1 when we take a right branch. Let's build these character codes (bit sequences):

\*(d, e) | \*(\*(a, b), f)
:---: | :---:
0 | 1

Again we do this recursively (I'll expand both sides of the table simultaneously for brevity):

d | e | \*(a, b) | f
:---: | :---: | :---: | :---:
00 | 01 | 10 | 11

And finally we arrive at our complete table:

d | e | a | b | f
:---: | :---: | :---: | :---: | :---:
00 | 01 | 100 | 101 | 11

This is our encoding. Inverting the table gives us our decoding:

00 | 01 | 100 | 101 | 11
:---: | :---: | :---: | :---: | :---:
d | e | a | b | f

We decode by taking bigger and bigger initial chunks off of the encoded data and checking if those chunks are in our decoding table. Each of these chunks is called a prefix. Huffman codes are prefix codes because no code is a prefix of another.

# optimality and correctness

So intuitively why is this optimal? We build from the bottom of the tree upwards. This means that groups of characters that have lower frequency occur closer to the bottom of the tree. Since we traverse from the top down when constructing our code table the character groups with the highest frequency are reached first. By combining individual character nodes into stand-in nodes (and also stand-in nodes into other stand-in nodes), we ensure that the most used nodes (representing a character or groups of characters) bubble to the top. 

Correctness for lossless compression means that we don't make any errors when decoding encoded data (i.e. our decoded encoded data is equal to our data. How could we make errors? Let's say this was our code table:

d | e | a
:---: | :---: | :---:
100 | 10 | 01

If we try to encode the string "deed" we end up with the bits 1001010100. As we decode we are using the inverse of this table:

100 | 10 | 01
:---: | :---: | :---:
d | e | a

So as we look at prefixes of increasing length we end up with a decoding that looks like

1. 1001010100
2. e 01010100
3. ea 010100
4. eaa 0100
5. eaaa 00

You will notice that we incorrectly decode the first character because the code for 'e' is a prefix for the code for 'd'. As long as no code is a prefix for another and each code is unique (only corresponds with one character) our encoding/decoding will work correctly.

When developing a Huffman coding, because we terminate each code when we reach a leaf (a single character node) no code can be a prefix of another code. Because we constructed our tree such that each individual character is in its own leaf node, no two characters will correspond to the same code.

# [code](https://github.com/jalexvig/comprustion)

This started as an exercise in learning Rust so let's look at some code. I am going to use the `HashMap` and `BinaryHeap` structs from the standard library. Also I will use the `Ordering` trait since the `BinaryHeap` is a max binary heap. My changing methods in the `Ordering` trait and its traits we can easily get a min heap. Lastly I am going to use the [bit-vec crate](https://github.com/contain-rs/bit-vec) in order to do bitwise operations and concatenations. This crate uses ints (size may be specified) as the underlying storage for the bits.

{% highlight rust %}
use std::collections::{HashMap, BinaryHeap};
use std::cmp::Ordering;

use bit_vec::BitVec;
{% endhighlight %}

First let's get character counts from our data.

{% highlight rust %}
fn get_char_counts(data: &str) -> HashMap<char, i32> {

    let mut char_counts = HashMap::new();

    for c in data.chars() {
        let count = char_counts.entry(c).or_insert(0);
        *count += 1;
    }

    char_counts
}
{% endhighlight %}

Now let's build a min heap with all of these characters in it.

{% highlight rust %}
fn heapify(map: HashMap<char, i32>) -> BinaryHeap<Box<Tree>> {

    let mut heap = BinaryHeap::new();

    for (letter, count) in map.iter() {
        let t = Tree::new(*letter, *count);
        heap.push(t);
    }

    heap
}
{% endhighlight %}

Now let's build our huffman binary tree from this heap by popping off the smallest two nodes, attaching them to a parent node, and pushing the parent node back onto the heap.

{% highlight rust %}
fn create_huffman_tree(mut heap: BinaryHeap<Box<Tree>>) -> Box<Tree> {
    while heap.len() > 1 {
        let t1 = heap.pop().unwrap();
        let t2 = heap.pop().unwrap();

        let t_new = Tree::combine(t1, t2);
        heap.push(t_new);
    }

    heap.pop().unwrap()
}
{% endhighlight %}

Now we can get our codes!

{% highlight rust %}
fn huffman_codes_from_tree(opt: &Option<Box<Tree>>, prefix: BitVec, mut map: HashMap<char, BitVec>) -> HashMap<char, BitVec> {

    if let Some(ref tree) = *opt {
        match tree.value {
            Some(c) => {
                map.insert(c, prefix);
            },
            None => {
                let mut prefix_left = prefix.clone();
                prefix_left.push(true);
                let map = huffman_codes_from_tree(&tree.left, prefix_left, map);
                let mut prefix_right = prefix.clone();
                prefix_right.push(false);
                return huffman_codes_from_tree(&tree.right, prefix_right, map);
            }
        }
    }

    return map;
}
{% endhighlight %}

All together in a function this flow is:

{% highlight rust %}
pub fn get_huffman_codes(data: &str) -> HashMap<char, BitVec> {

    let char_counts = get_char_counts(data);

    let heap = heapify(char_counts);

    let ht = create_huffman_tree(heap);

    return huffman_codes_from_tree(&Some(ht), BitVec::new(), HashMap::new());
}
{% endhighlight %}

To encode something now all we need to do is map these codes over our data.

{% highlight rust %}
pub fn encode(data: &str, huffman_codes: &HashMap<char, BitVec>) -> BitVec {

    let mut nbits = 0;
    for c in data.chars() {
        nbits += huffman_codes.get(&c).unwrap().len();
    }

    let mut res = BitVec::with_capacity(nbits);

    for c in data.chars() {
        let bv = huffman_codes.get(&c).unwrap();
        for bit in bv.iter() {
            res.push(bit);
        }
    }

    res
}
{% endhighlight %}

To decode we will need to invert our huffman codes as mentioned.

{% highlight rust %}
fn invert_huffman_codes(codes: &HashMap<char, BitVec>) -> HashMap<BitVec, char> {

    let mut res = HashMap::new();

    for (k, v) in codes.iter() {
        res.insert(v.clone(), k.clone());
    }

    res
}
{% endhighlight %}

Then we can decode the same way we encoded... by mapping over the data.

{% highlight rust %}
pub fn decode(bits: BitVec, huffman_codes: &HashMap<char, BitVec>) -> String {

    let hci = invert_huffman_codes(huffman_codes);

    let mut res = String::new();
    let mut bv = BitVec::new();

    let mut start = 0;
    for bit in bits.iter() {
        bv.push(bit);
        if hci.contains_key(&bv) {
            res.push(hci.get(&bv).unwrap().clone());
            bv = BitVec::new();
        }
    }

    res
}
{% endhighlight %}

# other resources

* Huffman coding is covered well conceptually at [geeksforgeeks](http://www.geeksforgeeks.org/greedy-algorithms-set-3-huffman-coding/).

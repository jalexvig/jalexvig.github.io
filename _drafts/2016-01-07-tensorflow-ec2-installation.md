---
layout: post
title: "tensorflow ec2 install"
categories: blog
excerpt: "Installing tensor for Python3 on a GPU enabled g2.2xlarge Amazon ec2 instance"
tags: [tensorflow, ec2, python3]
---

# intro

So I don't have a GPU, and I want to run models that I've built in tensorflow faster. Luckily we can spin up an ec2 instance that has a GPU!

Some of this information comes from [the relevant secion on the tensorflow docs](https://www.tensorflow.org/versions/master/get_started/os_setup.html#optional-install-cuda-gpus-on-linux).

# setup

Update and upgrade.

{% highlight bash %}
sudo apt-get update
sudo apt-get dist-upgrade
{% endhighlight %}

Download essentials.

{% highlight bash %}
sudo apt-get install git python3-pip libopenblas-dev swig
sudo pip3 install numpy
sudo apt-get install gcc g++ gfortran build-essential linux-image-generic  python3-dev
{% endhighlight %}

Download and install CUDA.

{% highlight bash %}
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.0-28_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb
rm cuda-repo-ubuntu1404_7.0-28_amd64.deb
sudo apt-get update
sudo apt-get install -y cuda-7-0
{% endhighlight %}

Now this is a pain... You need to [download cuDNN-6.5 v2](https://developer.nvidia.com/rdp/cudnn-archive) (the version that is compatible with tensorflow) which involves registering with NVIDIA. Once you've done that, copy the file over to ec2 instance (using `scp`, cloud storage, etc.).

{% highlight bash %}
tar xvzf cudnn-6.5-linux-x64-v2.tgz
sudo cp cudnn-6.5-linux-x64-v2/cudnn.h /usr/local/cuda/include
sudo cp cudnn-6.5-linux-x64-v2/libcudnn* /usr/local/cuda/lib64
rm cudnn-6.5-linux-x64-v2.tgz
{% endhighlight %}

{% highlight bash %}
echo "export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
{% endhighlight %}

{% highlight bash %}
sudo reboot
{% endhighlight %}

Unfortunately, at the time of this writing, AWS instances [only support CUDA comput capability 3.0](https://groups.google.com/a/tensorflow.org/forum/#!msg/discuss/jRkkvsB1iWA/fZn2PdPgBQAJ). So we are going to need to build and install tensorflow from source.

First clone the latest version of tensorflow (this will also download Google's protobuf as a git submodule).

{% highlight bash %}
git clone --recurse-submodules https://github.com/tensorflow/tensorflow
cd tensorflow
{% endhighlight %}

According to [the docs](https://www.tensorflow.org/versions/master/get_started/os_setup.html#enabling-cuda-30) we have to enable an unofficial setting before configuring.

{% highlight bash %}
TF_UNOFFICIAL_SETTING=1 ./configure
{% endhighlight %}

Now we need to [install Bazel](http://bazel.io/docs/install.html) (Google's build tool) to build tensorflow.

{% highlight bash %}
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install oracle-java8-installer
sudo apt-get install pkg-config zip g++ zlib1g-dev unzip
wget https://github.com/bazelbuild/bazel/releases/download/0.1.3/bazel-0.1.3-installer-linux-x86_64.sh
chmod +x bazel-0.1.3-installer-linux-x86_64.sh
./bazel-0.1.3-installer-linux-x86_64.sh --user
rm bazel-0.1.3-installer-linux-x86_64.sh
{% endhighlight %}

Finally we can build tensorflow.

{% highlight bash %}
bazel build -c opt --config=cuda --spawn_strategy=standalone //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip3 install /tmp/tensorflow_pkg/tensorflow-0.6.0-cp27-none-linux_x86_64.whl
{% endhighlight %}


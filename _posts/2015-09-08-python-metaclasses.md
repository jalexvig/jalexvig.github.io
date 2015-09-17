---
layout: post
title: "metaclasses"
modified:
categories: blog
excerpt: "A quick tutorial on metaclasses"
tags: [python3, metaclass]
image:
  feature:
date: 2015-09-08
---

# [type](<https://docs.python.org/3/library/functions.html#type>)

To understand metaclasses we need to understand Python's `type` function. `type` has two function signatures.

1. With one argument passed, `type` returns the type (class) of the object passed. Keep in mind that classes in Python are objects.

2. With three arguments passed, `type` returns a class. This is the one we will be talking about.

As the documentation states:

> With three arguments, return a new type object. This is essentially a dynamic form of the class statement. The name string is the class name and becomes the __name__ attribute; the bases tuple itemizes the base classes and becomes the __bases__ attribute; and the dict dictionary is the namespace containing definitions for class body and becomes the __dict__ attribute. For example, the following two statements create identical type objects:
> {% highlight python %}
class X:
    a = 1
X = type('X', (object,), dict(a=1))
{% endhighlight %}

# hooks

Metaclasses are subclasses of `type`. Just as you can subclass another class, you can also subclass `type`. Overriding hooks in this subclass will allow you to dynamically change class creation.

1. `__prepare__` is used to initialize the class's namespace. It should return a dictionary-like object (one supporting the `__getitem__` and `__setitem__` methods) that will hold the class's namespace. This hook only exists in Python3.

2. `__new__` is used to create the new metaclass. It is passed the populated namespace.

3. `__init__` is used to initialize attributes of the class.

# example

Here is an example of a metaclass.

{% highlight python %}
   
import collections

class MyMetaClass(type):

    @classmethod
    def __prepare__(mcls, clsname, bases, **kwds):
        print("__prepare__ called")
        print("mcls = ", mcls)
        print("clsname = ", clsname)
        print("bases = ", bases)
        print("kwds = ", kwds, '\n')
        return collections.OrderedDict()

    def __new__(mcls, clsname, bases, namespace, **kwds):

        print("__new__ called")
        print("mcls = ", mcls)
        print("clsname = ", clsname)
        print("bases = ", bases)
        print("namespace = ", namespace)
        print("kwds = ", kwds, '\n')
        upper_attr_dict = {}
        for key, val in namespace.items():
            key_mod = key.upper() if key[:2] != '__' else key
            upper_attr_dict[key_mod] = val

        result = super().__new__(mcls, clsname, bases, upper_attr_dict)
        result.original_namespace = namespace
        return result

    def __init__(cls, clsname, bases, namespace, **kwds):

        print("__init__ called")
        print("cls = ", cls)
        print("clsname = ", clsname)
        print("bases = ", bases)
        print("namespace = ", namespace)
        print("kwds = ", kwds, '\n')

        return super().__init__(clsname, bases, namespace)

    def __call__(cls):
        print("__call__ called")
        print("cls = ", cls)
        print("super = ", super(), '\n')
        return super().__call__()
{% endhighlight %}

The `__prepare__` hook is returning an ordered dictionary. This means that the order in which the classes attributes are defined matters.

The `__new__` hook is changing all non "magic" attributes to be uppercased and storing the original namespace as a lower cased attribute.

The `__init__` hook does what you would guess it does but for one small thing. The namespace that it is passed is the one returned from `__prepare__` not the one created in `__new__` (i.e. the one the class will have).

Now we can use this metaclass to dynamically change how our classes are defined.

{% highlight python %}

class Foo(object, metaclass=MyMetaClass, temp='hi'):

def __new__(cls):
    print("Foo __new__ called")
    return super().__new__(cls)

def __init__(self):
    print("Foo __init__ called")

def my_func(self):
    print("My function")
{% endhighlight %}

By passing the `metaclass` keyword argument to the class, we change the class's creation mechanism from `type` to `MyMetaClass`. Note: metaclass's are specified as a keyword argument in Python3 and using the `__metaclass__` class attribute in Python2.

At the end of our module...

{% highlight python %}

if __name__ == '__main__':

    print('Start executing commands...\n')

    foo = Foo()
    print('Foo.__dict__ = ', Foo.__dict__)
{% endhighlight %}

The resulting output is

{% highlight python %}

__prepare__ called
mcls =  <class '__main__.MyMetaClass'>
clsname =  Foo
bases =  (<class 'object'>,)
kwds =  {'temp': 'hi'} 

__new__ called
mcls =  <class '__main__.MyMetaClass'>
clsname =  Foo
bases =  (<class 'object'>,)
namespace =  OrderedDict([('__module__', '__main__'), ('__qualname__', 'Foo'), ('__new__', <function Foo.__new__ at 0x7f908e96d6a8>), ('__init__', <function Foo.__init__ at 0x7f908d469730>), ('my_func', <function Foo.my_func at 0x7f908d4697b8>), ('a', 3)])
kwds =  {'temp': 'hi'} 

__init__ called
cls =  <class '__main__.Foo'>
clsname =  Foo
bases =  (<class 'object'>,)
namespace =  OrderedDict([('__module__', '__main__'), ('__qualname__', 'Foo'), ('__new__', <function Foo.__new__ at 0x7f908e96d6a8>), ('__init__', <function Foo.__init__ at 0x7f908d469730>), ('my_func', <function Foo.my_func at 0x7f908d4697b8>), ('a', 3)])
kwds =  {'temp': 'hi'} 

Start executing commands...

__call__ called
cls =  <class '__main__.Foo'>
super =  <super: <class 'MyMetaClass'>, <MyMetaClass object>> 

Foo __new__ called

Foo __init__ called

Foo.__dict__ =  {'MY_FUNC': <function Foo.my_func at 0x7f908d4697b8>, '__dict__': <attribute '__dict__' of 'Foo' objects>, 'original_namespace': OrderedDict([('__module__', '__main__'), ('__qualname__', 'Foo'), ('__new__', <function Foo.__new__ at 0x7f908e96d6a8>), ('__init__', <function Foo.__init__ at 0x7f908d469730>), ('my_func', <function Foo.my_func at 0x7f908d4697b8>), ('a', 3)]), '__module__': '__main__', 'A': 3, '__new__': <staticmethod object at 0x7f908e962978>, '__init__': <function Foo.__init__ at 0x7f908d469730>, '__doc__': None, '__weakref__': <attribute '__weakref__' of 'Foo' objects>}

{% endhighlight %}
	
On class definition, the metaclass's three hook methods are called in the correct order. I passed the keyword argument for funsies to show how it works and propagates through the calls to the hook methods.

Then when the `Foo` class is instantiated, the `__call__` hook of the metaclass fires before the class's `__new__` and `__init__` class hooks do. Remember though, if you do override a metaclass's `__call__` hook, the class's `__new__` hook will not be called unless you do it.

# uses

Usually none. Metaclasses are fun to play around with, but unless you are doing something really fancy, you probably won't need them. If you think you do, think again for a bit. How could use superclasses, class attributes, or properties to solve the problem?

Their canonical use case is developing an [ORM](<https://en.wikipedia.org/wiki/Object-relational_mapping>) where you want class creation to trigger table creation in a database. [SQLAlchemy](<http://www.sqlalchemy.org/>) is an example of a Python ORM that uses metaclasses.

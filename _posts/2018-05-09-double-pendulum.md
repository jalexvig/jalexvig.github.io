---
layout: post
title: "double pendulum motion"
categories: blog
excerpt: "Derive the equations of motion for a system of coupled pendulums"
tags: [physics]
---

# intro

This is an implementation of the double pendulum example given in Leonard Susskind's [lecture](https://youtu.be/7SiW_x3cUBo?t=27m4s) on classical mechanics. The goal of this post is to explain the derivation and implementation of the equations of motion for coupled pendulums.

The accompanying Python implementation is [available on GitHub](https://github.com/jalexvig/double_pendulum).

I'll start with the drawing given in Professor Susskind's lecture with one exception. The angle $\alpha$ will be called $\theta_2$ and it will be the angle with respect to the rod securing the apparatus (same relationship as $\theta_1$) rather than the angle with respect to $\theta_1$.

So here's our system:

{:refdef: style="text-align: center;"}
![System](/images/double_pendulum.svg)
*Image courtesy of [JabberWok](https://en.wikipedia.org/wiki/Double_pendulum#/media/File:Double-Pendulum.svg).*
{: refdef}

# derivation

Using the image above and some basic trigonometry we can derive the following equations for the $x$, $y$ positions of the pendulums:

$$
\begin{align}
x_1 &= l_1 \sin \theta_1 \\
y_1 &= l_1 \cos \theta_1 \\
x_2 &= x_1 + l_2 \sin \theta_2 \\
y_2 &= y_1 + l_2 \cos \theta_2
\end{align}
$$

**Note**: In this inertial reference frame the positive y direction is down.

Differentiating with respect to time we get:

$$
\begin{align}
\dot x_1 &= l_1 \dot \theta_1 \cos t1 \\
\dot y_1 &= -l_1 \dot \theta_1 \sin t1 \\
\dot x_2 &= \dot x_1 + l_2 \dot \theta_2 \cos \theta_2 \\
\dot y_2 &= \dot y_1 - l_2 \dot \theta_2 \sin \theta_2
\end{align}
$$

Our Lagrangian is of the form $T-V$ where $T$ is the kinetic energy and $V$ is the potential energy.

Let's calculate the kinetic energy $T$:

$$
\begin{align}

T &= \frac{1}{2} m_1 \left( \dot x_1^2 + \dot y_1^2 \right) + \frac{1}{2} m_2 \left( \dot x_2^2 + \dot y_2^2 \right) \\

  &= \frac{1}{2} m_1 l_1^2 \dot \theta_1^2 + \frac{1}{2} m_2 \left( l_1^2 \dot \theta_1^2 + 2 l_1 \dot \theta_1 l_2 \dot \theta_2 \left( \cos \theta_1 \cos \theta_2 + \sin \theta_1 \sin \theta_2 \right) + l_2^2 \dot \theta_2^2 \right) \\

  &= \frac{1}{2} l_1^2 \dot \theta_1^2 \left( m_1 + m_2 \right) + \frac{1}{2} m_2 \left( 2 l_1 \dot \theta_1 l_2 \dot \theta_2 \cos \left( \theta_1 - \theta_2 \right) + l_2^2 \dot \theta_2^2 \right) \\

\end{align}
$$

And now the potential energy $V$:

$$
\begin{align}

V &= -m_1 g y_1 - m_2 g y_2 \\

  &= -\left( m_1 + m_2 \right) g l_1 \cos \theta_1 - m_2 g l_2 \cos \theta_2 \\

\end{align}
$$

Now we can compute our Lagrangian $\mathcal{L}$:

$$
\begin{align}

\mathcal{L} &= T - V \\

            &= \frac{1}{2} l_1^2 \dot \theta_1^2 \left( m_1 + m_2 \right) + \frac{1}{2} m_2 \left( 2 l_1 \dot \theta_1 l_2 \dot \theta_2 \cos \left( \theta_1 - \theta_2 \right) + l_2^2 \dot \theta_2^2 \right) + \left( m_1 + m_2 \right) g l_1 \cos \theta_1 + m_2 g l_2 \cos \theta_2

\end{align}
$$

Time to get the terms for the Euler-Lagrange equation for the coordinate $\theta_1$:

$$
\begin{align}

P_{\theta_1} &= l_1^2 \dot \theta_1 \left( m_1 + m_2 \right) + m_2 l_1 l_2 \dot \theta_2 \cos \left( \theta_1 - \theta_2 \right) \\

\dot P_{\theta_1} &= l_1^2 \ddot \theta_1 \left( m_1 + m_2 \right) + m_2 l_1 l_2 \left( \ddot \theta_2 \cos \left( \theta_1 - \theta_2 \right) - \dot \theta_2 \sin \left( \theta_1 - \theta_2 \right) \left( \dot \theta_1 - \dot \theta_2 \right) \right) \\

\frac {\partial{\mathcal{L}}} {\partial{\theta_1}} &= -m_2 l_1 td1 l_2 td2 \sin (\theta_1 - \theta_2) - (m_1 + m_2) g l_1 \sin \theta_1 \\

0 &= \frac {\partial{\mathcal{L}}} {\partial{\theta_1}} - \dot P_{\theta_1} \\

  &= l_1 tdd1 (m_1 + m_2) + m_2 l_2 (tdd2 \cos (\theta_1 - \theta_2) + td2^2 \sin (\theta_1 - \theta_2)) + (m_1 + m_2) g \sin \theta_1

\end{align}
$$

And now the terms for the Euler-Lagrange equation for the coordinate $\theta_2$:

$$
\begin{align}

P_{\theta_2} &= m_2 l_1 l_2 \dot \theta_1 \cos \left( \theta_1 - \theta_2 \right) + m_2 l_2^2 \dot \theta_2 \\

\dot P_{\theta_2} &= m_2 l_1 l_2 \left( \ddot \theta_1 \cos \left( \theta_1 - \theta_2 \right) - \dot \theta_1 \sin \left( \theta_1 - \theta_2 \right) \left( \dot \theta_1 - \dot \theta_2 \right) \right) + m_2 l_2^2 \ddot \theta_2 \\

\frac {\partial{\mathcal{L}}} {\partial{\theta_2}} &= m_2 l_1 l_2 \dot \theta_1 \dot \theta_2 \sin \left( \theta_1 - \theta_2 \right) - m_2 g l_2 \sin \theta_2 \\

0 &= \frac {\partial{\mathcal{L}}} {\partial{\theta_2}} - \dot P_{\theta_2} \\

  &= m_2 l_2 \ddot \theta_2 + m_2 l_1 \left( \ddot \theta_1 \cos \left( \theta_1 \theta_2 \right) - \dot \theta_1^2 \sin\left( \theta_1 - \theta_2 \right) \right) + m_2 g \sin \theta_2

\end{align}
$$

Solving for $\ddot \theta_1$ and $\ddot \theta_2$ we get:

$$
\begin{align}

\ddot \theta_1 &= \frac{g m_{1} \sin{\left (\theta_{1} \right )} + \frac{g m_{2}}{2} \sin{\left (\theta_{1} \right )} + \frac{g m_{2}}{2} \sin{\left (\theta_{1} - 2 \theta_{2} \right )} + \frac{l_{1} m_{2}}{2} \dot \theta_{1}^{2} \sin{\left (2 \theta_{1} - 2 \theta_{2} \right )} + l_{2} m_{2} \dot \theta_{2}^{2} \sin{\left (\theta_{1} - \theta_{2} \right )}}{l_{1} \left(- m_{1} + m_{2} \cos^{2}{\left (\theta_{1} - \theta_{2} \right )} - m_{2}\right)} \\

\ddot \theta_2 &= \frac{- g m_{1} \sin{\left (\theta_{2} \right )} + g m_{1} \sin{\left (2 \theta_{1} - \theta_{2} \right )} - g m_{2} \sin{\left (\theta_{2} \right )} + g m_{2} \sin{\left (2 \theta_{1} - \theta_{2} \right )} + 2 l_{1} m_{1} \dot \theta_{1}^{2} \sin{\left (\theta_{1} - \theta_{2} \right )} + 2 l_{1} m_{2} \dot \theta_{1}^{2} \sin{\left (\theta_{1} - \theta_{2} \right )} + l_{2} m_{2} \dot \theta_{2}^{2} \sin{\left (2 \theta_{1} - 2 \theta_{2} \right )}}{2 l_{2} \left(m_{1} - m_{2} \cos^{2}{\left (\theta_{1} - \theta_{2} \right )} + m_{2}\right)}

\end{align}
$$

**Note**: I didn't do some crazy mental variable substitutions/arithmetic. I got these expressions using [sympy](http://www.sympy.org/):

{% highlight python %}
import sympy

s = 'l_1 theta_dd_1 m_1 m_2 l_2 theta_dd_2 theta_1 theta_2 theta_d_2 theta_d_1 g'
l_1, theta_dd_1, m_1, m_2, l_2, theta_dd_2, theta_1, theta_2, theta_d_2, theta_d_1, g = sympy.symbols(s)
expr1 = l_1 * theta_dd_1 * (m_1 + m_2) + m_2 * l_2 * (theta_dd_2 * sympy.cos(theta_1 - theta_2) + theta_d_2 ** 2 * sympy.sin(theta_1 - theta_2)) + (m_1 + m_2) * g * sympy.sin(theta_1)
expr2 = m_2 * l_2 * theta_dd_2 + m_2 * l_1 * (theta_dd_1 * sympy.cos(theta_1 - theta_2) - theta_d_1 ** 2 * sympy.sin(theta_1 - theta_2)) + m_2 * g * sympy.sin(theta_2)
theta_dd_1_expr = sympy.solveset(expr1, theta_dd_1).args[0]
theta_dd_2_eval = sympy.solveset(expr2.subs(theta_dd_1, theta_dd_1_expr), theta_dd_2).args[0]
theta_dd_2_expr = sympy.solveset(expr2, theta_dd_2).args[0]
theta_dd_1_eval = sympy.solveset(expr1.subs(theta_dd_2, theta_dd_2_expr), theta_dd_1).args[0]

print(sympy.latex(sympy.simplify(theta_dd_1_eval)))
print(sympy.latex(sympy.simplify(theta_dd_2_eval)))
{% endhighlight %}

# solution

Now that we have coupled differential equations we can solve for motion. We will need to solve numerically and by reframing this system of second order ODEs as four first order differential equations we will be able to do so:

$$
\begin{align}

\omega_1 &= \dot \theta_1 \\

\omega_2 &= \dot \theta_2 \\

\dot \omega_1 &= \frac{g m_{1} \sin{\left (\theta_{1} \right )} + \frac{g m_{2}}{2} \sin{\left (\theta_{1} \right )} + \frac{g m_{2}}{2} \sin{\left (\theta_{1} - 2 \theta_{2} \right )} + \frac{l_{1} m_{2}}{2} \omega_{1}^{2} \sin{\left (2 \theta_{1} - 2 \theta_{2} \right )} + l_{2} m_{2} \omega_{2}^{2} \sin{\left (\theta_{1} - \theta_{2} \right )}}{l_{1} \left(- m_{1} + m_{2} \cos^{2}{\left (\theta_{1} - \theta_{2} \right )} - m_{2}\right)} \\

\dot \omega_2 &= \frac{- g m_{1} \sin{\left (\theta_{2} \right )} + g m_{1} \sin{\left (2 \theta_{1} - \theta_{2} \right )} - g m_{2} \sin{\left (\theta_{2} \right )} + g m_{2} \sin{\left (2 \theta_{1} - \theta_{2} \right )} + 2 l_{1} m_{1} \omega_{1}^{2} \sin{\left (\theta_{1} - \theta_{2} \right )} + 2 l_{1} m_{2} \omega_{1}^{2} \sin{\left (\theta_{1} - \theta_{2} \right )} + l_{2} m_{2} \omega_{2}^{2} \sin{\left (2 \theta_{1} - 2 \theta_{2} \right )}}{2 l_{2} \left(m_{1} - m_{2} \cos^{2}{\left (\theta_{1} - \theta_{2} \right )} + m_{2}\right)}

\end{align}
$$

We'll use the fourth order [Runge-Kutta method](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) as implemented in [the Scipy ODE solver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html):

{% highlight python %}

def double_pend(
    y: list,
    t: float,
    m_1: float,
    l_1: float,
    m_2: float,
    l_2: float,
    g: float
) -> list:
    """
    Calculate one step for coupled pendulums.

    Args:
        y: (theta_1, theta_d_1, theta_2, theta_d_2) at this timestep.
        t: Time (unused).
        m_1: Mass 1.
        l_1: Length of pendulum 1.
        m_2: Mass 2.
        l_2: Length of pendulum 2.
        g: Gravity constant.

    Returns:
        Derivative estimates for each element of y.
    """

    theta_1, theta_d_1, theta_2, theta_d_2 = y

    theta_dd_1 = (g*m_1*sin(theta_1) + g*m_2*sin(theta_1)/2 + g*m_2*sin(theta_1 - 2*theta_2)/2 + l_1*m_2*theta_d_1**2*sin(2*theta_1 - 2*theta_2)/2 + l_2*m_2*theta_d_2**2*sin(theta_1 - theta_2))/(l_1*(-m_1 + m_2*cos(theta_1 - theta_2)**2 - m_2))
    theta_dd_2 = (-g*m_1*sin(theta_2) + g*m_1*sin(2*theta_1 - theta_2) - g*m_2*sin(theta_2) + g*m_2*sin(2*theta_1 - theta_2) + 2*l_1*m_1*theta_d_1**2*sin(theta_1 - theta_2) + 2*l_1*m_2*theta_d_1**2*sin(theta_1 - theta_2) + l_2*m_2*theta_d_2**2*sin(2*theta_1 - 2*theta_2))/(2*l_2*(m_1 - m_2*cos(theta_1 - theta_2)**2 + m_2))

    dydt = [theta_d_1, theta_dd_1, theta_d_2, theta_dd_2]

    return dydt


import functools
import numpy as np

system_params = {
    'm_1': 1,
    'l_1': 1,
    'm_2': 1,
    'l_2': 1,
    'g': 9.81
}

pend_func = functools.partial(double_pend, **system_params)
y0 = [np.pi / 2, 0, np.pi, 0]
t = np.linspace(0, 20, 1000)

theta_1, theta_d_1, theta_2, theta_d_2 = odeint(pend_func, y0, t).T

{% endhighlight %}

We can then convert these into x, y coordinates:

{% highlight python %}

l_1, l_2 = system_params['l_1'], system_params['l_2']

x_1 = l_1 * np.sin(theta_1)
y_1 = l_1 * np.cos(theta_1)
x_2 = x_1 + l_2 * np.sin(theta_2)
y_2 = y_1 + l_2 * np.cos(theta_2)

{% endhighlight %}

Now we have an initial value problem so let's pick some initial values:

$$
\begin{align}

\theta_1 \left( 0 \right) &= \frac {\pi} {2} \\
\omega_1 \left( 0 \right) &= 0 \\
\theta_2 \left( 0 \right) &= \pi \\
\omega_2 \left( 0 \right) &= 0

\end{align}
$$

Here's an animation of this system (plotted with `matplotlib`):

<iframe width="560" height="315" src="https://www.youtube.com/embed/p5q6K3A98MY" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

Feel free to checkout the code and play with the system by changing the parameters and initial values.

# additional resources

* All the code used to generate the double pendulum motions can be found on [GitHub](https://github.com/jalexvig/double_pendulum).
* Leonard Susskind's [lecture](https://youtu.be/7SiW_x3cUBo?t=27m4s) that inspired this example and [the course](http://theoreticalminimum.com/courses/classical-mechanics/2011/fall/lecture-5) containing the lecture.
* [MyPhysicsLab](https://www.myphysicslab.com/pendulum/double-pendulum-en.html) solves the ODEs in javascript so you can change the initial conditions and see the corresponding motions in a browser.

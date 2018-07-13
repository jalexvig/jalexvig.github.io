---
layout: post
title: "backdoor roth IRAs"
categories: blog
excerpt: "The value of backdoor roth IRAs"
tags: [IRA, 401k]
---

# Intro

We should all be saving for later in life but how much will that money that we are squirreling away actually be worth in the future? We are going to discuss two methods of getting tax-free growth and how much that can save us. The two methods are:

1. [Backdoor Roth IRA through Traditional IRAs](#backdoor-roth-ira-through-traditional-iras)
2. [Backdoor Roth IRA through 401ks](#backdoor-roth-ira-through-401k)

The basic idea behind both of these is that we sometimes have the option of putting money into retirement accounts where the gains will be taxed (in the future when we draw on them). If we can convert to accounts where we don't have to pay those taxes, then we can save a surprising amount of money.

Abbreviations I will use:
* Traditional IRA (tIRA)
* Roth IRA (rIRA)

# Backdoor Roth IRA through Traditional IRAs

This is the more well known of the two techniques and is applicable for people who have exceeded the rIRA [modified adjusted gross income](https://turbotax.intuit.com/tax-tools/tax-tips/IRS-Tax-Return/What-Is-the-Difference-Between-AGI-and-MAGI-on-Your-Taxes-/INF22699.html) threshold (people whose income after deductions is above a certain amount). The [2017 threshold](https://www.irs.gov/retirement-plans/amount-of-roth-ira-contributions-that-you-can-make-for-2017) is \\$133,000 if you are filing as single and \\$186,000 if you are filing jointly. Note that for there is some wiggle room for partial contibutions if you make slightly less than the above numbers but we will ignore that as this same logic will apply.

So what to do if you make more than these thresholds? You can contribute after-tax dollars (money from your pocket) to a tIRA. Then you can then turn around and convert that tIRA into a rIRA. Sound weird? Sound like you are just contributing to a rIRA? It is and you are. The fact that this is allowed makes no sense. But it is allowed.

That means each year:

1. Contribute after-tax dollars to a tIRA.
2. Convert that tIRA to a rIRA. (You might have to pay a very small amount that corresponds to taxes on gains made between 1 and 2).

The one caveat to this process occurs when you have pre-tax dollars as well as after-tax dollars in the tIRA before you convert it to a rIRA. This won't happen once you get in the habit of doing steps 1 and 2 each year but you might start out in this situation. Annoyingly you can't just move the after-tax dollars to a rIRA. You would need to follow the ["pro-rata rule"](https://www.irs.gov/retirement-plans/rollovers-of-after-tax-contributions-in-retirement-plans). Essentially this rule taxes the amount you move proportionally to the ratio of pre-tax dollars to total dollars in the account. If this doesn't make sense don't worry about it because we shouldn't needing it. If you are in this boat though (you have pre-tax dollars in your tIRA already) try the following...

### Rollover your Traditional IRA into a 401k

Many 401ks allow you to rollover IRAs into them. In this case we can rollover the **pre-tax** dollars in the tIRA into your 401k. You can only roll-over pre-tax dollars according to [this IRS publication](https://www.irs.gov/pub/irs-pdf/p590a.pdf) (p. 21). Once you have done this you can convert the tIRA with the remaining after-tax dollars into a rIRA as explained above.

# Backdoor Roth IRA through 401k

This technique is less well known but even more valuable because you can convert a lot more money (and therefore you can save a lot more money). Many companies allow you to contribute after-tax dollars to a 401k. Luckily for us a lot of the time we can then convert this money into a rIRA. There are two ways this can work...

1. Our employer puts after-tax 401k contributions into a separate subaccount. This is the jackpot since that subaccount can be rolled over directly into a rIRA.
2. Our pre-tax and after-tax 401k contributions are in the same account.

Check with your company to see if 1 applies.

If not, check to see if you are allowed to convert your entire 401k to IRAs. Ask your benefits department if this is possible. The outgoing money is still subject to the "pro-rate rule" mentioned earlier if you only move some of it. The best thing to do then is to convert your entire 401k by moving your after-tax 401k dollars into a rIRA and your pre-tax 401k dollars into a tIRA. If you do these at the same time [no tax is imposed on any of that money](https://www.irs.gov/retirement-plans/rollovers-of-after-tax-contributions-in-retirement-plans)!

**Note**: You are allowed to convert your 401k to IRAs when you leave your company so if you are planning on leaving your company soon and none of the above works for you, then you can convert your entire 401k to IRAs when you leave.

# Results

So is doing all of this actually useful? In figuring out the answer we are going to assume a 7% annual return on investment invested ($r$) and a 2% annual rate of inflation ($i$). This will allow us to calculate the present value of the money we save. If the "present value of money" is confusing think about this: \\$100 is worth less today than it was 100 years ago (just imagine your grandfather saying "I used to be able a can of soda for a nickel..." and you get the idea).

We can calculate the amount of money that will be in the either of those accounts after $n$ years.

$$ Money = A * (1 + r - i) ^ n + A * (1 + r - i) ^ {n - 1} + A * (1 + r - i) ^ {n - 2} + ... + A * (1 + r - i) + A $$

Simplifying this gives us 

$$ Money = A * \frac{(1 + r - i) ^ {n + 1} - 1}{r - i} $$

where A is the present value of the amount of money that we contribute each year. We are assuming it is constant (which would mean that contribution limits scale with inflation... a reasonable assumption).

To get the taxable amount we need to subtract the amount of after-tax dollars we put in (since those won't be taxed when we take distributions).

$$ Taxable = Money - (A * n) $$

Let's assume a tax rate $t$ of 30% (between both state and federal taxes). Then $Tax$ is the amount we would owe in taxes on these retirement accounts... if we didn't do the two things suggested above.

$$ Tax = t * Taxable $$

### Backdoor Roth IRA through Traditional IRAs

Let's say we contribute the max to our tIRA but, because we are over the income threshold, put in after-tax dollars. At the end of 35 years

$$ Money = 5500 * \frac{(1 + 0.07 - 0.02) ^ {35 + 1} - 1}{0.07 - 0.02} = $527,100 $$

$$ Taxable = 527100 - (5500 * 35) = $334,600 $$

$$ Tax = 0.30 * 334600 = $100,380 $$

This means that we save just over \\$100,000 dollars by rolling over our tIRAs to rIRAs every year. Feel free to modify the variables to more accurately represent your circumstances.

**Note**: You can actually contribute more to your IRA (called a "catch-up" contribution) at age 50 or higher. This means that all of the calculated numbers above will be slightly increased. Because I didn't want to make an assumption of when an individual started to work, this is not included.

### Backdoor Roth IRA through 401k

Here's where things really start to compound. The maximum pre-tax contribution an individual can make to his/her 401k each year is currently \\$18,000. The maximum total contributions that can be made into a 401k is \\$53,000. Employers will typically match some percentage of the individual's pre-tax contribution. This varies wildly and you should check with your benefits department to figure this out. I am going to assume that an employer matches \\$5,000. So that means we have \\$30,000 (our $A$) that we can contribute to our 401k with after-tax dollars.

$$ Money = 30000 * \frac{(1 + 0.07 - 0.02) ^ {35 + 1} - 1}{0.07 - 0.02} = $2,875,090 $$

$$ Taxable = 2875090 - (30000 * 35) = $1,825,090 $$

$$ Tax = 0.30 * 1825090 = $547,527 $$

Note that you can also multiply the calculated numbers from the previous section by $\frac{30000}{5500}$ to scale for our new $A$ and you would get these same results.

This means that we save **over half a million dollars** by contributing after-tax dollars to our 401ks and then converting that money to rIRAs each year. Again, feel free to modify the variables to more accurately represent your circumstances.

# Other resources

* Backdoor rIRA through tIRAs is covered quite well [here](https://thefinancebuff.com/the-backdoor-roth-ira-a-complete-how-to.html).

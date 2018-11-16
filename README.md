# Study-AB-Testing

## [Intro: SO...Not Enough Samples ?] 
### Let's do inferential stats! Estimate population parameter from samples.
#### Estimation techniques for finding 'population parameter θ'
 - Maximum Likelihood Estimation
 - Method of Moments Estimation
 - Bayesian Estimation (easier to communicate with audience than awkawrd frequentists' methods)
 - *Aggregate approach (CI, Hypothesis_test)
 - *Individual approach (machine learning, but does need more samples, doesn't care finding "good statistics")

> Hypothesis tests and CI: 
<img src="https://user-images.githubusercontent.com/31917400/48592904-7aabe880-e942-11e8-9f7b-339edb6a60b2.jpg" />  

**Hypothesis tests**(Significance tests) take an `aggregate approach` towards the conclusions made based on data, as these tests are aimed at understanding population parameters(which are **aggregate population values** -_-;). `A/B testing` is another name for "the significance test for **two sample proportions**". For example, it's used for comparing `ClickThroughRates`. 
 - The critical region `α` (such as 0.10, 0.05, 0.01) for the hypothesis test is the region of rejection in the distribution. 
   - if one tailed (H0: `μ <= sth`): `1 - cdf` < `α`
   - if two tailed (H0: `μ = sth`): `-α/2` < `1 - cdf` < `+α/2`(too big difference)
   - In `N(0,1)`, 1.64, 1.96, 2.58 are called `threshold Z_value` or `cut_off_value` pointing significance level of 90%, 95%, 99%, and the test statistics of H0 is just a `Z_value`. 
 - **p_value of H0, a probability value `1 - F(x) where x is my test statistics`, is hinting about some information on our population parameter.** Is it wider then critical region `α` or not?  
 - Two Components
   - 1) **Test statistic of H0** for example, `some point(Z_value) in the N(0,1)` 
     - Your test statistics is all about `(obv - claim)/SE`  
   - 2) **Deriving its Null distribution**(the distribution of the test statistic under the assumption that the null hypothesis is true) for example, `N(0,1)`
   - Placing the `test statistics of H0` on the pdf chart and see where it is located. 
   
**CI** is used to bridge the gap between statistical sampling of an event and measuring every single instance of it. It tells you **how likely a small sample is to be correct when applied to the overall population.** They are derived using 'the bell curve concept' and aim to figure out **where most (90%) data would fall within 2 standard deviations from the mean.**
<img src="https://user-images.githubusercontent.com/31917400/47932798-8b804700-deca-11e8-82b6-b5261d7bde47.jpg" />  

> Machine learning techniques: 
 - It takes an `individual approach` towards making conclusions, as they attempt to predict an outcome for each specific data point.

### 1. Basic Sampling Test
 - When we have large sample sizes (n > 30), and variance is known: **z-test**
 - When we have less sample sizes(n < 30), and the variance is unknown: **t-test**
 - Both methods assume a **normal distribution** of the data(Independent, Identically distributed)
<img src="https://user-images.githubusercontent.com/31917400/38500148-76cd5148-3c01-11e8-85a0-a90adf7ed19e.jpg" />  

 - So in sampling distribution, always think about `sample size` and `variance`. In case of t-distribution,
<img src="https://user-images.githubusercontent.com/31917400/48620729-018bb000-e999-11e8-8c69-6b1d57b4f486.jpg" />  

__ > Note: How about `ClickThroughRates`??__
 - Why comparing **two sample means** instead of going directly to comparing **two sample proportions** ? Because two sample proportions are also two sample means. WTF?! Let me explain: When the RV follows a Bernoulli-Dist(1 / 0), then the **sample mean**(the size of '1' out of n times) becomes the sample proportion, and we can get z-statistics.
 - We can compare two sample means and in this case, but cannot use t-test. We are able to use t-test when the test statistic we have follows the Student's t-distribution under the assumption that the null hypothesis is true. However, here the test statistic's null distribution is not t-distribution, but z-distribution because it's about the proportion!!!
 
### 2. Chi-Sqr test: 
It's expected value: `E[x] = df` (몇개나 더했어?) - the sum of squares`∑(x-μ)^2` of independent standard normals is a random variable that fairly naturally arises in many contexts, and that is something we would like to have a name for. The degrees of freedom relates to the number of independent normals involved(or squared then summed) and each of those squared components has mean `1`. 
#### 1> In a contingency table `(along cols: categories, along rows: each group)`, values are all about `Countings`.
### We use Chi-Sqr to test relationships between `categorical variables`.
we want to know: 
 - between two groups(rows), there is any significant **difference**?
   - In this case our **df** is `r-1`(NO.of variables to compare, one way)
 - between the groups(rows) and the categorical variables(columns), there is any **association**?
   - `H0: No connection or independent` / `H1: dependent`
   - In this case our **df** is `(r-1)(c-1)`(NO.of variables to compare, two way)  
 - > This is not about population mean or variance, but about the correlation?  
 - > if the accounts in the categories are binary(Bernulli) or multinary(A,B,C,D,F), and all values are playing with **frequency**...we first assume H0 is true, then ...
   - Values in Chi-sqr Dist are always (+).
   - Like t-Dist, it only has one parameter: df
   - Location/Scale: 0 / 1 by default
<img src="https://user-images.githubusercontent.com/31917400/38503101-0c07da10-3c09-11e8-92f4-114707454eaa.jpg" />  

 - > What if we should compare more than 2 groups(2+ rows in a contingency table)?
   - For P_Value: even running the same experiment twice, the odds to get significant p_value would increase..(FP). This is why Frequentist's methods are awkawrd...Here, we need to fix it using..
     - Bonferroni's Correction: Alpha_new = Alpha / # of tests
   - For Method: 
     - **pairwise:** compare every group against every other group.
       - Alpha_new = Alpha / choose(N,2)
     - **One VS the rest:** 
     
#### 2> Population Variance Estimation
 - __From a single sample:__
 <img src="https://user-images.githubusercontent.com/31917400/47957643-d759eb80-dfb1-11e8-84c6-5153362a14c1.jpg" />  

   - Let's say we have set a timed-sales-goal where `population_SD` is less than 21 days(so variance upper limit is `441`). Then we randomly select 15 sales records. Based on this sample, the following is obtained: `n = 15`, `sample_mean = 162 days`, `sample_variance = 582`, `sample_SD = 24 days` and our focus is variance. So our sample_SD is 24 days which exceeds the goal. But this exceed is significant? so our goal `441` is too much(which means too small)? Tell me. 
   <img src="https://user-images.githubusercontent.com/31917400/47957077-9a88f700-dfa7-11e8-885f-d6830550de2d.jpg" />  

   - While the sampling distribution of `mean` follows the `Normal`, the sampling distribution of `variance` follows the `Chi-Sqr`.  
   <img src="https://user-images.githubusercontent.com/31917400/47957744-c1e5c100-dfb3-11e8-8f46-dbef4064f2ab.jpg" /> 
   
__[Note]: If From two samples,__ **F-Test** for Equality of two sample variances
 - Is the variance the same for both soup brands?
<img src="https://user-images.githubusercontent.com/31917400/47958023-55ba8b80-dfba-11e8-91b2-3eb73030f733.jpg" />

#### 3> Goodness of fit
 - Use when you have a single **categorical** sample from a population. It is used to determine whether sample data are consistent with a hypothesized distribution(**proportion distribution**), i.e to test the hypothesis H0 that a set of observations is consistent with a given **probability** distribution.`Does this sample come from this distribution?` Yeah, it claims about population proportion. It's a **Non-parametric** test. 
   - `sample size` in each level of the category > `5`
   - so..each category takes up some proportion area on the distribution(pdf) chart..and data point on x-axis belong to each category.. 
   - `H0: The data are consistent with a specified distribution.`
   <img src="https://user-images.githubusercontent.com/31917400/47964916-16cb1b00-e038-11e8-893f-805af7da9452.jpg" />

### 3. F-Test
F-Distribution(Variance-Ratio-Distribution) defines the ratio of the two variances(of the two normally distributed samples). It has a density = ratio of gamma function(the sum of exponential) and two parameters = df `m` for the `numerator` and df `n` for the `denominator`. Let's say we have a 'iid' sample_A and a 'iid' sample_B and both are **independent** (basic assumptions). 
<img src="https://user-images.githubusercontent.com/31917400/48623932-0ead9c80-e9a3-11e8-8377-5cae67faa5ae.jpg" />

 - Use when testing the hypothesis of the equality of two `sample variances` (Chi-Sqr test for a single population variance)
 - Use when testing the hypothesis of the equality of `multiple means` at the same time (ANOVA). 
 - Use when testing the overall significance of the mutiple regression model as a whole(A significant `F-value` indicates a linear relationship between the `Response` and at least one of the `Predictors` so have some hope!). 
  
### 4. Statistical Power
 - **Power** = Sensitivity(TPr) = P(reject H0 | H1 is True) = 1 - FNr
   - reject H0: 'pos'
   - don't reject H0: 'neg'
   - FP: 'type I' error, FPr=P(FP)
   - FN: 'type II' error, FNr=P(FN)
 - **High power** decreases the odds to get FN(type_II)
 - Why quantify power ? 
   - 2 keys: Effect_Size, Sample_Size
     - Effect_Size: 'the difference b/w two Grp'...becomes easy to detect
     - Sample_Size: Power helps determine the sample size we need(by P_Value)

### 5. Popular Questions
 - whether two variables (n = 2) are correlated (i.e., associated) => **Correlation test** between two variables. 
 - whether multiple variables (n > 2) are correlated => **Correlation matrix** between multiple variables. 
 - whether two groups (n = 2) of samples differ from each other => **t-test** (parametric: Need of Dist_Assumption). 
   - sample_mean VS population_mean
   - sample_mean VS sample_mean
 - whether multiple groups (n >= 2) of samples differ from each other => **ANOVA test** as an extension of t-test to compare more than two groups.
 - If things are in a contingency table(counts from the categorical) => **Chi-Sqr test** (parametric ? `No`) 
 - whether the variability of a single sample differ from population variance => **Chi-Sqr test** (parametric ? `No`)
 - whether the variability of two samples differ each other => **F-test** (parametric) to compare the variances of two groups.
 - __Simple Rule of Significance Test:__ 
   - Use `Chi-Sqr_test` if your predictor and your outcome are both **categorical**(e.g., purple vs. white). 
   - Use a `t-test` if your single predictor has **two categorical conditions**(means two samples) and your outcome is **continuous**(e.g., height, weight, etc).
     - it becomes a two sample test...(so use "One_way_ANOVA" for multi-sample`>=3` test")
   - Use `correlation test` or `regression` if both the predictor and the outcome are **continuous**.
### Sampling Distribution of proportion 
Let's say there are two samples - X and Y - and they are **independent Binomially? distributed** with parameters ~ `Bin(n, p)` and `Bin(m, p)`. You know what? `X+Y ~ Bin(n+m, p)`. By this logic, if `X1 ~ Bin(1,p)`, `X2 ~ Bin(1,p)`, ...., then `X1+X2+...Xn ~ Bin(n, p)`. Here, we know that 





### Popular Hypothesis testing (when the data are normally distributed)
 - 1.Testing a **population mean** (One sample t-test).
 - 2.Testing the difference in means (Two sample t-test) with two **independent** samples.
 - 3.Testing the difference before and after some treatment on an the same individual (Paired t-test) with two **dependent** samples.
 - 4.Testing a **population proportion** (One sample z-test).
 - 5.Testing the difference in proportions (Two sample z-test) with two **independent** samples.
 - 6.Comparing the means of multiple groupsssss (ANOVA Test, the involvement of categorical variables).
   - One-Way(factor) ANOVA Test (one response VS one predictor with multiple(`>=3`) conditions)
     - Using the F-distribution, it examines the influence of a single 'numerical' or 'categorical' input variable(X) on the 'numerical' response variable(Y)...whether the mean of some numeric variable differs across the levels of one categorical variable. Do any of the group means differ from one another? 
<img src="https://user-images.githubusercontent.com/31917400/40049943-fd17b908-582d-11e8-8c2e-8bc23c80114a.JPG" />  
     
   - Two-Way(factor) ANOVA Test (one response VS two predictors with multiple(`>=3`) conditions)
     - As an extension of the one-way ANOVA, it examines the influence of 2 different 'categorical' input variables(X) on the 'numerical' response variable(Y). The two-way ANOVA not only aims at assessing the main effect of each independent variable but also if there is any `interaction` between them. https://www.youtube.com/watch?v=ajLdnsLPErE&t=442s
<img src="https://user-images.githubusercontent.com/31917400/40051922-5a9fc8ee-5834-11e8-95c4-3945026a87cc.jpg" />  
     
   - MANOVA Test (Multivariate Analysis of Variance)
     - It helps to answer:
       - Do changes in the independent variable(s) have significant effects on the dependent variables?
       - What are the relationships among the dependent variables?
       - What are the relationships among the independent variables?

This gives t-statistics and P-value (with equal/unequal variance)
```
import scipy.stats as stats

stats.ttest_1samp(a - b, popmean=0)  ## one sample ##
stats.ttest_ind(df['A'], df['B'], equal_var = True)  ## two samples independent ##
stats.ttest_rel(df['A'], df['B'])  ## Paired dependent ##

index_dict = df.groupby('categorical_A').groups  ## it's housing all index of 'numeric_B_values' under the name of 'categ_values'
stats.f_oneway(df['numeric_B'][index_dict['categ_values']], df['numeric_B'][index_dict['categ_values']], ...) ## oneway ANOVA ##

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
formula = 'response ~ C(A) + C(B) + C(A):C(B)'
model = ols(formula, df).fit()
aov_table = anova_lm(model, typ=2) ## twoway ANOVA ##
```
The t-test formula depends on the **sample_mean** and the **sample_sd** of the data. It's basic form is `(The obv - the argu) / SE` where **'sample_sd'** quantifies scatter — how much the values vary from one another, while **'SE'** quantifies how precisely you know the true mean of the population. It takes into account both the value of the **sample_sd** and the **sample size**, thus by definition, **SE** is always smaller than the **sample_sd**.
 - Placing the `test statistics of H0` on the pdf chart and see where it is located. 
<img src="https://user-images.githubusercontent.com/31917400/34945069-df8f793a-f9f9-11e7-8372-3f00bab83b24.jpg" />  

----------------------------------------------------------------------------------------------------------
## However, using our computer,
#### Instead of memorizing how to perform all of these tests, we can find the statistics that best estimates the parameter(s) we want to estimate, we can bootstrap to simulate the sampling distribution. Then we can use our sampling distribution to assist in choosing the appropriate hypothesis.
 
#### Once we set up 'H0', we need to use our data to figure out which hypothesis is true. There are two ways to choose. 
 - __Using C.I:__ where we simulate sampling distribution of our statistics, then we could see if our hypothesis is consistent with what we observe in the sampling distribution.  
 - __Simulating what we believe to be a possible under the H0,__ then seeing if our data is consistent with that.  

# 1) Using C.I & Bootstrapping
__*When estimating population parameters, we build the confidence intervals.__
 - Basically;
   - Increasing your sample size will decrease the width of your confidence interval (by the law of large numbers).
   - Increasing your confidence level (say 95% to 99%) will increase the width of your confidence interval. 
 - C.I. that Capturing pop-mean/proportion
<img src="https://user-images.githubusercontent.com/31917400/34266269-6c003960-e670-11e7-857f-3a755839c4b9.jpg" width="200" height="150" />  

 - C.I. that Capturing the difference in pop-mean/proportion
<img src="https://user-images.githubusercontent.com/31917400/34266277-6e55239c-e670-11e7-924e-212e7c9f7876.jpg" width="290" height="150" /> 

 - C.I. with t-test as the Traditional mean comparison method with sampling - ex>"height"
```
df_samp = df.sample(200)
X1 = df_samp[df_samp[condition] == True]['height'] 
X2 = df_samp[df_samp[condition] == False]['height']

import statsmodels.stats.api as sm
cm = sm.CompareMeans(sm.DescrStatsW(X1), sm.DescrStatsW(X2))
cm.tconfint_diff(usevar='unequal')
```
All of these formula have underlying "assumptions" (Central Limit Theorem - regarding the sampling distribution ie.the distribution of statistics) that may or maynot be true. But **Bootstrapping** does not need the assumptions of these intervals. Bootstrapping only assumes the sample is representitive of the popluation. With large enough sample size, Bootstrapping and the traditional methods would provide the same Confidence Interval.    

__*Bootstrapping and C.I.__
 - We just use a bootstrapping with 10,000 iterations to build a confidence interval !
```
df_samp = df.sample(200)

A_means, B_means, diffs = [], [], []
for i in range(10000):
    bootsamp = df_samp.sample(200, replace = True)
    A_mean = bootsamp[bootsamp[condition] == True]['height'].mean()
    B_mean = bootsamp[bootsamp['condition'] == False]['height'].mean()
    A_means.append(A_mean)
    B_means.append(B_mean)
    diffs.append(A_mean - B_mean)   
```
 - Compute the C.I.
``` 
A_lower, A_upper = np.percentile(A_means, 2.5), np.percentile(A_means, 97.5)
B_lower, B_upper = np.percentile(B_means, 2.5), np.percentile(B_means, 97.5)
diff_lower, diff_upper = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
```
 - See the distribution of parameters and see if the H0 can be rejected. Does H0 is within the bound? 
``` 
plt.hist(A_means);
plt.axvline(x=A_lower, color='r', linewidth=2)
plt.axvline(x=A_upper, color='r', linewidth=2);

plt.hist(B_means);
plt.axvline(x=B_lower, color='r', linewidth=2)
plt.axvline(x=B_upper, color='r', linewidth=2);

plt.hist(diffs);
plt.axvline(x=diff_lower, color='r', linewidth=2)
plt.axvline(x=diff_upper, color='r', linewidth=2);
```

# 2) Simulating From the Null Hypothesis

__*We assume the Null is true, then see what the sampling distribution would look like if we were to simulate (from) the closest value under the Null (to) the Alternative. In this case, we simulate from a "Normal Distribution" because by the central limit theorem.  

 - Let's say
<img src="https://user-images.githubusercontent.com/31917400/34455227-605ab9b0-ed72-11e7-82b9-d8df9c0babbf.jpg" />  

 - The hypothesized mean at 70 and the SD of our sampling distribution would follow it. 

 - First, Get the SD
``` 
df_samp = df.sample(200)

mu_pool = []
for i in range(10000):
    bootsamp = df_samp.sample(200, replace=True)
    mu_pool.append(bootsamp.query('drinks_coffee==True')['height'].mean())
        
np.std(mu_pool)
```
 - Next, 10,000 Sampling from ~ N(mu, SD, size=10000)
```
null_vals = np.random.normal(70, np.std(mu_pool), 10000)
plt.hist(null_vals)
```
 - Now we can ask a question "where the sample mean falls in this distribution ?" 
```
sample_mean = df_samp.query('drinks_coffee==True')['height'].mean()
plt.hist(null_vals)
plt.axvline(sample_mean, color='r')
```
<img src="https://user-images.githubusercontent.com/31917400/34455271-62f4bdaa-ed73-11e7-9b0c-5b1ad4971d38.jpg" width="300" height="200" /> 

 - With our sample mean so far out in the tail, we intuitively (by eyeballing) we don't reject H0.
 - The definition of a p-value is the probability of the acception of null hypothesis. It is the area created by the t-statistics of the data. If we calculate 'P-value' here, the result is 1.0 
```
(null_vals > df_samp.query('drinks_coffee==True')['height'].mean()).mean() 
```
Note here, '<>' direction follows that of 'H1'. For example,  
<img src="https://user-images.githubusercontent.com/31917400/34455473-c3cc5abc-ed77-11e7-9fe2-78e0c605842e.jpg" />  
```
(null_vals < df_samp.query('drinks_coffee==True')['height'].mean()).mean() 
```
but
<img src="https://user-images.githubusercontent.com/31917400/34455494-50db9274-ed78-11e7-94f0-aa2d6bc0a4a7.jpg" />  
```
null_mean=70
sample_mean = df_samp.query('drinks_coffee==True')['height'].mean()

(null_vals < df_samp.query('drinks_coffee==True')['height'].mean()).mean() + (null_vals > null_mean + (null_mean-sample_mean)).mean()
```
 
## What if our sample is large?
One of the most important aspects of interpreting any statistical results (and one that is frequently overlooked) is assuring that your sample is **truly representative** of your population of interest. Particularly in the way that data is collected today in the age of computers, `response bias` is so important to keep in mind. In the 2016 U.S election, polls conducted by many news media suggested a staggering difference from the reality of poll results. 
> Two things to consider
 - a) Is my sample representative of the population?
 - b) What is the impact of large sample size on my result? (with large sizes, everything will be statistically significant..then we'd always choose H1---[Type-I. Error]) 

### Multi-testing Correction
When performing more than one hypothesis test, your type-I error compounds. In order to correct for this, a common technique is called the `Bonferroni correction`. This correction is very conservative, but says that your new type-I error rate should be the error rate you actually want divided by the number of tests you are performing. Therefore, if you would like to hold an **allowable type-I error rate of 1%** (99% confidence means alpha=0.01) for each of 20 hypothesis tests, the Bonferroni corrected rate would be 0.01/20 = 0.0005. This would be the new rate you should use as your comparison to the p-value for each of the 20 tests to make your decision.

### Finite Population Correction
When sampling **without replacement**(like HyperGeometric instead Binomial) from more than `5%` of a finite population, you need to multiply the SE by this correction `sqrt((N-n) / (N-1))` because under these circumstances, the Central Limit Theorem doesn’t hold. **FPC** captures the difference between sampling **with** replacement and sampling **without** replacement. 



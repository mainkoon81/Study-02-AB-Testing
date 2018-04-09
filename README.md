# Study-AB-Testing

## [Intro: SO...Not Enough Samples ?] 
#### Estimation techniques for finding "good statistics" or 'population parameter'
 - Maximum Likelihood Estimation
 - Method of Moments Estimation
 - Bayesian Estimation (easier to communicate with audience than awkawrd frequentists' methods)
 - *Aggregate approach (CI, Hypothesis_test)
 - *Individual approach (machine learning, but does need more samples, doesn't care finding "good statistics")

> Confidence intervals and Hypothesis tests: 
 - It takes an `aggregate approach` towards the conclusions made based on data, as these tests are aimed at understanding population parameters (which are aggregate population values).

> Machine learning techniques: 
 - It takes an `individual approach` towards making conclusions, as they attempt to predict an outcome for each specific data point.

#### Statistical Test
 - When we have large sample sizes (n > 30), and variance is known: **z-test**
 - When we have less sample sizes(n < 30), and the variance is unknown: **t-test**
 - Both methods assume a **normal distribution** of the data(Independent, Identically distributed)
<img src="https://user-images.githubusercontent.com/31917400/38500148-76cd5148-3c01-11e8-85a0-a90adf7ed19e.jpg" />  

#### Chi-Sqr test: 
 - In a contingency table(along cols: categories, along rows: each group), we want to know between two groups(rows), there is any significant difference..or between the groups(row) and the categorical variable(column), there is any association..(H0: No connection)
 - if the accounts in the categories are binary(Bernulli) or multinary(A,B,C,D,F), and all values are playing with **frequency**...we first assume H0 is true, then ...
   - Values in Chi-sqr Dist are always (+).
   - Like t-Dist, it only has one parameter: df
   - Location/Scale: 0 / 1 by default
<img src="https://user-images.githubusercontent.com/31917400/38503101-0c07da10-3c09-11e8-92f4-114707454eaa.jpg" />  

 - What if we should compare more than 2 groups(2+ rows in a contingency table)?
   - For P_Value: even running the same experiment twice, the odds to get significant p_value would increase...then
     - Bonferroni's Correction: Alpha_new = Alpha / # of tests
   - For Method: 
     - **pairwise:** compare every group against every other group.
       - Alpha_new = Alpha / choose(N,2)
     - **One VS the rest:** 
#### Statistical Power
 - Power = Sensitivity(TPr) = P(reject H0 | H1 is True) = 1 - FNr
   - reject H0: 'pos'
   - don't reject H0: 'neg'
   - FP: 'type I' error, FPr=P(FP)
   - FN: 'type II' error, FNr=P(FN)
   - High power decreases the odds to get FN(type_II)
 - Why quantify power ? 
   - 2 keys: Effect_Size, Sample_Size
   - Effect_Size: 'the difference b/w two Grp'...becomes easy to detect
   - Sample_Size: Power helps determine the sample size we need(by P_Value)

#### Popular Questions
 - whether two variables (n = 2) are correlated (i.e., associated) => **Correlation test** between two variables
 - whether multiple variables (n > 2) are correlated => **Correlation matrix** between multiple variables
 - whether two groups (n = 2) of samples differ from each other => **t-test** (parametric)
 - whether multiple groups (n >= 2) of samples differ from each other => **ANOVA test** as an extension of t-test to compare more than two groups.
 - whether the variability of two samples differ => **F-test** (parametric) to compare the variances of two groups.

#### Popular Hypothesis testing (when the data are normally distributed)
 - 1.Testing a population mean (One sample t-test).
 - 2.Testing the difference in means (Two sample t-test) with two **independent** samples
 - 3.Testing the difference before and after some treatment on an the same individual (Paired t-test) with two **dependent** samples
 - 4.Testing a population proportion (One sample z-test)
 - 5.Testing the difference between population proportions (Two sample z-test)
 - 6.Comparing the means of more than two groups (ANOVA Test)
   - One-Way ANOVA Test (one response VS one predictor)
     - Using the F distribution, it examines the influence of a single 'numerical' or 'categorical' input variable(X) on the 'numerical' response variable(Y)...whether the mean of some numeric variable differs across the levels of one categorical variable. Do any of the group means differ from one another? 
   - Two-Way ANOVA Test (one response VS two predictors)
     - As an extension of the one-way ANOVA, it examines the influence of 2 different 'categorical' input variables(X) on the 'numerical' response variable(Y). The two-way ANOVA not only aims at assessing the main effect of each independent variable but also if there is any interaction between them.
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
The t-test formula depends on the **mean** and the **SD** of the data. It's basic form is `(The obv - the argu) / SE` where **'SD'** quantifies scatter — how much the values vary from one another, while **'SE'** quantifies how precisely you know the true mean of the population. It takes into account both the value of the **SD** and the **sample size**, thus by definition, **SE** is always smaller than the **SD**.   
<img src="https://user-images.githubusercontent.com/31917400/34945069-df8f793a-f9f9-11e7-8372-3f00bab83b24.jpg" />  

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

#### Multi-testing Correction
When performing more than one hypothesis test, your type-I error compounds. In order to correct for this, a common technique is called the `Bonferroni correction`. This correction is very conservative, but says that your new type-I error rate should be the error rate you actually want divided by the number of tests you are performing. Therefore, if you would like to hold an **allowable type-I error rate of 1%** (99% confidence means alpha=0.01) for each of 20 hypothesis tests, the Bonferroni corrected rate would be 0.01/20 = 0.0005. This would be the new rate you should use as your comparison to the p-value for each of the 20 tests to make your decision.






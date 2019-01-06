# Study-AB-Testing

## [Intro: SO...Not Enough Samples ?] 
### Let's do inferential stats! Estimate population parameter from samples.
#### Estimation techniques for finding 'population parameter θ'
 - 1.Maximum Likelihood Estimation
   - Regular MLE(analytical)
   - Grid Search
   - Newton-Raphson Iteration 
   - EM Algorithm
 - 2.Method of Moments Estimation
 - 3.Bayesian Estimation (easier to communicate with audience than awkawrd frequentists' methods)
 - 4.Frequentist's *Aggregate approach (CI, Hypothesis_test)
   - parametric
   - Non-parametric
 - 5.*Individual approach (machine learning, but does need more samples, doesn't care finding "good statistics")

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
 - In LM, A coefficient describes the weight of the contribution of the corresponding independent variable. The parameter estimation refers to the process of using sample data to estimate the parameters of the selected distribution, in order to **minimize the cost function**. 

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

### Proportion Sampling and z-test
Let's say there are two samples - X and Y - and they are **independent Binomially? distributed** with parameters ~ `Bin(n, p)` and `Bin(m, p)`. You know what? `X+Y ~ Bin(n+m, p)`. By this logic, if `X1 ~ Bin(1,p)`, `X2 ~ Bin(1,p)`, ...., then `X1+X2+...Xn ~ Bin(n, p)`. (It's like a relationship between Bernoulli and Binomial. Like a Exponential and Gamma. Bur not like a Normal and Chi-Sqr? )  
<img src="https://user-images.githubusercontent.com/31917400/48653099-cb7f1800-e9fa-11e8-8333-319b2726643d.jpg" />

### 2. Chi_Sqr-test: 
It's expected value: `E[x] = df` (몇개나 더했어?) - the sum of squares`∑(x-μ)^2` or **SS** of independent standard normals is a random variable that fairly naturally arises in many contexts, and that is something we would like to have a name for. The degrees of freedom relates to the number of independent **normals** involved(or squared then summed) and each of those squared components has mean `1`. 
> In Goodness of Fit Test and Contingency Test, no parameters(mean, sd, etc) are required to compute and no assumptions are made about the underlying distribution. However, when we test if a sample variance is exactly equal to the population variance, this test is a parametric test because it makes assumptions about the underlying sample distribution (the data are normally distributed). 

> **[Note]**: The chi-squared test is essentially always a **one-sided test** (because of the essence is measuring the fittness: good/bad fit). ************************************************************************************************************ 

#### 1> Population Variance Estimation
 - __From a single sample:__
 <img src="https://user-images.githubusercontent.com/31917400/48678662-27be7500-eb7e-11e8-949e-396a7a87f092.jpg" />  

   - Let's say we have set a timed-sales-goal where `population_SD` is less than 21 days(so variance upper limit is `441`). Then we randomly select 15 sales records. Based on this sample, the following is obtained: `n = 15`, `sample_mean = 162 days`, `sample_SD = 24 days`(so sample_var is `582`) and our focus is variance. So our sample_SD is 24 days which exceeds the goal. But this exceed is significant? so our goal `441` is too much small? Tell me. 
   <img src="https://user-images.githubusercontent.com/31917400/47957077-9a88f700-dfa7-11e8-885f-d6830550de2d.jpg" />  

### Hey, While the sampling distribution of `sample_mean` follows the `Normal`, the sampling distribution of `sample_variance` follows the `Chi-Sqr`.  
   <img src="https://user-images.githubusercontent.com/31917400/47957744-c1e5c100-dfb3-11e8-8f46-dbef4064f2ab.jpg" /> 
   
__[Note]: If From two samples,__ **F-Test** for Equality of two sample variances
 - Is the variance the same for both soup brands?
<img src="https://user-images.githubusercontent.com/31917400/47958023-55ba8b80-dfba-11e8-91b2-3eb73030f733.jpg" />

### Next,
### We use Chi-Sqr to test relationships between `categorical variables`.
 - > This is not about population mean or variance, but about the correlation?  
 - > if the accounts in the categories are binary(Bernulli) or multinary(A,B,C,D,F), and all values are playing with **frequency**...we first assume H0 is true, then ...
   - Values in Chi-sqr Dist are always (+).
   - **Like t-Dist, it only has one parameter: `df`**
   - Location/Scale: 0 / 1 by default?
   
#### 2> Goodness_of_fit-Test: `one distribution VS one categorical sample` (values are all about `Countings` like a histogram).
<img src="https://user-images.githubusercontent.com/31917400/48679061-5ee35500-eb83-11e8-82fe-2216e0724115.jpg" />  

> we want to know: 
 - between `one population group`(distribution) and `one sample groups with multiple classes`, there is any **association**?
   - `Does this sample come from this distribution?`
   - `H0: No **difference** b/w the data and a specified distribution.`..(consistent)?
     - Doeas it mean...? H0: observed_values = expected values
   - In this case our **df** is `r-1`(NO.of variables to compare, one way)
 - Use when you have a single **categorical** sample (with multiple classes) from a population. It is used to determine whether sample data are consistent with a hypothesized distribution(**proportion distribution**), i.e to test the hypothesis H0 that a set of observations is consistent with a given **probability** distribution. 
   - Yeah, it claims about population `proportion`. 
   - It's a **Non-parametric** test. WHY? 
   - `sample size`(expected frequency) in each level of the category should be > `5`
   - so..each category takes up some `proportion area` on the distribution(pdf) chart..and data point on x-axis belong to each category..like a set of divisions 
   <img src="https://user-images.githubusercontent.com/31917400/48679735-d4532380-eb8b-11e8-8660-5bd890b30dac.jpg" />

#### 3> Independence-Test with Contingency: `two categorical samples` (values are all about `Countings` like a histogram).
<img src="https://user-images.githubusercontent.com/31917400/48679063-63a80900-eb83-11e8-8382-df9d11b0d641.jpg" />  

> we want to know: 
 - between the categorical variable(rows) and the categorical variable(columns), there is any **association**? Test two random variables if they are statistically independent? 
   - `H0: No **association** b/w the two categorocal variables`..(so independent) ?
     - the proportion of **category_A(1/2/3)** is equal between **category_B(Y/N)** : Independence between **category_A** and **category_B**
     - H0: `proportion(class_1)|Y = proportion(class_1)|N`,  `proportion(class_2)|Y = proportion(class_2)|N`, `proportion(class_3)|Y = proportion(class_3)|N`
   - In this case our **df** is `(r-1)(c-1)`(NO.of variables to compare, two way)  
<img src="https://user-images.githubusercontent.com/31917400/38503101-0c07da10-3c09-11e8-92f4-114707454eaa.jpg" />  

- > What if we should compare more than 2 groups(3,4,5...dimensional)?
  - Do pair-wise multiple test (compare every group against every other group).
   - For P_Value: even running the same experiment twice, the odds to get significant p_value would increase..(FP). This is why Frequentist's methods are awkawrd...Here, we need to fix it using..
     - Bonferroni's Correction: Alpha_new = Alpha / # of tests
       - Alpha_new = Alpha / choose(N,2)
     - **One VS the rest:** ?????   

### 3. F-Test
F-Distribution(Variance-Ratio-Distribution) defines the ratio of the two variances(of the two normally distributed samples). It has a density = ratio of gamma function(the sum of exponential) and two parameters = df `m` for the `numerator` and df `n` for the `denominator`. Let's say we have a 'iid' sample_A and a 'iid' sample_B and both are **independent** (basic assumptions). 
<img src="https://user-images.githubusercontent.com/31917400/48623932-0ead9c80-e9a3-11e8-8377-5cae67faa5ae.jpg" />

 - Use when testing the hypothesis of the equality of two `sample variances` (Note: Chi-Sqr test for a single population variance)
 - Use when testing the hypothesis of the equality of `multiple means` at the same time (ANOVA). 
 - Use when testing the overall significance of the mutiple regression model as a whole(A significant `F-value` indicates a linear relationship between the `Response` and at least one of the `Predictors` so have some hope!). 
  
### 4. Statistical Power and Effective Sample_size
<img src="https://user-images.githubusercontent.com/31917400/49330372-9986bd80-f585-11e8-8639-c0cabbf638ff.jpg" />

It refers the porbability to `reject H0` when it is correct to do so...**Not making type-II error:** `1 - β`. 
 - type-I error: `α=FP` 
 - type-II error: `β=FN`
 - **Power** = P(reject H0 | H0 is False) = `1 - FN` 
   - REJECT: 'pos'
   - ACCEPT: 'neg'
 - **High power** decreases the odds to get FN(type_II) so **more Rejection**
 - Why quantify power ? 
   - Power helps determine the sample size we need(by P_Value).
 - Four elements to prepare better hypothesis testing!
   - 1. **Effect_size**(`Δ`): 'the difference b/w two Grp means'.
   - 2. width of the distribution..SE..(`σ/sqrt(n)`)
   - 3. significance level(`α`)
   - 4. power you want(`1-β`)
 - so the effective sample_size`n` should be:
   - one tailed: `(σ/Δ)^2 * (Z<α> + Z<β>)^2`
   - two tailed: `(σ/Δ)^2 * (Z<α/2> + Z<β>)^2`

### 5. Popular Questions
 - whether two variables (n = 2) are correlated (i.e., associated) => **Correlation test** between two variables. 
 - whether multiple variables (n > 2) are correlated => **Correlation matrix** between multiple variables. 
 - whether two groups (n = 2) of samples differ from each other => **t-test**(parametric: Need of Dist_Assumption). 
   - sample_mean VS population_mean (one sample test)
   - sample_mean VS sample_mean (two sample test)
 - whether multiple groups (n >= 3) of samples differ from each other => **ANOVA test**(Multiple Sample Test) as an extension of t-test.
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
<img src="https://user-images.githubusercontent.com/31917400/50741219-f7946980-11f1-11e9-87bc-70d260f52a3b.jpg" />  

 - If there is one categorical variable with multiple classes and whether it is consistent with the population distribution => **Chi-Sqr test**(parametric ? `No`)
 - If things are in a contingency table(counts from the 2 categoricals) and whether they are correlated => **Chi-Sqr test**(parametric ? `No`) 
 - whether the variability of a single sample differ from population variance => **Chi-Sqr test**(parametric ? `Yes`)
 - whether the variability of two samples differ each other => **F-test**(parametric? `Yes`) 

### Popular Hypothesis testing (when the data are normally distributed)
 - __Simple Rule of Significance Test:__ 
   - Use `Chi-Sqr_test` if your predictor and your outcome are both **categorical**(e.g., purple vs. white). 
   - Use a `t-test` if your single categorical predictor has **only 2 classes** and your outcome is **continuous**(e.g., height, weight, etc)...two_sample_t-test
   - use "One_way_ANOVA" for multi-sample`>=3` test.
   - Use `correlation test` or `regression` if both the predictor and the outcome are **continuous**.
 - a)Testing a **population mean** (One sample t-test) with `1` sample.
 - b)Testing the difference in means (Two sample t-test) with `2` **independent** samples.
 - c)Testing the difference before and after some treatment on the same individual (Paired t-test) with `2` **dependent** samples.
 - d)Testing a **population proportion** (One sample z-test) with `1` sample.
 - e)Testing the difference in proportions (Two sample z-test) with `2` **independent** samples.
 - f)Comparing `**multiple samples**` (ANOVA Test, the involvement of categorical variables and **Response variable**).
   - Wtf is Multiple sample test? It involves the `categorical sample(s)` and the `response veriable` accordingly.
   - Doing ANOVA requires our data to meet the following assumptions:
     - Independent observations(IID): This often holds if each case contains a distinct person and the participants didn't interact.
     - Homogeneity: the population variances are all equal over sub-populations. Violation of this assumption is less serious insofar as sample sizes are equal.
     - Normality: the test variable must be normally distributed in each sub-population. This assumption becomes less important insofar as the sample sizes are larger.
   - __One-Way(factor) ANOVA Test__ (one response VS one predictor with multiple(`>=3`) classes)
     - Using the F-distribution, it examines the influence of a single 'categorical' input variable(X1) on the 'numerical' response variable(Y)...whether the mean of some numeric variable differs across the levels of one categorical variable. Do any of the group means differ from one another? 
     - H0: `mean(class_A) = mean(class_B) = mean(class_C) = ...`
     <img src="https://user-images.githubusercontent.com/31917400/48962784-342b3f00-ef7d-11e8-80e9-38071b0b474e.JPG" />  
     
   - __Two-Way(factor) ANOVA Test__ (one response VS two predictors with multiple(`>=3`) classes)
     - As an extension of the one-way ANOVA, it examines the influence of 2 different 'categorical' input variables(X1, X2) on the 'numerical' response variable(Y). The two-way ANOVA not only aims at assessing the main effect of each independent variable but also if there is any `interaction` between them. https://www.youtube.com/watch?v=ajLdnsLPErE&t=442s
     - [Notice]: We have 2 categorical variables, but we should generate **2** integrated categorical variables. In order to do this, note that `one of two original categorical variables cannot hold more than binary classes` because it's a two-way ANOVA. This binary classes are sacrificed and become `[World-I]` & `[World-II]`.
     - `within_MS`는 항상 비교대상...
     - 3 Null-hypothesis
       - H0: `mean(class_1) = mean(class_2) = mean(class_3)` from `factor_01` (like the **one-way ANOVA**).
       - H0: `mean(class_Y) = mean(class_N)` from `factor_02` (like the **one-way ANOVA**).
       - H0: `mean(class_1)|Y = mean(class_1)|N`,  `mean(class_2)|Y = mean(class_2)|N`, `mean(class_3)|Y = mean(class_3)|N` (like the **Chi-Sqr test for independence** with contingency tables)...this is the interaction test. 
         - What is the `interaction`? An interaction effect(boysXgirls) can be examined by asking if `Y|Age`(score) affects `girls` differently than `boys`. If `Y|Age_1`&`Y|Age_2`&`Y|Age_3` run parallel to each other across `girls` and `boys`, then we can say that `Age` categorical variable and `Gender` categorical variable are independent so have no relationship.  
     <img src="https://user-images.githubusercontent.com/31917400/48811341-35fac580-ed25-11e8-844f-47de2ed13413.jpg" />  
   
   - 3 way ANOVA
     - blah blah shit.

   
   - MANOVA Test (Multivariate Analysis of Variance)
     - ANOVA has only one dependent variable, while MANOVA has multiple dependent variables.
     - It helps to answer:
       - Do changes in the independent variable(s) have significant effects on the dependent variables?
       - What are the relationships among the dependent variables?
       - What are the relationships among the independent variables?

### **Randomized Block** Design VS **Two factor Factorial** Design
 - two way ANOVA can refer to two distinct but related models. [y and X1(a/b), y and x2(1/2/3...)]
 - What does it mean by "**blocking**"? Make experimental units homogeneous in a block. 
 <img src="https://user-images.githubusercontent.com/31917400/48944475-bdf7ef80-ef1e-11e8-85e6-442f6fa4ad2f.jpg" />  
 
 - What's the difference between a **randomized_block_design** and a **two-factorial_design**(given that they both use **two-way ANOVA**, and your **blocks** also can be your factor in two_factor_design) ?
   - In both cases, you have `2 categorical variables` and `1 numerical response variable` but in a **randomized block design** the second categorical variable is a **nuisance variable**(no interest thus become a block variable), while in the **two factor factorial design** the second categorical variable is also of **interest** and you would like to understand the **interaction**(In the randomised block design, the interaction term `αγ = δ` would be lumped in with the error term `ϵ`).
   - **two-way ANOVA** is a special case of **factorial_design** as it compares two categorical variables. 
   <img src="https://user-images.githubusercontent.com/31917400/50573353-41ccb380-0dca-11e9-9372-e3879ba008b4.jpg" />  
   
 - > Randomized Block Design  
   - `within_block` important and `between_block` not important
   - So it's a sort of **two-way ANOVA without interaction** !!!
   - Let's say we test on efficiency of 4 cutting tools. Data on measurements could be spread over several different materials such as wood, plastic, metal, etc. But we want to somehow eliminate the `effect of material` on cutting tools("material" is our nuisance variable). So we **block** measurements per material like..`block 1`means "wood", `block 2`means "plastic", `block 3`means "metal"...but we consider them homogeneous and assign randomly each treatment(cutting tool) but once in each block.        
 <img src="https://user-images.githubusercontent.com/31917400/50742171-6f1cc580-11ff-11e9-8b52-20c90c7d556b.jpg" />  

 - > How ANOVA is just a special case of Regression?
 <img src="https://user-images.githubusercontent.com/31917400/48967993-be58bf00-efe0-11e8-865f-781d43b9c85c.jpg" />  

   - In the ANOVA model, the **predictors** are often called `factors` or `grouping variables`, but just like in the regression case, we can call them the more generic “predictors.”
   - The subscript `i` indicates that each case has an individual value of **Y**. **ε** has an `i` subscript because there is one value per case. 
   - In the regression model, we use `X` to indicate the value of the predictor variables. This is flexible — if `X` is numerical, we plug in the **numerical** values. If `X` is categorical, we simply indicate which group someone was in with coded values of X1. The simplest would have a `1 for the treatment` group and a `0 for the control` group. `β` measures the treatment effect on `Y`.
   - ANOVA assumes that all the predictors are categorical (aka factors or grouping variables), those **predictors have a limited number of values**. Because of the limited number of values, the ANOVA model uses subscripts to indicate if someone is in the treatment or control group. Subscript `j` would have values of `1 for the treatment` and `0 for the control`. `α` measures the effect on `Y` of the treatment effect. Even those these `X` values aren’t written directly into the ANOVA model, they exist.
   -  In the regression model, the error term is called the **intercept** and denoted `β0` and in the ANOVA model, this is called the **grand mean** and denoted `μ`. 
   - Let's say we use a model with a single categorical predicter - employment - with 3 classes: managerial, clerical, and custodial. 
   - In the ANOVA, the categorical variable is `effect-coded`, which means that each classes’ mean is compared to the `grand mean`. In the regression, the categorical variable is `dummy-coded`, which means that each classes’ intercept is compared to the reference group’s `intercept`. 
   - The dummy coding creates two 1/0 variables: 
     - Clerical = 1 for the clerical class, 0 otherwise; 
     - Custodial = 1 for the custodial class, 0 otherwise.  
     - Observations in the Managerial class have a **0 value on both of these variables**, and this is known as the reference group.
   - Since the intercept is defined as the mean value **when all other predictors = 0**, and there are no other predictors, the 3 intercepts are just means. In both analyses, Job Category has an F=69.192, with a p < .001. Highly significant.
   - Let's say in the ANOVA, we find the means of the three groups are:
     - Clerical: 85.039
     - Custodial: 298.111
     - Manager: **77.619**
   - Let's say in the Regression, we find these coefficients:
     - Intercept: **77.619**
     - Clerical: 7.420
     - Custodial: 220.492
   - The `intercept` is simply the **mean of the reference group**, Managers. The coefficients for the other two groups are the **differences** in the mean between the `reference group` and the `other groups`. For example, that the regression coefficient for Clerical is the difference between the mean for Clerical, 85.039, and the Intercept, or mean for Manager (85.039 – 77.619 = 7.420). The same works for Custodial. 
   - ANOVA reports **each mean** and a `p-value that says at least two are significantly different`.  
   - Regression reports **only one mean**(as an intercept), and the **differences** between that one and all other means, but the `p-values evaluate those specific comparisons`. 
   - It’s all the same model; the same information but presented in different ways. 

### Random VS Fixed Effects
Let's say you have a model with a categorical predictor, which divides your observations into groups according to the category values. The model **coefficients**, or `"effects"`, associated to that predictor can be either **fixed** or **random**. Random effects are estimated with partial pooling, while fixed effects are not.

 - Random Effect with partial pooling: Grp_effect estimate will be based **partially** on data from other groups.
   - why partial pooling? 
     - if estimating an effect by completely pooling all groups, it masks **Grp-level variation**. 
     - The **sub-grps** are part of some **bigger grp** with a `common mean effect`. 
     - The `sub-grp means` can deviate a bit from the `bigger grp mean`, but not by an arbitrary amount. 
     - To formalize that idea, we posit that the deviations follow a **Normal distribution**. That's where the "random" in random effects comes in. 
   - Its goal is to estimate `var(Treatments)`  
 - Fixed Effect 
   - Treatment effect sum to 0. 
  - In practice, the distinction b/w **fixed** and **random** comes down to which variable has the levels of out interest: fixed, then other variables: random.
<img src="https://user-images.githubusercontent.com/31917400/50566760-1055d880-0d35-11e9-8937-699546536b1f.jpg" />  




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

 - C.I. with t-test as the Traditional mean comparison method 
```
df_samp = df.sample(200)
X1 = df_samp[df_samp[condition] == True]['height'] 
X2 = df_samp[df_samp[condition] == False]['height']

import statsmodels.stats.api as sm
cm = sm.CompareMeans(sm.DescrStatsW(X1), sm.DescrStatsW(X2))
cm.tconfint_diff(usevar='unequal')
```
All of these formula have underlying "assumptions" (Central Limit Theorem - regarding the sampling distribution ie.the distribution of statistics) that may or maynot be true. But **Bootstrapping** does not need the assumptions of these intervals. Bootstrapping is one of the re-sampling techniques. 
 - Re-sampling implies sampling from a sample in order to estimate empirical properties(such as variance, distribution, C.I. of an estimator) and to obtain Empirical DF of a test statistic. The common methods are Bootstrap, Jacknife, shuffling. They are effective when the distribution is unknown or complex. It's a Non-Parametric Method. 
 - **Bootstrapping** only assumes the sample is representitive of the popluation. With large enough sample size, Bootstrapping and the traditional methods would provide the same **Confidence Interval**. 

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
 - b) What is the impact of large sample size on my result? (with large sizes, everything will be statistically significant..then we'd always choose to `Reject H0` with probability of alpha---[Type-I. Error]:FP)
 - `Power: 1-β` ∝ sample_size ∝ Type-I(α)...always reject Ho. FP..Bitch!
 - too small sample_size causes Type-II(β)...always accept Ho. FN..Slut! 


### Multi-testing Correction
When performing more than one hypothesis test, your type-I error compounds. In order to correct for this, a common technique is called the `Bonferroni correction`. This correction is very conservative, but says that your new type-I error rate should be the error rate you actually want divided by the number of tests you are performing. Therefore, if you would like to hold an **allowable type-I error rate of 1%** (99% confidence means alpha=0.01) for each of 20 hypothesis tests, the Bonferroni corrected rate would be 0.01/20 = 0.0005. This would be the new rate you should use as your comparison to the p-value for each of the 20 tests to make your decision.

### Finite Population Correction
When sampling **without replacement**(like HyperGeometric instead of Binomial) from more than `5%` of a finite population, you need to multiply the SE by this correction `sqrt((N-n) / (N-1))` because under these circumstances, the Central Limit Theorem doesn’t hold. **FPC** captures the difference between sampling **with** replacement and sampling **without** replacement and shrinks the SE. 

-----------------------------------------------------------------------------------------------------
# Who does give a shit to parameters ???
## [Non-Parametric Statistics]
Nominal is Categorical. **`Ordinal`** is categorical but deals with the **ranking** of items in order. In your research, have you ever encountered one or more the following scenarios?
 - some non-linear data ?
 - some “chunked” data (1-4 cm, 2-5 cm, >5 cm…)
 - qualitative judgments measured on a ratings scale
 - **data that don’t follow a normal distribution**
 - **data that violate assumptions of ANOVA**
 
Non-parametric(distribution free) test is a statistical procedure whereby the data does not match a normal distribution. The data used in non-parametric test can be any types but be frequently of **`Ordinal`** data type, thus implying it does not depend on arithmetic properties. Consequently, all tests involving the ranking of data are non-parametric and also no statement about the distribution of data is made. Instead, we use empirical distributions. The questions are same:
 - One sample. A sample comes from a population with a specific(underlying) distribution?
 - Two samples. Are they coming from the same population with a specific(underlying) distribution? or the two datasets differ significantly?
 - In a non-parametric test, the observed sample is converted into ranks and then ranks are treated as a test statistic.

> **Pros**
 - 1. Non-parametric tests deliver accurate results even when the sample size is small.
 - 2. Non-parametric tests are more powerful than parametric tests when the assumptions of normality have been violated.
 - 3. They are suitable for all data types, such as nominal, ordinal, interval or the data which has outliers.
> **Cons**
 - 1. If there exists any parametric test for a data then using non-parametric test could be a terrible blunder.
 - 2. The critical value tables for non-parametric tests are not included in many computer software packages so these tests require more manual calculations...And "Special table" for smaller sample sizes required.
 - 3. When variability differs b/w groups (unequal variance), it does not deliver accurate results.
 - 4. In general, low power(hard to reject Null hypothesis). 
<img src="https://user-images.githubusercontent.com/31917400/50407742-33343a00-07d5-11e9-99b4-a834702eea34.png" />  

----------------------------------------------------------------------------------------------
### A. Non-Parametric Goodness of Fit: one or two sample: `Kolmogorov_Smirnov-test`    
`KS statistic` quantifies a **distance** 
 - b/w the **empirical cdf** of the 1 sample and the **cdf** of the reference distribution, or 
 - b/w the **empirical cdf** of 2 samples. 

and looks for consistency by comparing... overall shape, not parameters. `KS statistic` called **'D'** is simply the **maximum absolute difference** b/w the two cdf.  
 
### __(+)__
> 1. **KS-test** allows you to detect patterns(such as variance) that you can’t detect with t-test.
 - t-test calculates the P-value of `sample vs Normal population` or `sample vs sample`. But there is an issue with t-Test: samples must be shaped in a **normal distribution**. What if we work a lot with Poisson distributions??? Binomial distribution???, etc?  
 - If the mean and SD b/w two samples are highly similar, **t-test** would give a very high p-value. 
 - But **KS-test** can detect the **variance**. In the chart below, for example, the red distribution has a slightly binomial which KS detects.
 <img src="https://user-images.githubusercontent.com/31917400/50407190-2eb65400-07ca-11e9-8b21-5dda4ac43e8c.png" />  

   - t-test above says that there is 79.3% chances the two samples come from the same distribution.
   - KS-test above says that there are 1.6% chances the two samples come from the same distribution. 
### __(-)__
> 1. It wastes information in using only differences of greatest magnitude in cumulative form.
> 2. **KS-test** generally deals well with continuous data. Discrete data also possible, but test criteria is not exact, so can be inefficient. 
  - Chi-Sqr applies both continuous, discrete, but its “arbitrary” grouping can be a problem which affects "sensitivity" of H0 rejection. 
> 3. For two samples, it needs same sample sizes. 
> 4. the distinction between location/shape differences not established.. 

# Next, playing with medians. 
 - Your dataset is skewed? 
 - η is resistant to outliers !

### B. one sample Non-Parametric: `Sign-test`
It is also called the **`Binominal Sign-test`** with `p=0.5`.  
 - This Non-parametric test is based on ranks of the data samples, and test the hypotheses relating to quantiles of the probability distribution representing the population distribution from which the data are drawn. Specifically, the test concerns the population median`η` where `P(obv <= η) = 0.5` And ask some prevalence of the certain data points. It tests if the direction of change is random or not.
 - The sign test is considered a weaker test because it tests the pair value below or above the **median** and **it does not measure the pair difference**. 
 - One-Sided test
   - Its **test statistics `S`** is the count of observed data points(SIGNS) that corresponds `H1:Alternative Hypothesis`. Since the test statistic is expected to follow a binomial distribution, the standard binomial test is used to calculate significance. The p-value is defined by `P(x >= S)`  
 - Two-Sided test
   - Its **test statistics `S`** is `max{S1, S2}`where S1 and S2 are the counts of the observations less than, and greater than the some specified value `η0`. The p-value is defined by `2*P(x >= S)`   
 - The normal approximation to the binomial distribution can be used for large sample sizes: > 25. 
   - Do a continous correction such as `S+0.5` 
   - In this case, the **test statistics `Z`** is `(S+0.5 - n/2) / sqrt(npq)`where p=0.5, q=0.5
   - As for the direction, follow the eye in H0. 
 - For example, suppose we want to test if the haemoglobin level of vegans is likely to be less than 13g/dL. From 10 subjects, count **how many above(+) 13**, and **how many below(-) 13**. 
<img src="https://user-images.githubusercontent.com/31917400/50425772-c96b6d00-0874-11e9-88e8-c692a926d610.jpg" />  

 - H0: Median `η >= 13`, H1: Median `η < 13`
 - In the sample, we have 7 data points that < 13, so the **test statistics `S`** is 7.   
 - Compare it with the population distribution. From **`Bin(10, 0.5)`**...
   - P(x >= 7) = 0.172 which is greater than the 0.05 significance level, so we conclude that "do not reject H0".   

### Q. Dependent? levels absorbed into one column? Independent? levels become separate columns??????

### C1. paired sample(dependent) Non-Parametric: `Wilcoxon W-test (Sign*Rank)`
 - Examples:
   - Related Sample: Marital Satisfaction ratings given by husbands and wives
   - Matched Sample: Medical trial matching patients on age, gender
   - Repeated Sample: pre/post test score from same individuals..heart rate before/after exercise
 - **Steps**:
   - 1) We have two columns(before/after). Find the difference(After - Before).
   - 2) Based on the differences, assign the SIGN(`-1`:if negative, `1`:if positive, `0`:if no change).   
   - 3) Drop the rows with SIGN '0' coz it won't contribute to our analysis.
   - 4) Go back to the differences and Order the differences by absolute values.
   - 5) Assign **Ranks** to the absolute differences. 
   - 6) Multiply a rank by the sign (rank x sign = signed_rank) and add them all up = test statistics `W1` or `W2`
     - W1: `Sum of (+)ranks`, W2: `Sum of (-)ranks`
     - The Real W: `min(W1, W2)`
   - 7) Compare to W-distribution to calculate the P-value. 
<img src="https://user-images.githubusercontent.com/31917400/50445779-e2bdf900-0908-11e9-9534-4c73accdbab0.jpg" />  

 - H0: Median difference `Dη = 0`, H1: Median difference `Dη > 0`
 - In the sample, we have 12 data points.    
 - Compare it to the W distribution with the Lower tail. 
   - The test statistics is 10 which is smaller than the value at 0.05 significance level, so we conclude that "reject H0".     

### C2. two sample(independent) Non-Parametric: `Mann_Whitney_Wilcoxon U-test (Rank)`
It compares two independent samples but compare them by `rank`! It uses **U**-statistics: offering the degree of **overlap in ranks** b/w the two groups. 
 - Examples:
   - Let's say a pizza café owner wants to know who eats more slices of pizza: football or basketball players. With this information she will determine how much inventory she needs during football and basketball seasons. After collecting the data, you realize that there are some extreme outliers among basketball players that may **skew the results**. You determine to run a Mann_Whitney_Wilcoxon U-test. 
<img src="https://user-images.githubusercontent.com/31917400/50447541-b9569a80-0913-11e9-8133-bb7f085c6862.jpg" />  

 - **Steps**:
   - 1) Identify group with **smaller** summed_ranks.
   - 2) For each data point in the group of the smaller summed_rank, add up how many points in the other group are higher in rank.
     - test statistics: `U1` OR `U2`
     - U1: `Sum of all rank(gr1) - 0.5 x n1(n1 + 1)`, U2: `Sum of all rank(gr2) - 0.5 x n2(n2 + 1)`
     - The Real U: `min(U1, U2)`
   - 3) Compare U-statistics to the U-distribution.  
<img src="https://user-images.githubusercontent.com/31917400/50449942-7bad3e00-0922-11e9-9400-1ef1822fcaa6.jpg" />  

 - H0: Median `η1 = η2`, H1: Median `η1 > η2`
 - In the sample, we have 4 data points in group_A, and 3 in group_B.     
 - Compare it to the U-distribution. 
   - The test statistics is 1 which is bigger than the value at 0.05 significance level, so we conclude that "Do not reject H0". 

### D1. Three or more(dependent) Non-Parametric: `Friedman-test`
Just like one way ANOVA with repeated measurement or Randomized Blocks(two way ANOVA w/o interaction)? It's an extension of Wilcoxon paired W-test. 
 - Examples:
   - ratings of same performer(or a group of performers A,B,C..) on separate multiple test attempts(test, retest, re-test..)
   - same product(or a group of products A,B,C..) rated by several different judges
 -**Steps**:
   - 1) Give ranks to each data point **across `treatment`** (low->high)
   - 2) Sum up ranks **across `block`**(judge)
   - 3) Calculate Chi-Sqr statistics: 
<img src="https://user-images.githubusercontent.com/31917400/50458644-7bca2f80-095c-11e9-8b42-cf725e61afa4.jpg" />  

 - H0: Median `η1 = η2 = η3 = η4 = η5`, H1: Median `η1 != η2 != η3 != η4 != η5`
 - Compare it to the Chi-Sqr distribution.
   - The test statistics is 19.76 which is bigger than the value at 0.05 significance level, so we conclude that "reject H0". 

### D2. Three or more(independent) Non-Parametric: `Kruskal_Wallis-test`
Just like **one way ANOVA** or Completely Randomized Design. It's an extension of Mann_Whitney_Wilcoxon U-test. Use in case of the **unequal sample size**(one way ANOVA can do this too though).  
 - Examples:
   - Finding out how test anxiety affects exam scores. The predictor “test anxiety” has three levels: no anxiety, medium anxiety and high anxiety. The dependent variable is the exam score, rated from 0 to 100%.
   - Finding out how socioeconomic status affects attitude towards sales_tax increases. The predictor is “socioeconomic status” with three levels: working class, middle class and wealthy. The dependent variable is measured on a 5-point Likert scale (ratings) from strongly agree to strongly disagree.
   - To assess the **Effects of Expectation on the Perception of Aesthetic Quality**, an investigator randomly sorts 24 amateur wine aficionados into three groups, A, B, and C, of 8 subjects each`(so exetremely independent each other)`. Each subject is scheduled for an individual interview. Unfortunately, one of the subjects of group B and two of group C fail to show up for their interviews, so the investigator must make do with samples of **`unequal size: na=8, nb=7, nc=6, for a total of N=21`**. The subjects who do show up for their interviews are each asked to rate the overall quality of each of three wines on a 10-point scale, with "1" standing at the bottom of the scale and "10" at the top. As it happens, the three wines are the same for all subjects. The only difference is in the texture of the interview, which is designed to induce a relatively **high expectation of quality in the members of group A; a relatively low expectation in the members of group C; and a merely neutral state, tending in neither the one direction nor the other, for the members of group B**. Remember our data points are all ordinal... 
 - **Steps**:
   - 1) Give ranks to each data point **across `all data points`** (low->high)
   - 2) `(R)` Sum up ranks **across `rows`**
   - 3) `(n)` Find each Treatment size
   - 4) `(N)` Total data size
   - 5) Calculate Chi-Sqr statistics: 
<img src="https://user-images.githubusercontent.com/31917400/50460634-b6899300-096f-11e9-8234-f3fbf16150df.jpg" />  

 - H0: Median `η1 = η2 = η3`, H1: Median `η1 != η2 != η3`
 - Compare it to the Chi-Sqr distribution.
   - The test statistics is 9.84 which is bigger than the value at 0.05 significance level, so we conclude that "reject H0". 





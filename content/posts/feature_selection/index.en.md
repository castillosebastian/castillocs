---
weight: 2
title: "Feature Selection: Intro"
date: 2023-06-01T10:49:29-03:00
lastmod: 2023-06-01T21:29:01+08:00
draft: true
images: []
resources:
- name: "featured-image"
  src: "featured-image.jpg"
- name: "featured-image-preview"
  src: "featured-image-preview.jpg"
tags: ["feature selection", "machine learning", "metaheuristic"]
categories: ["evolutionary algorithms"]
lightgallery: true
toc:
  auto: false
---

RESUMEN

<!--more-->


Resto del artículo



## Unfearness and Responsible AI

The pervasive application of machine learning in many sensitive environments to make important and life-changing decisions, has heightened concerns about the fairness and ethical impact of these technologies.  More importantly, experiments that unveil biases and disparities inherent in these implementations (Mehrabi, 2022) have dismantled the idea of algorithmic 'neutrality', emphasizes the critical need for alignment with laws and values pertaining to human dignity. 

In this context, the concept of *responsible AI* has emerged as a crucial component of every AI project, underscoring the need for procedures that can facilitate the creation of safe, fair, and ethically grounded tools (Richardson, 2021). 

## Fair AI tools

We can view the concept of *fair AI tools* as pointing to software that is free from unintentional algorithmic bias. Fairness, as defined by Mehrabi et al. (2021), is *the absence of any prejudice or favoritism toward an individual or a group based on their inherent or acquired characteristics.* 

{{< admonition type=info title="What is the difference between individual and group fairness?" open=false >}}
## Individual an Group Fairness

A brief overview of the concepts of individual and group fairness as defined by Dwork et al. in their 2011 paper "Fairness Through Awareness."

1. **Individual Fairness:** According to Dwork et al., individual fairness is the principle that similar individuals should be treated similarly. This means that an algorithm is individually fair if it gives similar outputs for similar inputs. The definition of "similarity" can vary depending on the context, but it is generally defined in terms of a metric or distance function over the input space. 

   The formal definition of individual fairness is as follows: Given a metric space (X, d) and a function f: X → Y, we say that f is Lipschitz if for all x, x' ∈ X, d_Y(f(x), f(x')) ≤ d_X(x, x'). In the context of fairness, this means that the difference in the outputs of the function (i.e., the decisions made by the algorithm) should not be greater than the difference in the inputs (i.e., the individuals being considered).

2. **Group Fairness:** Group fairness, on the other hand, is the principle that different groups should be treated similarly on average. This means that an algorithm is group-fair if it gives similar outcomes for different groups, even if the individuals within those groups are not similar. 

   The formal definition of group fairness can vary depending on the specific notion of fairness being considered (e.g., demographic parity, equal opportunity, etc.). However, a common definition is that the decision rates (i.e., the proportion of positive outcomes) should be equal for different groups. For example, if we denote by P(Y=1|A=a) the probability of a positive outcome given group membership a, then demographic parity requires that P(Y=1|A=a) = P(Y=1|A=a') for all groups a, a'.
{{< /admonition >}}

To achieve fairness-related goals, we can approach them through both *product* development and the implementation of specific *procedures*:

- **Products**: refers to AI software that is designed and developed with fairness in mind. This could involve algorithms that mitigate bias or tools that promote transparency in AI decision-making. Regarding solutions, Richardson asserts that fair AI consists of *strategies to combat algorithmic bias*. These often include top-tier solutions drawn from research in explainability, transparency, interpretability, and accountability (Richardson, 2021).

- **Procedures**: refers to standardized activities or practices that ensure fairness. This could include ethical guidelines for AI development, rigorous testing for bias in AI systems, and policies for responsible AI use.

It's important to note that the specifics of these 'products' and 'procedures' can vary significantly depending on the context, the specific AI application, and the definition of 'fairness' in use. *Fairness* is a time-bound and context-dependent moral concept, so tools designed to ensure it must adapt to evolving standards. This means they must be flexible to changes in societal values and expectations over time. That is why the pursuit of *fair AI tools* is a continuous and context-specific endeavor, which rules out the possibility of universally applicable or one-size-fits-all solutions. As stated in Fairlearn project (Microsoft): 'because there are many complex sources of unfairness—some societal and some technical—it is not possible to fully “debias” a system or to guarantee fairness; the goal is to mitigate fairness-related harms as much as possible.'

## A fair tool: InFairness

Now, let's turn into a practical application of fairness in AI. We will be testing the 'fair-ml' algorithms developed by IBM Research, available in their [inFairness package](https://github.com/IBM/inFairness). These algorithms are designed with a focus on fairness, guided by the fairness metric proposed by Dwork et al., 2011.    

To explore these implementation we will follow the model [example](https://github.com/IBM/inFairness/tree/main/examples/adult-income-prediction) provided in the package. We are going to work with *Adult* dataset (Dua & Graff, 2017) used to predict whether income exceeds $50K/yr based on census data. Also known as "Census Income" dataset Train dataset contains 13 features and 30178 observations. Test dataset contains 13 features and 15315 observations. Target column is a binary factor where 1: <=50K and 2: >50K for annual income. 

### Libraries

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from inFairness.fairalgo import SenSeI
from inFairness import distances
from inFairness.auditor import SenSRAuditor, SenSeIAuditor
%load_ext autoreload
%autoreload 2
import metrics
```

### Bias exploration

```python
import pandas as pd
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
names = [
        'age', 'workclass', 'fnlwgt', 'education', 
        'education-num', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'capital-gain', 
        'capital-loss', 'hours-per-week', 'native-country',
        'annual-income'
    ]
data = pd.read_csv(url, sep=',', names=names)
```

{{< admonition type=info title="What are the 'Adult' dataset features?" open=false >}}
```python
data.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>annual-income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>77516</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>
{{< /admonition >}}

```python
data['annual-income'].value_counts()
```

    annual-income
     <=50K    24720
     >50K      7841
    Name: count, dtype: int64


The dataset is imbalanced: 25% make at least $50k per year. This imbalanced also appears in *sex* and *race* as shown here: 

```python
(imbal_sex := data.groupby(['annual-income', 'sex']).size() 
   .sort_values(ascending=False) 
   .reset_index(name='count')
   .assign(percentage = lambda df:100 * df['count']/df['count'].sum())   
   )
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>annual-income</th>
      <th>sex</th>
      <th>count</th>
      <th>percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>&lt;=50K</td>
      <td>Male</td>
      <td>15128</td>
      <td>46.460490</td>
    </tr>
    <tr>
      <th>1</th>
      <td>&lt;=50K</td>
      <td>Female</td>
      <td>9592</td>
      <td>29.458555</td>
    </tr>
    <tr>
      <th>2</th>
      <td>&gt;50K</td>
      <td>Male</td>
      <td>6662</td>
      <td>20.460060</td>
    </tr>
    <tr>
      <th>3</th>
      <td>&gt;50K</td>
      <td>Female</td>
      <td>1179</td>
      <td>3.620896</td>
    </tr>
  </tbody>
</table>
</div>


```python
(imbal_race := data.groupby(['annual-income', 'race']).size() 
   .sort_values(ascending=False) 
   .reset_index(name='count')
   .assign(percentage = lambda df:100 * df['count']/df['count'].sum())   
   )
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>annual-income</th>
      <th>race</th>
      <th>count</th>
      <th>percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>&lt;=50K</td>
      <td>White</td>
      <td>20699</td>
      <td>63.569915</td>
    </tr>
    <tr>
      <th>1</th>
      <td>&gt;50K</td>
      <td>White</td>
      <td>7117</td>
      <td>21.857437</td>
    </tr>
    <tr>
      <th>2</th>
      <td>&lt;=50K</td>
      <td>Black</td>
      <td>2737</td>
      <td>8.405761</td>
    </tr>
    <tr>
      <th>3</th>
      <td>&lt;=50K</td>
      <td>Asian-Pac-Islander</td>
      <td>763</td>
      <td>2.343294</td>
    </tr>
    <tr>
      <th>4</th>
      <td>&gt;50K</td>
      <td>Black</td>
      <td>387</td>
      <td>1.188538</td>
    </tr>
    <tr>
      <th>5</th>
      <td>&gt;50K</td>
      <td>Asian-Pac-Islander</td>
      <td>276</td>
      <td>0.847640</td>
    </tr>
    <tr>
      <th>6</th>
      <td>&lt;=50K</td>
      <td>Amer-Indian-Eskimo</td>
      <td>275</td>
      <td>0.844569</td>
    </tr>
    <tr>
      <th>7</th>
      <td>&lt;=50K</td>
      <td>Other</td>
      <td>246</td>
      <td>0.755505</td>
    </tr>
    <tr>
      <th>8</th>
      <td>&gt;50K</td>
      <td>Amer-Indian-Eskimo</td>
      <td>36</td>
      <td>0.110562</td>
    </tr>
    <tr>
      <th>9</th>
      <td>&gt;50K</td>
      <td>Other</td>
      <td>25</td>
      <td>0.076779</td>
    </tr>
  </tbody>
</table>
</div>

### Simple neural network model 

Folowing the IBM [example](https://github.com/IBM/inFairness/blob/main/examples/adult-income-prediction/adult_income_prediction.ipynb) in the income prediction task, we will test a simple neural network.


```python
class AdultDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data, label
    
    def __len__(self):
        return len(self.labels)
```

Note that the categorical variable are transformed into one-hot variables.


```python
import data
train_df, test_df = data.load_data()
X_train_df, Y_train_df = train_df
X_test_df, Y_test_df = test_df
X_train_df.head(1)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>education-num</th>
      <th>hours-per-week</th>
      <th>marital-status_Divorced</th>
      <th>marital-status_Married-AF-spouse</th>
      <th>marital-status_Married-civ-spouse</th>
      <th>marital-status_Married-spouse-absent</th>
      <th>marital-status_Never-married</th>
      <th>...</th>
      <th>relationship_Unmarried</th>
      <th>relationship_Wife</th>
      <th>sex_Male</th>
      <th>workclass_Federal-gov</th>
      <th>workclass_Local-gov</th>
      <th>workclass_Private</th>
      <th>workclass_Self-emp-inc</th>
      <th>workclass_Self-emp-not-inc</th>
      <th>workclass_State-gov</th>
      <th>workclass_Without-pay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.409331</td>
      <td>-0.14652</td>
      <td>-0.218253</td>
      <td>-1.613806</td>
      <td>-0.49677</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>

In the IBM-inFairness model [example](https://github.com/IBM/inFairness/blob/main/examples/adult-income-prediction/adult_income_prediction.ipynb) the protected attributes are dropped from the training and test data. That is usually the case in fairness-aware machine learning models,especially when dealing with known biased features. The aim is to prevent the model from directly using these sensitive attributes for decision-making, thereby avoiding potential discriminatory outcomes.

However, this approach has some limitations. Even when the protected attributes are removed, other features in the dataset might act as proxies for it, potentially retaining a strong signal of the biased information. As an example certain occupations, neighborhoods, or education levels might be disproportionately associated with certain racial groups due to societal factors. So, even without explicit information about race, the model might still end up learning patterns that indirectly reflect racial biases.

On the other hand, removing sensitives attributes makes it difficult to analyze the fairness of the model. If we don't know the race of the individuals in our dataset, we can't check whether our model is treating individuals of different races equally.

In some cases, it's important to consider sensitive attributes to ensure fairness. For example, in order to correct for historical biases or to achieve certain diversity and inclusion goals, it might be necessary to consider these attributes.

So, while removing sensitive attributes might seem like an easy fix, it doesn't necessarily solve the problem of bias and might introduce new problems. Instead, it's often better to use techniques that aim to ensure that the model treats similar individuals similarly (individual fairness), regardless of their sensitive attributes.


```python
protected_vars = ['race_White', 'sex_Male']
X_protected_df = X_train_df[protected_vars]
X_train_df = X_train_df.drop(columns=protected_vars)
X_test_df = X_test_df.drop(columns=protected_vars)
```

In assessing individual fairness, the example we are working with implements a variable consistency measure using the 'spouse' attribute. This involves flipping the 'spouse' variable in the dataset, essentially simulating a scenario where individuals with the same characteristics but different 'spouse' values are compared. The goal is to ensure that the model's predictions are consistent for individuals who are similar except for their 'spouse' attribute, thereby upholding the principle of individual fairness. This approach provides a practical way to audit the model's fairness by checking if similar individuals are treated similarly.


```python
X_test_df.relationship_Wife.values.astype(int)
```




    array([0, 1, 0, ..., 0, 0, 0])




```python

X_test_df_spouse_flipped = X_test_df.copy()
X_test_df_spouse_flipped.relationship_Wife = 1 - X_test_df_spouse_flipped.relationship_Wife
X_test_df_spouse_flipped.relationship_Wife.values
```




    array([1, 0, 1, ..., 1, 1, 1])




```python
device = torch.device('cpu')

# Convert all pandas dataframes to PyTorch tensors
X_train, y_train = data.convert_df_to_tensor(X_train_df, Y_train_df)
X_test, y_test = data.convert_df_to_tensor(X_test_df, Y_test_df)
X_test_flip, y_test_flip = data.convert_df_to_tensor(X_test_df_spouse_flipped, Y_test_df)
X_protected = torch.tensor(X_protected_df.values).float()

# Create the training and testing dataset
train_ds = AdultDataset(X_train, y_train)
test_ds = AdultDataset(X_test, y_test)
test_ds_flip = AdultDataset(X_test_flip, y_test_flip)

# Create train and test dataloaders
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=1000, shuffle=False)
test_dl_flip = DataLoader(test_ds_flip, batch_size=1000, shuffle=False)
```

We test a multilayer neural network as proposed in the IBM implementation example.


```python
class Model(nn.Module):

    def __init__(self, input_size, output_size):

        super().__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fcout = nn.Linear(100, output_size)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fcout(x)
        return x
```

### Standard training


```python
input_size = X_train.shape[1]
output_size = 2

network_standard = Model(input_size, output_size).to(device)
optimizer = torch.optim.Adam(network_standard.parameters(), lr=1e-3)
loss_fn = F.cross_entropy

EPOCHS = 10
```


```python
network_standard.train()

for epoch in tqdm(range(EPOCHS)):

    for x, y in train_dl:

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = network_standard(x).squeeze()
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
```

    100%|██████████| 10/10 [00:08<00:00,  1.24it/s]



```python
accuracy = metrics.accuracy(network_standard, test_dl, device)
balanced_acc = metrics.balanced_accuracy(network_standard, test_dl, device)
spouse_consistency = metrics.spouse_consistency(network_standard, test_dl, test_dl_flip, device)

print(f'Accuracy: {accuracy}')
print(f'Balanced accuracy: {balanced_acc}')
print(f'Spouse consistency: {spouse_consistency}')
```

    Accuracy: 0.8555948734283447
    Balanced accuracy: 0.7764129391420478
    Spouse consistency: 0.9636222910216719


The simple NN achieve .85 of accuracy. However, the inconsistency score of 0.04 on the 'spouse' variable suggests that the model is not treating similar individuals consistently, which is a violation of individual fairness. This inconsistency could be due to the fact that the model is learning to differentiate based on gender, despite the intention to avoid such bias.

### Individually fair training with LogReg fair metric

In the following section, a fair machine learning model is introduced. This model is said to be fair because its performance remains consistent under certain perturbations within a sensitive subspace, meaning it is robust to partial data variations.

To illustrate the authors' approach, let's consider the process of evaluating the fairness of a resume screening system. An auditor might alter the names on resumes of applicants from the ethnic majority group to those more commonly found among the ethnic minority group. If the system's performance declines upon reviewing the altered resumes (i.e., the evaluations become less favorable), one could infer that the model exhibits bias against applicants from the ethnic minority group.

To algorithmically address this issue, the authors propose a method to instill individual fairness during the training of ML models. This is achieved through *distributionally robust optimization* (DRO), an optimization technique that seeks the optimal solution while considering a fairness metric (inspired by Adversarial Robustness). 

### Learning fair metric from data and its hidden signals

The authors use Wasserstein distances to measure the similarity between individuals. Unlike Mahalanobis, Wasserstein distance can be used to compare two probability distributions and is defined as the minimum cost that must be paid to transform one distribution into the other.  The distances between data points are calculated in a way that takes into account protected attributes (in our example: gender or race). The goal is to ensure that similar individuals, as determined by Wasserstein distance, are treated similarly by the machine learning model.

To achieve this, the algorithm learn 'sensitive directions' in the data. These are directions in the feature space along which changes are likely to correspond to changes in protected attributes. These is a clever approach to uncover hidden biases by identifying subtle patterns that may correspond to changes in protected attributes, even if those attributes are not present in our model inputs. This allows the model to account for potential biases that might otherwise go unnoticed. 

For instance, to identify a sensitive direction associated with a particular attribute (e.g., gender), the algorithm use a logistic regression classifier to distinguish between classes (such as men and women in the data). The coefficients from this logistic regression model define a direction within the feature space. The performance of the machine learning model is assessed by its worst-case performance on hypothetical populations of users with perturbed sensitive attributes. By minimizing the loss function, the system is ensured to perform well on all such populations.

```python
# Same architecture we found
network_fair_LR = Model(input_size, output_size).to(device)
optimizer = torch.optim.Adam(network_fair_LR.parameters(), lr=1e-3)
lossfn = F.cross_entropy

# set the distance metric for instances similiraty detections
distance_x_LR = distances.LogisticRegSensitiveSubspace()
distance_y = distances.SquaredEuclideanDistance()

# train fair metric
distance_x_LR.fit(X_train, data_SensitiveAttrs=X_protected)
distance_y.fit(num_dims=output_size)

distance_x_LR.to(device)
distance_y.to(device)
```


```python
rho = 5.0
eps = 0.1
auditor_nsteps = 100
auditor_lr = 1e-3

fairalgo_LR = SenSeI(network_fair_LR, distance_x_LR, distance_y, lossfn, rho, eps, auditor_nsteps, auditor_lr)
```

### A fair objective function

The objective function that is minimized during the training of a fair machine learning model as proposed in the inFairness package is composed of two parts: the loss function and the fair metric (see [SenSeI](https://ibm.github.io/inFairness/_modules/inFairness/fairalgo/sensei.html#SenSeI)): 


```python
fair_loss = torch.mean(
            #--------1------------- + -----------------2-----------------------------# 
            self.loss_fn(Y_pred, Y) + self.rho * self.distance_y(Y_pred, Y_pred_worst)
        )
```

1. Loss Function: a classical loss function that measure of how well the model's predictions match the actual data. The goal of this metric is to adjust the model's parameters to minimize the loss score, and
2. Fair Metric (DIF): the fairness term is a measure of the difference between the model's predictions on the original data and its predictions on the worst-case examples. 

The model is trying to minimize this objective function, which means it's trying to make accurate and fair predictions.

It's important to note that due to the computation of a complex loss score, the training process becomes more resource-intensive.


```python
fairalgo_LR.train()

for epoch in tqdm(range(EPOCHS)):
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        result = fairalgo_LR(x, y)
        result.loss.backward()
        optimizer.step()
```

    100%|██████████| 10/10 [10:09<00:00, 60.90s/it]



```python
accuracy = metrics.accuracy(network_fair_LR, test_dl, device)
balanced_acc = metrics.balanced_accuracy(network_fair_LR, test_dl, device)
spouse_consistency = metrics.spouse_consistency(network_fair_LR, test_dl, test_dl_flip, device)

print(f'Accuracy: {accuracy}')
print(f'Balanced accuracy: {balanced_acc}')
print(f'Spouse consistency: {spouse_consistency}')
```

    Accuracy: 0.8401150107383728
    Balanced accuracy: 0.742399333699871
    Spouse consistency: 0.9997788589119858


### Results


```python
# Auditing using the SenSR Auditor + LR metric

audit_nsteps = 1000
audit_lr = 0.1

auditor_LR = SenSRAuditor(loss_fn=loss_fn, distance_x=distance_x_LR, num_steps=audit_nsteps, lr=audit_lr, max_noise=0.5, min_noise=-0.5)

audit_result_stdmodel = auditor_LR.audit(network_standard, X_test, y_test, lambda_param=10.0, audit_threshold=1.15)
audit_result_fairmodel_LR = auditor_LR.audit(network_fair_LR, X_test, y_test, lambda_param=10.0, audit_threshold=1.15)
print("="*100)
print("LR metric")
print(f"Loss ratio (Standard model) : {audit_result_stdmodel.lower_bound}. Is model fair: {audit_result_stdmodel.is_model_fair}")
print(f"Loss ratio (fair model - LogReg metric) : {audit_result_fairmodel_LR.lower_bound}. Is model fair: {audit_result_fairmodel_LR.is_model_fair}")
```
    
    LR metric
    Loss ratio (Standard model) : 2.1810670575586046. Is model fair: False
    Loss ratio (fair model - LogReg metric) : 1.0531351204682995. Is model fair: True   
    


{{< admonition type=success title="Findings">}}
Upon reviewing the overall results, we see that with a minor decrease in accuracy of 0.01, we have successfully constructed a fair model that is debiased with respect to gender and race.
{{< /admonition >}}

## Conclusion

We've explored a specific example of a fair-ML model using the tools provided by the inFairness package. While the results are promising, it's important to contextualize them within the broader challenges of responsible AI, particularly given the rapid evolution of ML tools and the dynamic nature of societal values.

Following Richardson, we can mention:

- **Conflicting Fairness Metrics**: Measurement is always a political activity in the sense that we must select, define, and prioritize certain dimensions of reality, setting aside others. Friedler et al. (2021) argue that fairness experts must explicitly state the priorities of each fairness metric to ensure practitioners make informed choices.
- **Metric Robustness**: Friedler et al. (2018) discovered that many fairness metrics lack robustness. Their study showed that by simply modifying dataset composition and changing train-test splits, many fairness criteria lacked stability.
- **Oversimplification of Fairness**: A major concern in the literature is the emphasis on technical solutions to algorithmic bias, which is a socio-technical problem. Madaio et al. (2020) referred to the exclusive use of technical solutions as "ethics washing," and Selbst et al. (2019) describe the failure to recognize that fairness cannot be solely achieved through mathematical formulation as the "formalism trap."
- **Operationalization of Ethical Concepts**: A significant challenge for fair AI is translating ethical reflections into actionable products and procedures that practitioners and institutions can implement. This difficulty is not unique to the AI field but affects every aspect of human activity where there is a need for ethical actions.

{{< admonition type=note title="Bibliography" open=false >}}
## Bibliography
- Dwork, Cynthia, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Rich Zemel. “Fairness Through Awareness.” arXiv, November 28, 2011. https://doi.org/10.48550/arXiv.1104.3913.
- Mehrabi, Ninareh, Fred Morstatter, Nripsuta Saxena, Kristina Lerman, and Aram Galstyan. “A Survey on Bias and Fairness in Machine Learning.” arXiv, January 25, 2022. https://doi.org/10.48550/arXiv.1908.09635.
- Richardson, Brianna, and Juan E. Gilbert. “A Framework for Fairness: A Systematic Review of Existing Fair AI Solutions.” arXiv, December 10, 2021. https://doi.org/10.48550/arXiv.2112.05700.
- Weerts, Hilde, Miroslav Dudík, Richard Edgar, Adrin Jalali, Roman Lutz, and Michael Madaio. “Fairlearn: Assessing and Improving Fairness of AI Systems.” arXiv, March 29, 2023. https://doi.org/10.48550/arXiv.2303.16626.
- Bird, Sarah., Dudík, Miro., Edgar, Richard., Horn, Brandon., Lutz, Roman., Milan, Vanessa., Sameki, Mehrnoosh., Wallach, Hanna., & Walker, Kathleen. "Fairlearn: A toolkit for assessing and improving fairness in AI." Microsoft, May 2020. https://www.microsoft.com/en-us/research/publication/fairlearn-a-toolkit-for-assessing-and-improving-fairness-in-ai.
- Floridi, Luciano., Cowls, Josh., Beltrametti, Monica., Chatila, Raja., Chazerand, Patrice., Dignum, Virginia., Luetge, Christoph., Madelin, Robert., Pagallo, Ugo., Rossi, Francesca., Schafer, Burkhard., Valcke, Peggy., & Vayena, Effy. "AI4People—An Ethical Framework for a Good AI Society: Opportunities, Risks, Principles, and Recommendations." Minds and Machines, December 1, 2018. https://doi.org/10.1007/s11023-018-9482-5. 
- Weerts, Hilde, Miroslav Dudík, Richard Edgar, Adrin Jalali, Roman Lutz, and Michael Madaio. “Fairlearn: Assessing and Improving Fairness of AI Systems.” arXiv, March 29, 2023. https://doi.org/10.48550/arXiv.2303.16626.
- Yurochkin, Mikhail, Amanda Bower, and Yuekai Sun. “Training Individually Fair ML Models with Sensitive Subspace Robustness.” arXiv, March 13, 2020. http://arxiv.org/abs/1907.00020.
{{< /admonition >}}

Pic by <a href="https://unsplash.com/@patrickian4?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Patrick Fore</a>, <a href="https://unsplash.com/es/s/fotos/balance?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
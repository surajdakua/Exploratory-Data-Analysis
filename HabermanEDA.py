'''
@author Suraj Dakua
Haberman Dataset/Kaggel to perform Exploratory Data Analysis(EDA).
EDA is for seeing what the data can tell us beyond the formal modelling and also summarize the data characteristics often with visual methods.  
EDA can be also used for better understanding of data and main features of data.
Machine learning is more about getting insights of data that is very very important.
In machine learning is not about how much lines of cde you write but it is about how much richness you bring to the analysis of the data.
'''

#import libraries required for the EDA.
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#seaborn is a very good data visualization tool used in python. 
import seaborn as sns

#load the dataset.
# Source: https://www.kaggle.com/gilsousa/habermans-survival-data-set
#dataset I'm using here is imbalanced dataset.
haberman = pd.read_csv('haberman.csv')
print(haberman.shape)  #prints the number of rows and columns in the dataset
print(haberman.columns)  #prints the objects present in the dataset.
print(haberman['year'].value_counts)

# haberman.plot(kind='scatter', x='age',y='status')

''' This plots are 2D Scatter plots'''
sns.set_style('whitegrid')
sns.FacetGrid(haberman, hue='status', size=5)\
    .map(plt.scatter, 'age', 'status')
plt.legend(labels=['Survived more than 5yrs','Died within 5yrs'])  


sns.set_style('whitegrid')
sns.FacetGrid(haberman, hue='year', size=5)\
    .map(plt.scatter, 'age', 'year')
plt.legend()    

'''Histogram plots'''
sns.FacetGrid(haberman, hue='year', size=5)\
    .map(sns.distplot, 'year')
plt.legend()

sns.FacetGrid(haberman, hue='age', size=5)\
    .map(sns.distplot, 'age')
plt.legend()

sns.FacetGrid(haberman, hue='status', size=5)\
    .map(sns.distplot, 'status')

''' Pair plots'''
#pairplots plots pairwise scatter plot.
#pair plot can be used to study the relationship between the two variables.
sns.pairplot(haberman,hue='year',size=4)

counts, bin_edges = np.histogram(haberman['age'],bins=5,density=True)

'''
To seperate two most usefull features amongst all the features in a plot
we use pair plots. 
'''

'''
univariate analysis as the name suggest one variable analysis.
which of the variable is more usefull as compared to my other variables is known as univariate analysis.
we only pick any one feature which is best suitable for the analysis.
'''

'''
Cummulative Density Function(CDF)
What percentage of Y-axis has value less than the corresponding
point on the X-axis.
It is the cummulative sum of the the probablities.
Area under the curve of the PDF till that point is called as CDF. 
If we differentiate CDF we get PDF and if we integrate our PDF we get 
our CDF.
'''
#to compute the pdf 
pdf = counts/(sum(counts))
print(pdf)
print(bin_edges)

#to compute the cdf
cdf = np.cumsum(pdf)  #cumsum = cummulative sum.
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

'''
Mean, Variance and Standard Deviation.
OUTLIER: Suppose we have 50 observtions and all observations lie between 1-10
but there is one observation which is 50 then such a point is called an OUTLIER
If we have an outlier then the mean will increase drastically this this will be an error
and we should take care of outliers using percentile std etc.
Varince is average square distance from the mean value.
And square rooot of variance  is nothing but the standard deviation.
Spread mathematicallly means standard deviation.
'''
print(np.mean(haberman['age']))
print(np.std(haberman['year']))

'''
Mean, Variance and Standard Deviation can be easily
corrupted easily if we have outliers. 
Median is the middle value of the observations
If more than 50% of the values are corrupted or have outlier then the median gets corrupted.
'''
print(np.median(haberman['age']))
#here 50 is the extreme value means an outlier we can see the difference.
print(np.median(np.append(haberman['age'],50)))

'''
Percentile: Suppose we want a 10th percentile value of the sorted array
the the percentile value tells us that how many points are less than the 10th index and how
many values are greater than the 10th index thats what a percentile is. 

Quantile: Break the sorted array into four parts(quant) is known as quantile.
'''
#Quantile with gap of 25 in a range of 0,100
print(np.percentile(haberman['age'],np.arange(0,100,25)))
#90th Percentile 
print(np.percentile(haberman['age'],90))

'''
Median Absolute Deviation
Numpy dont have mad function so we use statsmodel(robust) to do this.
'''
from statsmodels import robust
print(robust.mad(haberman['age']))

'''
IQR: Inter Quartile Range
Subtracting 75th value from 25th value gives the IQR range or value.
'''
'''
Box plot tells us what is the value for 25th percentile, 50th percentile
and 75th percentile.
Whiskers are the lines around the box plot.
SNS calculate whiskers values as 1.5xIQR
Box plot takes the mean,median and quartiles and put them in the box form.
'''
sns.boxplot(x='age',y='year', data=haberman)
'''
Violin Plots: Combinatiom of histogram,pdf and box plot is violin plot.
The thick box in the violin is the box plot with 25th.75th and 50th values.

'''
sns.violinplot(x='age',y='status',data=haberman,size=7)
'''
When we are looking at two variables and doing the analysis it 
is called bivariate analysis. Example is pair plots or scatetr plots.   
When we are looking for multiple variables then it is called multivariate analysis.
Machine learning is all about multivariate analysis.
'''
sns.jointplot(x='age',y='status',data=haberman,kind='kde ')
plt.show()







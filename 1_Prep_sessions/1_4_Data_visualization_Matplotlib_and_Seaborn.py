#!/usr/bin/env python
# coding: utf-8

# ## Contents:
# 
# 
# 1. [Intro_to_matplotlib](#Intro_to_matplotlib)<br>
# 2. [Case_study_Online_Retailer](#Case_study_Online_Retailer)<br>
# 3. [Plotting_Exercise_Numpy_Attributes](#Plotting_Exercise_Numpy_Attributes)<br>
# 4. [Practice_questions_footwear_dataset_solution](#Practice_questions_footwear_dataset_solution)<br>
# 5. [Case_Study_Google_Playstore](#Case_Study_Google_Playstore)<br>
# 6. [Case_study_cricket](#Case_study_cricket)<br>

# <a id='Intro_to_matplotlib'>Intro_to_matplotlib</a>

# In[19]:


get_ipython().run_line_magic('load_ext', 'IPython.core.magics')
import matplotlib.pyplot as plt
from IPython.display import HTML


# In[20]:


import numpy as np
x = np.linspace(0,5,11)
y = x**2

display(x)
display(y)

#functional 
plt.plot(x,y);
plt.xlabel('X label',color='white')
plt.ylabel('Y label',color='white')
plt.show()
plt.plot(y,x);
plt.show()


# In[21]:


# subplots (multi-plots on the same canvas)
# subplot(nrows, ncols, index, **kwargs)

plt.subplot(2,3,2)
#the figure has 2 row, 3 columns, and this plot is the plot - 2.
plt.plot(x,y,color="red")

plt.subplot(2,3,1)
#the figure has 2 row, 3 columns, and this plot is the plot - 1.
plt.plot(x,y,color="green")

plt.subplot(2,3,5)
#the figure has 2 row, 3 columns, and this plot is the plot - 5.
plt.plot(x,y,color="pink")

plt.subplot(1,3,3)
plt.plot(y,x,color="blue")

plt.subplot(3,1,3)
plt.plot(y,x,color="blue")


# In[22]:


# OO 

fig = plt.figure()
axes = fig.add_axes([-2, 0.1, 0.7,.7])
axes.plot(x,y)
plt.xlabel('X label',color='white')
plt.ylabel('Y label',color='white')
plt.title('Graph',color='white')


# In[23]:


fig = plt.figure()
axes1 = fig.add_axes([0.1,0.1,0.8,0.8])
axes2 = fig.add_axes([0.2,0.5,0.4,0.3])
# left, bottom, width, height
axes1.plot(x,y)
axes2.plot(y,x)
plt.title('Graph')
axes2.set_title("Nested graph")
axes1.set_title("Outer graph")


# <a id='Case_study_Online_Retailer'>Case_study_Online_Retailer</a>

# In[24]:


HTML('<h2 style = "color : Brown"> Case Study - Online Retailer </h2>')

HTML('<h4 style = "color : Sky blue"> Example - 1</h4> ') 

##### Bar Chart - Plot sales across each product category
print('''
- A bar chart uses bars to show comparisons between categories of data.
- A bar graph will always have two axis. 
- One axis will generally have numerical values or measures, 
- The other will describe the types of categories being compared or dimensions.
''')

import numpy as np
import matplotlib.pyplot as plt

product_category = np.array(['Furniture', 'Technology', 'Office Supplies'])
sales = np.array ([4110451.90, 4744557.50, 3787492.52] )

plt.bar(product_category, sales)
plt.show()


##### Bar Chart - Plot sales across each product category
print('''
1. Adding labels to Axes
2. Reducing the bar width
3. Giving Title to the chart
4. Modifying the ticks to show information in (million dollars)
''')

import numpy as np
import matplotlib.pyplot as plt

product_category = np.array(['Furniture', 'Technology', 'Office Supplies'])
sales = np.array ([4110451.90, 4744557.50, 3787492.52] )

# plotting bar chart and setting bar width to 0.5 and aligning it to center
plt.bar(product_category, sales, width= 0.5, align='center', edgecolor='Orange',color='cyan')

# Adding and formatting title
plt.title("Sales Across Product Categories\n", fontdict={'fontsize': 20, 'fontweight' : 5, 'color' : 'Green'})

# Labeling Axes
plt.xlabel("Product Category", fontdict={'fontsize': 12, 'fontweight' : 5, 'color' : 'Brown'})
plt.ylabel("Sales", fontdict={'fontsize': 12, 'fontweight' : 5, 'color' : 'Brown'})

# Modifying the ticks to show information in (million dollars)
ticks = np.arange(0, 6000000, 1000000)
labels = ["{}M".format(i//1000000) for i in ticks]
plt.yticks(ticks, labels)

plt.show()

HTML('<h4 style = "color : Sky blue"> Example - 2</h4>')

##### Scatter Chart - Plot Sales versus Profits across various Countries and Product Categories

print('''
Scatter plots are used when you want to show the relationship between two facts or measures.

- Profit is measured across each product category
''')

import numpy as np
import matplotlib.pyplot as plt

# Data
sales = np.array ([1013.14, 8298.48, 875.51, 22320.83, 9251.6, 4516.86, 585.16, 836154.03, 216748.48, 174.2, 27557.79, 563.25, 558.11, 37117.45, 357.36, 2206.96, 709.5, 35064.03, 7230.78, 235.33, 148.32, 3973.27, 11737.8, 7104.63, 83.67, 5569.83, 92.34, 107104.36, 1045.62, 9072.51, 42485.82, 5093.82, 14846.16, 943.92, 684.36, 15012.03, 38196.18, 2448.75, 28881.96, 13912.14, 4507.2, 4931.06, 12805.05, 67912.73, 4492.2, 1740.01, 458.04, 16904.32, 21744.53, 10417.26, 18665.33, 2808.42, 54195.57, 67332.5, 24390.95, 1790.43, 2234.19, 9917.5, 7408.14, 36051.99, 1352.22, 1907.7, 245722.14, 2154.66, 1078.21, 3391.65, 28262.73, 5177.04, 66.51, 2031.34, 1683.72, 1970.01, 6515.82, 1055.31, 1029.48, 5303.4, 1850.96, 1159.41, 39989.13, 1183.87, 96365.09, 8356.68, 7010.24, 23119.23, 46109.28, 146071.84, 242259.03, 9058.95, 1313.67, 31525.06, 2019.94, 703.04, 1868.79, 700.5, 55512.02, 243.5, 2113.18, 11781.81, 262189.49, 3487.29, 513.12, 312050.42, 5000.7, 121.02, 1302.78, 169.92, 124.29, 57366.05, 29445.93, 4614.3, 45009.98, 309.24, 3353.67, 41348.34, 2280.27, 61193.7, 1466.79, 12419.94, 445.12, 25188.65, 263514.92, 12351.23, 1152.3, 26298.81, 9900.78, 5355.57, 2325.66, 6282.81, 127707.92, 1283.1, 3560.15, 3723.84, 13715.01, 4887.9, 3396.89, 33348.42, 625.02, 1665.48, 32486.97, 340212.44, 20516.22, 8651.16, 13590.06, 2440.35, 6462.57, 1770.13, 7527.18, 1433.65, 423.3, 21601.72, 10035.72, 2378.49, 3062.38, 719469.32, 179366.79, 345.17, 30345.78, 300.71, 940.81, 36468.08, 1352.85, 1755.72, 2391.96, 19.98, 19792.8, 15633.88, 7.45, 521.67, 1118.24, 7231.68, 12399.32, 204.36, 23.64, 5916.48, 313.98, 108181.5, 9212.42, 27476.91, 1761.33, 289.5, 780.3, 15098.46, 813.27, 47.55, 8323.23, 22634.64, 1831.02, 28808.1, 10539.78, 588.99, 939.78, 7212.41, 15683.01, 41369.09, 5581.6, 403.36, 375.26, 12276.66, 15393.56, 76.65, 5884.38, 18005.49, 3094.71, 43642.78, 35554.83, 22977.11, 1026.33, 665.28, 9712.49, 6038.52, 30756.51, 3758.25, 4769.49, 2463.3, 160153.16, 967.11, 2311.74, 1414.83, 12764.91, 4191.24, 110.76, 637.34, 1195.12, 2271.63, 804.12, 196.17, 167.67, 131.77, 2842.05, 9969.12, 1784.35, 3098.49, 25005.54, 1300.1, 118697.39, 7920.54, 6471.78, 31707.57, 37636.47, 118777.77, 131170.76, 3980.88, 3339.39, 26563.9, 4038.73, 124.8, 196.65, 2797.77, 29832.76, 184.84, 79.08, 8047.83, 205313.25, 1726.98, 899.73, 224.06, 304763.54, 6101.31, 729.6, 896.07, 17.82, 26.22, 46429.78, 31167.27, 2455.94, 37714.3, 1506.93, 3812.78, 25223.34, 3795.96, 437.31, 41278.86, 2091.81, 6296.61, 468.82, 23629.64, 160435.53, 9725.46, 1317.03, 1225.26, 30034.08, 7893.45, 2036.07, 215.52, 3912.42, 82783.43, 253.14, 966.96, 3381.26, 164.07, 1984.23, 75.12, 25168.17, 3295.53, 991.12, 10772.1, 44.16, 1311.45, 35352.57, 245783.54, 20.49, 13471.06, 8171.16, 14075.67, 611.82, 3925.56, 981.84, 10209.84, 156.56, 243.06, 21287.52, 7300.51, 434.52, 6065.0, 741577.51, 132461.03, 224.75, 28953.6, 757.98, 528.15, 34922.41, 50.58, 2918.48, 1044.96, 22195.13, 3951.48, 6977.64, 219.12, 5908.38, 10987.46, 4852.26, 445.5, 71860.82, 14840.45, 24712.08, 1329.9, 1180.44, 85.02, 10341.63, 690.48, 1939.53, 20010.51, 914.31, 25223.82, 12804.66, 2124.24, 602.82, 2961.66, 15740.79, 74138.35, 7759.39, 447.0, 2094.84, 22358.95, 21734.53, 4223.73, 17679.53, 1019.85, 51848.72, 69133.3, 30146.9, 705.48, 14508.88, 7489.38, 20269.44, 246.12, 668.13, 768.93, 215677.35, 899.16, 2578.2, 4107.99, 20334.57, 366.84, 3249.27, 98.88, 3497.88, 3853.05, 786.75, 1573.68, 458.36, 1234.77, 1094.22, 2300.61, 970.14, 3068.25, 35792.85, 4277.82, 71080.28, 3016.86, 3157.49, 15888.0, 30000.36, 140037.89, 216056.25, 1214.22, 1493.94, 32036.69, 4979.66, 106.02, 46257.68, 1033.3, 937.32, 3442.62, 160633.45, 213.15, 338.88, 242117.13, 9602.34, 2280.99, 73759.08, 23526.12, 6272.74, 43416.3, 576.78, 1471.61, 20844.9, 3497.7, 56382.38, 902.58, 6235.26, 48.91, 32684.24, 276611.58, 13370.38, 10595.28, 4555.14, 10084.38, 267.72, 1012.95, 4630.5, 149433.51, 364.32, 349.2, 4647.56, 504.0, 10343.52, 5202.66, 2786.26, 34135.95, 2654.58, 24699.51, 339239.87, 136.26, 23524.51, 8731.68, 8425.86, 835.95, 11285.19])
profit = np.array([-1213.46, 1814.13, -1485.7, -2286.73, -2872.12, 946.8, 198.48, 145454.95, 49476.1, -245.56, 5980.77, -790.47, -895.72, -34572.08, 117.9, 561.96, 152.85, 1426.05, 1873.17, -251.03, 68.22, 635.11, 3722.4, -3168.63, 27.6, 952.11, 7.38, 20931.13, 186.36, -5395.38, 9738.45, 525.27, 3351.99, 120.78, 266.88, 3795.21, 8615.97, 609.54, 7710.57, 2930.43, 1047.96, -2733.32, 2873.73, -5957.89, -909.6, 163.41, -376.02, -6322.68, -10425.86, 2340.36, -28430.53, 756.12, 12633.33, 7382.54, -14327.69, 436.44, 683.85, -694.91, 1960.56, 10925.82, 334.08, 425.49, 53580.2, 1024.56, 110.93, 632.22, 8492.58, 1418.88, 19.26, -2567.57, 346.26, 601.86, 1318.68, 304.05, 428.37, 1416.24, -2878.18, 283.41, 12611.04, 261.95, -648.43, 1112.88, -2640.29, 6154.32, 11558.79, 15291.4, 56092.65, 1515.39, 342.03, -10865.66, -902.8, 351.52, 364.17, 87.72, 11565.66, 75.4, 289.33, 3129.63, 50795.72, 783.72, 215.46, 29196.89, 1147.26, 53.22, 286.56, 73.02, 42.24, 13914.85, 5754.54, 998.04, -1476.04, 86.58, -1636.35, 10511.91, 647.34, 13768.62, 338.67, 3095.67, 173.84, 5632.93, 64845.11, 3297.33, 338.61, 7246.62, 2255.52, 1326.36, 827.64, 1100.58, 9051.36, 412.23, 1063.91, 940.59, 3891.84, 1599.51, 1129.57, 8792.64, 6.24, 592.77, 8792.85, 47727.5, -4597.68, 2242.56, 3546.45, 321.87, 1536.72, -2463.29, 1906.08, -1916.99, 186.24, 3002.05, -3250.98, 554.7, 830.64, 122612.79, 33894.21, -559.03, 7528.05, -477.67, -1660.25, -33550.96, 481.68, 425.08, 450.3, 9.57, -3025.29, 2924.62, -11.84, 87.36, 26.51, 1727.19, -6131.18, 59.16, 3.06, 1693.47, 74.67, 24729.21, -4867.94, 6705.18, 410.79, 70.74, 101.7, 3264.3, 137.01, 6.18, 2100.21, 5295.24, 520.29, 7205.52, 2602.65, 116.67, 224.91, -5153.93, 3882.69, -6535.24, -1254.1, 84.56, -186.38, -3167.2, -7935.59, 37.02, 1908.06, -27087.84, 829.32, 8727.44, 2011.47, -11629.64, 234.96, 53.1, 1248.14, 1511.07, 7374.24, 1193.28, 1090.23, 553.86, 38483.86, 255.81, 528.54, 326.07, 3924.36, 1018.92, 36.48, 113.24, -1770.05, 527.64, 224.49, 79.53, 64.77, 38.08, 868.08, 2265.06, -2643.62, 833.73, 5100.03, 326.44, 18158.84, 1682.01, -3290.22, 8283.33, 7926.18, 1694.41, 30522.92, 1214.07, 900.6, -6860.8, -865.91, 26.16, 47.22, 863.52, 7061.26, 73.92, 33.12, 1801.23, 38815.44, 431.13, 216.81, 16.5, 53688.2, 1210.32, 236.94, 210.84, 3.18, 2.22, 10265.64, 7212.3, 343.56, 3898.28, 568.11, -1867.85, 5782.38, 697.29, -192.06, 10179.02, 616.32, 1090.47, 165.84, 6138.28, 39723.06, 2085.14, 90.0, 129.93, 7957.53, 2131.86, 562.44, 99.12, 1298.37, 7580.33, 113.73, 139.71, 456.0, 21.24, 292.68, 30.34, 5817.15, 1060.89, 252.9, 3060.61, 6.6, 219.09, 8735.82, 31481.09, 2.85, -3124.72, 2195.94, 3464.7, 141.12, 1125.69, -1752.03, 3281.52, -303.77, 114.18, -2412.63, -5099.61, 146.64, 660.22, 18329.28, 28529.84, -232.27, 7435.41, -1157.94, -746.73, -30324.2, 2.52, 1313.44, 213.72, -5708.95, 930.18, 1663.02, 31.59, 1787.88, -8219.56, 973.92, 4.32, 8729.78, -2529.52, 5361.06, 69.21, 519.3, 13.56, 2236.77, 213.96, 367.98, 5074.2, 206.61, 7620.36, 2093.19, 164.07, 230.01, -815.82, 4226.7, -3635.09, -3344.17, 167.26, 143.79, -8233.57, -4085.21, 919.35, -25232.35, 234.33, 12040.68, 7206.28, -15112.76, 206.04, -2662.49, 2346.81, 4461.36, 93.48, 82.11, 147.87, 10389.53, 395.58, 474.74, 1333.26, 3913.02, 117.36, 858.78, 6.9, -4628.49, 1170.6, 218.55, 539.58, -211.0, 438.87, 317.16, 310.8, -1578.09, 706.56, 6617.4, 803.84, 2475.26, 764.34, -1461.88, 3805.56, 7371.27, -1377.13, 42435.03, 472.47, 315.48, -11755.91, -2418.6, 6.36, 9317.76, 326.88, -287.31, 637.68, 17579.17, 70.83, 47.4, 26143.92, 1548.15, 612.78, 17842.76, 6735.39, 1206.5, -10035.74, 149.4, -777.85, 5566.29, 748.92, 14941.58, 348.93, 1944.06, -5.51, 7026.84, 46114.92, 2361.86, 2613.24, 1277.37, 2587.74, 103.08, 311.43, 1250.58, 13055.21, 18.21, 108.24, 709.44, 115.92, 1863.6, 1873.86, 817.32, 7577.64, 1019.19, 6813.03, 24698.84, 66.24, -10971.39, 2056.47, 2095.35, 246.33, 2797.89])
product_category = np.array(['Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture', 'Furniture'])
country = np.array(['Zimbabwe', 'Zambia', 'Yemen', 'Vietnam', 'Venezuela', 'Uzbekistan', 'Uruguay', 'United States', 'United Kingdom', 'United Arab Emirates', 'Ukraine', 'Uganda', 'Turkmenistan', 'Turkey', 'Tunisia', 'Trinidad and Tobago', 'Togo', 'Thailand', 'Tanzania', 'Tajikistan', 'Taiwan', 'Syria', 'Switzerland', 'Sweden', 'Swaziland', 'Sudan', 'Sri Lanka', 'Spain', 'South Sudan', 'South Korea', 'South Africa', 'Somalia', 'Singapore', 'Sierra Leone', 'Serbia', 'Senegal', 'Saudi Arabia', 'Rwanda', 'Russia', 'Romania', 'Qatar', 'Portugal', 'Poland', 'Philippines', 'Peru', 'Paraguay', 'Papua New Guinea', 'Panama', 'Pakistan', 'Norway', 'Nigeria', 'Niger', 'Nicaragua', 'New Zealand', 'Netherlands', 'Nepal', 'Namibia', 'Myanmar (Burma)', 'Mozambique', 'Morocco', 'Mongolia', 'Moldova', 'Mexico', 'Mauritania', 'Martinique', 'Mali', 'Malaysia', 'Madagascar', 'Luxembourg', 'Lithuania', 'Libya', 'Liberia', 'Lesotho', 'Lebanon', 'Kyrgyzstan', 'Kenya', 'Kazakhstan', 'Jordan', 'Japan', 'Jamaica', 'Italy', 'Israel', 'Ireland', 'Iraq', 'Iran', 'Indonesia', 'India', 'Hungary', 'Hong Kong', 'Honduras', 'Haiti', 'Guyana', 'Guinea-Bissau', 'Guinea', 'Guatemala', 'Guadeloupe', 'Greece', 'Ghana', 'Germany', 'Georgia', 'Gabon', 'France', 'Finland', 'Ethiopia', 'Estonia', 'Eritrea', 'Equatorial Guinea', 'El Salvador', 'Egypt', 'Ecuador', 'Dominican Republic', 'Djibouti', 'Denmark', 'Democratic Republic of the Congo', 'Czech Republic', 'Cuba', 'Croatia', "Cote d'Ivoire", 'Costa Rica', 'Colombia', 'China', 'Chile', 'Central African Republic', 'Canada', 'Cameroon', 'Cambodia', 'Burkina Faso', 'Bulgaria', 'Brazil', 'Bosnia and Herzegovina', 'Bolivia', 'Benin', 'Belgium', 'Belarus', 'Barbados', 'Bangladesh', 'Bahrain', 'Azerbaijan', 'Austria', 'Australia', 'Argentina', 'Angola', 'Algeria', 'Albania', 'Afghanistan', 'Zimbabwe', 'Zambia', 'Yemen', 'Western Sahara', 'Vietnam', 'Venezuela', 'Uzbekistan', 'Uruguay', 'United States', 'United Kingdom', 'United Arab Emirates', 'Ukraine', 'Uganda', 'Turkmenistan', 'Turkey', 'Tunisia', 'Trinidad and Tobago', 'Togo', 'The Gambia', 'Thailand', 'Tanzania', 'Tajikistan', 'Taiwan', 'Syria', 'Switzerland', 'Sweden', 'Swaziland', 'Suriname', 'Sudan', 'Sri Lanka', 'Spain', 'South Korea', 'South Africa', 'Somalia', 'Slovenia', 'Slovakia', 'Singapore', 'Sierra Leone', 'Serbia', 'Senegal', 'Saudi Arabia', 'Rwanda', 'Russia', 'Romania', 'Republic of the Congo', 'Qatar', 'Portugal', 'Poland', 'Philippines', 'Peru', 'Paraguay', 'Papua New Guinea', 'Panama', 'Pakistan', 'Oman', 'Norway', 'Nigeria', 'Niger', 'Nicaragua', 'New Zealand', 'Netherlands', 'Nepal', 'Namibia', 'Myanmar (Burma)', 'Mozambique', 'Morocco', 'Montenegro', 'Mongolia', 'Moldova', 'Mexico', 'Mauritania', 'Martinique', 'Mali', 'Malaysia', 'Madagascar', 'Macedonia', 'Luxembourg', 'Lithuania', 'Libya', 'Liberia', 'Lesotho', 'Lebanon', 'Laos', 'Kyrgyzstan', 'Kenya', 'Kazakhstan', 'Jordan', 'Japan', 'Jamaica', 'Italy', 'Israel', 'Ireland', 'Iraq', 'Iran', 'Indonesia', 'India', 'Hungary', 'Hong Kong', 'Honduras', 'Haiti', 'Guyana', 'Guinea-Bissau', 'Guinea', 'Guatemala', 'Guadeloupe', 'Greece', 'Ghana', 'Germany', 'Georgia', 'Gabon', 'French Guiana', 'France', 'Finland', 'Ethiopia', 'Estonia', 'Eritrea', 'Equatorial Guinea', 'El Salvador', 'Egypt', 'Ecuador', 'Dominican Republic', 'Djibouti', 'Denmark', 'Democratic Republic of the Congo', 'Czech Republic', 'Cyprus', 'Cuba', 'Croatia', "Cote d'Ivoire", 'Costa Rica', 'Colombia', 'China', 'Chile', 'Chad', 'Central African Republic', 'Canada', 'Cameroon', 'Cambodia', 'Burkina Faso', 'Bulgaria', 'Brazil', 'Botswana', 'Bosnia and Herzegovina', 'Bolivia', 'Bhutan', 'Benin', 'Belize', 'Belgium', 'Belarus', 'Barbados', 'Bangladesh', 'Bahrain', 'Azerbaijan', 'Austria', 'Australia', 'Armenia', 'Argentina', 'Angola', 'Algeria', 'Albania', 'Afghanistan', 'Zimbabwe', 'Zambia', 'Yemen', 'Western Sahara', 'Vietnam', 'Venezuela', 'Uzbekistan', 'Uruguay', 'United States', 'United Kingdom', 'United Arab Emirates', 'Ukraine', 'Uganda', 'Turkmenistan', 'Turkey', 'Tunisia', 'Trinidad and Tobago', 'Togo', 'Thailand', 'Tanzania', 'Taiwan', 'Syria', 'Switzerland', 'Sweden', 'Sudan', 'Sri Lanka', 'Spain', 'South Korea', 'South Africa', 'Somalia', 'Slovenia', 'Slovakia', 'Singapore', 'Sierra Leone', 'Senegal', 'Saudi Arabia', 'Rwanda', 'Russia', 'Romania', 'Republic of the Congo', 'Qatar', 'Portugal', 'Poland', 'Philippines', 'Peru', 'Paraguay', 'Papua New Guinea', 'Panama', 'Pakistan', 'Norway', 'Nigeria', 'Niger', 'Nicaragua', 'New Zealand', 'Netherlands', 'Nepal', 'Myanmar (Burma)', 'Mozambique', 'Morocco', 'Montenegro', 'Mongolia', 'Moldova', 'Mexico', 'Mauritania', 'Martinique', 'Mali', 'Malaysia', 'Malawi', 'Madagascar', 'Macedonia', 'Lithuania', 'Libya', 'Liberia', 'Lebanon', 'Laos', 'Kyrgyzstan', 'Kuwait', 'Kenya', 'Kazakhstan', 'Jordan', 'Japan', 'Jamaica', 'Italy', 'Israel', 'Ireland', 'Iraq', 'Iran', 'Indonesia', 'India', 'Hungary', 'Hong Kong', 'Honduras', 'Haiti', 'Guyana', 'Guatemala', 'Guadeloupe', 'Greece', 'Ghana', 'Germany', 'Georgia', 'Gabon', 'France', 'Finland', 'Estonia', 'El Salvador', 'Egypt', 'Ecuador', 'Dominican Republic', 'Djibouti', 'Denmark', 'Democratic Republic of the Congo', 'Czech Republic', 'Cuba', 'Croatia', "Cote d'Ivoire", 'Costa Rica', 'Colombia', 'China', 'Chile', 'Canada', 'Cameroon', 'Cambodia', 'Burundi', 'Burkina Faso', 'Bulgaria', 'Brazil', 'Botswana', 'Bosnia and Herzegovina', 'Bolivia', 'Benin', 'Belgium', 'Belarus', 'Barbados', 'Bangladesh', 'Azerbaijan', 'Austria', 'Australia', 'Armenia', 'Argentina', 'Angola', 'Algeria', 'Albania', 'Afghanistan'])

# plotting scatter chart

plt.scatter(profit, sales, alpha= 0.7, s = 50 )

# Adding and formatting title
plt.title("Sales versus Profits across various Countries and Product Categories\n", fontdict={'fontsize': 20, 'fontweight' : 5, 'color' : 'Green'})

# Labeling Axes
plt.xlabel("Profit", fontdict={'fontsize': 12, 'fontweight' : 5, 'color' : 'Brown'})
plt.ylabel("Sales", fontdict={'fontsize': 12, 'fontweight' : 5, 'color' : 'Brown'})

plt.show()

##### Scatter Chart - Plot Sales versus Profits across various Countries and Product Categories
print('''
- Represent product category using different colors
- Adding a Legend to Product Categories
''')

plt.scatter(profit[product_category == "Technology"], sales[product_category == "Technology"], 
            c= 'Green', alpha= 0.7, s = 150, label="Technology" )

plt.scatter(profit[product_category == "Office Supplies"], sales[product_category == "Office Supplies"], 
            c= 'Yellow', alpha= 0.7, s = 100, label="Office Supplies" )

plt.scatter(profit[product_category == "Furniture"], sales[product_category == "Furniture"], 
            c= 'Cyan', alpha= 0.7, s = 50, label="Furniture" )

# Adding and formatting title
plt.title("Sales versus Profits across various Countries and Product Categories\n", fontdict={'fontsize': 20, 'fontweight' : 5, 'color' : 'Green'})

# Labeling Axes
plt.xlabel("Profit", fontdict={'fontsize': 12, 'fontweight' : 5, 'color' : 'Brown'})
plt.ylabel("Sales", fontdict={'fontsize': 12, 'fontweight' : 5, 'color' : 'Brown'})

plt.legend()

plt.show()

plt.scatter(profit[product_category == "Technology"], sales[product_category == "Technology"], 
            c= 'Green', alpha= 0.7, s = 150, label="Technology" )

plt.scatter(profit[product_category == "Office Supplies"], sales[product_category == "Office Supplies"], 
            c= 'Yellow', alpha= 0.7, s = 100, label="Office Supplies" )

plt.scatter(profit[product_category == "Furniture"], sales[product_category == "Furniture"], 
            c= 'Cyan', alpha= 0.7, s = 50, label="Furniture" )

for xy in zip (profit[country == "India"], sales[country == "India"]):
	plt.annotate(text = "India", xy = xy)

# Adding and formatting title
plt.title("Sales versus Profits across various Countries and Product Categories\n", fontdict={'fontsize': 20, 'fontweight' : 5, 'color' : 'Green'})

# Labeling Axes
plt.xlabel("Profit", fontdict={'fontsize': 12, 'fontweight' : 5, 'color' : 'Brown'})
plt.ylabel("Sales", fontdict={'fontsize': 12, 'fontweight' : 5, 'color' : 'Brown'})

plt.legend()

plt.show()

HTML('<h4 style = "color : Sky blue"> Example - 3</h4>') 

##### Line Chart - Plot Sales across 2015

print('''
A line chart or line plot or line graph or curve chart is a type of chart which displays information as a series of data points called 'markers' connected by straight line segments. Most commonly used with time data.
''')

import numpy as np
import matplotlib.pyplot as plt

months = np.array(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
sales = np.array([241268.56, 184837.36, 263100.77, 242771.86, 288401.05, 401814.06, 258705.68, 456619.94, 481157.24, 422766.63, 555279.03, 503143.69])

plt.plot(months, sales)

# Adding and formatting title
plt.title("Sales across 2015\n", fontdict={'fontsize': 20, 'fontweight' : 5, 'color' : 'Green'})

# Labeling Axes
plt.xlabel("Months", fontdict={'fontsize': 12, 'fontweight' : 5, 'color' : 'Brown'})
plt.ylabel("Sales", fontdict={'fontsize': 12, 'fontweight' : 5, 'color' : 'Brown'} )

ticks = np.arange(0, 600000, 50000)
labels = ["{}K".format(i//1000) for i in ticks]
plt.yticks(ticks, labels)

plt.xticks(rotation=90)

plt.show()

##### Line Chart - Plot Sales across 2015

print('''
- Add labels to marks
''')

plt.plot(months, sales)

# Adding and formatting title
plt.title("Sales across 2015\n", fontdict={'fontsize': 20, 'fontweight' : 5, 'color' : 'Green'})

# Labeling Axes
plt.xlabel("Months", fontdict={'fontsize': 12, 'fontweight' : 5, 'color' : 'Brown'})
plt.ylabel("Sales", fontdict={'fontsize': 12, 'fontweight' : 5, 'color' : 'Brown'} )

ticks = np.arange(0, 600000, 50000)
labels = ["{}K".format(i//1000) for i in ticks]
plt.yticks(ticks, labels)

plt.xticks(rotation=90)

for xy in zip(months, sales):
    plt.annotate(text = "{}K".format(xy[1]//1000), xy = xy,  textcoords='data')

plt.show()

HTML('<h4 style = "color : Sky blue"> Example - 4</h4>') 

##### Box and Whisker Chart - Sales across Countries and Product Categories

print('''A Box and Whisker Plot (or Box Plot) is a convenient way of visually displaying the data distribution through their quartiles. The lines extending parallel from the boxes are known as the “whiskers”, which are used to indicate variability outside the upper and lower quartiles. Outliers are sometimes plotted as individual dots that are in-line with whiskers. Box Plots can be drawn either vertically or horizontally.
''')

import numpy as np
import matplotlib.pyplot as plt

# Data
sales_technology = np.array ([1013.14, 8298.48, 875.51, 22320.83, 9251.6, 4516.86, 585.16, 174.2, 27557.79, 563.25, 558.11, 37117.45, 357.36, 2206.96, 709.5, 35064.03, 7230.78, 235.33, 148.32, 3973.27, 11737.8, 7104.63, 83.67, 5569.83, 92.34, 1045.62, 9072.51, 42485.82, 5093.82, 14846.16, 943.92, 684.36, 15012.03, 38196.18, 2448.75, 28881.96, 13912.14, 4507.2, 4931.06, 12805.05, 67912.73, 4492.2, 1740.01, 458.04, 16904.32, 21744.53, 10417.26, 18665.33, 2808.42, 54195.57, 67332.5, 24390.95, 1790.43, 2234.19, 9917.5, 7408.14, 36051.99, 1352.22, 1907.7, 2154.66, 1078.21, 3391.65, 28262.73, 5177.04, 66.51, 2031.34, 1683.72, 1970.01, 6515.82, 1055.31, 1029.48, 5303.4, 1850.96, 1159.41, 39989.13, 1183.87, 96365.09, 8356.68, 7010.24, 23119.23, 46109.28, 9058.95, 1313.67, 31525.06, 2019.94, 703.04, 1868.79, 700.5, 55512.02, 243.5, 2113.18, 11781.81, 3487.29, 513.12, 5000.7, 121.02, 1302.78, 169.92, 124.29, 57366.05, 29445.93, 4614.3, 45009.98, 309.24, 3353.67, 41348.34, 2280.27, 61193.7, 1466.79, 12419.94, 445.12, 25188.65, 12351.23, 1152.3, 26298.81, 9900.78, 5355.57, 2325.66, 6282.81, 1283.1, 3560.15, 3723.84, 13715.01, 4887.9, 3396.89, 33348.42, 625.02, 1665.48, 32486.97, 20516.22, 8651.16, 13590.06, 2440.35, 6462.57])
sales_office_supplies = np.array ([1770.13, 7527.18, 1433.65, 423.3, 21601.72, 10035.72, 2378.49, 3062.38, 345.17, 30345.78, 300.71, 940.81, 36468.08, 1352.85, 1755.72, 2391.96, 19.98, 19792.8, 15633.88, 7.45, 521.67, 1118.24, 7231.68, 12399.32, 204.36, 23.64, 5916.48, 313.98, 9212.42, 27476.91, 1761.33, 289.5, 780.3, 15098.46, 813.27, 47.55, 8323.23, 22634.64, 1831.02, 28808.1, 10539.78, 588.99, 939.78, 7212.41, 15683.01, 41369.09, 5581.6, 403.36, 375.26, 12276.66, 15393.56, 76.65, 5884.38, 18005.49, 3094.71, 43642.78, 35554.83, 22977.11, 1026.33, 665.28, 9712.49, 6038.52, 30756.51, 3758.25, 4769.49, 2463.3, 967.11, 2311.74, 1414.83, 12764.91, 4191.24, 110.76, 637.34, 1195.12, 2271.63, 804.12, 196.17, 167.67, 131.77, 2842.05, 9969.12, 1784.35, 3098.49, 25005.54, 1300.1, 7920.54, 6471.78, 31707.57, 37636.47, 3980.88, 3339.39, 26563.9, 4038.73, 124.8, 196.65, 2797.77, 29832.76, 184.84, 79.08, 8047.83, 1726.98, 899.73, 224.06, 6101.31, 729.6, 896.07, 17.82, 26.22, 46429.78, 31167.27, 2455.94, 37714.3, 1506.93, 3812.78, 25223.34, 3795.96, 437.31, 41278.86, 2091.81, 6296.61, 468.82, 23629.64, 9725.46, 1317.03, 1225.26, 30034.08, 7893.45, 2036.07, 215.52, 3912.42, 82783.43, 253.14, 966.96, 3381.26, 164.07, 1984.23, 75.12, 25168.17, 3295.53, 991.12, 10772.1, 44.16, 1311.45, 35352.57, 20.49, 13471.06, 8171.16, 14075.67, 611.82, 3925.56])
sales_furniture = np.array ([981.84, 10209.84, 156.56, 243.06, 21287.52, 7300.51, 434.52, 6065.0, 224.75, 28953.6, 757.98, 528.15, 34922.41, 50.58, 2918.48, 1044.96, 22195.13, 3951.48, 6977.64, 219.12, 5908.38, 10987.46, 4852.26, 445.5, 71860.82, 14840.45, 24712.08, 1329.9, 1180.44, 85.02, 10341.63, 690.48, 1939.53, 20010.51, 914.31, 25223.82, 12804.66, 2124.24, 602.82, 2961.66, 15740.79, 74138.35, 7759.39, 447.0, 2094.84, 22358.95, 21734.53, 4223.73, 17679.53, 1019.85, 51848.72, 69133.3, 30146.9, 705.48, 14508.88, 7489.38, 20269.44, 246.12, 668.13, 768.93, 899.16, 2578.2, 4107.99, 20334.57, 366.84, 3249.27, 98.88, 3497.88, 3853.05, 786.75, 1573.68, 458.36, 1234.77, 1094.22, 2300.61, 970.14, 3068.25, 35792.85, 4277.82, 71080.28, 3016.86, 3157.49, 15888.0, 30000.36, 1214.22, 1493.94, 32036.69, 4979.66, 106.02, 46257.68, 1033.3, 937.32, 3442.62, 213.15, 338.88, 9602.34, 2280.99, 73759.08, 23526.12, 6272.74, 43416.3, 576.78, 1471.61, 20844.9, 3497.7, 56382.38, 902.58, 6235.26, 48.91, 32684.24, 13370.38, 10595.28, 4555.14, 10084.38, 267.72, 1012.95, 4630.5, 364.32, 349.2, 4647.56, 504.0, 10343.52, 5202.66, 2786.26, 34135.95, 2654.58, 24699.51, 136.26, 23524.51, 8731.68, 8425.86, 835.95, 11285.19])


plt.boxplot([sales_technology, sales_office_supplies, sales_furniture])

# Adding and formatting title
plt.title("Sales across Countries and Product Categories\n", fontdict={'fontsize': 20, 'fontweight' : 5, 'color' : 'Green'})

# Labeling Axes
plt.xlabel("Product Category", fontdict={'fontsize': 12, 'fontweight' : 5, 'color' : 'Brown'})
plt.ylabel("Sales", fontdict={'fontsize': 12, 'fontweight' : 5, 'color' : 'Brown'})

plt.xticks((1,2,3),["Technology", "Office Supplies", "Furniture"])

plt.show()

HTML('<h4 style = "color : Sky blue"> Example - 5</h4>')

##### Histogram - Plot a histogram for Sales distribution across Countries.


print('''A histogram is a plot that lets you discover, and show, the underlying frequency distribution (shape) of a set of continuous data.''')

import numpy as np
import matplotlib.pyplot as plt

# Data   
profit = np.array([-5428.79, 7001.73, -3706.46, 300.42, -1697.31, -11222.71, 1648.14, 1689.34, -1036.86, 20944.23, -2426.09, -3302.71, 602.1, 2300.48, 816.87, 9.57, -7308.2, 5727.97, -262.87, 1818.6, 693.21, 7237.47, -17519.37, 86.76, 3.06, 3619.5, 86.37, 54390.12, 186.36, -12792.83, 21804.69, 1005.27, 590.04, 115.26, 8853.06, 471.75, 273.06, 6263.4, 18985.41, 1336.44, 22536.45, 7626.27, 280.74, 1502.88, -8703.06, 10983.12, -16128.23, -5507.88, 415.23, -418.61, -17723.45, -22446.65, 37.02, 5167.77, 1819.77, 33401.44, 16600.28, 877.44, 736.95, -2109.26, 5818.44, 22761.42, 1286.76, 1506.42, 1127.22, 1675.95, 1114.21, 2291.55, 16329.96, 117.36, 3296.58, 43.38, 132.5, -8966.12, 2044.5, 1044.9, 1398.21, 908.4, -172.92, 1735.32, 317.16, 3992.1, -7099.9, 1823.7, 24328.47, 1392.23, 19985.68, 3559.23, -7392.38, 18243.21, 26856.24, 15608.68, 3201.93, 1558.11, -29482.37, -4187.31, 384.04, 411.39, 951.24, 27944.69, 476.2, 35.14, 5568.54, 1285.68, 479.67, 16.5, 3905.73, 290.16, 1110.18, 76.2, 44.46, 42023.24, 19702.23, 2548.1, -7613.5, 804.09, -4282.05, 21860.58, 2093.55, -192.06, 38889.22, 1303.92, 6130.2, 334.17, 18798.05, 7744.33, 90.0, 468.54, 17817.39, 5664.75, 4476.54, 103.08, 1238.19, 3649.53, 29686.9, 131.94, 660.18, 2229.35, 21.24, 1349.19, 30.34, 11572.59, 4534.26, 2199.79, 19430.89, 12.84, 1831.05, 24341.7, 69.09, -18693.8, 6494.97, 9106.5, 709.32, 5460.3])


plt.hist(profit, bins = 100,edgecolor='Orange',color='cyan')

plt.show() 

HTML('<h4 style = "color : Sky blue"> Example - 6</h4>')  

##### SubPlots - Plot Sales for various markets for years 2012 to 2015

import numpy as np
import matplotlib.pyplot as plt

years = np.array(['2012', '2013', '2014', '2015'])

sales_africa = np.array([127187.27, 144480.70, 229068.79, 283036.44])

sales_USCA = np.array([492756.60, 486629.30, 627634.98, 757108.13])

sales_LATAM = np.array([385098.15, 464733.29, 608140.77, 706632.93])

sales_Asia_Pacific = np.array([713658.22, 863983.97, 1092231.65, 1372784.40])

sales_Europe = np.array([540750.63, 717611.40, 848670.24, 1180303.95])

##### Subplots are shown in the same graph as line charts, identified by different colours

fig, ax = plt.subplots()

europe, = ax.plot(years, sales_Europe)
europe.set_label("Europe")

usca, = ax.plot(years, sales_USCA)
usca.set_label("USCA")
usca.set_dashes([2, 2, 2, 2])

africa, = ax.plot(years, sales_africa)
africa.set_label("Africa")
africa.set_dashes([2, 2, 5, 2])

asia, = ax.plot(years, sales_Asia_Pacific)
asia.set_label("Asia Pacific")
asia.set_dashes([2, 2, 10, 2])

latam, = ax.plot(years, sales_LATAM)
latam.set_label("LATAM")
latam.set_dashes([2, 5, 5, 2])

plt.legend()

plt.legend(bbox_to_anchor=(1.31,0.4))

plt.show()

##### Subplots are shown in seperate graphs. Each chart can be of different types.

fig, ax = plt.subplots(ncols=3, nrows=2, sharex=True, sharey=True)

europe, = ax[0][0].plot(years, sales_Europe)
europe.set_label("Europe")

usca = ax[0][1].bar(years, sales_USCA)
usca.set_label("USCA")

africa = ax[0][2].scatter(years, sales_africa)
africa.set_label("Africa")

asia, = ax[1][0].plot(years, sales_Asia_Pacific)
asia.set_label("Asia Pacific")
asia.set_dashes([2, 2, 10, 2])

latam, = ax[1][1].plot(years, sales_LATAM)
latam.set_label("LATAM")
latam.set_dashes([2, 5, 5, 2])


plt.show()

fig, ax = plt.subplots(ncols=4, sharey=True)

europe, = ax[0].plot(years, sales_Europe)
europe.set_label("Europe")
europe.set_color("red")
ax[0].set_title('Sales in Europe')

usca = ax[1].bar(years, sales_USCA)
usca.set_label("USCA")
ax[1].set_title('Sales in USCA')

africa = ax[2].scatter(years, sales_africa)
africa.set_label("Africa")
ax[2].set_title('Sales in Africa')

asia = ax[3].bar(years, sales_Asia_Pacific, width = 0.5, color='royalblue')

latam = ax[3].bar(years, sales_LATAM, width = 0.5, color='seagreen')

ax[3].set_title('Sales in Asia Pacific and LATAM')
ax[3].legend( (asia[0], latam[0]), ('Asia Pacific', 'LATAM') )

fig.set_size_inches(20.5, 5.5, forward=True)

plt.show()



# <a id='Plotting_Exercise_Numpy_Attributes'> Plotting Exercise(Numpy Attributes)</a>

# In[25]:


from IPython.display import HTML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

HTML('<h1 style = "color : Sky blue"> Plotting Exercise(Numpy Attributes)</h1>')

print('''As you learnt in the session, data visualisation is an essential skill for a data scientist to have. Charts and graphs are the easiest way to communicate data to different kinds of consumers. The assessments in this notebook will test your ability to represent information. The questions will be investigative and will not have detailed instructions about the execution; you will have to figure that part out yourself. Don't worry, there are no wrong answers here, only happy accidents. Try these questions on your own, to verify if you have done the task correctly, look at the solution as a person who does not know the data and is looking at it for the first time. You will be able to identify better ways to represent data automatically.

The dataset given for this notebook is the pricing data for diamonds based on the carat, cut, colour and so on. Take a look at the data below and also the data dictionary for a better understanding of the data and the attributes.
''')

# Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the data into a dataframe
data = pd.read_csv("https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/1_Prep_sessions/data_visualization_material/diamonds.csv", header = 0)

# As a preparatory step, let's drop Unnamed column.
#data = data.drop("Unnamed: 0", axis =1 )

data.head()

print('''### Data Dictionary:

1. **carat**: Weight of the diamond (0.2--5.01)

2.  **cut**: Quality of the cut (Fair, Good, Very Good, Premium, Ideal)

3. **color**: Diamond colour, from J (worst) to D (best)

4. **clarity**: A measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))

5. **depth**: Total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)

6. **table**: Width of top of diamond relative to widest point (43--95)

7. **price**: Price in US dollars (326 dollars -18,823 dollars)

8. **x**: Length in mm (0--10.74)

9. **y**: Width in mm (0--58.9)

10. **z**: Depth in mm (0--31.8)
''')

HTML('<h3 style = "color : Brown"> Question </h3>')

print('''Investigate the variation of prices of diamonds and note your observations.''')

print('''###### Solution

You have been asked to investigate the prices of diamonds, not its variation with respect to any other variable. What information can be gained by studying just one variable? Think about it for a moment. Can you find the highest price of a diamond? Yes. You can also find the distribution of prices. What kind of plot can be used to show the distribution of a variable? The best-situated plot is a histogram. You can use a histogram to create 'bins' in the variable and plot the count of data points in each bin. A histogram will give you the distribution of data.
''')

plt.hist(data["price"], bins = 150)

# Adding and formatting title
plt.title("Frequency Plot: Prices of Diamonds\n", fontdict={'fontsize': 20, 'fontweight' : 20, 'color' : 'Green'})

# Labeling Axes
plt.xlabel("Price", fontdict={'fontsize': 12, 'fontweight' : 20, 'color' : 'Brown'})
plt.ylabel("Count", fontdict={'fontsize': 12, 'fontweight' : 20, 'color' : 'Brown'})

plt.show()

print('''Observations from the **histogram**

1. The histogram has a long tail towards the right side.
2. There are two peaks in the data, one near 500-1000 USD and the other near 4000-5000 USD.
''')

HTML('<h3 style = "color : Brown"> Question </h3>')

print('''Perform a similar variation analysis on the quality of cut.''')

print('''##### Solution

The observations in the variable cut are categories, not values like price or carat. So you will not be able to use a histogram to do a count analysis as you did earlier. For this kind of variable, you have to use a bar plot to compare the counts of observations in each category. You can use other plot types like pie chart doughnut chart, but it isn't easy to compare the counts of each class.
''')

# Before plotting extract the required data by performing necessary data manipulation
groupdata = data.groupby("cut").count()
groupdata

# Create lists which hold the necessary data
x_vals = []
y_vals = []
for i in [0, 1, 4, 3, 2]:
    # The elements in the list are arranged in increasing order of the quality of cut.
    # This information can be gathered from the data dictionary.
    x_vals.append(groupdata.index[i])
    y_vals.append(groupdata.iloc[i,0])

plt.bar(x_vals, y_vals)

# Adding and formatting title
plt.title("Frequency Plot: Cut of Diamonds\n", fontdict={'fontsize': 20, 'fontweight' : 20, 'color' : 'Green'})

# Labeling Axes
plt.xlabel("Cut", fontdict={'fontsize': 12, 'fontweight' : 20, 'color' : 'Brown'})
plt.ylabel("Count", fontdict={'fontsize': 12, 'fontweight' : 20, 'color' : 'Brown'})

plt.show()

print('''Observations:
1. The number of diamonds goes on increasing with increasing quality of cut.
''')

HTML('<h3 style = "color : Brown"> Question </h3>')

print('''Represent the same information as above in a pie chart.''')

explode = (0, 0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Very good')
plt.figure(figsize = [8,8])

plt.pie(y_vals, explode=explode, labels=x_vals, autopct='%1.1f%%',
        textprops={'fontsize': 16, 'fontweight' : 20, 'color' : 'Black'}, startangle=90)

# Adding and formatting title
plt.title("Distribution based on quality of cut\n", fontdict={'fontsize': 20, 'fontweight' : 20, 'color' : 'Green'})

plt.show()

print('''As you can see, a pie chart, although visually attractive, is not a good tool to represent information. The changes in the different sectors are not noticeable. It is absolutely necessary to have written values. Now compare the same with the bar diagram earlier, the relative differences in the distribution are very clear even without the data labels.
''')

HTML('<h3 style = "color : Brown"> Question </h3>')

print('''Investigate the variation of price of the diamonds, with respect to the carats.

In cases like these, where you investigate the relationship between two continuous variables, a scatter plot is the best device
''')

plt.scatter(data = data , x = 'carat', y = 'price')
plt.show()

print('''The scatter plot with default settings looks very crowded, that is because there are a lot of data points, there are a couple of ways to solve this problem; you can either draw a smaller sample (you will learn sampling techniques in latter modules) or use smaller and more transparent markers. You can also make the plot larger.
''')

plt.figure(figsize = [14,10])
plt.scatter(data = data , x = 'carat', y = 'price', alpha = 0.5, marker="x")

# Adding and formatting title
plt.title("Weight (in carats) vs Price\n", fontdict={'fontsize': 20, 'fontweight' : 20, 'color' : 'Green'})

# Labeling Axes
plt.xlabel("Weight (in Carats) ", fontdict={'fontsize': 12, 'fontweight' : 20, 'color' : 'Brown'})
plt.ylabel("Price", fontdict={'fontsize': 12, 'fontweight' : 20, 'color' : 'Brown'})

plt.show()

print('''Observations:
1. A general trend observed is as the weight in carats goes on increasing so does the price.
2. There are specific weights to which diamonds are manufactured.
3. There is also a lot of variation in price with the same weight.
''')

HTML('<h3 style = "color : Brown"> Question </h3>')

print('''Investigate the relationship of price with respect to the cut of the diamonds.

As you already know, "cut" is a categorical variable, and it will not be possible to use a scatter plot to analyse the variation. So you can explore the data in a lot of different ways like use colors to the display cut information in the above plot that is an excellent idea, you can try that on your own. You can also divide the data according to the cut and then plot box plots for each of the cuts. Such a plot will allow you to investigate the variation of statistical price data across various cuts.
''')

# Create the required dataframes
data_fair = data[data['cut'] == 'Fair']
data_good = data[data['cut'] == 'Good']
data_vgood = data[data['cut'] == 'Very Good']
data_premium = data[data['cut'] == 'Premium']
data_ideal = data[data['cut'] == 'Ideal']

plt.boxplot([data_fair['price'], data_good['price'], data_vgood['price'], data_premium['price'], data_ideal['price']])
plt.show()

print('''It looks like there are a lot of outliers in the data, which are making the data hard to analyse. Let's drop all the observations **above 12500 USD** as all of them are outliers.
''')

data_fair = data[(data['cut'] == 'Fair') & (data['price'] < 12500) ]
data_good = data[(data['cut'] == 'Good') & (data['price'] < 12500)]
data_vgood = data[(data['cut'] == 'Very Good') & (data['price'] < 12500)]
data_premium = data[(data['cut'] == 'Premium') & (data['price'] < 12500)]
data_ideal = data[(data['cut'] == 'Ideal') & (data['price'] < 12500)]

plt.figure(figsize = [16,10])
plt.boxplot([data_fair['price'], data_good['price'], data_vgood['price'], data_premium['price'], data_ideal['price']])

plt.xticks((1,2,3,4,5), ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])

# Adding and formatting title
plt.title("Price vs Type of Cut\n", fontdict={'fontsize': 20, 'fontweight' : 20, 'color' : 'Green'})

# Labeling Axes
plt.xlabel("Type of Cut", fontdict={'fontsize': 12, 'fontweight' : 20, 'color' : 'Brown'})
plt.ylabel("Price", fontdict={'fontsize': 12, 'fontweight' : 20, 'color' : 'Brown'})

plt.show()

print('''Be cautious while making such modifications because the overall impact on the data is not very well explored yet, it might be possible that the average of different classes is affected differently. Luckily, in this case, the trend before and after the modification looks similar, but again caution is advised.

Observations:
1. The median price of the ideal cut is the least among all the five.
2. There are a lot of outliers in the ideal cut.

Go over all the plots that have been created and make them more presentable.

A few codes in this exercise were more labour intensive than smart. The next library you will study that is seaborn is a much more sophisticated library which will give you all the flexibility you need. You needed to learn two things here, use of matplotlib and the thought process behind visualising data.

Hopefully, you were able to solve all the questions to produce an outcome of the desired level. **Great job!**
''')


# <a id='Practice_questions_footwear_dataset_solution'>Practice_questions_footwear_dataset_solution</a>

# In[26]:


from IPython.display import HTML
import numpy as np
import pandas as pd
import seaborn as sns

HTML('<h1>Footwear</h1>')

print('''The given dataset contains the profits generated(in %) by all the suppliers of a footwear company in 4 major cities of India - Delhi, Mumbai, Jaipur and Hyderabad. The company wants to invest more money in the city that is showing the most promise. Analyse the dataset and answer the following questions.

# Import all the necessary libraries
''')

# Import all the necessary libraries here
import numpy as np
import pandas as pd
import seaborn as sns

# loading data
df=pd.read_csv("https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/1_Prep_sessions/data_visualization_material/Footwear_v2.csv")

df.head()

# we note that there are no null values, and the values are treated as objects and not floats, we will have to clean the 
# '%' sign at the end of all and change it to float
# we will write a function to do this like the last session
def clean(string):
    clean="".join(filter(lambda x: x!='%', string))
    return float(clean)

# you can also use replace
# def clean(val):
#    return float(val.replace("%",""))


# we also see the supplier column has few 'S' as upper case and few lowercase
# lets clean that too
def supply_cleaner(string):
    return string.lower()

# clean the df
df['Supplier']=df['Supplier'].apply(supply_cleaner)
df['Mumbai']=df['Mumbai'].apply(clean)
df['Delhi']=df['Delhi'].apply(clean)
df['Jaipur']=df['Jaipur'].apply(clean)
df['Hyderabad']=df['Hyderabad'].apply(clean)

# ## 1. Average
# Q1)The absolute difference in the average profit percentages of Delhi and Mumbai comes out to be approximately ____

# - a) 1.67

# - b) 1.57

# - **c) 1.77**

# - d) 1.47

# Solution Q1
# We simply call describe on df
df.describe()

#6.324-4.555

HTML('<h3>Question</h3>')

print('''## 2. Box Plots

Plot a box plot to analyse the spread of profits for each of the cities. Which city has the highest profit value at the upper fence in the box plot?

- a) Delhi

- b) Mumbai

- **c) Hyderabad**

- d) Jaipur
''')

# Box plot for cities
sub_df = df[['Delhi', 'Mumbai', 'Jaipur', 'Hyderabad']]
sub_df.boxplot()

print('''# Crypto Currencies

The following datasets contain the prices of some popular cryptocurrencies such as bitcoin, litecoin, ethereum, monero, neo, quantum and ripple.Now, you would like to know how the prices of these currencies vary with each other.
The cryptocurrencies and the corresponding columns in the dataset are as follows:

- bitcoin (Close_btc)
- litecoin(Close_ltc)
- ethereum(Close_et)
- monero(Close_mon)
- neo(Close_neo)

# loading data
df=pd.read_csv("https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/1_Prep_sessions/data_visualization_material/crypto.csv")
''')

# Loading data
df = pd.read_csv("https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/1_Prep_sessions/data_visualization_material/crypto.csv")

HTML('<h3>Question</h3>')

print('''## 1. Correct Statements
Q1)Create a pair plot with all these columns and choose the correct statements from the given ones:

I)There is a good trend between litecoin and monero, one increases as the other

II)There is a weak trend between bitcoin and neo.

- **a)I**

- b)II

- c)Both I and II

- d)None of the above

# Your code here
sns.pairplot(df)

As you can see the correlation between Close_ltc and Close_mon is positive and a good trend

And Close_btc and Close_neo also show a strong trend positive line
''')

# Pair plot for crypto prices
sns.pairplot(df)

HTML('<h3>Question</h3>')

print('''## 2. Heatmap
Q2)As mentioned earlier, Heat Maps are predominantly utilised for analysing Correlation Matrix. A high positive correlation (values near 1) means a good positive trend - if one increases, then the other also increases. A negative correlation on the other hand(values near -1) indicate good negative trend - if one increases, then the other decreases. A value near 0 indicates no correlation, as in one variable doesn’t affect the other.

##### Correlation Matrix
Here, you can create a correlation matrix of the closing prices by passing the **df.corr()** function, (where df is the dataframe's name) and storing it in a variable. The code will be as follows

df2 = df.corr()

After that use this variable df2 to plot a heatmap and choose the correct option. 

Check out this link for creating a correlation matrix: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html

Create a correlation matrix for all the prices and then plot a heatmap to analyse the trends. Which of the following options is/are correct?

- **a)Ethereum and Quantum have high correlation**

- b)Neo and Bitcoin have pretty low correlation

- **c)Ethereum has similar correlation with litecoin and neo**

# Solution
df2 = df.corr()
sns.heatmap(df2, cmap="Greens", annot=True)

Close_et and Close_qt have high corr (0.79)

Close_neo and Close_btc have a high corr (0.73)

Close_et and Close_ltc have corr of (0.49)

Close_et and Close_neo have a corr of (0.48)
''')

# Correlation matrix and heatmap for crypto prices
df2 = df.corr()
sns.heatmap(df2, cmap="Greens", annot=True)


# <a id='Case_Study_Google_Playstore'>Case_Study_Google_Playstore</a>

# In[27]:


from IPython.display import display, HTML, Markdown
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

# Displaying HTML content
display(HTML('<h1>Google Playstore Case Study</h1>'))
display(Markdown('''
In this module, you’ll be learning data visualisation with the help of a case study. This will enable you to understand how visualisation aids you in solving business problems.

**Problem Statement**

The team at Google Play Store wants to develop a feature that would enable them to boost visibility for the most promising apps. Now, this analysis would require a preliminary understanding of the features that define a well-performing app. You can ask questions like:
- Does a higher size or price necessarily mean that an app would perform better than the other apps?
- Or does a higher number of installs give a clear picture of which app would have a better rating than others?

### Session 1 - Introduction to Data Visualisation
'''))

# Import the libraries
import pandas as pd
import numpy as np

# Read the dataset and check the first five rows
inp0 = pd.read_csv("https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/1_Prep_sessions/data_visualization_material/googleplaystore_v2.csv")
display(inp0.head())

# Check the shape of the dataframe
display(inp0.shape)

display(Markdown('''
### Data Handling and Cleaning

The first few steps involve making sure that there are no __missing values__ or __incorrect data types__ before we proceed to the analysis stage. These aforementioned problems are handled as follows:

- For Missing Values: Some common techniques to treat this issue are
  - Dropping the rows containing the missing values
  - Imputing the missing values
  - Keep the missing values if they don't affect the analysis

- Incorrect Data Types:
  - Clean certain values
  - Clean and convert an entire column
'''))

# Check the datatypes of all the columns of the dataframe
display(inp0.info())

#### Missing Value Treatment
# Check the number of null values in the columns
display(inp0.isnull().sum())

# Handling missing values for rating
# - Ratings is the target variable
# - drop the records

# Drop the rows having null values in the Rating field
inp1 = inp0[~inp0.Rating.isnull()]

# Check the shape of the dataframe
display(inp1.shape)

# Check the number of nulls in the Rating field again to cross-verify
display(inp1.Rating.isnull().sum())

# Check the number of nulls in the dataframe again and find the total number of null values
display(inp1.isnull().sum())

# Inspect the nulls in the Android Version column
display(inp1[inp1['Android Ver'].isnull()])

# Drop the row having shifted values
display(inp1.loc[10472,:])
display(inp1[(inp1['Android Ver'].isnull() & (inp1.Category == "1.9"))])
inp1 = inp1[~(inp1['Android Ver'].isnull() & (inp1.Category == "1.9"))]
# Check the nulls again in Android version column to cross-verify
display(inp1[inp1['Android Ver'].isnull()])

# Imputing Missing Values

# - For numerical variables use mean and median
# - For categorical variables use mode

# Check the most common value in the Android version column
display(inp1['Android Ver'].value_counts())

# Fill up the nulls in the Android Version column with the above value
inp1['Android Ver'] = inp1['Android Ver'].fillna(inp1['Android Ver'].mode()[0])

# Check the nulls in the Android version column again to cross-verify
display(inp1['Android Ver'].isnull().sum())

# Check the nulls in the entire dataframe again
display(inp1.isnull().sum())

# Check the most common value in the Current version column
display(inp1['Current Ver'].value_counts())

# Replace the nulls in the Current version column with the above value
inp1['Current Ver'] = inp1['Current Ver'].fillna(inp1['Current Ver'].mode()[0])

# Question : Check the most common value in the Current version column again
display(inp1['Current Ver'].value_counts())

#### Handling Incorrect Data Types
# Check the datatypes of all the columns
display(inp1.dtypes)

# Question - Try calculating the average price of all apps having the Android version as "4.1 and up"
display(inp1.head())

# Analyse the Price column to check the issue
display(inp1.Price.value_counts())

# Write the function to make the changes
inp1.Price = inp1.Price.apply(lambda x: 0 if x=="0" else float(x[1:]))

# Verify the dtype of Price once again
display(inp1.Price.dtype)

# Analyse the Reviews column
display(inp1.Reviews.value_counts())

# Change the dtype of this column
inp1.Reviews = inp1.Reviews.astype("int32")
# Check the quantitative spread of this dataframe
display(inp1.Reviews.describe())

# Analyse the Installs Column
display(inp1.Installs.head())

# Question Clean the Installs Column and find the approximate number of apps at the 50th percentile.
def clean_installs(val):
    return int(val.replace(",","").replace("+",""))
type(clean_installs("3,000+"))
inp1.Installs = inp1.Installs.apply(clean_installs)
display(inp1.Installs.describe())

#### Sanity Checks

# The data that we have needs to make sense and therefore you can perform certain sanity checks on them to ensure they are factually correct as well. Some sanity checks can be:

# - Rating is between 1 and 5 for all the apps.
# - Number of Reviews is less than or equal to the number of Installs.
# - Free Apps shouldn’t have a price greater than 0.

# Perform the sanity checks on the Reviews column
display(inp1[(inp1.Reviews > inp1.Installs)].shape)
display(inp1[(inp1.Reviews > inp1.Installs)])
inp1 = inp1[inp1.Reviews <= inp1.Installs]

# Perform the sanity checks on prices of free apps
display(inp1[(inp1.Type == "Free") & (inp1.Price>0)])

#### Outliers Analysis Using Boxplot

display(Markdown('''
Now you need to start identifying and removing extreme values or __outliers__ from our dataset. These values can tilt our analysis and often provide us with a biased perspective of the data available. This is where you’ll start utilising visualisation to achieve your tasks. And the best visualisation to use here would be the box plot. Boxplots are one of the best ways of analysing the spread of a numeric variable.

Using a box plot you can identify the outliers as follows:

![BoxPlots to Identify Outliers](https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/1_Prep_sessions/data_visualization_material/images/Boxplot.png)

- Outliers in data can arise due to genuine reasons or because of dubious entries. In the latter case, you should go ahead and remove such entries immediately. Use a boxplot to observe, analyse and remove them.
- In the former case, you should determine whether or not removing them would add value to your analysis procedure.

- You can create a box plot directly from pandas dataframe or the matplotlib way as you learnt in the previous session. Check out their official documentation here:
   - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.boxplot.html
   - https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.boxplot.html
'''))

# Create a box plot for the price column
plt.boxplot(inp1.Price)
plt.show()

# Check the apps with price more than 200
display(inp1[inp1.Price > 200])

# Clean the Price column
inp1 = inp1[inp1.Price < 200]
display(inp1.Price.describe())

# Create a box plot for paid apps
inp1[inp1.Price>0].Price.plot.box()
plt.show()

# Check the apps with price more than 30
display(inp1[inp1.Price>30])

# Clean the Price column again
inp1 = inp1[inp1.Price <= 30]
display(inp1.shape)

#### Histograms

display(Markdown('''
Histograms can also be used in conjunction with boxplots for data cleaning and data handling purposes. You can use it to check the spread of a numeric variable. Histograms generally work by bucketing the entire range of values that a particular variable takes to specific __bins__. After that, it uses vertical bars to denote the total number of records in a specific bin, which is also known as its __frequency__.

![Histogram](https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/1_Prep_sessions/data_visualization_material/images/Histogram.png)

You can adjust the number of bins to improve its granularity

![Bins change](https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/1_Prep_sessions/data_visualization_material/images/Granular.png)

You'll be using plt.hist() to plot a histogram. Check out its official documentation:https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.hist.html
'''))

# Create a histogram of the Reviews
plt.hist(inp1.Reviews)
plt.show()

# Create a boxplot of the Reviews column
plt.boxplot(inp1.Reviews)
plt.show()

# Check records with 1 million reviews
display(inp1[inp1.Reviews >= 1000000])

# Drop the above records
inp1 = inp1[inp1.Reviews <= 1000000]
display(inp1.shape)

# Question - Create a histogram again and check the peaks
plt.hist(inp1.Reviews)
plt.show()

# Question - Create a box plot for the Installs column and report back the IQR
plt.boxplot(inp1.Installs)
plt.show()

display(inp1.Installs.describe())

# Question - Clean the Installs by removing all the apps having more than or equal to 100 million installs
inp1 = inp1[inp1.Installs <= 100000000]
display(inp1.shape)

# Plot a histogram for Size as well.
inp1.Size.plot.hist()
plt.show()

display(inp1.Size.describe())

# Question - Create a boxplot for the Size column and report back the median value
plt.boxplot(inp1.Size)
plt.show()


# <a id='Case_study_cricket'>Case_study_cricket</a>

# In[28]:


from IPython.display import display, HTML, Markdown
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Display introductory HTML content
display(HTML('<h1>Virat Kohli Dataset</h1>'))

# Displaying the introduction and problem statement
display(Markdown('''
### I - Virat Kohli Dataset

The dataset contains information about the runs scored by Virat Kohli in various matches. We will analyze this data to answer the following questions.

**Question 1:** Analyse the spread of Runs scored by Virat in all his matches and report the difference between the scores at the 50th percentile and the 25th percentile respectively.

    a) 16.5
    b) 22.5
    c) 26.5
    d) 32.5
'''))

# Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv("https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/1_Prep_sessions/data_visualization_material/virat.csv")

# Display the first few rows of the dataframe
display(df.head())

# Clean the Runs column
df['Runs'] = df['Runs'].apply(lambda x: int(x[:-1]) if x[-1] == "*" else int(x))

# Use the describe function to get percentiles
describe_df = df['Runs'].describe(percentiles=[0.25, 0.5, 1])
display(describe_df)

# Calculate the difference between the 50th percentile and the 25th percentile
diff = describe_df['50%'] - describe_df['25%']
display(Markdown(f"**Answer:** The difference between the 50th percentile and the 25th percentile is {diff}"))

display(Markdown('''
**Question 2:** Plot a Box Plot to analyse the spread of Runs that Virat has scored. The upper fence in the box plot lies in which interval?

    a) 100-120
    b) 120-140
    c) 140-160
    d) 160-180
'''))

# Box plot for Runs
plt.boxplot(df.Runs)
plt.title("Box Plot of Runs")
plt.ylabel("Runs")
plt.show()

display(Markdown('''
**Question 3:** Consider the following statements and choose the correct option

     I - Virat has played the maximum number of matches in 2011
     II - Virat has the highest run average in the year 2017
     III - Virat has the maximum score in a single match and the highest run average in the year 2016.

Which of the above statements is/are false?

    a) I and II
    b) I and III
    c) II
    d) III
'''))

# Extract the year from the Start Date column
df['Start Date'] = df['Start Date'].apply(lambda x: (x[-2:]))

# For verifying statement 1
year_counts = df['Start Date'].value_counts()
display(year_counts)

# For verifying statement 2
avg_runs_per_year = pd.pivot_table(df, values='Runs', columns=['Start Date'], aggfunc=np.mean)
display(avg_runs_per_year)

# For verifying statement 3
max_runs_per_year = pd.pivot_table(df, values='Runs', columns=['Start Date'], aggfunc=np.max)
display(max_runs_per_year)

display(Markdown('''
**Question 4:** Plot a histogram for the Mins column with 15 bins. Among the three ranges mentioned below, which one has the highest frequency?

A - [54.6,68)

B - [68,81.4)

C - [121.6,135)

    a) A - [54.6,68)
    b) B - [68,81.4)
    c) C - [121.6,135)
    d) All the bin ranges have the same frequency
'''))

# Clean the Mins column
df2 = df[~(df['Mins'] == "-")]
df2['Mins'] = df2['Mins'].apply(lambda x: int(x))

# Plotting the histogram
plt.hist(df2.Mins, bins=15)
plt.title("Histogram of Mins")
plt.xlabel("Mins")
plt.ylabel("Frequency")
plt.show()

# Clearly, C [121.6 - 135) has the highest frequency
display(Markdown("**Answer:** Clearly, C [121.6 - 135) has the highest frequency"))


# In[29]:


from datetime import datetime
import pytz

# Define the IST timezone
ist = pytz.timezone('Asia/Kolkata')

# Get the current time in UTC
utc_now = datetime.now(pytz.utc)

# Convert the current time to IST
ist_now = utc_now.astimezone(ist)

# Print the current time in IST
print("Current Time in IST:", ist_now.strftime('%Y-%m-%d %H:%M:%S'))


# In[ ]:




